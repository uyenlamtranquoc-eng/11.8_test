# type: ignore  # Ignore all type checking for traci
import os
import sys
import importlib
import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces

# 延迟选择后端：优先在 _init_sumo 中按 use_gui/可用性决定
try:
    import traci as _traci_mod  # type: ignore
except Exception:
    _traci_mod = None
traci = _traci_mod  # type: ignore


class CityVSLEnv(gym.Env):
    """
    城市走廊（单向，3个信号）可变限速的RL环境
    - 分段限速：upstream / mid / down 三段
    - 只控制 CAV：typeID == "CAV" 的车
    - 每 decision_interval 秒做一次决策
    - 状态包含：当前运行量 + 信号相位 + CAV渗透率 + 简单预测量（对应论文的state extension）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sumo_cfg_path: str,
        use_gui: bool = False,
        backend: str = "auto",  # auto/libsumo/traci
        lane_groups: Optional[Dict[str, List[str]]] = None,
        lane_groups_wb: Optional[Dict[str, List[str]]] = None,
        tls_ids: Optional[List[str]] = None,
        discrete_speeds_kph: Optional[List[int]] = None,
        decision_interval: int = 5,
        cav_type_id: str = "CAV",
        max_sim_seconds: int = 3600,
        max_delta_kph: float = 10.0,  # 限速变化硬约束，论文式
        reward_speed_weight: float = 1.0,
        reward_congestion_weight: float = 1.0,
        reward_change_penalty: float = 0.1,
        warmup_steps: int = 5,
        reward_clip_min: Optional[float] = None,
        reward_clip_max: Optional[float] = None,
    ):
        super().__init__()

        self.sumo_cfg_path = sumo_cfg_path
        self.use_gui = use_gui
        self.backend_pref = (backend or "auto").lower()
        self.decision_interval = int(decision_interval)
        self.cav_type_id = cav_type_id
        self.max_sim_seconds = max_sim_seconds
        self.max_delta_kph = max_delta_kph
        self.warmup_steps = int(warmup_steps)

        # 评估指标累计器（按仿真步聚合）
        self.metrics = {
            "queue_sum": 0.0,
            "queue_steps": 0,
            "arrived_total": 0,
            "sim_seconds": 0,
        }

        # 三段路（东向）
        self.lane_groups = lane_groups or {
            "upstream": [],
            "mid": [],
            "down": [],
        }
        self.group_keys = ["upstream", "mid", "down"]

        # 三段路（西向，可选）：与 group_keys 对齐键名
        self.wb_lane_groups = lane_groups_wb or {
            "upstream": [],
            "mid": [],
            "down": [],
        }

        # 三个信号
        self.tls_ids = tls_ids or ["J0", "J1", "J2"]

        # 离散车速
        self.discrete_speeds_kph = discrete_speeds_kph or [30, 35, 40, 45, 50, 55]
        self.speed_levels_mps = [v / 3.6 for v in self.discrete_speeds_kph]

        # 构建动作空间：6 个独立离散动作（每段一档）
        n_per_segment = len(self.speed_levels_mps) if self.speed_levels_mps else 1
        self.action_space = spaces.MultiDiscrete(np.array([n_per_segment] * 6, dtype=np.int64))

        # 观测空间简化（城市VSL基础版）：
        # 东向3段 + 西向3段：每段 occ + speed = 2 * 6 = 12
        # 每信号: phase + remaining = 2 * 3 = 6
        # CAV ratio = 1
        # 总计: 12 + 6 + 1 = 19
        self.obs_dim = 2 * len(self.group_keys) * 2 + 2 * len(self.tls_ids) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # reward 权重：对应论文的多目标
        self.rw = {
            "efficiency": float(reward_speed_weight),      # 整体速度（双向平均）
            "congestion": float(reward_congestion_weight), # 停车/排队（双向合并）
            "action_change": float(reward_change_penalty), # 动作变化（硬约束下保持轻惩罚）
        }

        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max

        # 归一化参考：速度按离散档位最大值（至少 20 m/s）；信号按实际程序配置
        self.max_ref_speed_mps = max(20.0, max(self.speed_levels_mps)) if self.speed_levels_mps else 20.0
        self._tls_phase_meta: Dict[str, Dict[str, float]] = {}
        self._green_phase_cache: Dict[str, set] = {}

        # 仿真时间（秒，按 SUMO 实时累计）
        self.sim_time = 0.0
        self._prev_time_s = 0.0
        self._last_action_index = -1
        # 初始动作速度：取第0档（若存在），否则 20 m/s 兜底
        init_speed = (self.speed_levels_mps[0] if self.speed_levels_mps else 20.0)
        self._last_action_speeds = (init_speed,) * 6  # m/s（东向3段 + 西向3段）

        # 控制集追踪与默认速度缓存（用于避免“限速残留”）
        self._controlled_veh_ids: set = set()
        self._veh_type_max_speed: Dict[str, float] = {}

        # 决策周期语义：按秒；运行时按 step-length 折算为仿真步数
        self._decision_interval_s: float = float(self.decision_interval)

        # 每步指标缓存：减少 reward 与 _observe 重复的 TraCI 读取
        # 结构示例：
        # {
        #   "lane_metrics": {
        #       "eb": {"upstream": {"speed": 10.0, "halts": 0.5, "occ": 0.2}, ...},
        #       "wb": {"upstream": {"speed": 9.5,  "halts": 0.7, "occ": 0.25}, ...}
        #   }
        # }
        self._step_cache: Dict[str, Dict] = {}

        # 每次运行使用独立时间戳子目录：outputs/YYYYMMDD_HHMMSS
        self.run_output_dir = self._make_run_output_dir()
        self._run_additional_path = None

        # 起 SUMO
        self._init_sumo()

    # --------------------------------------------------------
    def _init_sumo(self):
        if "SUMO_HOME" not in os.environ or not os.environ.get("SUMO_HOME"):
            raise RuntimeError("Please set SUMO_HOME before running the environment.")
        # 确保 tools 在 sys.path 以支持 traci 导入
        tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)

        # 选择后端：GUI 强制使用 TraCI+sumo-gui；非 GUI 优先 libsumo（失败回退 TraCI）
        global traci
        backend_used = None
        # 使用原始 sumocfg；为本次运行生成 additional 覆盖文件，将 e2 输出重定向到运行目录
        sumocfg_to_use = self.sumo_cfg_path
        try:
            self._run_additional_path = self._write_run_additional_file()
        except Exception:
            self._run_additional_path = None
        # 日志文件：将 SUMO 控制台消息写入运行目录
        message_log_path = os.path.join(self.run_output_dir, "sumo_messages.log")

        if self.use_gui:
            traci = importlib.import_module("traci")
            sumo_cmd = ["sumo-gui", "-c", sumocfg_to_use, "--start"]
            # 写入 SUMO 日志到运行目录并静音控制台
            sumo_cmd += [
                "--message-log", message_log_path,
                "--log", message_log_path,
                "--error-log", message_log_path,
                "--no-warnings", "true",
                "--no-step-log", "true",
            ]
            # 覆盖 additional 文件（若生成成功）
            if self._run_additional_path:
                sumo_cmd += ["--additional-files", self._run_additional_path]
            traci.start(sumo_cmd)
            backend_used = "traci-gui"
        else:
            pref = self.backend_pref
            if pref in ("auto", "libsumo"):
                try:
                    traci = importlib.import_module("libsumo")
                    # libsumo 使用 load(...) 载入仿真，无需指定外部进程名
                    load_args = ["-c", sumocfg_to_use, "--start"]
                    load_args += [
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ]
                    if self._run_additional_path:
                        load_args += ["--additional-files", self._run_additional_path]
                    if hasattr(traci, "isLoaded") and traci.isLoaded():
                        traci.close()
                    traci.load(load_args)
                    backend_used = "libsumo"
                except Exception:
                    traci = importlib.import_module("traci")
                    sumo_cmd = ["sumo", "-c", sumocfg_to_use, "--start"]
                    sumo_cmd += [
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ]
                    if self._run_additional_path:
                        sumo_cmd += ["--additional-files", self._run_additional_path]
                    traci.start(sumo_cmd)
                    backend_used = "traci"
            elif pref == "traci":
                traci = importlib.import_module("traci")
                sumo_cmd = ["sumo", "-c", sumocfg_to_use, "--start"]
                sumo_cmd += [
                    "--message-log", message_log_path,
                    "--log", message_log_path,
                    "--error-log", message_log_path,
                    "--no-warnings", "true",
                    "--no-step-log", "true",
                ]
                if self._run_additional_path:
                    sumo_cmd += ["--additional-files", self._run_additional_path]
                traci.start(sumo_cmd)
                backend_used = "traci"
            else:
                # 未识别：回到 auto
                try:
                    traci = importlib.import_module("libsumo")
                    # 与上面保持一致，libsumo 使用 load(...) 而不是 start(...)
                    load_args = ["-c", sumocfg_to_use, "--start"]
                    load_args += [
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ]
                    if self._run_additional_path:
                        load_args += ["--additional-files", self._run_additional_path]
                    if hasattr(traci, "isLoaded") and traci.isLoaded():
                        traci.close()
                    traci.load(load_args)
                    backend_used = "libsumo"
                except Exception:
                    traci = importlib.import_module("traci")
                    traci.start([
                        "sumo", "-c", sumocfg_to_use, "--start",
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ] + (["--additional-files", self._run_additional_path] if self._run_additional_path else []))
                    backend_used = "traci"
        # 不在控制台打印后端信息；如需调试，可写入消息日志
        # 缓存信号相位元数据用于归一化
        self._compute_tls_phase_meta()

    def _compute_tls_phase_meta(self) -> None:
        self._tls_phase_meta = {}
        for tid in self.tls_ids:
            try:
                # 选择当前程序对应的逻辑，避免弃用 API 警告
                try:
                    curr_prog = traci.trafficlight.getProgram(tid)
                except Exception:
                    curr_prog = None
                try:
                    logics = traci.trafficlight.getAllProgramLogics(tid)
                except Exception:
                    logics = []

                logic = None
                if logics:
                    if curr_prog is not None:
                        for L in logics:
                            pid = L.getProgramID() if hasattr(L, "getProgramID") else getattr(L, "programID", None)
                            if pid == curr_prog:
                                logic = L
                                break
                    if logic is None:
                        logic = logics[0]

                phase_count = 0
                max_dur = 1.0
                green_phases = set()
                if logic is not None:
                    phases = logic.getPhases() if hasattr(logic, "getPhases") else getattr(logic, "phases", [])
                    phase_count = len(phases) if phases else 0
                    if phases:
                        try:
                            max_dur = max(
                                float(
                                    getattr(p, "duration", 0.0)
                                    if hasattr(p, "duration") else getattr(p, "maxDur", 0.0)
                                )
                                for p in phases
                            ) or 1.0
                        except Exception:
                            max_dur = 60.0
                        # 识别绿灯相位：相位定义中包含'G'
                        for idx, p in enumerate(phases):
                            state = getattr(p, "state", "") if hasattr(p, "state") else str(p)
                            if 'G' in state or 'g' in state:
                                green_phases.add(idx)
                if phase_count <= 0:
                    phase_count = max(1, int(traci.trafficlight.getPhase(tid)) + 1)
                    remain = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
                    max_dur = max(1.0, float(remain))
                self._tls_phase_meta[tid] = {
                    "phase_count": float(phase_count), 
                    "max_phase_duration": float(max_dur)
                }
                self._green_phase_cache[tid] = green_phases
            except traci.TraCIException:
                self._tls_phase_meta[tid] = {"phase_count": 8.0, "max_phase_duration": 60.0}
                self._green_phase_cache[tid] = set()
            except Exception:
                self._tls_phase_meta[tid] = {"phase_count": 8.0, "max_phase_duration": 60.0}
                self._green_phase_cache[tid] = set()
    
    def _is_green_phase(self, tls_id: str, phase_idx: int) -> bool:
        """判断指定信号的指定相位是否为绿灯。"""
        return phase_idx in self._green_phase_cache.get(tls_id, set())

    # --------------------------------------------------------
    def _make_run_output_dir(self) -> str:
        """创建并返回运行时输出目录：<repo_root>/outputs/YYYYMMDD_HHMMSS"""
        proj_root = os.path.abspath(os.path.join(os.path.dirname(self.sumo_cfg_path), ".."))
        out_root = os.path.join(proj_root, "outputs")
        os.makedirs(out_root, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_root, stamp)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _get_sumocfg_additional_files_value(self) -> Optional[str]:
        try:
            with open(self.sumo_cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "<additional-files" in line and "value=" in line:
                        q1 = line.find('"')
                        q2 = line.find('"', q1 + 1)
                        if q1 != -1 and q2 != -1:
                            return line[q1 + 1:q2]
        except Exception:
            return None
        return None

    def _write_run_additional_file(self) -> Optional[str]:
        """生成 additional.generated.xml，覆盖所有 laneAreaDetector 的 file 指向运行目录。"""
        add_rel = self._get_sumocfg_additional_files_value()
        if not add_rel:
            return None
        base_dir = os.path.dirname(self.sumo_cfg_path)
        src_path = os.path.join(base_dir, add_rel)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None
        e2_out_path = os.path.join(self.run_output_dir, "grid1x3_e2.out.xml")
        import re
        content_new = re.sub(r'file="[^\"]+"', f'file="{e2_out_path}"', content)
        out_path = os.path.join(self.run_output_dir, "additional.generated.xml")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content_new)
            return out_path
        except Exception:
            return None

    # --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        self._init_sumo()

        # 重置仿真时间累计（使用 SUMO 的真实时间）
        self.sim_time = 0.0
        self._prev_time_s = 0.0
        self._last_action_index = -1
        init_speed = (self.speed_levels_mps[0] if self.speed_levels_mps else 20.0)
        self._last_action_speeds = (init_speed,) * 6

        # 清空评估累计
        self.metrics = {
            "queue_sum": 0.0,
            "queue_steps": 0,
            "arrived_total": 0,
            "sim_seconds": 0,
        }
        self._controlled_veh_ids.clear()
        self._veh_type_max_speed.clear()

        # 暖机步数：推进仿真但不计入指标
        for _ in range(self.warmup_steps):
            traci.simulationStep()
        # 暖机完成后，以 SUMO 当前时间初始化累计
        try:
            self.sim_time = float(traci.simulation.getTime())
        except Exception:
            self.sim_time = float(self.warmup_steps)
        self._prev_time_s = self.sim_time

        obs = self._observe()
        return obs, {}

    # --------------------------------------------------------
    def step(self, action):
        """
        标准step方法，包含基本的错误处理和内存管理
        
        主要改进：
        1. 基本TraCI异常处理
        2. 内存泄漏防护和性能监控
        3. 动作历史记录（有限长度）
        """
        # 1. MultiDiscrete 动作解析与越界保护：期望长度为 6 的整数向量
        try:
            indices = np.array(action, dtype=np.int64).reshape(-1)
        except Exception as e:
            print(f"[错误] 动作解析失败: {e}")
            indices = np.zeros(6, dtype=np.int64)
        if indices.size != 6:
            indices = np.pad(indices, (0, max(0, 6 - indices.size)), mode="constant")[:6]
        n_per_segment = len(self.speed_levels_mps) if self.speed_levels_mps else 1
        indices = np.clip(indices, 0, max(0, n_per_segment - 1))

        # 2. 取目标速度（东向3段 + 西向3段），按档位映射到 m/s
        #    indices[0:3] -> EB 三段；indices[3:6] -> WB 三段
        try:
            target_speeds = [self.speed_levels_mps[int(i)] for i in indices.tolist()]  # m/s
        except Exception as e:
            print(f"[错误] 速度映射失败: {e}")
            target_speeds = [20.0] * 6  # 默认速度

        # 3. 做论文式"限速变化≤Δv"的硬约束（按段，6 段）
        max_delta_mps = self.max_delta_kph / 3.6
        prev_speeds = self._last_action_speeds
        clipped_speeds = []
        for prev_v, new_v in zip(prev_speeds, target_speeds):
            if abs(new_v - prev_v) > max_delta_mps:
                if new_v > prev_v:
                    new_v = prev_v + max_delta_mps
                else:
                    new_v = prev_v - max_delta_mps
            clipped_speeds.append(new_v)
        target_speeds = tuple(clipped_speeds)

        ep_reward = 0.0

        # 重置本决策周期的每步缓存
        self._step_cache = {}

        # 4. 决策周期内：以目标时间 self.sim_time + decision_interval 为准，逐步推进至该目标
        #    这样避免依赖 deltaT 的不确定性（某些平台返回异常值导致步数过大）
        target_end_time = float(self.sim_time) + float(self._decision_interval_s)
        safety_iter = 0
        while True:
            if float(self.sim_time) >= target_end_time:
                break
            safety_iter += 1
            if safety_iter > 10000:
                # 安全退出：防止异常 deltaT 或仿真停滞导致死循环
                print(f"[警告] 仿真步数超限: {safety_iter}, 提前结束决策周期")
                break
            
            # 4.1 下控制：只控 CAV；东向与西向各用自己的3段速度
            eb_speeds = target_speeds[0:3]
            wb_speeds = target_speeds[3:6]
            curr_controlled = set()
            
            # 东向：分段应用限速，同时记录进入控制集的车辆并缓存其类型默认最大速度
            controlled_count_eb = 0
            for gname, target_speed in zip(self.group_keys, eb_speeds):
                for lid in self.lane_groups.get(gname, []):
                    try:
                        vehs = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vehs = []
                    
                    for vid in vehs:
                        try:
                            if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                                curr_controlled.add(vid)
                                controlled_count_eb += 1
                                if vid not in self._veh_type_max_speed:
                                    try:
                                        vtype = traci.vehicle.getTypeID(vid)
                                        tmax = float(traci.vehicletype.getMaxSpeed(vtype))
                                    except Exception:
                                        try:
                                            tmax = float(traci.vehicle.getMaxSpeed(vid))
                                        except Exception:
                                            tmax = float(target_speed)
                                    self._veh_type_max_speed[vid] = tmax
                                traci.vehicle.setMaxSpeed(vid, float(target_speed))
                        except Exception:
                            pass  # 简化错误处理，忽略单个车辆设置失败
            
            # 西向：同理
            controlled_count_wb = 0
            for gname, target_speed in zip(self.group_keys, wb_speeds):
                for lid in self.wb_lane_groups.get(gname, []):
                    try:
                        vehs = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vehs = []
                    
                    for vid in vehs:
                        try:
                            if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                                curr_controlled.add(vid)
                                controlled_count_wb += 1
                                if vid not in self._veh_type_max_speed:
                                    try:
                                        vtype = traci.vehicle.getTypeID(vid)
                                        tmax = float(traci.vehicletype.getMaxSpeed(vtype))
                                    except Exception:
                                        try:
                                            tmax = float(traci.vehicle.getMaxSpeed(vid))
                                        except Exception:
                                            tmax = float(target_speed)
                                    self._veh_type_max_speed[vid] = tmax
                                traci.vehicle.setMaxSpeed(vid, float(target_speed))
                        except Exception:
                            pass  # 简化错误处理，忽略单个车辆设置失败
            
            # 调试输出：每100步打印一次控制信息
            # 4.1.1 恢复：离开控制区的车辆恢复默认类型最大速度，避免"限速残留"
            exited = self._controlled_veh_ids - curr_controlled
            for vid in list(exited):
                try:
                    default_v = float(self._veh_type_max_speed.pop(vid, 27.78))
                    traci.vehicle.setMaxSpeed(vid, default_v)
                except traci.TraCIException as e:
                    print(f"[TraCI异常] 恢复车辆速度失败(vid={vid}): {e}")
                except Exception as e:
                    print(f"[异常] 恢复车辆速度失败(vid={vid}): {e}")
                finally:
                    self._controlled_veh_ids.discard(vid)
            # 更新当前控制集
            self._controlled_veh_ids = curr_controlled

            # 推进 1 个仿真步（基本错误处理）
            try:
                traci.simulationStep()
            except Exception as e:
                print(f"[警告] 仿真步失败: {e}")
                # 简单错误处理，不尝试重启
                break
            
            # 更新时间与步长（秒）
            try:
                t_now = float(traci.simulation.getTime())
            except Exception:
                t_now = self.sim_time + 1.0  # 使用估计时间
            
            dt_s = max(0.0, t_now - self._prev_time_s)
            self.sim_time = t_now
            self._prev_time_s = t_now

            # —— 评估累计：队列、能耗、距离、到达、时间损失 ——
            # 1) 队列：各段车道的"停止车辆数"（双向合并）
            queue_count = 0.0
            for gname in self.group_keys:
                lane_ids = self.lane_groups.get(gname, []) + self.wb_lane_groups.get(gname, [])
                for lid in lane_ids:
                    try:
                        queue_count += float(traci.lane.getLastStepHaltingNumber(lid))
                    except Exception:
                        pass  # 简化错误处理，忽略单个检测器失败
            self.metrics["queue_sum"] += queue_count
            self.metrics["queue_steps"] += 1

            # 2) 车辆能耗 / 距离 / 时间损失（删除，不参与奖励计算）

            # 3) 本步到达车辆：仅计数
            try:
                arrived_ids = traci.simulation.getArrivedIDList()
            except Exception:
                arrived_ids = []
                
            for vid in arrived_ids:
                self.metrics["arrived_total"] += 1

            # 4) 仿真时长累计（秒）
            self.metrics["sim_seconds"] += dt_s

            # 计算奖励（基本错误处理）
            try:
                step_reward = self._compute_reward_per_step()
                ep_reward += step_reward
            except Exception:
                ep_reward += 0.0  # 使用默认奖励

        # 5. 决策结束：加一次动作变化惩罚
        try:
            penalty = self._compute_action_change_penalty(tuple(target_speeds))
            ep_reward += penalty
        except Exception:
            pass  # 忽略惩罚计算失败

        # 6. 更新last
        self._last_action_index = 0  # 仅用于"首次不惩罚"的语义
        self._last_action_speeds = tuple(target_speeds)

        # 7. 观测 & done
        try:
            # 在生成观测前补齐占有率到缓存，避免 _observe 重复读取
            try:
                self._collect_lane_group_metrics(need_occ=True)
            except Exception:
                pass
            obs = self._observe()
        except Exception:
            obs = np.zeros(self.obs_dim, dtype=np.float32)  # 返回默认观测
            
        # 以"真实秒"为终止标准
        terminated = float(self.sim_time) >= float(self.max_sim_seconds)
        truncated = False

        info = {
            "sim_time": self.sim_time,
            "action_speeds_mps": tuple(target_speeds),
            "raw_action_levels": indices.tolist(),
        }

        return obs, ep_reward, terminated, truncated, info

    # --------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        """
        返回当前累计指标的汇总：
        - avg_queue_veh: 平均排队车辆数（辆）= 每步停止车辆数平均
        - throughput_veh_per_hour: 通行流量（辆/小时）= 到达数 / 时长 * 3600
        """
        q_steps = self.metrics.get("queue_steps", 0)
        sim_sec = self.metrics.get("sim_seconds", 0)
        arrived = self.metrics.get("arrived_total", 0)

        avg_queue = (self.metrics.get("queue_sum", 0.0) / q_steps) if q_steps > 0 else 0.0
        throughput = (arrived / sim_sec * 3600.0) if sim_sec > 0 else 0.0

        return {
            "avg_queue_veh": avg_queue,
            "throughput_veh_per_hour": throughput,
            "arrived_total": arrived,
            "sim_seconds": sim_sec,
        }

    # --------------------------------------------------------
    def _collect_lane_group_metrics(self, need_occ: bool = True) -> None:
        """采集并缓存每个车道组（东/西向）的平均速度、停止车辆数、占有率。

        参数：
        - need_occ: 是否采集占有率；奖励仅需速度/停止车辆数时可以为 False。
        """
        lm = self._step_cache.setdefault("lane_metrics", {"eb": {}, "wb": {}})

        for g in self.group_keys:
            # 东向
            eb_lanes = self.lane_groups.get(g, [])
            eb_speeds, eb_halts, eb_occs = [], [], []
            for lid in eb_lanes:
                try:
                    eb_speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                except Exception:
                    pass
                try:
                    eb_halts.append(traci.lane.getLastStepHaltingNumber(lid))
                except Exception:
                    pass
                if need_occ:
                    try:
                        eb_occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
            eb_speed_avg = float(np.mean(eb_speeds)) if eb_speeds else 0.0
            eb_halts_avg = float(np.mean(eb_halts)) if eb_halts else 0.0
            data_eb = {"speed": eb_speed_avg, "halts": eb_halts_avg}
            if need_occ:
                data_eb["occ"] = float(np.mean(eb_occs)) if eb_occs else 0.0
            lm["eb"][g] = data_eb

            # 西向
            wb_lanes = self.wb_lane_groups.get(g, [])
            wb_speeds, wb_halts, wb_occs = [], [], []
            for lid in wb_lanes:
                try:
                    wb_speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                except Exception:
                    pass
                try:
                    wb_halts.append(traci.lane.getLastStepHaltingNumber(lid))
                except Exception:
                    pass
                if need_occ:
                    try:
                        wb_occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
            wb_speed_avg = float(np.mean(wb_speeds)) if wb_speeds else 0.0
            wb_halts_avg = float(np.mean(wb_halts)) if wb_halts else 0.0
            data_wb = {"speed": wb_speed_avg, "halts": wb_halts_avg}
            if need_occ:
                data_wb["occ"] = float(np.mean(wb_occs)) if wb_occs else 0.0
            lm["wb"][g] = data_wb

    # --------------------------------------------------------
    def _observe(self) -> np.ndarray:
        feats: List[float] = []

        # 简化版观测：东向3段 + 西向3段，每段仅提供 occ + speed = 2个特征
        curr_occ_eb = {}
        curr_speed_eb = {}
        curr_occ_wb = {}
        curr_speed_wb = {}

        # 东向
        for gname in self.group_keys:
            lane_ids = self.lane_groups.get(gname, [])
            if not lane_ids:
                feats.extend([0.0, 0.0])  # occ, speed
                curr_occ_eb[gname] = 0.0
                curr_speed_eb[gname] = 0.0
                continue
            # 使用缓存或直接读取
            lm_local = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})
            cached_eb = lm_local.get("eb", {}).get(gname, {})
            avg_occ = cached_eb.get("occ", None)
            avg_speed = cached_eb.get("speed", None)
            if avg_occ is None:
                occs: List[float] = []
                for lid in lane_ids:
                    try:
                        occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
                avg_occ = float(np.mean(occs)) if occs else 0.0
                lm_local.setdefault("eb", {}).setdefault(gname, {})["occ"] = avg_occ
            if avg_speed is None:
                speeds: List[float] = []
                for lid in lane_ids:
                    try:
                        speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                    except Exception:
                        pass
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                lm_local.setdefault("eb", {}).setdefault(gname, {})["speed"] = avg_speed
            avg_speed_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
            
            # 仅输出当前占有率和速度
            feats.append(avg_occ)
            feats.append(avg_speed_norm)
            curr_occ_eb[gname] = avg_occ
            curr_speed_eb[gname] = avg_speed_norm

        # 西向
        for gname in self.group_keys:
            lane_ids = self.wb_lane_groups.get(gname, [])
            if not lane_ids:
                feats.extend([0.0, 0.0])
                curr_occ_wb[gname] = 0.0
                curr_speed_wb[gname] = 0.0
                continue
            lm_local = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})
            cached_wb = lm_local.get("wb", {}).get(gname, {})
            avg_occ = cached_wb.get("occ", None)
            avg_speed = cached_wb.get("speed", None)
            if avg_occ is None:
                occs = []
                for lid in lane_ids:
                    try:
                        occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
                avg_occ = float(np.mean(occs)) if occs else 0.0
                lm_local.setdefault("wb", {}).setdefault(gname, {})["occ"] = avg_occ
            if avg_speed is None:
                speeds = []
                for lid in lane_ids:
                    try:
                        speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                    except Exception:
                        pass
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                lm_local.setdefault("wb", {}).setdefault(gname, {})["speed"] = avg_speed
            avg_speed_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
            
            feats.append(avg_occ)
            feats.append(avg_speed_norm)
            curr_occ_wb[gname] = avg_occ
            curr_speed_wb[gname] = avg_speed_norm

        # 2) 三个信号：phase + remaining（简化版）
        for tid in self.tls_ids:
            try:
                phase_index = traci.trafficlight.getPhase(tid)
                remain = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
            except traci.TraCIException:
                phase_index = 0
                remain = 0.0
            meta = self._tls_phase_meta.get(tid, {"phase_count": 8.0, "max_phase_duration": 60.0})
            phase_den = float(max(int(meta["phase_count"]) - 1, 1))
            phase_norm = min(float(phase_index) / phase_den, 1.0)
            remain = max(remain, 0.0)
            remain_norm = min(float(remain) / float(max(meta["max_phase_duration"], 1.0)), 1.0)
            
            feats.extend([phase_norm, remain_norm])

        # 3) CAV渗透率
        feats.append(self._get_cav_ratio())

        return np.array(feats, dtype=np.float32)

    # --------------------------------------------------------
    def _compute_reward_per_step(self) -> float:
        """
        简单、可验证的奖励函数（轻量版）
        
        设计目标：
        - 稳定：避免动态权重与过多分量引入的噪声
        - 高效：减少 TraCI 调用，提升步速
        - 可解释：显式记录各奖励分量，便于可视化与调参
        
        组成：
        - 效率奖励 r_efficiency：双向三段平均速度 / 参考最大速度（∈[0,1]）
        - 拥堵惩罚 r_congestion：双向三段平均停止车辆数（按 5 辆/车道归一，∈[0,1]）
        - 合成：total = w_eff*r_efficiency + w_cong*(-r_congestion)
        """
        try:
            return self._compute_reward_per_step_simple()
        except Exception as e:
            print(f"[奖励计算] 简单版奖励计算失败: {e}")
            return 0.0
    # --------------------------------------------------------

    # --------------------------------------------------------
    def _compute_reward_per_step_simple(self) -> float:
        """轻量版奖励：速度效率 - 拥堵惩罚。

        - 速度效率：各段平均速度相对参考最大速度的归一化（越快越好）
        - 拥堵惩罚：各段停止车辆的平均值按经验上限（5 辆/车道）归一化（越少越好）
        - 明确记录分量到 metrics，便于训练时可视化与排查
        """
        # 优先使用缓存的速度/停止车辆数，避免重复 TraCI 读取；缓存缺失时回退读取并写回缓存
        lm = self._step_cache.get("lane_metrics")
        eb_speeds: List[float] = []
        wb_speeds: List[float] = []
        eb_halts: List[float] = []
        wb_halts: List[float] = []

        cache_used = lm is not None
        if cache_used:
            for g in self.group_keys:
                eb_val = lm.get("eb", {}).get(g, {})
                wb_val = lm.get("wb", {}).get(g, {})
                eb_speeds.append(float(eb_val.get("speed", 0.0)))
                wb_speeds.append(float(wb_val.get("speed", 0.0)))
                eb_halts.append(float(eb_val.get("halts", 0.0)))
                wb_halts.append(float(wb_val.get("halts", 0.0)))
        else:
            # 缓存不存在：读取一次并写回缓存（不采集占有率，观测阶段再补齐）
            self._collect_lane_group_metrics(need_occ=False)
            lm = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})
            for g in self.group_keys:
                eb_val = lm.get("eb", {}).get(g, {})
                wb_val = lm.get("wb", {}).get(g, {})
                eb_speeds.append(float(eb_val.get("speed", 0.0)))
                wb_speeds.append(float(wb_val.get("speed", 0.0)))
                eb_halts.append(float(eb_val.get("halts", 0.0)))
                wb_halts.append(float(wb_val.get("halts", 0.0)))

        mean_speed = float(np.mean([*eb_speeds, *wb_speeds])) if (eb_speeds or wb_speeds) else 0.0
        avg_halting = float(np.mean([*eb_halts, *wb_halts])) if (eb_halts or wb_halts) else 0.0

        # 归一化与权重
        speed_norm = max(0.0, min(mean_speed / max(self.max_ref_speed_mps, 1e-6), 1.0))
        # 经验归一：每车道 0~5 辆停止车，>5 视为重拥堵
        halting_norm = max(0.0, min(avg_halting / 5.0, 1.0))

        r_efficiency = self.rw.get("efficiency", 1.0) * speed_norm
        r_congestion = self.rw.get("congestion", 1.0) * (-halting_norm)

        total = r_efficiency + r_congestion

        # 可选裁剪
        if (self._reward_clip_min is not None) or (self._reward_clip_max is not None):
            clip_min = self._reward_clip_min if self._reward_clip_min is not None else -float("inf")
            clip_max = self._reward_clip_max if self._reward_clip_max is not None else float("inf")
            total = float(np.clip(total, clip_min, clip_max))

        # 记录奖励分量（用于分析/可视化）
        try:
            self.metrics["r_efficiency"] = r_efficiency
            self.metrics["r_congestion"] = r_congestion
            self.metrics["r_total"] = total
            self.metrics["r_speed_mean_mps"] = mean_speed
            self.metrics["r_halting_avg"] = avg_halting
        except Exception:
            pass

        return total
    

    # --------------------------------------------------------
    def _compute_action_change_penalty(self, new_speeds_mps: Tuple[float, float, float, float, float, float]) -> float:
        # 每个决策周期算一次
        if self._last_action_index < 0:
            return 0.0
        prev = np.array(self._last_action_speeds)
        curr = np.array(new_speeds_mps)
        diff_kph = np.abs(curr - prev) * 3.6
        # 归一化到 [0,1]：按每段最大允许变化 self.max_delta_kph 聚合（6 段）
        denom = float(max(len(self.group_keys) * 2 * max(self.max_delta_kph, 1e-6), 1.0))
        norm = float(np.sum(diff_kph)) / denom
        norm = max(0.0, min(norm, 1.0))
        return -self.rw["action_change"] * norm

    # --------------------------------------------------------
    def _get_cav_ratio(self) -> float:
        try:
            veh_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return 0.0
        if not veh_ids:
            return 0.0
        cav_num = sum(1 for vid in veh_ids if traci.vehicle.getTypeID(vid) == self.cav_type_id)
        return cav_num / len(veh_ids)

    # 简化的错误处理：不再实现检查点保存和加载功能
    
    def close(self):
        # 先恢复所有被控车辆的默认类型最大速度，避免"限速残留"影响后续回合
        try:
            for vid in list(self._controlled_veh_ids):
                try:
                    default_v = float(self._veh_type_max_speed.pop(vid, self.max_ref_speed_mps))
                    traci.vehicle.setMaxSpeed(vid, default_v)
                except Exception:
                    pass
                finally:
                    self._controlled_veh_ids.discard(vid)
        except Exception:
            pass
        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass


if __name__ == "__main__":
    env = CityVSLEnv(
        sumo_cfg_path=os.path.join("..", "sumo", "grid1x3.sumocfg"),
        use_gui=True,
        lane_groups={
            "upstream": ["J0_J1_0", "J0_J1_1", "J0_J1_2"],
            "mid": ["J1_J2_0", "J1_J2_1", "J1_J2_2"],
            "down": ["J2_E_0", "J2_E_1", "J2_E_2"],
        },
        tls_ids=["J0", "J1", "J2"],
        discrete_speeds_kph=[30, 35, 40, 45, 50],
        decision_interval=5,
    )
    obs, _ = env.reset()
    print("obs:", obs.shape)
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, done, trunc, info = env.step(a)
        print("r=", r, "info=", info)
        if done:
            break
    env.close()
