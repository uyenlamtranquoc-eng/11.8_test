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
        detector_ids: Optional[List[str]] = None,
        same_speed_for_all: bool = False,
        max_delta_kph: float = 10.0,  # 限速变化硬约束，论文式
        reward_speed_weight: float = 1.0,
        reward_bottleneck_weight: float = 1.0,
        reward_congestion_weight: float = 1.0,
        reward_change_penalty: float = 0.1,
        # 双向队列惩罚权重（可在 env.yaml 配置）：默认东向更关注下游堆积
        reward_queue_weight_eb: float = 0.5,
        reward_queue_weight_wb: float = 0.3,
        # 城市VSL优化：绿波匹配奖励权重
        reward_green_wave_weight: float = 3.0,
        # 拥堵惩罚平滑函数参数（占有率起始阈值与幂次）
        occ_penalty_start: float = 0.7,
        occ_penalty_power: float = 2.0,
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
        self.same_speed_for_all = same_speed_for_all
        self.max_delta_kph = max_delta_kph
        self.warmup_steps = int(warmup_steps)

        # 评估指标累计器（按仿真步聚合）
        self.metrics = {
            "queue_sum": 0.0,
            "queue_steps": 0,
            "arrived_total": 0,
            "sim_seconds": 0,
            "energy_total_j": 0.0,
            "distance_total_m": 0.0,
            "time_loss_total_s": 0.0,
        }
        # 每车累计字典
        self._veh_energy_j: Dict[str, float] = {}
        self._veh_prev_dist: Dict[str, float] = {}
        self._veh_dist_m: Dict[str, float] = {}
        self._veh_time_loss_s: Dict[str, float] = {}

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
        
        # 绿灯相位定义（根据SUMO配置，绿灯相位通常包含'G'字符）
        self._green_phase_cache: Dict[str, set] = {}  # {tls_id: {绿灯相位索引集合}}

        # 离散车速
        self.discrete_speeds_kph = discrete_speeds_kph or [30, 35, 40, 45, 50, 55]
        self.speed_levels_mps = [v / 3.6 for v in self.discrete_speeds_kph]

        # 构建动作空间：6 个独立离散动作（每段一档）
        n_per_segment = len(self.speed_levels_mps) if self.speed_levels_mps else 1
        self.action_space = spaces.MultiDiscrete(np.array([n_per_segment] * 6, dtype=np.int64))

        # 观测空间扩展（城市VSL增强）：
        # 东向3段 + 西向3段：每段 occ + speed + pred_occ + pred_speed + eta_to_signal + distance_to_signal = 6 * 6 = 36
        # 每信号: phase + remaining + is_green = 3 * 3 = 9
        # CAV ratio = 1
        # 总计: 36 + 9 + 1 = 46
        self.obs_dim = 6 * len(self.group_keys) * 2 + 3 * len(self.tls_ids) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # 信号距离缓存（米）：每个段到下游信号的距离（将在SUMO初始化后计算）
        self._segment_to_signal_distance: Dict[str, float] = {}

        # detector
        self.detector_ids = detector_ids  # 可以是 None

        # reward 权重：对应论文的多目标
        self.rw = {
            "efficiency": float(reward_speed_weight),      # 整体速度 / TTS proxy（双向平均）
            "bottleneck": float(reward_bottleneck_weight), # 下游速度（双向平均）
            "congestion": float(reward_congestion_weight), # 占有率/排队（双向合并）
            "queue_eb": float(reward_queue_weight_eb),     # 东向队列惩罚权重
            "queue_wb": float(reward_queue_weight_wb),     # 西向队列惩罚权重
            "action_change": float(reward_change_penalty), # 动作变化（较论文小一点，因为我们有硬约束）
            "green_wave": float(reward_green_wave_weight), # 城市VSL优化：绿波命中率正奖励
        }

        # 拥堵惩罚平滑参数
        self._occ_penalty_start = float(occ_penalty_start)
        self._occ_penalty_power = float(occ_penalty_power)

        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max

        # 预测量需要记住上一步
        # 预测量需要记住上一步（东西向分开）
        self._prev_occ_eb = {g: 0.0 for g in self.group_keys}
        self._prev_speed_eb = {g: 0.0 for g in self.group_keys}
        self._prev_occ_wb = {g: 0.0 for g in self.group_keys}
        self._prev_speed_wb = {g: 0.0 for g in self.group_keys}
        self._ema_alpha = 0.6  # 简单预测用的系数

        # 归一化参考：速度按离散档位最大值（至少 20 m/s）；信号按实际程序配置
        self.max_ref_speed_mps = max(20.0, max(self.speed_levels_mps)) if self.speed_levels_mps else 20.0
        self._tls_phase_meta: Dict[str, Dict[str, float]] = {}

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

        # 每次运行使用独立时间戳子目录：outputs/YYYYMMDD_HHMMSS
        self.run_output_dir = self._make_run_output_dir()
        self._run_additional_path = None

        # 起 SUMO
        self._init_sumo()
        
    # --------------------------------------------------------
    def _compute_segment_distances(self) -> Dict[str, float]:
        """计算每个段到下游信号的平均距离（米）。
        
        基于 lane ID 命名约定：J0_J1 -> 下游信号为 J1
        """
        distances = {}
        # 东向段
        segment_to_tls = {
            "upstream": "J1",   # J0_J1 -> J1
            "mid": "J2",        # J1_J2 -> J2
            "down": "E",         # J2_E -> 出口，用特殊标记
        }
        for seg_name in self.group_keys:
            lane_ids = self.lane_groups.get(seg_name, [])
            if not lane_ids:
                distances[f"eb_{seg_name}"] = 0.0
                continue
            # 取第一条lane长度作为代表（假设同段同长）
            try:
                lane_length = traci.lane.getLength(lane_ids[0])
                distances[f"eb_{seg_name}"] = float(lane_length)
            except Exception:
                distances[f"eb_{seg_name}"] = 300.0  # 默认300米
        
        # 西向段（逆序）
        for seg_name in self.group_keys:
            lane_ids = self.wb_lane_groups.get(seg_name, [])
            if not lane_ids:
                distances[f"wb_{seg_name}"] = 0.0
                continue
            try:
                lane_length = traci.lane.getLength(lane_ids[0])
                distances[f"wb_{seg_name}"] = float(lane_length)
            except Exception:
                distances[f"wb_{seg_name}"] = 300.0
        
        return distances

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
        # 计算段到信号的距离（SUMO初始化后）
        self._segment_to_signal_distance = self._compute_segment_distances()

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
    def _build_actions(self) -> List[Tuple[float, float, float, float, float, float]]:
        actions: List[Tuple[float, float, float, float, float, float]] = []
        if self.same_speed_for_all:
            # 所有6段统一速度
            for s in self.speed_levels_mps:
                actions.append((s, s, s, s, s, s))
        else:
            # 东向 upstream/mid/down 与 西向 upstream/mid/down 独立组合
            for su_e in self.speed_levels_mps:
                for sm_e in self.speed_levels_mps:
                    for sd_e in self.speed_levels_mps:
                        for su_w in self.speed_levels_mps:
                            for sm_w in self.speed_levels_mps:
                                for sd_w in self.speed_levels_mps:
                                    actions.append((su_e, sm_e, sd_e, su_w, sm_w, sd_w))
        return actions

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
        self._prev_occ_eb = {g: 0.0 for g in self.group_keys}
        self._prev_speed_eb = {g: 0.0 for g in self.group_keys}
        self._prev_occ_wb = {g: 0.0 for g in self.group_keys}
        self._prev_speed_wb = {g: 0.0 for g in self.group_keys}

        # 清空评估累计
        self.metrics = {
            "queue_sum": 0.0,
            "queue_steps": 0,
            "arrived_total": 0,
            "sim_seconds": 0,
            "energy_total_j": 0.0,
            "distance_total_m": 0.0,
            "time_loss_total_s": 0.0,
        }
        self._veh_energy_j = {}
        self._veh_prev_dist = {}
        self._veh_dist_m = {}
        self._veh_time_loss_s = {}
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
        
        # 初始化动作历史记录（如果不存在）
        if not hasattr(self, '_action_history'):
            self._action_history = []

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
            if safety_iter % 100 == 0 and (controlled_count_eb > 0 or controlled_count_wb > 0):
                print(f"[VSL控制] 时间={self.sim_time:.0f}s, 东向CAV={controlled_count_eb}, 西向CAV={controlled_count_wb}, "
                      f"总控制={len(curr_controlled)}, 东向速度={[f'{s*3.6:.0f}km/h' for s in eb_speeds]}, "
                      f"西向速度={[f'{s*3.6:.0f}km/h' for s in wb_speeds]}")

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

            # 2) 车辆能耗 / 距离 / 时间损失（逐车）
            try:
                veh_ids = traci.vehicle.getIDList()
            except Exception:
                veh_ids = []
                
            for vid in veh_ids:
                # 能耗（尽力读取电/油耗）
                e_j = 0.0
                try:
                    e_wh = traci.vehicle.getElectricityConsumption(vid)  # Wh/步（模型Energy）
                    if e_wh is not None:
                        e_j += max(0.0, float(e_wh)) * 3600.0
                except Exception:
                    pass  # 简化错误处理，忽略单个车辆能耗获取失败
                    
                try:
                    fuel_ml = traci.vehicle.getFuelConsumption(vid)  # mL/步（HBEFA 等）
                    # 汽油低位发热值 ~34.2 MJ/L
                    e_j += max(0.0, float(fuel_ml)) / 1000.0 * 34.2e6
                except Exception:
                    pass  # 简化错误处理，忽略单个车辆油耗获取失败
                    
                if e_j > 0.0:
                    self._veh_energy_j[vid] = self._veh_energy_j.get(vid, 0.0) + e_j

                # 距离累计（以车辆里程为准）
                try:
                    dist = float(traci.vehicle.getDistance(vid))
                except Exception:
                    dist = 0.0  # 简化错误处理，使用默认值
                    
                prev = self._veh_prev_dist.get(vid, dist)
                delta = max(0.0, dist - prev)
                self._veh_prev_dist[vid] = dist
                if delta > 0.0:
                    self._veh_dist_m[vid] = self._veh_dist_m.get(vid, 0.0) + delta

                # 时间损失（累积）
                try:
                    tl = float(traci.vehicle.getTimeLoss(vid))
                    self._veh_time_loss_s[vid] = tl
                except Exception:
                    pass  # 简化错误处理，忽略单个车辆时间损失获取失败

            # 3) 本步到达车辆：汇总并移除跟踪
            try:
                arrived_ids = traci.simulation.getArrivedIDList()
            except Exception:
                arrived_ids = []
                
            for vid in arrived_ids:
                self.metrics["arrived_total"] += 1
                self.metrics["time_loss_total_s"] += self._veh_time_loss_s.pop(vid, 0.0)
                self.metrics["energy_total_j"] += self._veh_energy_j.pop(vid, 0.0)
                self.metrics["distance_total_m"] += self._veh_dist_m.pop(vid, 0.0)
                self._veh_prev_dist.pop(vid, None)

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
        
        # 7. 记录动作历史（用于重启恢复）
        self._action_history.append((action, tuple(target_speeds)))
        # 限制历史记录长度，避免内存泄漏
        max_history_length = 100
        if len(self._action_history) > max_history_length:
            self._action_history = self._action_history[-max_history_length:]

        # 8. 定期内存清理（每100步清理一次）
        if safety_iter % 100 == 0:
            self._cleanup_memory()

        # 9. 观测 & done
        try:
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
        - avg_delay_s: 平均延误（s/车）= 总时间损失 / 到达车辆数
        - avg_queue_veh: 平均排队车辆数（辆）= 每步停止车辆数平均
        - throughput_veh_per_hour: 通行流量（辆/小时）= 到达数 / 时长 * 3600
        - avg_energy_j_per_km: 平均能耗（J/km）= 总能耗 / 总里程(km)
        """
        q_steps = self.metrics.get("queue_steps", 0)
        sim_sec = self.metrics.get("sim_seconds", 0)
        arrived = self.metrics.get("arrived_total", 0)
        dist_m = self.metrics.get("distance_total_m", 0.0)

        avg_queue = (self.metrics.get("queue_sum", 0.0) / q_steps) if q_steps > 0 else 0.0
        throughput = (arrived / sim_sec * 3600.0) if sim_sec > 0 else 0.0
        avg_delay = (self.metrics.get("time_loss_total_s", 0.0) / arrived) if arrived > 0 else 0.0
        avg_energy = (self.metrics.get("energy_total_j", 0.0) / (dist_m / 1000.0)) if dist_m > 0 else 0.0

        return {
            "avg_delay_s": avg_delay,
            "avg_queue_veh": avg_queue,
            "throughput_veh_per_hour": throughput,
            "avg_energy_j_per_km": avg_energy,
            # 也返回原始累计，便于调试
            "arrived_total": arrived,
            "sim_seconds": sim_sec,
            "energy_total_j": self.metrics.get("energy_total_j", 0.0),
            "distance_total_m": dist_m,
        }

    # --------------------------------------------------------
    def _observe(self) -> np.ndarray:
        feats: List[float] = []

        # 1) 三段（东向 + 西向）：当前 + 预测 + ETA + 距离（分开）
        curr_occ_eb = {}
        curr_speed_eb = {}
        curr_occ_wb = {}
        curr_speed_wb = {}

        # 东向
        for gname in self.group_keys:
            lane_ids = self.lane_groups.get(gname, [])
            if not lane_ids:
                # 缺失段：填充6个0（occ, speed, pred_occ, pred_speed, eta, distance）
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                curr_occ_eb[gname] = 0.0
                curr_speed_eb[gname] = 0.0
                continue
            occs, speeds = [], []
            for lid in lane_ids:
                occs.append(traci.lane.getLastStepOccupancy(lid))
                speeds.append(traci.lane.getLastStepMeanSpeed(lid))
            avg_occ = float(np.mean(occs)) if occs else 0.0
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            avg_speed_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
            
            # 预测量（EMA）
            prev_occ = self._prev_occ_eb[gname]
            prev_speed = self._prev_speed_eb[gname]
            pred_occ = self._ema_alpha * avg_occ + (1 - self._ema_alpha) * prev_occ
            prev_speed_norm = (prev_speed / self.max_ref_speed_mps) if prev_speed else 0.0
            pred_speed = self._ema_alpha * avg_speed_norm + (1 - self._ema_alpha) * prev_speed_norm
            
            # ETA到下游信号（秒，归一化）
            distance_m = self._segment_to_signal_distance.get(f"eb_{gname}", 300.0)
            distance_norm = min(distance_m / 1000.0, 1.0)  # 最大1km归一
            if avg_speed > 0.1:
                eta_s = distance_m / max(avg_speed, 0.1)
                eta_norm = min(eta_s / 120.0, 1.0)  # 最大120秒归一
            else:
                eta_norm = 1.0  # 静止时设为最大
            
            feats.extend([avg_occ, avg_speed_norm, pred_occ, min(pred_speed, 1.0), eta_norm, distance_norm])
            curr_occ_eb[gname] = avg_occ
            curr_speed_eb[gname] = avg_speed

        # 西向
        for gname in self.group_keys:
            lane_ids = self.wb_lane_groups.get(gname, [])
            if not lane_ids:
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                curr_occ_wb[gname] = 0.0
                curr_speed_wb[gname] = 0.0
                continue
            occs, speeds = [], []
            for lid in lane_ids:
                occs.append(traci.lane.getLastStepOccupancy(lid))
                speeds.append(traci.lane.getLastStepMeanSpeed(lid))
            avg_occ = float(np.mean(occs)) if occs else 0.0
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            avg_speed_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
            
            prev_occ = self._prev_occ_wb[gname]
            prev_speed = self._prev_speed_wb[gname]
            pred_occ = self._ema_alpha * avg_occ + (1 - self._ema_alpha) * prev_occ
            prev_speed_norm = (prev_speed / self.max_ref_speed_mps) if prev_speed else 0.0
            pred_speed = self._ema_alpha * avg_speed_norm + (1 - self._ema_alpha) * prev_speed_norm
            
            distance_m = self._segment_to_signal_distance.get(f"wb_{gname}", 300.0)
            distance_norm = min(distance_m / 1000.0, 1.0)
            if avg_speed > 0.1:
                eta_s = distance_m / max(avg_speed, 0.1)
                eta_norm = min(eta_s / 120.0, 1.0)
            else:
                eta_norm = 1.0
            
            feats.extend([avg_occ, avg_speed_norm, pred_occ, min(pred_speed, 1.0), eta_norm, distance_norm])
            curr_occ_wb[gname] = avg_occ
            curr_speed_wb[gname] = avg_speed

        # 写回 prev（用于下一次预测）
        for gname in self.group_keys:
            self._prev_occ_eb[gname] = curr_occ_eb.get(gname, 0.0)
            self._prev_speed_eb[gname] = curr_speed_eb.get(gname, 0.0)
            self._prev_occ_wb[gname] = curr_occ_wb.get(gname, 0.0)
            self._prev_speed_wb[gname] = curr_speed_wb.get(gname, 0.0)

        # 2) 信号：相位 + 剩余时间 + 是否绿灯
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
            
            # 判断是否绿灯（通过相位定义）
            is_green = self._is_green_phase(tid, phase_index)
            is_green_float = 1.0 if is_green else 0.0
            
            feats.extend([phase_norm, remain_norm, is_green_float])

        # 3) CAV渗透率
        feats.append(self._get_cav_ratio())

        return np.array(feats, dtype=np.float32)

    # --------------------------------------------------------
    def _compute_reward_per_step(self) -> float:
        """
        城市VSL奖励函数（增强版）
        
        使用基于交通流理论的分层奖励函数，包含：
        1. 分层奖励架构：主目标(绿波效率) + 次要目标(能耗、舒适度) + 约束满足
        2. 动态权重自适应：根据交通状态动态调整各目标权重
        3. 连续化奖励信号：消除红灯期零惩罚的负面影响
        4. 自适应队列惩罚：基于实际交通状况的动态阈值
        5. 交通需求变化自适应：根据流量变化动态调整策略
        """
        try:
            # 使用增强版奖励函数
            return self._compute_reward_per_step_enhanced()
        except Exception as e:
            print(f"[奖励计算] 增强版奖励计算失败: {e}")
            # 返回默认奖励
            return 0.0
    # --------------------------------------------------------
    def _compute_reward_per_step_enhanced(self) -> float:
        """
        基于交通流理论的分层奖励函数
        
        核心改进：
        1. 分层奖励架构：主目标(绿波效率) + 次要目标(能耗、舒适度) + 约束满足
        2. 动态权重自适应：根据交通状态动态调整各目标权重
        3. 连续化奖励信号：消除红灯期零惩罚的负面影响
        4. 自适应队列惩罚：基于实际交通状况的动态阈值
        5. 交通需求变化自适应：根据流量变化动态调整策略
        
        返回：
            float: 当前步的总奖励
        """
        # 获取当前交通状态特征
        traffic_state_features = self._extract_traffic_state_features()
        
        # 计算动态权重
        primary_weight, secondary_weights = self._compute_dynamic_weights(traffic_state_features)
        
        # 计算主目标：绿波效率
        primary_reward = self._compute_primary_green_wave_reward(traffic_state_features)
        
        # 计算次要目标：能耗、舒适度、排放
        secondary_reward = self._compute_secondary_objectives(secondary_weights)
        
        # 计算约束满足度
        constraint_reward = self._compute_constraint_satisfaction()
        
        # 计算连续化奖励信号
        continuity_reward = self._compute_reward_continuity(traffic_state_features)
        
        # 分层奖励聚合（secondary_reward已经包含权重，直接相加）
        total_reward = (
            primary_weight * primary_reward +
            secondary_reward["energy"] +
            secondary_reward["comfort"] +
            secondary_reward["emission"] +
            constraint_reward +
            continuity_reward
        )
        
        # 奖励归一化与量纲统一
        normalized_reward = self._normalize_reward(total_reward, traffic_state_features)
        
        return normalized_reward
    
    def _extract_traffic_state_features(self) -> Dict[str, float]:
        """提取当前交通状态特征用于动态权重调整"""
        # 获取基础交通指标
        speeds_norm_eb = []
        speeds_norm_wb = []
        occ_pen = 0.0
        
        # 获取各信号灯当前状态
        tls_states = {}
        for tid in self.tls_ids:
            try:
                phase_idx = int(traci.trafficlight.getPhase(tid))
                is_green = self._is_green_phase(tid, phase_idx)
                remain = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
                tls_states[tid] = {
                    'phase': phase_idx,
                    'is_green': is_green,
                    'remain': max(0.0, remain)
                }
            except Exception:
                tls_states[tid] = {'phase': 0, 'is_green': False, 'remain': 0.0}
        
        # 计算东向和西向的交通指标
        for gname in self.group_keys:
            # 东向
            lane_ids_eb = self.lane_groups.get(gname, [])
            if lane_ids_eb:
                occs, speeds = [], []
                for lid in lane_ids_eb:
                    try:
                        occs.append(traci.lane.getLastStepOccupancy(lid))
                        speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                    except Exception:
                        pass
                avg_occ = float(np.mean(occs)) if occs else 0.0
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                s_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
                speeds_norm_eb.append(s_norm)
                if gname == "down":
                    down_speed_norm_eb = s_norm
            
            # 西向
            lane_ids_wb = self.wb_lane_groups.get(gname, [])
            if lane_ids_wb:
                occs, speeds = [], []
                for lid in lane_ids_wb:
                    try:
                        occs.append(traci.lane.getLastStepOccupancy(lid))
                        speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                    except Exception:
                        pass
                avg_occ = float(np.mean(occs)) if occs else 0.0
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                s_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
                speeds_norm_wb.append(s_norm)
                if gname == "down":
                    down_speed_norm_wb = s_norm
        
        eff_eb = float(np.mean(speeds_norm_eb)) if speeds_norm_eb else 0.0
        eff_wb = float(np.mean(speeds_norm_wb)) if speeds_norm_wb else 0.0
        eff = float(np.mean([v for v in [eff_eb, eff_wb] if v is not None]))
        
        # 计算交通强度指标
        traffic_intensity = (eff_eb + eff_wb) / 2.0  # 平均效率作为交通强度指标
        congestion_level = max(0.0, min(1.0, traffic_intensity))  # 归一化到[0,1]
        
        # 计算需求变化指标
        demand_change = self._compute_demand_change()
        
        return {
            'traffic_intensity': traffic_intensity,
            'congestion_level': congestion_level,
            'demand_change': demand_change,
            'avg_speed_eb': eff_eb,
            'avg_speed_wb': eff_wb,
            'tls_states': tls_states,
        }
    
    def _compute_dynamic_weights(self, traffic_features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """根据交通状态计算动态权重"""
        congestion_level = traffic_features['congestion_level']
        demand_change = traffic_features['demand_change']
        
        # 基于交通强度分类
        if congestion_level < 0.3:  # 自由流
            primary_weight = 0.7
            secondary_weights = {"energy": 0.2, "comfort": 0.05, "emission": 0.05}
        elif congestion_level < 0.7:  # 轻度拥堵
            primary_weight = 0.8
            secondary_weights = {"energy": 0.15, "comfort": 0.03, "emission": 0.02}
        else:  # 重度拥堵
            primary_weight = 0.9
            secondary_weights = {"energy": 0.1, "comfort": 0.0, "emission": 0.0}
        
        # 根据需求变化调整权重
        if demand_change > 0.2:  # 需求激增
            primary_weight += 0.05  # 提高绿波权重
            secondary_weights["energy"] -= 0.02  # 降低能耗权重
        elif demand_change < -0.2:  # 需求骤降
            primary_weight -= 0.05  # 降低绿波权重，提高稳定性权重
            secondary_weights["comfort"] += 0.02  # 提高舒适度权重
        
        # 确保权重在合理范围
        primary_weight = max(0.5, min(1.0, primary_weight))
        for key in secondary_weights:
            secondary_weights[key] = max(0.0, min(0.3, secondary_weights[key]))
        
        return primary_weight, secondary_weights
    
    def _compute_primary_green_wave_reward(self, traffic_features: Dict[str, float]) -> float:
        """计算基于交通流理论的绿波效率奖励"""
        # 队列消散效率
        queue_dissipation_efficiency = self._compute_queue_dissipation_efficiency(traffic_features)
        
        # 信号协调质量
        signal_coordination_quality = self._compute_signal_coordination_quality(traffic_features)
        
        # 旅行时间稳定性
        travel_time_stability = self._compute_travel_time_stability(traffic_features)
        
        # 综合绿波效率奖励
        green_wave_reward = (
            0.4 * queue_dissipation_efficiency +
            0.3 * signal_coordination_quality +
            0.3 * travel_time_stability
        )
        
        return green_wave_reward
    
    def _compute_queue_dissipation_efficiency(self, traffic_features: Dict[str, float]) -> float:
        """计算队列消散效率：实际消散率/理论消散率"""
        # 理论消散率基于当前限速和道路容量
        theoretical_dissipation_rate = self._compute_theoretical_dissipation_rate(traffic_features)
        
        # 实际消散率基于交通流观测
        actual_dissipation_rate = self._compute_actual_dissipation_rate(traffic_features)
        
        # 效率 = 实际/理论，限制在[0,1]范围
        efficiency = min(1.0, actual_dissipation_rate / max(theoretical_dissipation_rate, 0.1))
        
        return efficiency
    
    def _compute_theoretical_dissipation_rate(self, traffic_features: Dict[str, float]) -> float:
        """计算理论队列消散率"""
        avg_speed = (traffic_features['avg_speed_eb'] + traffic_features['avg_speed_wb']) / 2.0
        
        # 基于LWR理论，消散率 = min(流量, 容量)
        # 简化：使用当前速度作为流量的代理
        theoretical_rate = min(avg_speed, self.max_ref_speed_mps) / self.max_ref_speed_mps
        
        return theoretical_rate
    
    def _compute_actual_dissipation_rate(self, traffic_features: Dict[str, float]) -> float:
        """计算实际队列消散率"""
        # 基于检测器数据计算实际消散
        if not self.detector_ids:
            return 0.5  # 默认值
        
        total_vehicles = 0
        total_halting_vehicles = 0
        
        for det in self.detector_ids:
            try:
                veh_num = traci.lanearea.getLastStepVehicleNumber(det)
                halting_veh = traci.lanearea.getLastStepHaltingNumber(det)
                total_vehicles += veh_num
                total_halting_vehicles += halting_veh
            except traci.TraCIException:
                pass
        
        # 实际消散率 = (总车辆数 - 停车数) / 总车辆数
        if total_vehicles > 0:
            actual_rate = (total_vehicles - total_halting_vehicles) / total_vehicles
        else:
            actual_rate = 1.0
            
        return actual_rate
    
    def _compute_signal_coordination_quality(self, traffic_features: Dict[str, float]) -> float:
        """计算信号协调质量：绿波带匹配度与相位偏移合理性"""
        tls_states = traffic_features['tls_states']
        
        # 绿波带匹配度
        green_wave_match_score = 0.0
        for tid in self.tls_ids:
            if tid in tls_states:
                is_green = tls_states[tid]['is_green']
                remain = tls_states[tid]['remain']
                
                # 绿灯期且剩余时间适中时给予奖励
                if is_green and 10.0 <= remain <= 50.0:
                    green_wave_match_score += 0.3
                elif is_green and remain < 10.0:
                    green_wave_match_score += 0.1  # 即将切换到红灯，适度奖励
        
        # 相位偏移合理性
        phase_offset_reasonableness = self._compute_phase_offset_reasonableness(tls_states)
        
        # 综合协调质量
        coordination_quality = 0.6 * green_wave_match_score + 0.4 * phase_offset_reasonableness
        
        return coordination_quality
    
    def _compute_phase_offset_reasonableness(self, tls_states: Dict) -> float:
        """计算相位偏移合理性"""
        # 简化实现：基于相邻信号灯的相位差
        if len(self.tls_ids) < 2:
            return 0.5  # 默认值
        
        # 计算相邻信号灯的相位差合理性
        total_reasonableness = 0.0
        count = 0
        
        for i in range(len(self.tls_ids) - 1):
            tid1 = self.tls_ids[i]
            tid2 = self.tls_ids[i + 1]
            
            if tid1 in tls_states and tid2 in tls_states:
                is_green1 = tls_states[tid1]['is_green']
                is_green2 = tls_states[tid2]['is_green']
                
                # 相邻信号灯相位差合理性：理想情况下应该有一定的相位差
                if is_green1 and not is_green2:
                    total_reasonableness += 0.8  # 理想的绿波带
                elif not is_green1 and is_green2:
                    total_reasonableness += 0.8  # 理想的绿波带
                elif is_green1 and is_green2:
                    total_reasonableness += 0.5  # 同时绿灯，不太理想
                else:
                    total_reasonableness += 0.2  # 同时红灯，不太理想
                
                count += 1
        
        if count > 0:
            return total_reasonableness / count
        else:
            return 0.5
    
    def _compute_travel_time_stability(self, traffic_features: Dict[str, float]) -> float:
        """计算旅行时间稳定性：惩罚过大的旅行时间波动"""
        # 简化实现：基于速度稳定性
        avg_speed_eb = traffic_features['avg_speed_eb']
        avg_speed_wb = traffic_features['avg_speed_wb']
        
        # 计算速度差异作为旅行时间不稳定性的代理
        speed_variance = abs(avg_speed_eb - avg_speed_wb)
        
        # 稳定性奖励：速度差异越小越好
        stability = max(0.0, 1.0 - speed_variance)
        
        return stability
    
    def _compute_secondary_objectives(self, weights: Dict[str, float]) -> Dict[str, float]:
        """计算次要目标：能耗、舒适度、排放"""
        energy_reward = self._compute_energy_efficiency_reward(weights["energy"])
        comfort_reward = self._compute_driving_comfort_reward(weights["comfort"])
        emission_reward = self._compute_emission_reduction_reward(weights["emission"])
        
        return {
            "energy": energy_reward,
            "comfort": comfort_reward,
            "emission": emission_reward,
        }
    
    def _compute_energy_efficiency_reward(self, weight: float) -> float:
        """计算能耗效率奖励：鼓励平滑驾驶"""
        try:
            veh_ids = traci.vehicle.getIDList()
            total_energy = 0.0
            total_acceleration_penalty = 0.0
            
            for vid in veh_ids:
                try:
                    # 获取能耗
                    energy = traci.vehicle.getElectricityConsumption(vid)
                    if energy is not None:
                        total_energy += max(0.0, float(energy)) * 3600.0  # 转换为J/h
                    
                    # 获取加速度并计算急加速惩罚
                    speed = traci.vehicle.getSpeed(vid)
                    accel = traci.vehicle.getAcceleration(vid)
                    if accel is not None and abs(accel) > 2.0:  # 急加速阈值
                        total_acceleration_penalty -= 0.1
                        
                except traci.TraCIException:
                    pass
            
            # 能耗效率奖励：低能耗和平滑驾驶
            if len(veh_ids) > 0:
                avg_energy = total_energy / len(veh_ids)
                # 归一化到[0,1]范围，低能耗获得高奖励
                energy_reward = weight * max(0.0, 1.0 - avg_energy / 1000.0)
            else:
                energy_reward = 0.0
                
        except traci.TraCIException:
            energy_reward = 0.0
            
        return energy_reward
    
    def _compute_driving_comfort_reward(self, weight: float) -> float:
        """计算驾驶舒适度奖励：减少急加速/急减速"""
        try:
            veh_ids = traci.vehicle.getIDList()
            total_discomfort = 0.0
            
            for vid in veh_ids:
                try:
                    accel = traci.vehicle.getAcceleration(vid)
                    if accel is not None:
                        # 急加速/急减速都会降低舒适度
                        discomfort = abs(accel)
                        total_discomfort += discomfort
                except traci.TraCIException:
                    pass
            
            # 舒适度奖励：低加速度变化获得高奖励
            if len(veh_ids) > 0:
                avg_discomfort = total_discomfort / len(veh_ids)
                # 归一化到[0,1]范围，低不适获得高奖励
                comfort_reward = weight * max(0.0, 1.0 - avg_discomfort / 5.0)
            else:
                comfort_reward = 0.0
                
        except traci.TraCIException:
            comfort_reward = 0.0
            
        return comfort_reward
    
    def _compute_emission_reduction_reward(self, weight: float) -> float:
        """计算排放减少奖励：减少怠速和急加速"""
        try:
            veh_ids = traci.vehicle.getIDList()
            total_emission_penalty = 0.0
            
            for vid in veh_ids:
                try:
                    speed = traci.vehicle.getSpeed(vid)
                    # 怠速惩罚：速度接近0时
                    if speed < 1.0:  # 怠速阈值
                        total_emission_penalty -= 0.05
                    
                    # 急加速惩罚（已在舒适度中考虑，这里轻微重复）
                    accel = traci.vehicle.getAcceleration(vid)
                    if accel is not None and abs(accel) > 3.0:
                        total_emission_penalty -= 0.02
                        
                except traci.TraCIException:
                    pass
            
            # 排放减少奖励：低排放获得高奖励
            if len(veh_ids) > 0:
                avg_emission_penalty = total_emission_penalty / len(veh_ids)
                # 归一化到[0,1]范围，低排放获得高奖励
                emission_reward = weight * max(0.0, 1.0 + avg_emission_penalty)
            else:
                emission_reward = 0.0
                
        except traci.TraCIException:
            emission_reward = 0.0
            
        return emission_reward
    
    def _compute_constraint_satisfaction(self) -> float:
        """计算约束满足度：奖励遵守物理约束"""
        constraint_reward = 0.0
        
        # 检查速度变化约束满足情况
        if hasattr(self, '_last_action_speeds'):
            for i, prev_speed in enumerate(self._last_action_speeds):
                # 这里简化处理，实际应该从环境获取当前速度
                try:
                    # 获取当前路段的车辆
                    if i < 3:  # 东向段
                        lane_ids = self.lane_groups.get(self.group_keys[i], [])
                    else:  # 西向段
                        lane_ids = self.wb_lane_groups.get(self.group_keys[i-3], [])
                    
                    if lane_ids:
                        speeds = []
                        for lid in lane_ids:
                            try:
                                speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                            except traci.TraCIException:
                                pass
                        
                        if speeds:
                            curr_speed = float(np.mean(speeds))
                            # 速度变化约束满足奖励
                            speed_diff = abs(curr_speed - prev_speed)
                            max_delta_mps = self.max_delta_kph / 3.6
                            if speed_diff <= max_delta_mps:
                                constraint_reward += 0.1  # 满足约束获得小奖励
                            else:
                                constraint_reward -= 0.2  # 违反约束给予惩罚
                except Exception:
                    pass
        
        return constraint_reward
    
    def _compute_reward_continuity(self, traffic_features: Dict[str, float]) -> float:
        """计算连续化奖励信号：避免相位切换导致的奖励不连续"""
        # 相位切换成本
        phase_switch_cost = self._compute_phase_switch_cost(traffic_features)
        
        # 交通流平滑度
        traffic_flow_smoothness = self._compute_traffic_flow_smoothness(traffic_features)
        
        # 队列长度连续性
        queue_length_continuity = self._compute_queue_length_continuity(traffic_features)
        
        # 连续化奖励
        continuity_reward = (
            -0.3 * phase_switch_cost +
            0.4 * traffic_flow_smoothness +
            0.3 * queue_length_continuity
        )
        
        return continuity_reward
    
    def _compute_phase_switch_cost(self, traffic_features: Dict[str, float]) -> float:
        """计算相位切换成本：惩罚过于频繁的切换"""
        tls_states = traffic_features['tls_states']
        
        total_switch_cost = 0.0
        switch_frequency_penalty = 0.0
        inappropriate_timing_penalty = 0.0
        
        for tid in self.tls_ids:
            if tid in tls_states:
                is_green = tls_states[tid]['is_green']
                remain = tls_states[tid]['remain']
                
                # 切换频率惩罚：频繁切换惩罚
                if not is_green and remain < 5.0:  # 即将切换到绿灯
                    switch_frequency_penalty += 0.1
                
                # 不当时机惩罚：在交通流未充分释放时切换
                if is_green and remain > 50.0:  # 绿灯时间过长，可能造成浪费
                    inappropriate_timing_penalty += 0.1
        
        total_switch_cost = switch_frequency_penalty + inappropriate_timing_penalty
        
        return total_switch_cost
    
    def _compute_traffic_flow_smoothness(self, traffic_features: Dict[str, float]) -> float:
        """计算交通流平滑度：奖励稳定的交通流"""
        avg_speed_eb = traffic_features['avg_speed_eb']
        avg_speed_wb = traffic_features['avg_speed_wb']
        
        # 交通流平滑度：速度变化越小越好
        speed_variance = abs(avg_speed_eb - avg_speed_wb)
        
        # 平滑度奖励：低方差获得高奖励
        smoothness = max(0.0, 1.0 - speed_variance)
        
        return smoothness
    
    def _compute_queue_length_continuity(self, traffic_features: Dict[str, float]) -> float:
        """计算队列长度连续性：避免队列长度突变"""
        # 简化实现：基于占有率变化
        if not self.detector_ids:
            return 0.0
        
        # 计算当前总队列长度
        total_queue_length = 0.0
        total_vehicles = 0
        
        for det in self.detector_ids:
            try:
                veh_num = traci.lanearea.getLastStepVehicleNumber(det)
                halting_veh = traci.lanearea.getLastStepHaltingNumber(det)
                total_vehicles += veh_num
                # 使用停车车辆数作为队列长度的代理
                total_queue_length += halting_veh
            except traci.TraCIException:
                pass
        
        if total_vehicles > 0:
            # 队列长度连续性：队列比例越低越好
            queue_ratio = total_queue_length / total_vehicles
            # 归一化：队列比例低获得高奖励
            queue_stability = max(0.0, 1.0 - queue_ratio)
        else:
            queue_stability = 0.0
            
        return queue_stability
    
    def _compute_demand_change(self) -> float:
        """计算交通需求变化"""
        # 简化实现：基于最近的速度变化
        if not hasattr(self, '_last_action_speeds'):
            return 0.0
        
        # 计算速度变化率
        prev_speeds = list(self._last_action_speeds)
        curr_speeds = []
        
        # 获取当前速度（简化处理）
        for i in range(6):
            try:
                if i < 3:  # 东向段
                    lane_ids = self.lane_groups.get(self.group_keys[i], [])
                else:  # 西向段
                    lane_ids = self.wb_lane_groups.get(self.group_keys[i-3], [])
                
                if lane_ids:
                    speeds = []
                    for lid in lane_ids:
                        try:
                            speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                        except traci.TraCIException:
                            pass
                    
                    if speeds:
                        curr_speeds.append(float(np.mean(speeds)))
            except Exception:
                curr_speeds.append(0.0)
        
        # 计算变化率
        if prev_speeds and curr_speeds:
            speed_changes = [abs(curr - prev) for curr, prev in zip(curr_speeds, prev_speeds)]
            avg_change = float(np.mean(speed_changes))
            # 归一化到[0,1]范围
            demand_change = min(1.0, avg_change / 10.0)  # 假设最大速度差为10m/s
        else:
            demand_change = 0.0
            
        return demand_change
    
    def _normalize_reward(self, reward: float, traffic_features: Dict[str, float]) -> float:
        """奖励归一化与量纲统一"""
        # 基于交通强度动态调整奖励范围
        congestion_level = traffic_features['congestion_level']
        
        # 动态裁剪范围：拥堵时更保守，畅通时更宽松
        if congestion_level > 0.7:  # 重度拥堵
            clip_min, clip_max = -2.0, 2.0
        elif congestion_level > 0.3:  # 轻度拥堵
            clip_min, clip_max = -3.0, 3.0
        else:  # 自由流
            clip_min, clip_max = -5.0, 5.0
        
        # 应用动态裁剪（修复None值比较问题）
        if self._reward_clip_min is not None:
            clip_min = max(clip_min, self._reward_clip_min)
        if self._reward_clip_max is not None:
            clip_max = min(clip_max, self._reward_clip_max)
        
        normalized_reward = max(clip_min, min(clip_max, reward))
        
        return normalized_reward
    
    # --------------------------------------------------------
    def _compute_green_wave_reward(self, tls_states: Dict) -> float:
        """
        绿波命中率奖励：如果车辆按当前速度到达ETA时恰好遇到绿灯。
        
        逻辑：
        - 计算各段到下游信号的ETA
        - 如果ETA < 剩余绿灯时间，说明可以在绿灯期通过，给予正奖励
        - 如果ETA接近下一个绿灯周期，也给予奖励（简化版本）
        """
        total_gw_reward = 0.0
        
        # 东向段
        for gname in self.group_keys:
            if gname == "down":  # down段无信号
                continue
                
            lane_ids = self.lane_groups.get(gname, [])
            if not lane_ids:
                continue
            
            # 获取当前段平均速度
            speeds = []
            for lid in lane_ids:
                try:
                    speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                except Exception:
                    pass
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            
            if avg_speed < 0.1:  # 静止，无法计算ETA
                continue
            
            # 计算ETA
            distance = self._segment_to_signal_distance.get(f"eb_{gname}", 300.0)
            eta_s = distance / max(avg_speed, 0.1)
            
            # 获取下游信号灯状态
            downstream_tls = {"upstream": "J1", "mid": "J2"}
            tls_id = downstream_tls.get(gname)
            
            if tls_id and tls_id in tls_states:
                is_green = tls_states[tls_id]['is_green']
                remain_s = tls_states[tls_id]['remain']
                
                if is_green:
                    # 当前是绿灯：如果ETA < 剩余绿灯，奖励
                    if eta_s < remain_s:
                        # 匹配度：越接近剩余时间越好
                        match_score = 1.0 - abs(eta_s - remain_s * 0.5) / max(remain_s, 1.0)
                        total_gw_reward += max(0.0, match_score)
                    # 否则，可能赶不上，不奖励
                else:
                    # 当前是红灯：需要估计下一个绿灯时间（简化：假设周期60s，绿灯30s）
                    # 如果ETA接近remain_s + 30s，可以小奖励
                    cycle_s = 60.0
                    next_green_s = remain_s + 30.0  # 简化估计
                    if abs(eta_s - next_green_s) < 15.0:  # 15s容差
                        match_score = 1.0 - abs(eta_s - next_green_s) / 15.0
                        total_gw_reward += max(0.0, match_score) * 0.5  # 次级奖励
        
        # 西向同理（简化版本）
        for gname in self.group_keys:
            if gname == "down":
                continue
                
            lane_ids = self.wb_lane_groups.get(gname, [])
            if not lane_ids:
                continue
            
            speeds = []
            for lid in lane_ids:
                try:
                    speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                except Exception:
                    pass
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            
            if avg_speed < 0.1:
                continue
            
            distance = self._segment_to_signal_distance.get(f"wb_{gname}", 300.0)
            eta_s = distance / max(avg_speed, 0.1)
            
            downstream_tls_wb = {"upstream": "J1", "mid": "J0"}
            tls_id = downstream_tls_wb.get(gname)
            
            if tls_id and tls_id in tls_states:
                is_green = tls_states[tls_id]['is_green']
                remain_s = tls_states[tls_id]['remain']
                
                if is_green:
                    if eta_s < remain_s:
                        match_score = 1.0 - abs(eta_s - remain_s * 0.5) / max(remain_s, 1.0)
                        total_gw_reward += max(0.0, match_score)
                else:
                    next_green_s = remain_s + 30.0
                    if abs(eta_s - next_green_s) < 15.0:
                        match_score = 1.0 - abs(eta_s - next_green_s) / 15.0
                        total_gw_reward += max(0.0, match_score) * 0.5
        
        return total_gw_reward

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

    # --------------------------------------------------------
    def _cleanup_memory(self):
        """清理内存，防止长时间运行时的内存泄漏"""
        try:
            # 清理车辆跟踪字典中的已离开车辆
            current_vehicles = set()
            try:
                current_vehicles = set(traci.vehicle.getIDList())
            except Exception:
                pass
            
            # 清理已离开车辆的跟踪数据
            vehicles_to_remove = []
            for vid in self._veh_energy_j.keys():
                if vid not in current_vehicles:
                    vehicles_to_remove.append(vid)
            for vid in vehicles_to_remove:
                self._veh_energy_j.pop(vid, None)
                self._veh_prev_dist.pop(vid, None)
                self._veh_dist_m.pop(vid, None)
                self._veh_time_loss_s.pop(vid, None)
                self._veh_type_max_speed.pop(vid, None)
                self._controlled_veh_ids.discard(vid)
            
            if vehicles_to_remove:
                print(f"[内存清理] 清理{len(vehicles_to_remove)}辆已离开车辆的跟踪数据")
            
            # 限制历史记录长度
            max_history_length = 100
            if hasattr(self, '_action_history'):
                if len(self._action_history) > max_history_length:
                    self._action_history = self._action_history[-max_history_length:]
            
            # 监控内存使用情况
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > 2000:  # 超过2GB时发出警告
                    print(f"[内存警告] 当前内存使用: {memory_mb:.1f}MB")
            except ImportError:
                # psutil未安装，跳过内存监控
                pass
            except Exception as e:
                print(f"[内存监控] 获取内存使用失败: {e}")
                
        except Exception as e:
            print(f"[内存清理] 清理过程失败: {e}")

    # 简化的错误处理：不再实现SUMO重启机制
    # SUMO崩溃时直接抛出异常，由训练脚本处理
    
    def _capture_restart_state(self) -> dict:
        """捕获重启所需的状态信息"""
        try:
            # 捕获基本状态
            restart_state = {
                'target_time': self.sim_time,
                'last_action_speeds': self._last_action_speeds,
                'last_action_index': self._last_action_index,
                'controlled_vehicles': list(self._controlled_veh_ids),
                'veh_type_max_speed': dict(self._veh_type_max_speed),
                'prev_occ_eb': dict(self._prev_occ_eb),
                'prev_speed_eb': dict(self._prev_speed_eb),
                'prev_occ_wb': dict(self._prev_occ_wb),
                'prev_speed_wb': dict(self._prev_speed_wb),
                'metrics': dict(self.metrics),
                'action_history': getattr(self, '_action_history', []),
            }
            
            # 捕获车辆快照
            try:
                vehicle_snapshot = {}
                all_vehicles = traci.vehicle.getIDList()
                for vid in all_vehicles:
                    try:
                        vehicle_snapshot[vid] = {
                            'type_id': traci.vehicle.getTypeID(vid),
                            'lane_id': traci.vehicle.getLaneID(vid),
                            'position': traci.vehicle.getPosition(vid),
                            'speed': traci.vehicle.getSpeed(vid),
                            'max_speed': traci.vehicle.getMaxSpeed(vid),
                        }
                    except Exception:
                        continue
                restart_state['vehicle_snapshot'] = vehicle_snapshot
            except Exception as e:
                print(f"[SUMO重启] 捕获车辆快照失败: {e}")
                restart_state['vehicle_snapshot'] = {}
            
            print(f"[SUMO重启] 捕获状态: 时间={restart_state['target_time']:.1f}s, "
                  f"控制车辆={len(restart_state['controlled_vehicles'])}, "
                  f"总车辆={len(restart_state['vehicle_snapshot'])}")
            return restart_state
            
        except Exception as e:
            print(f"[SUMO重启] 捕获重启状态失败: {e}")
            return {}
    
    def _advance_to_target_time(self, target_time: float):
        """推进仿真到目标时间点（不包含暖机）"""
        current_time = 0.0
        safety_steps = 0
        max_safety_steps = int(target_time * 2)  # 安全步数上限
        
        # 推进到目标时间（不执行暖机，因为暖机已在_init_sumo中完成）
        while current_time < target_time and safety_steps < max_safety_steps:
            try:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                safety_steps += 1
            except Exception as e:
                print(f"[SUMO重启] 仿真推进失败: {e}")
                break
        
        if current_time < target_time:
            print(f"[SUMO重启] 警告：未能完全推进到目标时间 {target_time:.1f}s，当前时间 {current_time:.1f}s")
    
    def _restore_restart_state(self, restart_state: dict):
        """恢复重启状态，包括车辆快照"""
        # 恢复基本状态
        self.sim_time = restart_state['target_time']
        self._last_action_speeds = restart_state['last_action_speeds']
        self._last_action_index = restart_state['last_action_index']
        self._controlled_veh_ids = set()
        self._veh_type_max_speed = {}
        self._prev_occ_eb = restart_state['prev_occ_eb']
        self._prev_speed_eb = restart_state['prev_speed_eb']
        self._prev_occ_wb = restart_state['prev_occ_wb']
        self._prev_speed_wb = restart_state['prev_speed_wb']
        self.metrics = restart_state['metrics']
        
        # 恢复动作历史
        self._action_history = restart_state['action_history']
        
        # 恢复车辆快照（关键改进）
        self._restore_vehicle_snapshot(restart_state.get('vehicle_snapshot', {}))
        
        # 恢复被控车辆状态
        self._restore_controlled_vehicles(
            restart_state['controlled_vehicles'],
            restart_state['veh_type_max_speed']
        )
    
    def _reapply_action_history(self, action_history: list):
        """
        重新应用历史动作（修复时间重复推进问题）
        
        关键修复：不再推进仿真时间，因为时间已经在_advance_to_target_time中同步
        只需要重新应用速度限制，确保车辆控制状态一致
        """
        if not action_history:
            return
        
        print(f"[SUMO重启] 重新应用 {len(action_history)} 个历史动作（仅速度控制，不推进时间）")
        
        # 只应用最后一个动作，确保当前速度限制正确
        # 因为时间已经推进到崩溃前，不需要重复应用所有历史动作
        if action_history:
            last_action, last_speeds = action_history[-1]
            try:
                # 只应用最后一个动作的速度设置
                self._apply_action_to_environment(last_action, last_speeds)
                print(f"[SUMO重启] 已应用最新动作速度设置: {[f'{s*3.6:.1f}km/h' for s in last_speeds]}")
            except Exception as e:
                print(f"[SUMO重启] 应用最新动作失败: {e}")
    
    def _apply_action_to_environment(self, action, speeds):
        """将动作应用到环境"""
        # 东向：分段应用限速
        eb_speeds = speeds[0:3]
        for gname, target_speed in zip(self.group_keys, eb_speeds):
            for lid in self.lane_groups.get(gname, []):
                try:
                    vehs = traci.lane.getLastStepVehicleIDs(lid)
                    for vid in vehs:
                        if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                            if vid not in self._veh_type_max_speed:
                                vtype = traci.vehicle.getTypeID(vid)
                                tmax = float(traci.vehicletype.getMaxSpeed(vtype))
                                self._veh_type_max_speed[vid] = tmax
                            traci.vehicle.setMaxSpeed(vid, float(target_speed))
                            self._controlled_veh_ids.add(vid)
                except Exception:
                    continue
        
        # 西向：同理
        wb_speeds = speeds[3:6]
        for gname, target_speed in zip(self.group_keys, wb_speeds):
            for lid in self.wb_lane_groups.get(gname, []):
                try:
                    vehs = traci.lane.getLastStepVehicleIDs(lid)
                    for vid in vehs:
                        if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                            if vid not in self._veh_type_max_speed:
                                vtype = traci.vehicle.getTypeID(vid)
                                tmax = float(traci.vehicletype.getMaxSpeed(vtype))
                                self._veh_type_max_speed[vid] = tmax
                            traci.vehicle.setMaxSpeed(vid, float(target_speed))
                            self._controlled_veh_ids.add(vid)
                except Exception:
                    continue
    
    def _restore_controlled_vehicles(self, controlled_vehicles: List[str], veh_type_max_speed: Dict[str, float]):
        """
        恢复被控车辆状态
        
        参数：
            controlled_vehicles: 之前被控制的车辆ID列表
            veh_type_max_speed: 车辆类型最大速度映射
        """
        try:
            # 获取当前仿真中的车辆
            current_vehicles = set()
            try:
                current_vehicles = set(traci.vehicle.getIDList())
            except Exception as e:
                print(f"[恢复车辆] 获取当前车辆列表失败: {e}")
                return
            
            # 找出仍然存在的车辆
            existing_vehicles = [vid for vid in controlled_vehicles if vid in current_vehicles]
            
            print(f"[恢复车辆] 尝试恢复{len(controlled_vehicles)}辆车，实际存在{len(existing_vehicles)}辆")
            
            # 恢复车辆控制状态
            for vid in existing_vehicles:
                try:
                    # 检查是否为CAV
                    if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                        # 恢复最大速度缓存
                        if vid in veh_type_max_speed:
                            self._veh_type_max_speed[vid] = veh_type_max_speed[vid]
                        
                        # 重新应用当前限速
                        if len(self._last_action_speeds) >= 6:
                            # 确定车辆所在段并应用相应限速
                            lane_id = traci.vehicle.getLaneID(vid)
                            
                            # 确定车辆所在段和方向
                            target_speed = 20.0  # 默认速度
                            
                            # 东向段判断
                            for i, gname in enumerate(self.group_keys):
                                lane_ids = self.lane_groups.get(gname, [])
                                if lane_id in lane_ids:
                                    target_speed = self._last_action_speeds[i]
                                    break
                            
                            # 西向段判断
                            if target_speed == 20.0:  # 如果东向未找到，尝试西向
                                for i, gname in enumerate(self.group_keys):
                                    lane_ids = self.wb_lane_groups.get(gname, [])
                                    if lane_id in lane_ids:
                                        target_speed = self._last_action_speeds[3 + i]
                                        break
                            
                            # 应用限速
                            traci.vehicle.setMaxSpeed(vid, float(target_speed))
                            self._controlled_veh_ids.add(vid)
                            
                except Exception as e:
                    print(f"[恢复车辆] 恢复车辆{vid}失败: {e}")
            
            print(f"[恢复车辆] 成功恢复{len(self._controlled_veh_ids)}辆CAV的控制状态")
            
        except Exception as e:
            print(f"[恢复车辆] 恢复过程失败: {e}")
    
    def _restore_vehicle_snapshot(self, vehicle_snapshot: dict):
        """
        恢复车辆快照：重建车辆分布和状态
        
        参数：
            vehicle_snapshot: 车辆快照字典 {vid: {type_id, lane_id, position, speed, max_speed}}
        """
        if not vehicle_snapshot:
            print("[车辆快照] 没有车辆快照可恢复")
            return
        
        try:
            current_vehicles = set()
            try:
                current_vehicles = set(traci.vehicle.getIDList())
            except Exception as e:
                print(f"[车辆快照] 获取当前车辆列表失败: {e}")
                return
            
            print(f"[车辆快照] 尝试恢复{len(vehicle_snapshot)}辆车，当前存在{len(current_vehicles)}辆")
            
            # 对于快照中的每辆车，尝试恢复其状态
            restored_count = 0
            for vid, snapshot in vehicle_snapshot.items():
                if vid not in current_vehicles:
                    # 车辆已不存在，可能已经离开仿真，跳过
                    continue
                
                try:
                    # 恢复车辆速度
                    current_speed = traci.vehicle.getSpeed(vid)
                    snapshot_speed = snapshot.get('speed', current_speed)
                    
                    # 如果速度差异较大，进行调整
                    if abs(current_speed - snapshot_speed) > 1.0:  # 速度差超过1m/s
                        traci.vehicle.setSpeed(vid, snapshot_speed)
                        restored_count += 1
                    
                    # 恢复车辆类型最大速度
                    snapshot_max_speed = snapshot.get('max_speed', 27.78)
                    self._veh_type_max_speed[vid] = snapshot_max_speed
                    
                    # 如果是CAV，重新应用当前限速
                    if snapshot.get('type_id') == self.cav_type_id:
                        # 确定车辆所在段并应用相应限速
                        lane_id = traci.vehicle.getLaneID(vid)
                        target_speed = 20.0  # 默认速度
                        
                        # 东向段判断
                        for i, gname in enumerate(self.group_keys):
                            lane_ids = self.lane_groups.get(gname, [])
                            if lane_id in lane_ids:
                                target_speed = self._last_action_speeds[i]
                                break
                        
                        # 西向段判断
                        if target_speed == 20.0:  # 如果东向未找到，尝试西向
                            for i, gname in enumerate(self.group_keys):
                                lane_ids = self.wb_lane_groups.get(gname, [])
                                if lane_id in lane_ids:
                                    target_speed = self._last_action_speeds[3 + i]
                                    break
                        
                        # 应用限速
                        traci.vehicle.setMaxSpeed(vid, float(target_speed))
                        self._controlled_veh_ids.add(vid)
                        
                except Exception as e:
                    print(f"[车辆快照] 恢复车辆{vid}失败: {e}")
                    continue
            
            print(f"[车辆快照] 成功恢复{restored_count}辆车的状态，控制{len(self._controlled_veh_ids)}辆CAV")
            
        except Exception as e:
            print(f"[车辆快照] 恢复车辆快照失败: {e}")
    
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
        detector_ids=None,
        decision_interval=5,
        same_speed_for_all=False,
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
