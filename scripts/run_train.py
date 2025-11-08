import os
import sys
import random
from collections import deque
import argparse
import datetime

import numpy as np
import torch
import yaml
import xml.etree.ElementTree as ET

# package root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.city_vsl_env import CityVSLEnv  # noqa
from rl.sac_agent import MultiHeadSACAgent  # 改进版多头离散SAC
from rl.replay_buffer import ReplayBuffer  # 你现有的
from rl.prioritized_replay_buffer import PrioritizedReplayBuffer  # PER（可选）
import pickle


def set_seed(seed: int = 0, worker_id: int = 0):
    """
    设置全局随机种子，确保实验可重现性
    
    参数：
        seed: 基础随机种子值
        worker_id: 工作进程ID，用于多环境并行时生成独立种子
    """
    # 为每个worker生成独立的种子
    worker_seed = seed + worker_id * 10000  # 确保不同worker有足够差异
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
        # 确保CUDA操作确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量确保SUMO可重现性
    os.environ['PYTHONHASHSEED'] = str(worker_seed)
    os.environ['SUMO_RANDOM'] = "false"  # 禁用SUMO的随机性
    
    print(f"[随机种子] 设置种子: 基础={seed}, worker={worker_id}, 实际={worker_seed}")
    
def _save_complete_checkpoint(episode: int, step: int, env, agent, save_dir: str):
    """
    保存完整的训练检查点，关联环境和智能体状态
    
    参数：
        episode: 当前回合数
        step: 当前步数
        env: 环境实例
        agent: 智能体实例
        save_dir: 保存目录
    """
    try:
        checkpoint_dir = os.path.join(save_dir, "complete_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_name = f"complete_checkpoint_ep{episode}_step{step}.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # 保存环境状态
        env_checkpoint_data = {
            'episode': episode,
            'step': step,
            'sim_time': getattr(env, 'sim_time', 0.0),
            'last_action_speeds': getattr(env, '_last_action_speeds', (20.0,) * 6),
            'last_action_index': getattr(env, '_last_action_index', -1),
            'controlled_veh_ids': list(getattr(env, '_controlled_veh_ids', set())),
            'veh_type_max_speed': dict(getattr(env, '_veh_type_max_speed', {})),
            'prev_occ_eb': dict(getattr(env, '_prev_occ_eb', {})),
            'prev_speed_eb': dict(getattr(env, '_prev_speed_eb', {})),
            'prev_occ_wb': dict(getattr(env, '_prev_occ_wb', {})),
            'prev_speed_wb': dict(getattr(env, '_prev_speed_wb', {})),
            'metrics': dict(getattr(env, 'metrics', {})),
            'veh_energy_j': dict(getattr(env, '_veh_energy_j', {})),
            'veh_prev_dist': dict(getattr(env, '_veh_prev_dist', {})),
            'veh_dist_m': dict(getattr(env, '_veh_dist_m', {})),
            'veh_time_loss_s': dict(getattr(env, '_veh_time_loss_s', {})),
            'action_history': getattr(env, '_action_history', []),
            'restart_count': getattr(env, '_restart_count', 0),
            'worker_id': getattr(env, 'worker_id', 0),
            'base_seed': getattr(env, 'base_seed', 0),
            'actual_seed': getattr(env, 'actual_seed', 0),
        }
        
        # 保存智能体状态
        agent_state = agent.get_state() if hasattr(agent, 'get_state') else {}
        
        # 合并环境和智能体状态
        complete_checkpoint_data = {
            'env_state': env_checkpoint_data,
            'agent_state': agent_state,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(complete_checkpoint_data, f)
        
        # 只保留最近的5个完整检查点
        _cleanup_old_complete_checkpoints(checkpoint_dir, keep_latest=5)
        
        print(f"[完整检查点] 保存完整检查点: {checkpoint_path}")
        
    except Exception as e:
        print(f"[完整检查点] 保存完整检查点失败: {e}")


def _cleanup_old_complete_checkpoints(checkpoint_dir: str, keep_latest: int = 5):
    """清理旧的完整检查点文件，只保留最新的几个"""
    try:
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "complete_checkpoint_*.pkl"))
        
        # 按修改时间排序，保留最新的几个
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # 删除多余的检查点
        for checkpoint_file in checkpoint_files[keep_latest:]:
            try:
                os.remove(checkpoint_file)
                print(f"[完整检查点] 删除旧完整检查点: {checkpoint_file}")
            except Exception as e:
                print(f"[完整检查点] 删除完整检查点失败: {e}")
                
    except Exception as e:
        print(f"[完整检查点] 清理完整检查点失败: {e}")


def _load_complete_checkpoint(checkpoint_path: str):
    """
    加载完整的训练检查点
    
    参数：
        checkpoint_path: 检查点文件路径
    
    返回：
        tuple: (episode, step, env_state, agent_state)
    """
    try:
        with open(checkpoint_path, 'rb') as f:
            complete_checkpoint_data = pickle.load(f)
        
        env_state = complete_checkpoint_data.get('env_state', {})
        agent_state = complete_checkpoint_data.get('agent_state', {})
        
        episode = env_state.get('episode', 0)
        step = env_state.get('step', 0)
        
        print(f"[完整检查点] 加载完整检查点成功: 回合{episode}, 步数{step}")
        return episode, step, env_state, agent_state
        
    except Exception as e:
        print(f"[完整检查点] 加载完整检查点失败: {e}")
        return 0, 0, {}, {}




def _load_yaml(path: str) -> dict:
    # 优先识别 BOM，避免误解码；同时清理遗留的 BOM/NUL 控制字符
    with open(path, "rb") as fb:
        data = fb.read()
    # 识别 BOM
    enc = None
    if data.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    elif data.startswith(b"\xff\xfe"):
        enc = "utf-16-le"
    elif data.startswith(b"\xfe\xff"):
        enc = "utf-16-be"
    else:
        enc = "utf-8"
    text = ""
    try:
        text = data.decode(enc)
    except Exception:
        # 退一步尝试更宽松编码
        decoded = False
        for fallback in ("utf-8", "gb18030", "latin1"):
            try:
                text = data.decode(fallback)
                decoded = True
                break
            except Exception:
                continue
        if not decoded:
            raise RuntimeError(f"Failed to decode {path} with any encoding")
    # 清理不可见控制字符（BOM/NUL）
    text = text.replace("\ufeff", "").replace("\x00", "")
    return yaml.safe_load(text)


def _get_route_files_value(sumo_cfg_path: str) -> str:
    # 解析 sumo 配置中的 <route-files value="..."/>
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if "<route-files" in line and "value=" in line:
                quote_start = line.find('"')
                quote_end = line.find('"', quote_start + 1)
                if quote_start != -1 and quote_end != -1:
                    return line[quote_start + 1:quote_end]
    raise RuntimeError("Cannot find <route-files> in SUMO config")


def _strip_pen_suffix(route_rel: str) -> str:
    """从文件名中剥离重复的 .penXX 后缀，回到固定基准文件名。
    例如：grid1x3_weighted_mt.rou.pen30.pen45.xml -> grid1x3_weighted_mt.rou.xml
    """
    import re
    name, ext = os.path.splitext(route_rel)
    base_name = re.sub(r"(\.pen\d{2})+$", "", name)
    return base_name + ext


def _patch_sumocfg_route(sumo_cfg_path: str, route_rel_value: str) -> None:
    # Replace the value inside <route-files value="..."/>
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    replaced = False
    for line in lines:
        if "<route-files" in line:
            # naive replace of value attribute
            start = line.find("value=")
            if start != -1:
                prefix = line[:start]
                # find closing quote after value=
                quote_start = line.find('"', start)
                quote_end = line.find('"', quote_start + 1)
                if quote_start != -1 and quote_end != -1:
                    new_line = prefix + f"value=\"{route_rel_value}\"" + line[quote_end + 1:]
                    new_lines.append(new_line)
                    replaced = True
                    continue
        new_lines.append(line)
    if not replaced:
        raise RuntimeError("Failed to patch route-files in SUMO config: tag not found")
    with open(sumo_cfg_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[sumocfg] patched {sumo_cfg_path} -> route-files={route_rel_value}")


def _generate_penetrated_routes_from_base(base_route_path: str, output_path: str, cav_rate: float, cfg: dict) -> None:
    # 参考 生成rou文件.py 的核心思想：读取现有车辆列表，按渗透率随机赋予 CAV/HV 类型，保持原有时序与路线不变
    tree = ET.parse(base_route_path)
    root_in = tree.getroot()

    routes_out = ET.Element("routes")
    vdist = ET.SubElement(routes_out, "vTypeDistribution", id="typedist1")
    # 从配置文件读取CAV/HV参数，确保训练与评估一致
    cav_tau = str(cfg.get("cav_tau", 0.5))
    cav_accel = str(cfg.get("cav_accel", 3.0))
    cav_decel = str(cfg.get("cav_decel", 4.5))
    cav_min_gap = str(cfg.get("cav_min_gap", 0.50))
    cav_speed_dev = str(cfg.get("cav_speed_dev", 0.0))
    cav_sigma = str(cfg.get("cav_sigma", 0))
    
    hv_tau = str(cfg.get("hv_tau", 1.5))
    hv_accel = str(cfg.get("hv_accel", 2.5))
    hv_decel = str(cfg.get("hv_decel", 4.5))
    hv_min_gap = str(cfg.get("hv_min_gap", 0.50))
    hv_speed_dev = str(cfg.get("hv_speed_dev", 0.05))
    
    # 对齐 CAV=AV 参数（从配置读取）
    ET.SubElement(
        vdist,
        "vType",
        id="CAV",
        length="5.0",
        color="0,0,255",
        minGap=cav_min_gap,
        maxSpeed="27.78",
        carFollowModel="IDM",
        accel=cav_accel,
        decel=cav_decel,
        emergencyDecel="6.5",
        tau=cav_tau,
        speedDev=cav_speed_dev,
        sigma=cav_sigma,
        laneChangeModel="LC2013",
        lcAssertive="1.0",
        lcStrategic="1.0",
        lcCooperative="1.0",
        probability="1.0",
        electricBatteryCapacity="50",
        vehicleClass="electric",
        emissionClass="Energy",
    )
    # 对齐 HV=HDV-G 参数（从配置读取）
    ET.SubElement(
        vdist,
        "vType",
        id="HV",
        length="5.0",
        color="255,0,0",
        minGap=hv_min_gap,
        maxSpeed="27.78",
        carFollowModel="IDM",
        accel=hv_accel,
        decel=hv_decel,
        emergencyDecel="6.5",
        tau=hv_tau,
        speedDev=hv_speed_dev,
        laneChangeModel="LC2013",
        lcAssertive="1.0",
        lcStrategic="1.0",
        lcCooperative="1.0",
        probability="0.0",
    )

    # 遍历所有车辆，复制 id/depart/route，并随机赋型
    import random
    for veh in root_in.findall("vehicle"):
        vid = veh.get("id")
        depart = veh.get("depart")
        if vid is None or depart is None:
            continue
        r = veh.find("route")
        edges = r.get("edges") if r is not None else None
        if not edges:
            # 跳过没有路线的车辆（罕见）
            continue
        vtype = "CAV" if random.random() < cav_rate else "HV"
        new_v = ET.SubElement(routes_out, "vehicle", id=vid, depart=depart, type=vtype, vClass="passenger")
        ET.SubElement(new_v, "route", edges=edges)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ET.ElementTree(routes_out).write(output_path, xml_declaration=True, encoding="UTF-8")
    print(f"[routes] wrote {output_path} | base={os.path.basename(base_route_path)} | CAV penetration={cav_rate}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=str, default="false")
    parser.add_argument("--episodes", type=int, default=None)
    # 可以将来传不同rou做不同CAV比例
    parser.add_argument("--route_suffix", type=str, default="")
    # 设备选择：auto/cuda/cpu（优先命令行，其次配置文件）
    parser.add_argument("--device", type=str, default=None, help="训练设备：auto/cuda/cpu，默认auto")
    # 参数扫描支持
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认config/env.yaml）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认outputs/run_YYYYMMDD_HHMMSS）")
    # 检查点和恢复相关参数
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认0）")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--checkpoint_interval", type=int, default=None, help="检查点保存间隔（回合数）")
    parser.add_argument("--max_errors_per_episode", type=int, default=10, help="每回合最大错误次数")
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # 支持自定义配置文件路径
    cfg_path = args.config if args.config else os.path.join(root, "config", "env.yaml")
    cfg = _load_yaml(cfg_path)

    use_gui = str(args.gui).lower() in ("1", "true", "yes")
    # 设备解析：命令行优先，其次 env.yaml，默认 auto
    dev_pref = (args.device or str(cfg.get("device", "auto"))).lower()
    if dev_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev_pref in ("cuda", "cpu"):
        device = dev_pref
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] --device=cuda 指定，但当前环境未检测到可用 CUDA，回退到 CPU")
            device = "cpu"
    else:
        print(f"[warn] 未识别的设备 '{dev_pref}'，使用 auto 策略")
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[device] 使用设备: {device}")

    # 使用 env.yaml 的路径；默认回退到 grid1x3
    sumo_cfg_rel = cfg.get("sumo_cfg_relpath", "sumo/grid1x3.sumocfg")
    # 如果想做多渗透率，可以在 cfg 里写多个 cfg，再根据 route_suffix 切
    sumo_cfg_path = os.path.join(root, sumo_cfg_rel)
    # 固定基准：剥离任何 .penXX 叠加后缀，始终以原始文件作为生成基准
    cur_route_rel = _get_route_files_value(sumo_cfg_path)
    base_route_rel = _strip_pen_suffix(cur_route_rel)
    base_route_abs = os.path.join(os.path.dirname(sumo_cfg_path), base_route_rel)

    pen = float(cfg.get("penetration", 0.3))
    pen = max(0.0, min(1.0, pen))
    # 输出文件名：在原文件名上追加 .penXX
    name, ext = os.path.splitext(base_route_rel)
    out_rel = f"{name}.pen{int(pen*100):02d}{ext}"
    out_abs = os.path.join(os.path.dirname(sumo_cfg_path), out_rel)

    # 仅首次生成；若文件已存在则直接复用，不再重复生成
    if not os.path.exists(out_abs):
        _generate_penetrated_routes_from_base(base_route_abs, out_abs, pen, cfg)
    else:
        print(f"[routes] reuse existing {out_rel}")
    # 指向规范化的渗透率文件，避免名称层层叠加
    if cur_route_rel != out_rel:
        _patch_sumocfg_route(sumo_cfg_path, out_rel)

    lane_groups = cfg.get(
        "lane_groups",
        {"upstream": ["J0_J1_0", "J0_J1_1", "J0_J1_2"], "mid": ["J1_J2_0", "J1_J2_1", "J1_J2_2"], "down": ["J2_E_0", "J2_E_1", "J2_E_2"]},
    )
    tls_ids = cfg.get("tls_ids", ["J0", "J1", "J2"])

    speeds_kph = cfg.get("discrete_speeds_kph")
    if not speeds_kph:
        mps = cfg.get("action_speeds_mps", [8.33, 10.0, 12.0, 14.0])
        speeds_kph = [round(v * 3.6) for v in mps]

    decision_interval = int(cfg.get("decision_interval", 5))

    detector_ids = cfg.get("detector_ids", None)
    # 奖励裁剪配置
    _rc_min = cfg.get("reward_clip_min", None)
    _rc_max = cfg.get("reward_clip_max", None)

    # 为多环境并行支持，获取worker ID
    worker_id = int(os.environ.get('WORKER_ID', '0'))
    
    # 设置独立种子
    actual_seed = set_seed(args.seed, worker_id)
    
    env = CityVSLEnv(
        sumo_cfg_path=sumo_cfg_path,
        use_gui=use_gui,
        backend=str(cfg.get("backend", "auto")),
        lane_groups=lane_groups,
        lane_groups_wb=cfg.get("lane_groups_wb", {"upstream": [], "mid": [], "down": []}),
        tls_ids=tls_ids,
        discrete_speeds_kph=speeds_kph,
        decision_interval=decision_interval,
        detector_ids=detector_ids,
        same_speed_for_all=False,
        max_sim_seconds=int(cfg.get("max_sim_seconds", 3600)),
        # 论文式硬约束：一次不超过10km/h
        max_delta_kph=float(cfg.get("max_delta_kph", 10.0)),
        # 从 env.yaml 接入奖励权重与暖机步数
        reward_speed_weight=float(cfg.get("reward_speed_weight", 1.0)),
        reward_bottleneck_weight=float(cfg.get("reward_bottleneck_weight", 1.0)),
        reward_congestion_weight=float(cfg.get("reward_congestion_weight", 1.0)),
        reward_change_penalty=float(cfg.get("reward_change_penalty", 0.1)),
        reward_queue_weight_eb=float(cfg.get("queue_weight_eb", 0.5)),
        reward_queue_weight_wb=float(cfg.get("queue_weight_wb", 0.3)),
        reward_green_wave_weight=float(cfg.get("reward_green_wave_weight", 3.0)),  # 城市VSL优化
        occ_penalty_start=float(cfg.get("occ_penalty_start", 0.7)),
        occ_penalty_power=float(cfg.get("occ_penalty_power", 2.0)),
        warmup_steps=int(cfg.get("warmup_steps", 5)),
        reward_clip_min=_rc_min,
        reward_clip_max=_rc_max,
    )
    
    # 将worker ID和种子信息存储在环境中，便于检查点保存
    env.worker_id = worker_id
    env.base_seed = args.seed
    env.actual_seed = actual_seed

    obs, _ = env.reset()
    # 类型注解：确保observation_space不为None
    assert env.observation_space is not None, "observation_space cannot be None"
    obs_dim = env.observation_space.shape[0]  # type: ignore
    # MultiDiscrete: nvec（长度=6，每段的档位数）
    if hasattr(env.action_space, "nvec"):
        nvec = list(map(int, env.action_space.nvec))  # type: ignore
    else:
        nvec = [int(env.action_space.n)]  # type: ignore

    # 构建SAC Agent（多头MultiDiscrete）
    sac_alpha_lr_val = cfg.get("sac_alpha_lr")
    alpha_lr = float(sac_alpha_lr_val) if sac_alpha_lr_val is not None else 1e-3
    
    target_entropy_val = cfg.get("sac_target_entropy")
    target_entropy = float(target_entropy_val) if target_entropy_val is not None else None  # type: ignore
    
    auto_alpha = bool(cfg.get("sac_auto_alpha", True))
    alpha_min_val = cfg.get("sac_alpha_min")
    alpha_min = float(alpha_min_val) if auto_alpha and alpha_min_val is not None else None  # type: ignore
    
    alpha_max_val = cfg.get("sac_alpha_max")
    alpha_max = float(alpha_max_val) if auto_alpha and alpha_max_val is not None else None  # type: ignore
    
    agent = MultiHeadSACAgent(
        obs_dim=obs_dim,
        nvec=nvec,
        lr=float(cfg.get("learning_rate", 5e-4)),
        actor_lr=float(cfg.get("actor_lr", cfg.get("learning_rate", 5e-4))),
        critic_lr=float(cfg.get("critic_lr", cfg.get("learning_rate", 5e-4))),
        alpha_lr=alpha_lr,
        gamma=float(cfg.get("gamma", 0.99)),
        target_update_interval=int(cfg.get("sac_target_update_interval", cfg.get("target_update_interval", 300))),
        device=device,
        tau=float(cfg.get("sac_tau", 0.01)),
        alpha=float(cfg.get("sac_alpha", 0.2)),
        auto_alpha=auto_alpha,
        target_entropy=target_entropy,  # type: ignore
        hidden_size=int(cfg.get("hidden_size", 256)),
        max_grad_norm=float(cfg.get("grad_clip_norm", 5.0)),
        alpha_min=alpha_min,  # type: ignore
        alpha_max=alpha_max,  # type: ignore
        # 改进参数
        actor_detach_q=bool(cfg.get("sac_actor_detach_q", True)),
        policy_delay=int(cfg.get("sac_policy_delay", 1)),
        dropout=float(cfg.get("sac_dropout", 0.0)),
        weight_decay=float(cfg.get("sac_weight_decay", 0.0)),
    )
    print(
        f"[SAC] 使用 Polyak 软更新，tau={float(cfg.get('sac_tau', 0.01)):.4f} | policy_delay={int(cfg.get('sac_policy_delay', 1))} | dropout={float(cfg.get('sac_dropout', 0.0)):.2f} | wd={float(cfg.get('sac_weight_decay', 0.0)):.1e}"
    )

    use_per = bool(cfg.get("use_per", False))
    if use_per:
        buffer = PrioritizedReplayBuffer(
            capacity=int(cfg.get("buffer_size", 200000)),
            alpha=float(cfg.get("per_alpha", 0.6)),
            beta_start=float(cfg.get("per_beta_start", 0.4)),
            beta_frames=int(cfg.get("per_beta_frames", 100000)),
            epsilon=float(cfg.get("per_epsilon", 1e-5)),
            n_step=int(cfg.get("n_step", 1)),
            gamma=float(cfg.get("gamma", 0.99)),
        )
        print(f"[PER] 启用优先级经验回放 | alpha={float(cfg.get('per_alpha', 0.6)):.2f} | beta_start={float(cfg.get('per_beta_start', 0.4)):.2f} | n_step={int(cfg.get('n_step', 1))}")
    else:
        buffer = ReplayBuffer(
            capacity=int(cfg.get("buffer_size", 200000)),
            obs_dim=obs_dim,
        )
    batch_size = int(cfg.get("batch_size", 128))

    # 预填充经验池（随机策略）
    prefill_steps = int(cfg.get("prefill_steps", 0) or 0)
    if prefill_steps > 0:
        print(f"[prefill] start prefill_steps={prefill_steps}")
        filled = 0
        obs, _ = env.reset()
        while filled < prefill_steps:
            try:
                action = env.action_space.sample()
            except Exception:
                # 回退：若环境不支持 sample，则使用均匀随机
                import numpy as _np
                action = _np.array([int(_np.random.randint(0, n)) for n in nvec], dtype=_np.int64)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            # 统一使用条件分支调用正确的方法
            if use_per:
                buffer.add(obs, action, reward, next_obs, bool(terminated or truncated), error=None)  # type: ignore
            else:
                buffer.push(obs, action, reward, next_obs, bool(terminated or truncated))  # type: ignore
            obs = next_obs if not (terminated or truncated) else env.reset()[0]
            filled += 1
        print(f"[prefill] done filled={filled}")

    # 仅 SAC 不使用 epsilon，移除DDQN相关代码
    global_steps = 0

    num_episodes = int(args.episodes or cfg.get("episodes", 10))
    # 周期备份频率：每 N 回合保存一次 *_ep{ep}.pt；0/None 表示关闭
    save_every = int(cfg.get("save_every_n_episodes", 0) or 0)
    # 统一决策周期语义（按秒）：按 step-length 折算，每个决策步覆盖 decision_interval 秒
    max_sim_seconds = int(cfg.get("max_sim_seconds", 3600))
    from math import ceil
    max_decision_steps_per_ep = max(1, int(ceil(max_sim_seconds / float(max(decision_interval, 1)))))
    print(f"[timing] decision_interval={decision_interval}s | max_sim_seconds={max_sim_seconds}s => max_decision_steps={max_decision_steps_per_ep}")

    save_dir = args.output_dir if args.output_dir else os.path.join(root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    reward_window = deque(maxlen=20)
    # CSV 指标输出（每回合）
    metrics_dir = args.output_dir if args.output_dir else env.run_output_dir
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv = os.path.join(metrics_dir, "train_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("ep,algo,dueling,tau,reward,avg20,eps_or_alpha,avg_delay_s,avg_queue_veh,throughput_veh_per_hour,avg_energy_j_per_km,arrived_total,sim_seconds,episode_errors,total_errors\n")
    best_avg20 = -1e18
    best_ckpt = None

    # 检查点间隔：优先使用命令行参数，其次配置文件，最后默认10回合
    checkpoint_interval = args.checkpoint_interval or int(cfg.get("checkpoint_interval", 10))
    
    # 错误计数器
    episode_errors = 0
    total_errors = 0
    max_errors_per_episode = args.max_errors_per_episode
    
    # 如果指定了恢复检查点，则加载
    start_episode = 1
    if args.resume:
        try:
            # 首先尝试加载完整检查点
            if os.path.exists(args.resume) and args.resume.endswith('.pkl'):
                # 完整检查点文件
                env_episode, env_step, env_state, agent_state = _load_complete_checkpoint(args.resume)
                
                # 恢复环境状态
                if env_state:
                    env.sim_time = env_state.get('sim_time', 0.0)
                    env._last_action_speeds = env_state.get('last_action_speeds', (20.0,) * 6)
                    env._last_action_index = env_state.get('last_action_index', -1)
                    env._controlled_veh_ids = set(env_state.get('controlled_veh_ids', []))
                    env._veh_type_max_speed = env_state.get('veh_type_max_speed', {})
                    env._prev_occ_eb = env_state.get('prev_occ_eb', {})
                    env._prev_speed_eb = env_state.get('prev_speed_eb', {})
                    env._prev_occ_wb = env_state.get('prev_occ_wb', {})
                    env._prev_speed_wb = env_state.get('prev_speed_wb', {})
                    env.metrics = env_state.get('metrics', {
                        "queue_sum": 0.0,
                        "queue_steps": 0,
                        "arrived_total": 0,
                        "sim_seconds": 0,
                        "energy_total_j": 0.0,
                        "distance_total_m": 0.0,
                        "time_loss_total_s": 0.0,
                    })
                    env._veh_energy_j = env_state.get('veh_energy_j', {})
                    env._veh_prev_dist = env_state.get('veh_prev_dist', {})
                    env._veh_dist_m = env_state.get('veh_dist_m', {})
                    env._veh_time_loss_s = env_state.get('veh_time_loss_s', {})
                    env._action_history = env_state.get('action_history', [])
                    env._restart_count = env_state.get('restart_count', 0)
                    env.worker_id = env_state.get('worker_id', 0)
                    env.base_seed = env_state.get('base_seed', 0)
                    env.actual_seed = env_state.get('actual_seed', 0)
                    
                    # 恢复被控车辆状态
                    env._restore_controlled_vehicles(
                        list(env._controlled_veh_ids),
                        dict(env._veh_type_max_speed)
                    )
                
                # 恢复智能体状态
                if agent_state and hasattr(agent, 'load_state'):
                    agent.load_state(agent_state)
                
                # 设置起始回合
                start_episode = env_episode + 1
                print(f"[恢复] 从完整检查点恢复训练: 回合{env_episode}, 步数{env_step}")
            else:
                # 尝试加载普通检查点文件
                if os.path.exists(args.resume):
                    agent.load(args.resume)
                    print(f"[恢复] 从普通检查点恢复训练: {args.resume}")
                else:
                    print(f"[恢复] 检查点文件不存在: {args.resume}, 从头开始训练")
        except Exception as e:
            print(f"[恢复] 加载检查点失败: {e}, 从头开始训练")
            start_episode = 1

    for ep in range(start_episode, num_episodes + 1):
        obs, _ = env.reset()
        # 调试：打印回合开始时的 SUMO 时间（应接近暖机后时间，如 ~500s）
        try:
            print(f"[reset] EP {ep:03d} | sim_time={getattr(env, 'sim_time', 0.0):.2f}s")
        except Exception:
            pass
        ep_reward = 0.0
        episode_errors = 0  # 重置回合错误计数

        for di_step in range(max_decision_steps_per_ep):
            try:
                # SAC随机动作选择（传递前一时刻动作用于约束）
                action = agent.select_action(obs, deterministic=False, prev_actions=getattr(agent, '_prev_actions', None))
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # 检查错误计数
                step_errors = info.get('step_errors', 0)
                if step_errors > 0:
                    episode_errors += step_errors
                    total_errors += step_errors
                    print(f"[错误] 回合{ep}步{di_step+1}发生{step_errors}个错误，回合总错误{episode_errors}")
                    
                    # 如果错误次数过多，提前结束回合
                    if episode_errors >= max_errors_per_episode:
                        print(f"[警告] 回合{ep}错误次数过多({episode_errors}>={max_errors_per_episode})，提前结束回合")
                        break

                done_flag = bool(terminated or truncated)
                if use_per:
                    # 初始加入优先级使用最大优先级（error=None）；采样后再按TD误差更新
                    buffer.add(obs, action, reward, next_obs, done_flag, error=None)  # type: ignore
                else:
                    buffer.push(obs, action, reward, next_obs, done_flag)  # type: ignore

                if len(buffer) >= batch_size:
                    if use_per:
                        states, actions, rewards, next_states, dones, is_weights, indices = buffer.sample(batch_size, device=device)  # type: ignore
                        metrics = agent.update((states, actions, rewards, next_states, dones), is_weights=is_weights)
                        # 按TD误差更新优先级
                        td_errors = metrics.get('td_errors', None)
                        if td_errors is not None:
                            buffer.update_priorities(indices, td_errors)  # type: ignore
                        loss = metrics.get('total_loss', None)
                    else:
                        loss = agent.update(buffer.sample(batch_size))
                else:
                    loss = None

                ep_reward += reward
                obs = next_obs

                # 计步
                global_steps += 1

                if terminated or truncated:
                    # 调试：明确打印本回合在第几个决策步终止，以及当前仿真时间
                    try:
                        sim_t = info.get('sim_time', getattr(env, 'sim_time', None))
                        print(
                            f"[term] EP {ep:03d} | di_step={di_step+1} | global_steps={global_steps} | sim_time={sim_t:.2f}s | terminated={int(bool(terminated))} | truncated={int(bool(truncated))} | errors={episode_errors}"
                        )
                    except Exception:
                        pass
                    break
                    
            except Exception as e:
                episode_errors += 1
                total_errors += 1
                print(f"[异常] 回合{ep}步{di_step+1}发生未捕获异常: {e}")
                
                # 如果异常次数过多，提前结束回合
                if episode_errors >= max_errors_per_episode:
                    print(f"[警告] 回合{ep}异常次数过多({episode_errors}>={max_errors_per_episode})，提前结束回合")
                    break
                continue

        reward_window.append(ep_reward)
        avg20 = float(np.mean(reward_window)) if reward_window else ep_reward
        print(
            f"[EP {ep:03d}] reward={ep_reward:.2f} | avg20={avg20:.2f} | alpha={getattr(agent, 'temperature', 0.0):.3f} | heads={len(nvec)} | nvec={nvec} | errors={episode_errors} | total_errors={total_errors}"
        )

        # 追加 CSV 指标
        m = env.get_metrics()
        alpha_val = getattr(agent, "temperature", 0.0)
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},sac,0,{float(cfg.get('sac_tau', 0.0) or 0.0):.6f},{ep_reward:.6f},{avg20:.6f},{alpha_val:.6f},{m.get('avg_delay_s',0.0):.6f},{m.get('avg_queue_veh',0.0):.6f},{m.get('throughput_veh_per_hour',0.0):.6f},{m.get('avg_energy_j_per_km',0.0):.6f},{int(m.get('arrived_total',0) or 0):d},{float(m.get('sim_seconds',0.0) or 0.0):.6f},{episode_errors:d},{total_errors:d}\n"
            )

        # 保存环境检查点
        try:
            env._save_checkpoint(ep, global_steps)
        except Exception as e:
            print(f"[检查点] 保存环境检查点失败: {e}")
        
        # 保存完整检查点（关联环境和智能体状态）
        try:
            _save_complete_checkpoint(ep, global_steps, env, agent, save_dir)
        except Exception as e:
            print(f"[完整检查点] 保存完整检查点失败: {e}")

        # 保存最佳模型（按 avg20）
        if avg20 > best_avg20:
            best_avg20 = avg20
            best_name = "sac.pt"
            best_ckpt = os.path.join(save_dir, best_name)
            agent.save(best_ckpt)
            compat_best_name = "best_sac.pt"
            agent.save(os.path.join(save_dir, compat_best_name))
            print(f"[save] new best checkpoint: {best_ckpt} (also wrote {compat_best_name}) | avg20={best_avg20:.2f}")

        # 周期备份：每 N 回合保存 *_ep{ep}.pt
        if save_every > 0 and (ep % save_every == 0):
            ep_name = f"sac_ep{ep}.pt"
            ep_ckpt = os.path.join(save_dir, ep_name)
            agent.save(ep_ckpt)
            print(f"[backup] wrote periodic checkpoint: {ep_ckpt}")
            
        # 检查点间隔保存
        if checkpoint_interval > 0 and (ep % checkpoint_interval == 0):
            checkpoint_name = f"sac_checkpoint_ep{ep}.pt"
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            agent.save(checkpoint_path)
            print(f"[检查点] 保存训练检查点: {checkpoint_path}")
            
            # 同时保存完整检查点
            try:
                _save_complete_checkpoint(ep, global_steps, env, agent, save_dir)
            except Exception as e:
                print(f"[完整检查点] 保存完整检查点失败: {e}")

    env.close()


if __name__ == "__main__":
    main()
