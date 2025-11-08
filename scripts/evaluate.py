import os
import sys
import argparse
import random
import yaml
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

import numpy as np
import torch
from scipy import stats
# 兼容两种后端的导入（评估脚本自身不直接使用，但避免在仅有 libsumo 的环境下导入失败）
try:
    import libsumo as traci  # noqa: F401
except Exception:
    import traci  # noqa: F401

# package root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.city_vsl_env import CityVSLEnv  # noqa
from rl.sac_agent import MultiHeadSACAgent  # 改进版多头离散SAC Agent


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    rewards: List[float]
    delays: List[float]
    queues: List[float]
    throughputs: List[float]
    energies: List[float]
    
    def get_means(self) -> Dict[str, float]:
        """计算各指标均值"""
        return {
            'reward': float(np.mean(self.rewards)) if self.rewards else 0.0,
            'delay': float(np.mean(self.delays)) if self.delays else 0.0,
            'queue': float(np.mean(self.queues)) if self.queues else 0.0,
            'throughput': float(np.mean(self.throughputs)) if self.throughputs else 0.0,
            'energy': float(np.mean(self.energies)) if self.energies else 0.0
        }
    
    def get_stds(self) -> Dict[str, float]:
        """计算各指标标准差"""
        return {
            'reward': float(np.std(self.rewards)) if self.rewards else 0.0,
            'delay': float(np.std(self.delays)) if self.delays else 0.0,
            'queue': float(np.std(self.queues)) if self.queues else 0.0,
            'throughput': float(np.std(self.throughputs)) if self.throughputs else 0.0,
            'energy': float(np.std(self.energies)) if self.energies else 0.0
        }
    
    def get_counts(self) -> Dict[str, int]:
        """获取各指标样本数"""
        return {
            'reward': len(self.rewards),
            'delay': len(self.delays),
            'queue': len(self.queues),
            'throughput': len(self.throughputs),
            'energy': len(self.energies)
        }


class StatisticalAnalyzer:
    """统计分析器，用于评估结果显著性检验"""
    
    @staticmethod
    def t_test(sample1: List[float], sample2: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        执行独立双样本t检验
        
        Args:
            sample1: 第一个样本数据
            sample2: 第二个样本数据
            alpha: 显著性水平
            
        Returns:
            包含t检验结果的字典
        """
        if len(sample1) < 2 or len(sample2) < 2:
            return {
                'valid': False,
                'reason': '样本数不足，至少需要2个样本'
            }
        
        try:
            # 执行t检验
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            
            # 计算置信区间
            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
            n1, n2 = len(sample1), len(sample2)
            
            # 合并标准误差
            se = np.sqrt(std1**2/n1 + std2**2/n2)
            
            # 自由度
            df = n1 + n2 - 2
            
            # 临界值
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # 置信区间
            margin = t_critical * se
            ci_lower = (mean1 - mean2) - margin
            ci_upper = (mean1 - mean2) + margin
            
            # 效应大小 (Cohen's d)
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / df)
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # 解释效应大小
            if abs(cohens_d) < 0.2:
                effect_size = "很小"
            elif abs(cohens_d) < 0.5:
                effect_size = "小"
            elif abs(cohens_d) < 0.8:
                effect_size = "中等"
            else:
                effect_size = "大"
            
            return {
                'valid': True,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'mean_difference': float(mean1 - mean2),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'cohens_d': float(cohens_d),
                'effect_size_interpretation': effect_size,
                'sample1_stats': {
                    'mean': float(mean1),
                    'std': float(std1),
                    'n': n1
                },
                'sample2_stats': {
                    'mean': float(mean2),
                    'std': float(std2),
                    'n': n2
                }
            }
        except Exception as e:
            return {
                'valid': False,
                'reason': f'计算错误: {str(e)}'
            }
    
    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """
        计算置信区间
        
        Args:
            data: 数据样本
            confidence: 置信水平
            
        Returns:
            包含置信区间的字典
        """
        if len(data) < 2:
            return {
                'lower': 0.0,
                'upper': 0.0,
                'mean': 0.0,
                'valid': False
            }
        
        try:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)
            
            # 计算临界值
            alpha = 1 - confidence
            t_critical = stats.t.ppf(1 - alpha/2, n-1)
            
            # 标准误差
            se = std / np.sqrt(n)
            
            # 置信区间
            margin = t_critical * se
            
            return {
                'lower': float(mean - margin),
                'upper': float(mean + margin),
                'mean': float(mean),
                'std': float(std),
                'margin': float(margin),
                'valid': True
            }
        except Exception as e:
            return {
                'lower': 0.0,
                'upper': 0.0,
                'mean': 0.0,
                'valid': False,
                'error': str(e)
            }
    
    @staticmethod
    def compare_policies(baseline_metrics: EvaluationMetrics,
                        best_metrics: EvaluationMetrics,
                        alpha: float = 0.05) -> Dict[str, Any]:
        """
        比较两种策略的评估结果
        
        Args:
            baseline_metrics: 基线策略评估指标
            best_metrics: 最优策略评估指标
            alpha: 显著性水平
            
        Returns:
            包含比较结果的字典
        """
        comparison_results = {}
        
        # 对每个指标进行比较
        for metric_name in ['reward', 'delay', 'queue', 'throughput', 'energy']:
            baseline_data = getattr(baseline_metrics, f"{metric_name}s")
            best_data = getattr(best_metrics, f"{metric_name}s")
            
            # 计算置信区间
            baseline_ci = StatisticalAnalyzer.confidence_interval(baseline_data)
            best_ci = StatisticalAnalyzer.confidence_interval(best_data)
            
            # 执行t检验
            t_test_result = StatisticalAnalyzer.t_test(best_data, baseline_data, alpha)
            
            # 计算改进百分比
            baseline_mean = baseline_ci['mean']
            best_mean = best_ci['mean']
            
            if metric_name in ['reward', 'throughput']:
                # 对于奖励和吞吐量，越高越好
                improvement_pct = ((best_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
                direction = "提升" if improvement_pct > 0 else "下降"
            else:
                # 对于延迟、队列和能耗，越低越好
                improvement_pct = ((baseline_mean - best_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
                direction = "降低" if improvement_pct > 0 else "增加"
            
            comparison_results[metric_name] = {
                'baseline_mean': baseline_mean,
                'best_mean': best_mean,
                'improvement_percentage': improvement_pct,
                'improvement_direction': direction,
                'baseline_ci': baseline_ci,
                'best_ci': best_ci,
                't_test': t_test_result,
                'significant': t_test_result.get('significant', False),
                'effect_size': t_test_result.get('effect_size_interpretation', '未知')
            }
        
        return comparison_results
    
    @staticmethod
    def generate_comparison_report(comparison_results: Dict[str, Any]) -> str:
        """
        生成比较报告
        
        Args:
            comparison_results: 比较结果字典
            
        Returns:
            格式化的比较报告字符串
        """
        report = "\n" + "="*80 + "\n"
        report += "                    策略比较统计显著性报告\n"
        report += "="*80 + "\n\n"
        
        for metric_name, results in comparison_results.items():
            metric_display_name = {
                'reward': '奖励',
                'delay': '平均延迟(s)',
                'queue': '平均队列长度',
                'throughput': '交通流量(veh/h)',
                'energy': '能耗(J/km)'
            }.get(metric_name, metric_name)
            
            report += f"【{metric_display_name}】\n"
            report += f"  基线策略: {results['baseline_mean']:.4f} (95% CI: [{results['baseline_ci']['lower']:.4f}, {results['baseline_ci']['upper']:.4f}])\n"
            report += f"  最优策略: {results['best_mean']:.4f} (95% CI: [{results['best_ci']['lower']:.4f}, {results['best_ci']['upper']:.4f}])\n"
            report += f"  改进幅度: {results['improvement_percentage']:.2f}% {results['improvement_direction']}\n"
            
            if results['t_test']['valid']:
                report += f"  统计显著性: {'显著' if results['significant'] else '不显著'} (p={results['t_test']['p_value']:.4f}, α=0.05)\n"
                report += f"  效应大小: {results['effect_size']} (Cohen's d={results['t_test']['cohens_d']:.3f})\n"
            else:
                report += f"  统计检验: 无效 ({results['t_test']['reason']})\n"
            
            report += "\n"
        
        # 总体结论
        significant_metrics = [name for name, results in comparison_results.items() if results['significant']]
        if significant_metrics:
            report += "【总体结论】\n"
            report += f"在 {len(significant_metrics)}/{len(comparison_results)} 个指标上观察到统计显著的改进:\n"
            for metric in significant_metrics:
                metric_display_name = {
                    'reward': '奖励', 'delay': '平均延迟', 'queue': '平均队列长度',
                    'throughput': '交通流量', 'energy': '能耗'
                }.get(metric, metric)
                report += f"  - {metric_display_name}: {comparison_results[metric]['improvement_percentage']:.2f}% {comparison_results[metric]['improvement_direction']}\n"
        else:
            report += "【总体结论】\n"
            report += "未观察到统计显著的改进。\n"
        
        report += "="*80 + "\n"
        
        return report
    
    @staticmethod
    def save_comparison_results(comparison_results: Dict[str, Any], output_path: str) -> None:
        """
        保存比较结果到JSON文件
        
        Args:
            comparison_results: 比较结果字典
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            print(f"[results] 比较结果已保存到: {output_path}")
        except Exception as e:
            print(f"[error] 保存比较结果失败: {str(e)}")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str) -> dict:
    # BOM优先解码并清理控制字符(BOM/NUL),避免 YAML 解析异常
    with open(path, "rb") as fb:
        data = fb.read()
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
    text = text.replace("\ufeff", "").replace("\x00", "")
    return yaml.safe_load(text)


def _get_route_files_value(sumo_cfg_path: str) -> str:
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if "<route-files" in line and "value=" in line:
                q1 = line.find('"')
                q2 = line.find('"', q1 + 1)
                if q1 != -1 and q2 != -1:
                    return line[q1 + 1:q2]
    raise RuntimeError("Cannot find <route-files> in SUMO config")


def _patch_sumocfg_route(sumo_cfg_path: str, route_rel_value: str) -> None:
    with open(sumo_cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    replaced = False
    for line in lines:
        if "<route-files" in line:
            start = line.find("value=")
            if start != -1:
                prefix = line[:start]
                qs = line.find('"', start)
                qe = line.find('"', qs + 1)
                if qs != -1 and qe != -1:
                    new_lines.append(prefix + f"value=\"{route_rel_value}\"" + line[qe + 1:])
                    replaced = True
                    continue
        new_lines.append(line)
    if not replaced:
        raise RuntimeError("Failed to patch route-files in SUMO config: tag not found")
    with open(sumo_cfg_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[sumocfg] patched {sumo_cfg_path} -> route-files={route_rel_value}")


def _strip_pen_suffix(route_rel: str) -> str:
    """剥离重复的 .penXX 后缀，得到固定基准文件名。"""
    import re
    name, ext = os.path.splitext(route_rel)
    base_name = re.sub(r"(\.pen\d{2})+$", "", name)
    return base_name + ext


def _find_latest_checkpoint_by_algo(root_dir: str, algo: str) -> str:
    """查找SAC检查点（DDQN已移除），优先加载“最优模型”：
    选择优先级：
    1) best_sac.pt（显式最佳命名）
    2) sac.pt（约定为最佳模型）
    3) 最新的 sac_ep*.pt（周期备份）
    4) 回退到 sac.pt
    """
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    pattern = "sac_ep"
    fallback_name = "sac.pt"
    best_name = "best_sac.pt"
    if not os.path.isdir(ckpt_dir):
        return os.path.join(root_dir, "checkpoints", fallback_name)
    # 1) 显式最佳命名的 best_*.pt
    best_path = os.path.join(ckpt_dir, best_name)
    if os.path.exists(best_path):
        return best_path
    # 2) 约定最佳模型文件 sac.pt
    best_conv_path = os.path.join(ckpt_dir, fallback_name)
    if os.path.exists(best_conv_path):
        return best_conv_path
    latest_path = None
    latest_ep = -1
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(pattern) and fn.endswith(".pt"):
            try:
                num = int(fn.replace(pattern, "").replace(".pt", ""))
            except Exception:
                num = -1
            if num > latest_ep:
                latest_ep = num
                latest_path = os.path.join(ckpt_dir, fn)
    if latest_path:
        return latest_path
    fallback = os.path.join(ckpt_dir, fallback_name)
    return fallback


def _generate_penetrated_routes_from_base(base_route_path: str, output_path: str, cav_rate: float, cfg: dict) -> None:
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
    
    # CAV 对齐 AV 参数（从配置读取）
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
    # HV 对齐 HDV-G 参数（从配置读取）
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

    import random as pyrand
    for veh in root_in.findall("vehicle"):
        vid = veh.get("id")
        depart = veh.get("depart")
        if vid is None or depart is None:
            continue
        r = veh.find("route")
        edges = r.get("edges") if r is not None else None
        if not edges:
            continue
        vtype = "CAV" if pyrand.random() < cav_rate else "HV"
        new_v = ET.SubElement(routes_out, "vehicle", id=vid, depart=depart, type=vtype, vClass="passenger")
        ET.SubElement(new_v, "route", edges=edges)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ET.ElementTree(routes_out).write(output_path, xml_declaration=True, encoding="UTF-8")
    print(f"[routes] wrote {output_path} | base={os.path.basename(base_route_path)} | CAV penetration={cav_rate}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=str, default="false")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--policy", type=str, default="both", choices=["best", "baseline", "both", "latest"], help="评估策略：best=加载最优模型，baseline=无控制，both=两者都评估；latest 同 best（兼容别名）")
    parser.add_argument("--device", type=str, default=None, help="评估设备：auto/cuda/cpu，默认auto")
    parser.add_argument("--repeat", type=int, default=1, help="重复评估次数（不同随机种子）")
    parser.add_argument("--seed_base", type=int, default=0, help="随机种子起点")
    parser.add_argument("--alpha", type=float, default=0.05, help="统计显著性水平（默认0.05）")
    parser.add_argument("--save_results", type=str, default=None, help="保存详细评估结果的JSON文件路径")
    parser.add_argument("--confidence", type=float, default=0.95, help="置信区间水平（默认0.95）")
    args = parser.parse_args()

    set_seed(0)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = _load_yaml(os.path.join(root, "config", "env.yaml"))
    use_gui = str(args.gui).lower() in ("1", "true", "yes")
    # 设备解析：命令行优先，其次配置文件，默认 auto
    dev_pref = (args.device or str(cfg.get("device", "auto")).lower())
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
    print(f"[device] 评估使用设备: {device}")

    sumo_cfg_rel = cfg.get("sumo_cfg_relpath", "sumo/grid1x3.sumocfg")
    sumo_cfg_path = os.path.join(root, sumo_cfg_rel)

    # 评估启动前：固定基准（剥离 .penXX），仅首次生成渗透率路由
    cur_route_rel = _get_route_files_value(sumo_cfg_path)
    base_route_rel = _strip_pen_suffix(cur_route_rel)
    base_route_abs = os.path.join(os.path.dirname(sumo_cfg_path), base_route_rel)

    pen = float(cfg.get("penetration", 0.3))
    pen = max(0.0, min(1.0, pen))
    name, ext = os.path.splitext(base_route_rel)
    out_rel = f"{name}.pen{int(pen*100):02d}{ext}"
    out_abs = os.path.join(os.path.dirname(sumo_cfg_path), out_rel)

    if not os.path.exists(out_abs):
        _generate_penetrated_routes_from_base(base_route_abs, out_abs, pen, cfg)
    else:
        print(f"[routes] reuse existing {out_rel}")
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

    # 将 'latest' 兼容映射为 'best'
    _pol = args.policy
    if _pol == "latest":
        print("[warn] --policy=latest 已弃用，按 'best' 处理")
        _pol = "best"
    policies = [_pol] if _pol != "both" else ["baseline", "best"]
    # 固定使用SAC算法
    algo = "sac"
    # 按参数设置评估回合数，保证至少 1 回合
    try:
        num_episodes = int(args.episodes)
    except (TypeError, ValueError):
        num_episodes = 1
    num_episodes = max(1, num_episodes)
    # 统一决策周期语义（按秒）：评估时按秒折算决策步数
    from math import ceil
    max_steps = max(1, int(ceil(int(cfg.get("max_sim_seconds", 3600)) / float(max(int(cfg.get("decision_interval", 5)), 1)))))
    print(f"[timing] decision_interval={int(cfg.get('decision_interval',5))}s | max_sim_seconds={int(cfg.get('max_sim_seconds',3600))}s => max_decision_steps={max_steps}")
    eps = 0.0
    
    # 存储所有策略的评估结果
    policy_results = {}

    for pol in policies:
        print(f"[eval] policy={pol}")
        if pol == "baseline":
            print(f"[baseline] 启用baseline模式：cav_type_id='__NO_CTRL__'，所有动作被忽略（零向量仅为占位）")
        cav_type = "__NO_CTRL__" if pol == "baseline" else "CAV"
        # 奖励裁剪配置
        _rc_min = cfg.get("reward_clip_min", None)
        _rc_max = cfg.get("reward_clip_max", None)
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
            max_delta_kph=float(cfg.get("max_delta_kph", 10.0)),
            cav_type_id=cav_type,
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

        obs, _ = env.reset()
        assert env.observation_space is not None, "observation_space cannot be None"
        obs_dim = env.observation_space.shape[0]  # type: ignore
        # MultiDiscrete: nvec（长度=6，每段的档位数）；若为单维离散则退化为 1 头
        if hasattr(env.action_space, "nvec"):
            nvec = list(map(int, env.action_space.nvec))  # type: ignore
        else:
            nvec = [int(env.action_space.n)]  # type: ignore

        agent = None
        ckpt_path = None
        if pol == "best":
            # 创建SAC Agent
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
            )
            ckpt_path = args.ckpt if args.ckpt else _find_latest_checkpoint_by_algo(root, algo)
            ckpt_path = os.path.join(root, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
            if os.path.exists(ckpt_path):
                agent.load(ckpt_path)
                print(f"[eval] loaded checkpoint: {ckpt_path}")
            else:
                print(f"[eval] checkpoint not found, evaluating with randomly-initialized agent: {ckpt_path}")

        rewards = []
        delays = []
        queues = []
        throughputs = []
        energies = []

        # 用多种随机种子重复评估
        for rep in range(int(args.repeat)):
            seed = int(args.seed_base) + rep
            set_seed(seed)
            for ep in range(1, num_episodes + 1):
                obs, _ = env.reset()
                ep_reward = 0.0
                for _ in range(max_steps):
                    if agent is None:
                        # baseline：无控制（cav_type 不匹配），动作对环境无效；MultiDiscrete 用零向量占位
                        import numpy as _np
                        action = _np.zeros(len(nvec), dtype=_np.int64)
                    else:
                        # SAC确定性动作选择
                        action = agent.select_action(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, _info = env.step(action)
                    ep_reward += reward
                    obs = next_obs
                    if terminated or truncated:
                        break

                rewards.append(ep_reward)
                m = env.get_metrics()
                delays.append(m.get("avg_delay_s", 0.0))
                queues.append(m.get("avg_queue_veh", 0.0))
                throughputs.append(m.get("throughput_veh_per_hour", 0.0))
                energies.append(m.get("avg_energy_j_per_km", 0.0))

                print(
                    f"[EVAL {pol} seed={seed} EP {ep:03d}] reward={ep_reward:.2f} | delay={m.get('avg_delay_s',0.0):.2f}s | queue={m.get('avg_queue_veh',0.0):.2f} | flow={m.get('throughput_veh_per_hour',0.0):.1f} veh/h | energy={m.get('avg_energy_j_per_km',0.0):.1f} J/km"
                )

        # 保存评估结果
        metrics = EvaluationMetrics(
            rewards=rewards,
            delays=delays,
            queues=queues,
            throughputs=throughputs,
            energies=energies
        )
        policy_results[pol] = metrics

        # 汇总输出
        def _avg(x):
            return float(np.mean(x)) if x else 0.0

        print(
            f"[EVAL {pol}] avg_reward={_avg(rewards):.2f} | avg_delay={_avg(delays):.2f}s | avg_queue={_avg(queues):.2f} | avg_flow={_avg(throughputs):.1f} veh/h | avg_energy={_avg(energies):.1f} J/km"
        )
        
        # 输出置信区间
        for metric_name, metric_data in [('reward', rewards), ('delay', delays), ('queue', queues), ('throughput', throughputs), ('energy', energies)]:
            ci_result = StatisticalAnalyzer.confidence_interval(metric_data, args.confidence)
            if ci_result['valid']:
                metric_display_name = {
                    'reward': '奖励', 'delay': '平均延迟', 'queue': '平均队列',
                    'throughput': '交通流量', 'energy': '能耗'
                }.get(metric_name, metric_name)
                print(f"[CI {pol}] {metric_display_name}: {ci_result['mean']:.4f} ({int(args.confidence*100)}% CI: [{ci_result['lower']:.4f}, {ci_result['upper']:.4f}])")

        env.close()
    
    # 如果评估了两种策略，进行统计比较
    if len(policy_results) == 2 and 'baseline' in policy_results and 'best' in policy_results:
        print("\n" + "="*80)
        print("正在进行统计显著性检验...")
        print("="*80 + "\n")
        
        # 执行比较
        comparison_results = StatisticalAnalyzer.compare_policies(
            policy_results['baseline'],
            policy_results['best'],
            args.alpha
        )
        
        # 生成并打印报告
        report = StatisticalAnalyzer.generate_comparison_report(comparison_results)
        print(report)
        
        # 保存结果到文件
        if args.save_results:
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = args.save_results if args.save_results.endswith('.json') else f"{args.save_results}_{timestamp}.json"
            StatisticalAnalyzer.save_comparison_results(comparison_results, output_path)
            
            # 同时保存报告文本
            report_path = output_path.replace('.json', '_report.txt')
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"[report] 详细报告已保存到: {report_path}")
            except Exception as e:
                print(f"[error] 保存报告失败: {str(e)}")


if __name__ == "__main__":
    main()
