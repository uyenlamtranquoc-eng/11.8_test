import os
from typing import Sequence, List, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CoordinatedPolicyNetwork(nn.Module):
    """联合动作建模 + 分层决策架构的策略网络
    
    特点：
    1. 共享trunk提取全局特征
    2. 全局协调策略网络输出协调向量
    3. 联合策略网络输出sum(nvec)维动作logits
    4. 支持Dropout正则化
    """

    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.nvec = list(int(n) for n in nvec)
        self.total_act = int(sum(self.nvec))
        self.num_segments = len(self.nvec)
        self.coord_dim = hidden // 4  # 协调向量维度
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 共享trunk提取全局特征
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
        )
        
        # 全局协调策略网络
        self.coordination_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.num_segments * self.coord_dim),
        )
        
        # 联合策略网络（输入：共享特征 + 协调向量）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden + self.coord_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, self.total_act),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 提取共享特征
        shared_features = self.shared_trunk(x)
        
        # 生成协调向量
        coordination_vec = self.coordination_head(shared_features)
        coordination_vec = coordination_vec.view(-1, self.num_segments, self.coord_dim)
        
        # 为每个段生成策略（使用对应的协调向量）
        segment_logits = []
        for i in range(self.num_segments):
            segment_coord = coordination_vec[:, i, :]
            segment_input = torch.cat([shared_features, segment_coord], dim=-1)
            # 为每个段生成对应维度的logits
            segment_logit_full = self.policy_head(segment_input)
            # 从总logits中提取当前段对应的维度
            start_idx = sum(self.nvec[:i])
            end_idx = sum(self.nvec[:i+1])
            segment_logit = segment_logit_full[:, start_idx:end_idx]
            segment_logits.append(segment_logit)
        
        return segment_logits


class CoordinatedQNetwork(nn.Module):
    """联合动作建模 + 分层决策架构的Q网络
    
    特点：
    1. 与策略网络共享相同的架构
    2. 双Q网络结构
    3. 支持Dropout正则化
    """

    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.nvec = list(int(n) for n in nvec)
        self.total_act = int(sum(self.nvec))
        self.num_segments = len(self.nvec)
        self.coord_dim = hidden // 4
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 共享trunk（与策略网络相同）
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
        )
        
        # 全局协调策略网络（与策略网络相同）
        self.coordination_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.num_segments * self.coord_dim),
        )
        
        # 联合Q网络（输入：共享特征 + 协调向量）
        self.q_head = nn.Sequential(
            nn.Linear(hidden + self.coord_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, self.total_act),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 提取共享特征
        shared_features = self.shared_trunk(x)
        
        # 生成协调向量
        coordination_vec = self.coordination_head(shared_features)
        coordination_vec = coordination_vec.view(-1, self.num_segments, self.coord_dim)
        
        # 为每个段生成Q值（使用对应的协调向量）
        segment_q_values = []
        for i in range(self.num_segments):
            segment_coord = coordination_vec[:, i, :]
            segment_input = torch.cat([shared_features, segment_coord], dim=-1)
            # 为每个段生成对应维度的Q值
            segment_q_full = self.q_head(segment_input)
            # 从总Q值中提取当前段对应的维度
            start_idx = sum(self.nvec[:i])
            end_idx = sum(self.nvec[:i+1])
            segment_q = segment_q_full[:, start_idx:end_idx]
            segment_q_values.append(segment_q)
        
        return segment_q_values


class MultiScaleValueNetwork(nn.Module):
    """多尺度价值函数网络
    
    特点：
    1. 局部价值：评估单段控制效果
    2. 全局价值：评估整体绿波效果
    3. 绿波协调价值：专门优化信号协调
    4. 动态权重：根据交通状态调整
    """
    
    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.nvec = list(int(n) for n in nvec)
        self.total_act = int(sum(self.nvec))
        self.num_segments = len(self.nvec)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 共享特征提取器
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
        )
        
        # 局部价值网络（每段独立）
        self.local_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, n),  # 每段的动作数
            ) for n in self.nvec
        ])
        
        # 全局价值网络
        self.global_value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.total_act),
        )
        
        # 绿波协调价值网络
        self.greenwave_value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.total_act),
        )
        
        # 动态权重网络
        self.dynamic_weight_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 3),  # [local_weight, global_weight, greenwave_weight]
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 提取共享特征
        shared_features = self.shared_trunk(x)
        
        # 局部价值
        local_values = []
        for i, head in enumerate(self.local_value_heads):
            local_val = head(shared_features)
            local_values.append(local_val)
        
        # 全局价值
        global_value = self.global_value_head(shared_features)
        
        # 绿波协调价值
        greenwave_value = self.greenwave_value_head(shared_features)
        
        # 动态权重
        dynamic_weights = self.softmax(self.dynamic_weight_head(shared_features))  # [B, 3]
        local_weight = dynamic_weights[:, 0:1]  # [B, 1]
        global_weight = dynamic_weights[:, 1:2]  # [B, 1]
        greenwave_weight = dynamic_weights[:, 2:3]  # [B, 1]
        
        # 加权组合
        combined_value = (local_weight * torch.stack(local_values, dim=1).mean(dim=1) +
                       global_weight * global_value +
                       greenwave_weight * greenwave_value)
        
        return {
            'local_values': local_values,
            'global_value': global_value,
            'greenwave_value': greenwave_value,
            'combined_value': combined_value,
            'weights': {
                'local': local_weight,
                'global': global_weight,
                'greenwave': greenwave_weight,
            }
        }


class MultiObjectiveOptimizationNetwork(nn.Module):
    """多目标优化网络：直接优化核心性能指标
    
    特点：
    1. 延迟预测网络
    2. 能耗预测网络
    3. 速度波动预测网络
    4. 帕累托权重学习
    """
    
    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.nvec = list(int(n) for n in nvec)
        self.total_act = int(sum(self.nvec))
        self.num_segments = len(self.nvec)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 共享特征提取器
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
        )
        
        # 延迟预测网络
        self.delay_predictor = nn.Sequential(
            nn.Linear(hidden + self.total_act, hidden // 2),  # 输入包含状态+动作
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # 预测总延迟
        )
        
        # 能耗预测网络
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden + self.total_act, hidden // 2),  # 输入包含状态+动作
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # 预测总能耗
        )
        
        # 速度波动预测网络
        self.speed_variance_predictor = nn.Sequential(
            nn.Linear(hidden + self.total_act, hidden // 2),  # 输入包含状态+动作
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # 预测速度波动
        )
        
        # 帕累托权重网络（动态调整各目标权重）
        self.pareto_weight_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 3),  # [delay_weight, energy_weight, speed_var_weight]
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播：预测性能指标并计算带权奖励"""
        # 提取共享特征
        shared_features = self.shared_trunk(x)
        
        # 融合动作信息：将动作与状态特征结合
        # actions是one-hot编码，维度为[B, total_act]
        action_features = torch.cat([shared_features, actions], dim=-1)
        
        # 使用动作-状态融合特征预测各性能指标
        pred_delay = self.delay_predictor(action_features)
        pred_energy = self.energy_predictor(action_features)
        pred_speed_var = self.speed_variance_predictor(action_features)
        
        # 获取动态权重（基于状态特征）
        pareto_weights = self.softmax(self.pareto_weight_head(shared_features))  # [B, 3]
        delay_weight = pareto_weights[:, 0:1]  # [B, 1]
        energy_weight = pareto_weights[:, 1:2]  # [B, 1]
        speed_var_weight = pareto_weights[:, 2:3]  # [B, 1]
        
        # 计算多目标奖励（最小化各指标）
        multi_obj_reward = -(delay_weight * pred_delay +
                           energy_weight * pred_energy +
                           speed_var_weight * pred_speed_var)
        
        return {
            'pred_delay': pred_delay,
            'pred_energy': pred_energy,
            'pred_speed_var': pred_speed_var,
            'pareto_weights': pareto_weights,
            'multi_obj_reward': multi_obj_reward,
            'delay_weight': delay_weight,
            'energy_weight': energy_weight,
            'speed_var_weight': speed_var_weight,
        }


class TemporalAwarenessNetwork(nn.Module):
    """时序感知网络：捕捉交通流的时序依赖性
    
    特点：
    1. LSTM/GRU时序建模
    2. 注意力机制捕捉关键时刻
    3. 时序特征融合
    4. 周期性模式识别
    """
    
    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: int = 256,
                 seq_len: int = 5, dropout: float = 0.0):
        super().__init__()
        self.nvec = list(int(n) for n in nvec)
        self.total_act = int(sum(self.nvec))
        self.num_segments = len(self.nvec)
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 时序编码器（LSTM）
        self.temporal_encoder = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0.0 else 0.0,
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
        # 时序特征融合器
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        
        # 时序特征到观测空间的映射（用于计算时序损失）
        self.temporal_to_obs = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, obs_dim),  # 映射回观测维度
        )
        
        # 周期性模式识别
        self.periodic_detector = nn.Sequential(
            nn.Linear(obs_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),  # 周期性强度
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 当前状态 [B, obs_dim]
            history: 历史状态序列 [B, seq_len, obs_dim] (可选)
        
        Returns:
            dict: {
                'temporal_features': 时序特征,
                'attention_weights': 注意力权重,
                'periodic_strength': 周期性强度,
            }
        """
        batch_size = x.size(0)
        
        # 如果没有提供历史，用当前状态填充
        if history is None:
            history = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 时序编码
        temporal_out, (hidden, cell) = self.temporal_encoder(history)
        
        # 注意力机制
        attended_out, attention_weights = self.attention(
            temporal_out, temporal_out, temporal_out
        )
        
        # 时序特征融合（使用最后一个时间步的输出）
        temporal_features = self.temporal_fusion(attended_out[:, -1, :])
        
        # 将时序特征映射回观测空间（用于计算时序损失）
        temporal_obs_pred = self.temporal_to_obs(temporal_features)
        
        # 周期性模式检测
        periodic_strength = self.periodic_detector(x)
        
        return {
            'temporal_features': temporal_features,
            'temporal_obs_pred': temporal_obs_pred,  # 映射回观测空间的特征
            'attention_weights': attention_weights,
            'periodic_strength': periodic_strength,
            'hidden_state': hidden[-1],  # 最后一层的隐藏状态
        }


class CoordinationRegularizationNetwork(nn.Module):
    """协调正则化网络：确保路段间的协调一致性
    
    特点：
    1. 协调一致性损失
    2. 冲突检测与缓解
    3. 协调强度学习
    4. 空间相关性建模
    """
    
    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.nvec = list(int(n) for n in nvec)
        self.total_act = int(sum(self.nvec))
        self.num_segments = len(self.nvec)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 空间相关性建模
        self.spatial_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            self.dropout if self.dropout else nn.Identity(),
            nn.Linear(hidden, hidden // 2),
        )
        
        # 协调强度学习网络
        self.coordination_strength_net = nn.Sequential(
            nn.Linear(hidden // 2, hidden),  # 修正：直接使用spatial_features
            nn.ReLU(),
            nn.Linear(hidden, self.num_segments * self.num_segments),  # 协调矩阵
            nn.Sigmoid(),  # 确保协调强度在[0,1]
        )
        
        # 冲突检测网络
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden // 2 * 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),  # 冲突概率
        )
        
        # 协调正则化权重
        self.coordination_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            states: 状态 [B, obs_dim]
            actions: 动作 [B, num_segments]
        
        Returns:
            dict: {
                'coordination_loss': 协调正则化损失,
                'coordination_matrix': 协调矩阵,
                'conflict_probabilities': 冲突概率,
                'coordination_strength': 平均协调强度,
            }
        """
        batch_size = states.size(0)
        
        # 提取空间特征
        spatial_features = self.spatial_encoder(states)  # [B, hidden//2]
        
        # 计算协调矩阵
        # 修正：直接使用spatial_features，不需要view操作
        # 因为spatial_features已经是[B, hidden//2]，而coordination_strength_net期望的输入维度是hidden//2
        coordination_matrix_raw = self.coordination_strength_net(spatial_features)
        coordination_matrix = coordination_matrix_raw.view(
            batch_size, self.num_segments, self.num_segments
        )
        
        # 对称化协调矩阵（确保i对j的协调等于j对i的协调）
        coordination_matrix = (coordination_matrix + coordination_matrix.transpose(1, 2)) / 2.0
        
        # 计算协调损失
        coordination_loss = 0.0
        conflict_probabilities = []
        
        for i in range(self.num_segments):
            for j in range(i + 1, self.num_segments):
                # 获取两个路段的动作
                action_i = actions[:, i].float().unsqueeze(1)
                action_j = actions[:, j].float().unsqueeze(1)
                
                # 获取协调强度
                coord_strength = coordination_matrix[:, i, j].unsqueeze(1)
                
                # 计算动作差异
                action_diff = torch.abs(action_i - action_j)
                
                # 协调损失：动作差异越大，协调强度越高时损失越大
                pair_coordination_loss = coord_strength * action_diff
                coordination_loss += torch.mean(pair_coordination_loss)
                
                # 冲突检测
                # 修正：spatial_features已经是[B, hidden//2]，不需要按段切分
                # 直接使用原始空间特征进行冲突检测
                conflict_prob = self.conflict_detector(spatial_features)
                conflict_probabilities.append(conflict_prob)
        
        # 平均协调强度
        avg_coordination_strength = torch.mean(coordination_matrix)
        
        # 总协调损失（考虑协调权重）
        total_coordination_loss = self.coordination_weight * coordination_loss
        
        return {
            'coordination_loss': total_coordination_loss,
            'coordination_matrix': coordination_matrix,
            'conflict_probabilities': torch.cat(conflict_probabilities, dim=1),
            'coordination_strength': avg_coordination_strength,
        }


class MultiHeadSACAgent:
    """改进版离散SAC（联合动作建模 + 分层决策架构）：
    
    核心改进：
    1. 联合动作建模：共享trunk + 协调向量，自动捕捉段间耦合
    2. 分层决策架构：全局协调策略 + 局部执行网络
    3. 约束优化框架：物理约束 + 拉格朗日乘子法
    4. 分组α参数：东向/西向各一个，简化理论复杂性
    5. HuberLoss替代MSE（稳定TD误差）
    6. 增强日志监控（每段entropy、alpha、loss）
    7. 支持策略延迟更新
    8. 支持权重衰减和Dropout
    """

    def __init__(
        self,
        obs_dim: int,
        nvec: Sequence[int],
        lr: float = 1e-3,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        alpha_lr: Optional[float] = None,
        gamma: float = 0.99,
        target_update_interval: int = 500,
        device: Optional[str] = None,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = False,
        target_entropy: Optional[float] = None,
        hidden_size: int = 256,
        max_grad_norm: float = 5.0,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        actor_detach_q: bool = True,  # 建议默认True：离散策略中Q不对策略反向传播更稳定
        policy_delay: int = 1,  # 新增：策略延迟更新
        dropout: float = 0.0,  # 新增：Dropout正则化
        weight_decay: float = 0.0,  # 新增：权重衰减
    ):
        self.obs_dim = obs_dim
        self.nvec = list(int(n) for n in nvec)
        self.num_heads = len(self.nvec)
        self.gamma = gamma
        self.tau = tau
        # 目标网络更新间隔：None或0表示不使用硬更新（依赖tau软更新）
        if target_update_interval is None or int(target_update_interval) <= 0:
            target_interval = None
        else:
            target_interval = int(target_update_interval)
        self.target_interval = target_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_delay = int(policy_delay)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)

        # 网络（联合动作建模 + 分层决策架构）
        self.policy = CoordinatedPolicyNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=dropout).to(self.device)
        self.q1 = CoordinatedQNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=dropout).to(self.device)
        self.q2 = CoordinatedQNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=dropout).to(self.device)
        self.q1_target = CoordinatedQNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=0.0).to(self.device)
        self.q2_target = CoordinatedQNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=0.0).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for n in [self.q1_target, self.q2_target]:
            n.eval()
        
        # 多目标优化网络
        self.multi_obj_network = MultiObjectiveOptimizationNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=dropout).to(self.device)
        self.multi_obj_opt = optim.Adam(self.multi_obj_network.parameters(), lr=alp_lr * 0.5, weight_decay=weight_decay)
        
        # 多尺度价值网络
        self.multiscale_value = MultiScaleValueNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=dropout).to(self.device)
        self.multiscale_value_target = MultiScaleValueNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=0.0).to(self.device)
        self.multiscale_value_target.load_state_dict(self.multiscale_value.state_dict())
        self.multiscale_value_target.eval()
        self.multiscale_opt = optim.Adam(self.multiscale_value.parameters(), lr=c_lr, weight_decay=weight_decay)
        
        # 时序感知网络
        self.temporal_network = TemporalAwarenessNetwork(obs_dim, self.nvec, hidden=hidden_size, seq_len=5, dropout=dropout).to(self.device)
        self.temporal_opt = optim.Adam(self.temporal_network.parameters(), lr=c_lr * 0.5, weight_decay=weight_decay)
        
        # 协调正则化网络
        self.coordination_network = CoordinationRegularizationNetwork(obs_dim, self.nvec, hidden=hidden_size, dropout=dropout).to(self.device)
        self.coordination_opt = optim.Adam(self.coordination_network.parameters(), lr=c_lr * 0.5, weight_decay=weight_decay)
        
        # 状态历史缓冲区（用于时序感知）
        self.state_history = []
        self.max_history_len = 5

        # 优化器（带权重衰减）
        a_lr = float(actor_lr) if actor_lr is not None else float(lr)
        c_lr = float(critic_lr) if critic_lr is not None else float(lr)
        alp_lr = float(alpha_lr) if alpha_lr is not None else (a_lr * 0.5)
        
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=a_lr, weight_decay=weight_decay)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=c_lr, weight_decay=weight_decay)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=c_lr, weight_decay=weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.alpha_min = float(alpha_min) if alpha_min is not None else None
        self.alpha_max = float(alpha_max) if alpha_max is not None else None
        self.detach_actor_q = bool(actor_detach_q)

        # 分组熵系数（东向/西向各一个）
        self.auto_alpha = bool(auto_alpha)
        self.num_groups = 2  # 东向和西向
        if self.auto_alpha:
            # 分组目标熵：每组的目标熵 = log(平均动作数)
            if target_entropy is None:
                avg_actions_per_group = float(np.mean(self.nvec))
                group_target_entropy = np.log(avg_actions_per_group)
                self.target_entropies = torch.tensor(
                    [group_target_entropy] * self.num_groups,
                    dtype=torch.float32, device=self.device
                )
            else:
                # 如果提供单一值，分配到各组
                self.target_entropies = torch.tensor(
                    [float(target_entropy)] * self.num_groups,
                    dtype=torch.float32, device=self.device
                )
            
            # 分组的 log_alpha（东向/西向各一个）
            self.log_alphas = torch.nn.Parameter(
                torch.tensor([np.log(alpha)] * self.num_groups, dtype=torch.float32, device=self.device)
            )
            self.alpha_opt = optim.Adam([self.log_alphas], lr=alp_lr)
            
            print(f"[SAC-Coordinated] 启用分组自适应温度：")
            for k, te in enumerate(self.target_entropies):
                group_name = "东向" if k == 0 else "西向"
                print(f"      {group_name}: target_entropy={te.item():.3f}, alpha_init={alpha:.3f}")
            print(f"      Alpha学习率={alp_lr:.6f}, 策略延迟={self.policy_delay}, Dropout={dropout:.2f}")
        else:
            self.alpha = float(alpha)
            self.log_alphas = None
            self.alpha_opt = None
            self.target_entropies = None

        self._train_steps = 0
        
        # 约束优化框架参数
        self.max_delta_kph = 10.0  # 最大速度变化约束
        self.constraint_weight = 1.0  # 约束违反惩罚权重
        
        # 速度档位映射（用于约束计算）
        self.speed_levels_mps = [8.33, 10.0, 12.0, 14.0]  # 默认速度档位
        
        # 监控指标（新增）
        self.metrics = {
            'q1_loss': 0.0,
            'q2_loss': 0.0,
            'pi_loss': 0.0,
            'alpha_loss': 0.0,
            'constraint_loss': 0.0,  # 约束违反损失
            'multiscale_value_loss': 0.0,  # 多尺度价值损失
            'multi_obj_loss': 0.0,  # 多目标优化损失
            'pred_delay': 0.0,  # 预测延迟
            'pred_energy': 0.0,  # 预测能耗
            'pred_speed_var': 0.0,  # 预测速度波动
            'delay_weight': 0.0,  # 延迟权重
            'energy_weight': 0.0,  # 能耗权重
            'speed_var_weight': 0.0,  # 速度波动权重
            'temporal_loss': 0.0,  # 时序感知损失
            'coordination_loss': 0.0,  # 协调正则化损失
            'periodic_strength': 0.0,  # 周期性强度
            'coordination_strength': 0.0,  # 协调强度
            'conflict_probability': 0.0,  # 冲突概率
            'per_segment_entropy': np.zeros(self.num_heads),
            'per_group_alpha': np.zeros(self.num_groups),
            'per_segment_q_loss': np.zeros(self.num_heads),
            'per_segment_pi_loss': np.zeros(self.num_heads),
        }
        
        # 初始化前一时刻动作
        self._prev_actions = None

    @property
    def temperature(self) -> float:
        """返回平均温度（用于日志显示）"""
        if self.auto_alpha and self.log_alphas is not None:
            return float(self.log_alphas.exp().mean().item())
        else:
            return float(self.alpha)
    
    def get_per_head_alphas(self) -> np.ndarray:
        """获取每段的alpha值（用于日志）"""
        if self.auto_alpha and self.log_alphas is not None:
            # 将分组α分配到各段（东向段用组0，西向段用组1）
            group_alphas = self.log_alphas.exp().detach().cpu().numpy()
            segment_alphas = []
            for i in range(self.num_heads):
                group_idx = 0 if i < self.num_heads // 2 else 1
                segment_alphas.append(group_alphas[group_idx])
            return np.array(segment_alphas)
        else:
            return np.array([self.alpha] * self.num_heads)
    
    def get_per_group_alphas(self) -> np.ndarray:
        """获取每组的alpha值（用于日志）"""
        if self.auto_alpha and self.log_alphas is not None:
            return self.log_alphas.exp().detach().cpu().numpy()
        else:
            return np.array([self.alpha] * self.num_groups)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前训练指标"""
        return self.metrics.copy()
    
    def set_train(self):
        """切换到训练模式"""
        self.policy.train()
        self.q1.train()
        self.q2.train()
        self.multi_obj_network.train()
        self.multiscale_value.train()
        self.temporal_network.train()
        self.coordination_network.train()
        # 目标网络始终保持eval模式
        self.q1_target.eval()
        self.q2_target.eval()
        self.multiscale_value_target.eval()
    
    def set_eval(self):
        """切换到评估模式（禁用Dropout等）"""
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
        self.multi_obj_network.eval()
        self.multiscale_value.eval()
        self.temporal_network.eval()
        self.coordination_network.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        self.multiscale_value_target.eval()

    def _policy_logits(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.policy(x)

    def _softmax_and_log(self, logits: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        logp = torch.log(torch.clamp(probs, min=1e-8))
        return probs, logp

    def select_action(self, state: np.ndarray, deterministic: bool = False,
                    prev_actions: Optional[np.ndarray] = None) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            logits_list = self._policy_logits(s)
            
            # 状态感知的探索强度调整
            exploration_scales = self._compute_exploration_scales(state)
            
            acts = []
            for i, lg in enumerate(logits_list):
                if deterministic:
                    a = int(torch.argmax(lg, dim=1).item())
                else:
                    # 应用状态感知的探索强度
                    adjusted_logits = lg * exploration_scales[i]
                    probs = torch.softmax(adjusted_logits, dim=1)
                    dist = torch.distributions.Categorical(probs)
                    a = int(dist.sample().item())
                acts.append(a)
            
            actions = np.array(acts, dtype=np.int64)
            
            # 应用物理约束
            if prev_actions is not None:
                actions = self._apply_physical_constraints(actions, prev_actions)
            
            return actions
    
    def _compute_exploration_scales(self, state: np.ndarray) -> np.ndarray:
        """计算状态感知的探索强度调整因子
        
        根据交通状态动态调整探索强度：
        - 拥堵时减少探索，避免加重拥堵
        - 畅通时增加探索，寻找更优策略
        """
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            
            # 提取交通状态特征（假设状态向量中包含占有率、速度等信息）
            # 这里简化处理：使用状态向量的前几个维度作为交通状态指标
            traffic_features = s[:, :6]  # 假设前6维是交通状态特征
            
            # 计算拥堵程度（简化：使用特征的平均值）
            congestion_level = torch.mean(traffic_features, dim=1)  # [B]
            
            # 动态调整探索强度：拥堵时降低，畅通时提高
            # 使用sigmoid函数平滑过渡
            base_scale = 1.0
            congestion_factor = 1.0 - torch.sigmoid(congestion_level * 2.0 - 1.0)  # 拥堵时接近0，畅通时接近1
            
            # 为每个段计算探索强度
            exploration_scales = base_scale * congestion_factor
            
            return exploration_scales.cpu().numpy().flatten()
    
    def _apply_physical_constraints(self, actions: np.ndarray, prev_actions: np.ndarray) -> np.ndarray:
        """应用物理约束：速度变化不超过最大值"""
        constrained_actions = actions.copy()
        
        # 将动作索引转换为速度档位
        for i in range(len(actions)):
            current_speed_idx = actions[i]
            prev_speed_idx = prev_actions[i]
            
            # 获取对应的实际速度值（假设速度档位是递增的）
            if i < len(self.nvec):  # 确保索引有效
                max_speed_idx = self.nvec[i] - 1
                min_speed_idx = 0
                
                # 计算最大允许的变化（以档位计）
                max_delta_idx = int(self.max_delta_kph / 3.6 / (self.speed_levels_mps[1] - self.speed_levels_mps[0]) if len(self.speed_levels_mps) > 1 else 1)
                
                # 应用约束
                if current_speed_idx > prev_speed_idx + max_delta_idx:
                    constrained_actions[i] = min(prev_speed_idx + max_delta_idx, max_speed_idx)
                elif current_speed_idx < prev_speed_idx - max_delta_idx:
                    constrained_actions[i] = max(prev_speed_idx - max_delta_idx, min_speed_idx)
        
        return constrained_actions
    
    def compute_constraint_loss(self, actions: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """计算约束违反损失"""
        # 计算速度变化（以档位计）
        action_diff = torch.abs(actions - prev_actions)
        
        # 计算最大允许的变化（以档位计）
        max_delta_idx = self.max_delta_kph / 3.6 / (self.speed_levels_mps[1] - self.speed_levels_mps[0]) if len(self.speed_levels_mps) > 1 else 1
        
        # 计算约束违反量
        constraint_violation = torch.relu(action_diff - max_delta_idx)
        
        # 返回约束违反损失
        return self.constraint_weight * torch.mean(constraint_violation ** 2)

    def update(self, batch, is_weights=None) -> Dict[str, float]:
        """更新网络，返回详细指标用于监控
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
            is_weights: 重要性采样权重 (optional, for PER)
                       shape: [B] or None
        
        Returns:
            dict: {
                'total_loss': float,
                'q1_loss': float,
                'q2_loss': float,
                'pi_loss': float,
                'alpha_loss': float,
                'avg_alpha': float,
                'td_errors': np.ndarray  # 用于PER优先级更新
            }
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        # 形状健壮性修复
        if actions.dim() == 2 and actions.size(1) == 1 and actions.size(0) % self.num_heads == 0:
            actions = actions.view(actions.size(0) // self.num_heads, self.num_heads)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 获取每组alpha（向量）
        if self.auto_alpha:
            group_alphas = self.log_alphas.exp()  # type: ignore  # [num_groups]
            # 将分组α分配到各段
            alphas = []
            for i in range(self.num_heads):
                group_idx = 0 if i < self.num_heads // 2 else 1
                alphas.append(group_alphas[group_idx])
            alphas = torch.stack(alphas)  # [num_heads]
        else:
            alphas = torch.full((self.num_heads,), self.alpha, device=self.device)

        # ------- 计算目标软值 V(s') -------
        with torch.no_grad():
            next_logits = self._policy_logits(next_states)
            q1_t_heads = self.q1_target(next_states)
            q2_t_heads = self.q2_target(next_states)
            multiscale_values = self.multiscale_value_target(next_states)
            v_next_list = []
            for k in range(self.num_heads):
                probs_next, logp_next = self._softmax_and_log(next_logits[k])
                min_q_next = torch.minimum(q1_t_heads[k], q2_t_heads[k])
                
                # 融合多尺度价值函数
                combined_q_next = min_q_next + 0.1 * multiscale_values['combined_value']
                
                # 使用每头独立的alpha_k
                v_next = torch.sum(probs_next * (combined_q_next - alphas[k] * logp_next), dim=1)
                v_next_list.append(v_next)
            y_list = [rewards + (1.0 - dones) * self.gamma * v for v in v_next_list]

        # ------- 评论家（双Q）更新：使用HuberLoss + IS权重 -------
        q1_heads = self.q1(states)
        q2_heads = self.q2(states)
        multiscale_values = self.multiscale_value(states)
        q1_losses = []
        q2_losses = []
        multiscale_losses = []  # 多尺度价值损失
        td_errors_list = []  # 用于PER优先级更新
        
        for k in range(self.num_heads):
            ak = actions[:, k].view(-1, 1)
            q1_sa = q1_heads[k].gather(1, ak).squeeze(1)
            q2_sa = q2_heads[k].gather(1, ak).squeeze(1)
            yk = y_list[k]
            
            # 计算TD误差（用于PER）
            with torch.no_grad():
                td_error_k = torch.abs(q1_sa - yk)  # 使用Q1的TD误差
                td_errors_list.append(td_error_k)
            
            # HuberLoss (Smooth L1) 替代 MSE
            if is_weights is not None:
                # 使用重要性采样权重（PER）
                q1_loss_k = torch.mean(is_weights * torch.nn.functional.smooth_l1_loss(q1_sa, yk, reduction='none'))
                q2_loss_k = torch.mean(is_weights * torch.nn.functional.smooth_l1_loss(q2_sa, yk, reduction='none'))
            else:
                q1_loss_k = torch.nn.functional.smooth_l1_loss(q1_sa, yk)
                q2_loss_k = torch.nn.functional.smooth_l1_loss(q2_sa, yk)
            
            # 多尺度价值损失
            multiscale_loss_k = torch.nn.functional.smooth_l1_loss(multiscale_values['combined_value'], yk)
            
            q1_losses.append(q1_loss_k)
            q2_losses.append(q2_loss_k)
            multiscale_losses.append(multiscale_loss_k)
            
            # 记录每段的Q损失
            self.metrics['per_segment_q_loss'][k] = float((q1_loss_k.item() + q2_loss_k.item()) / 2.0)
        
        # 汇总TD误差（取各头平均值）
        td_errors = torch.stack(td_errors_list, dim=1).mean(dim=1)  # [B]
        
        q1_loss = sum(q1_losses) / float(max(self.num_heads, 1))
        q2_loss = sum(q2_losses) / float(max(self.num_heads, 1))
        multiscale_loss = sum(multiscale_losses) / float(max(self.num_heads, 1))

        self.q1_opt.zero_grad()
        q1_loss.backward()  # type: ignore
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()  # type: ignore
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_opt.step()
        
        # 更新多尺度价值网络
        self.multiscale_opt.zero_grad()
        multiscale_loss.backward(retain_graph=True)  # type: ignore
        torch.nn.utils.clip_grad_norm_(self.multiscale_value.parameters(), self.max_grad_norm)
        self.multiscale_opt.step()

        # 记录评论家损失
        self.metrics['q1_loss'] = float(q1_loss.item())  # type: ignore
        self.metrics['q2_loss'] = float(q2_loss.item())  # type: ignore
        self.metrics['multiscale_value_loss'] = float(multiscale_loss.item())  # type: ignore

        # ------- 策略（actor）更新：支持延迟更新 -------
        pi_loss = torch.tensor(0.0, device=self.device)
        constraint_loss = torch.tensor(0.0, device=self.device)
        
        if self._train_steps % self.policy_delay == 0:
            logits = self._policy_logits(states)
            if self.detach_actor_q:
                with torch.no_grad():
                    q1_heads = self.q1(states)
                    q2_heads = self.q2(states)
            else:
                q1_heads = self.q1(states)
                q2_heads = self.q2(states)
            
            pi_losses = []
            entropies = []
            
            # 采样当前动作用于约束计算
            with torch.no_grad():
                sampled_actions = []
                for k in range(self.num_heads):
                    probs = torch.softmax(logits[k], dim=1)
                    dist = torch.distributions.Categorical(probs)
                    sampled_action = dist.sample()
                    sampled_actions.append(sampled_action)
                sampled_actions = torch.stack(sampled_actions, dim=1)  # [B, num_heads]
            
            for k in range(self.num_heads):
                probs, logp = self._softmax_and_log(logits[k])
                min_q = torch.minimum(q1_heads[k], q2_heads[k])
                if self.detach_actor_q:
                    min_q = min_q.detach()
                # 使用每段对应的alpha_k
                pi_obj = torch.sum(probs * (alphas[k] * logp - min_q), dim=1)
                pi_loss_k = torch.mean(pi_obj)
                pi_losses.append(pi_loss_k)
                
                # 记录每段的熵
                entropy_k = -torch.mean(torch.sum(probs * logp, dim=1))
                entropies.append(entropy_k)
                self.metrics['per_segment_entropy'][k] = float(entropy_k.item())
                self.metrics['per_segment_pi_loss'][k] = float(pi_loss_k.item())
            
            pi_loss = sum(pi_losses) / float(max(self.num_heads, 1))
            
            # 计算约束损失（如果有前一时刻动作）
            if hasattr(self, '_prev_actions') and self._prev_actions is not None:
                prev_actions_tensor = torch.from_numpy(self._prev_actions).to(self.device)
                constraint_loss = self.compute_constraint_loss(sampled_actions, prev_actions_tensor)
            
            # 更新前一时刻动作
            with torch.no_grad():
                self._prev_actions = sampled_actions.detach().cpu().numpy()

            self.policy_opt.zero_grad()
            pi_loss.backward()  # type: ignore
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_opt.step()

            # ------- 自适应 α 更新：分组更新，归一化梯度 -------
            if self.auto_alpha:
                # 计算每组的平均熵
                group_entropies = []
                for g in range(self.num_groups):
                    if g == 0:  # 东向
                        group_segments = range(0, self.num_heads // 2)
                    else:  # 西向
                        group_segments = range(self.num_heads // 2, self.num_heads)
                    
                    group_entropy = torch.mean(torch.stack([entropies[i] for i in group_segments]))
                    group_entropies.append(group_entropy)
                
                alpha_losses = []
                for g in range(self.num_groups):
                    # 每组的alpha损失：J(α) = α * (H - T)
                    entropy_g = group_entropies[g].detach()
                    alpha_loss_g = self.log_alphas[g].exp() * (entropy_g - self.target_entropies[g])  # type: ignore
                    alpha_losses.append(alpha_loss_g)
                
                # 按组取均值
                alpha_loss = sum(alpha_losses) / float(max(self.num_groups, 1))
                
                self.alpha_opt.zero_grad()  # type: ignore
                alpha_loss.backward()  # type: ignore
                self.alpha_opt.step()  # type: ignore
                
                # 钳位alpha到合理范围
                if (self.alpha_min is not None) or (self.alpha_max is not None):
                    with torch.no_grad():
                        min_log = float(np.log(max(self.alpha_min, 1e-8))) if (self.alpha_min is not None) else float(-1e18)
                        max_log = float(np.log(max(self.alpha_max, 1e-8))) if (self.alpha_max is not None) else float(1e18)
                        self.log_alphas.data.clamp_(min=min_log, max=max_log)  # type: ignore
                
                # 记录每组的alpha
                self.metrics['per_group_alpha'] = self.get_per_group_alphas()
                self.metrics['alpha_loss'] = float(alpha_loss.item())  # type: ignore
        else:
            # policy_delay时保持上一帧指标（不更新）
            # per_head_entropy和per_head_alpha已在上次更新中设置，此处不需操作
            pass

        self.metrics['pi_loss'] = float(pi_loss.item())  # type: ignore
        self.metrics['constraint_loss'] = float(constraint_loss.item())  # type: ignore
        
        # ------- 多目标优化更新 -------
        # 将动作转换为one-hot编码用于多目标网络
        actions_one_hot = []
        for k in range(self.num_heads):
            ak = actions[:, k].view(-1, 1)
            ak_one_hot = torch.zeros(ak.size(0), self.nvec[k], device=self.device)
            ak_one_hot.scatter_(1, ak, 1)
            actions_one_hot.append(ak_one_hot)
        actions_one_hot = torch.cat(actions_one_hot, dim=1)  # [B, total_act]
        
        # 多目标网络前向传播
        multi_obj_outputs = self.multi_obj_network(states, actions_one_hot)
        
        # 多目标损失：最小化预测的性能指标
        # 使用实际奖励作为监督信号的一部分
        multi_obj_loss = torch.nn.functional.mse_loss(multi_obj_outputs['multi_obj_reward'], rewards.view(-1, 1))
        
        # 更新多目标网络
        self.multi_obj_opt.zero_grad()
        multi_obj_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.multi_obj_network.parameters(), self.max_grad_norm)
        self.multi_obj_opt.step()
        
        # 记录多目标优化指标
        self.metrics['multi_obj_loss'] = float(multi_obj_loss.item())
        self.metrics['pred_delay'] = float(multi_obj_outputs['pred_delay'].mean().item())
        self.metrics['pred_energy'] = float(multi_obj_outputs['pred_energy'].mean().item())
        self.metrics['pred_speed_var'] = float(multi_obj_outputs['pred_speed_var'].mean().item())
        self.metrics['delay_weight'] = float(multi_obj_outputs['delay_weight'].mean().item())
        self.metrics['energy_weight'] = float(multi_obj_outputs['energy_weight'].mean().item())
        self.metrics['speed_var_weight'] = float(multi_obj_outputs['speed_var_weight'].mean().item())
        
        # ------- 时序感知更新 -------
        # 更新状态历史缓冲区
        current_states_np = states.detach().cpu().numpy()
        batch_size = current_states_np.shape[0]
        
        # 为每个样本维护独立的历史缓冲区（简化处理）
        # 这里使用全局历史，但扩展到批次维度
        for i in range(current_states_np.shape[0]):
            self.state_history.append(current_states_np[i])
            if len(self.state_history) > self.max_history_len:
                self.state_history.pop(0)
        
        # 准备历史状态序列，确保与当前批次对齐
        if len(self.state_history) >= self.max_history_len:
            history_tensor = torch.tensor(np.array(self.state_history[-self.max_history_len:]),
                                         dtype=torch.float32, device=self.device)
            # 扩展到批次维度：[seq_len, obs_dim] -> [batch_size, seq_len, obs_dim]
            history_tensor = history_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            # 如果历史不足，用当前状态填充
            history_tensor = states.unsqueeze(1).repeat(1, self.max_history_len, 1)
        
        # 时序网络前向传播
        temporal_outputs = self.temporal_network(states, history_tensor)
        
        # 时序损失：鼓励时序特征与当前状态特征的一致性
        # 使用映射回观测空间的时序特征与原始状态计算差异
        temporal_loss = torch.nn.functional.mse_loss(temporal_outputs['temporal_obs_pred'], states)
        
        # 更新时序网络
        self.temporal_opt.zero_grad()
        temporal_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.temporal_network.parameters(), self.max_grad_norm)
        self.temporal_opt.step()
        
        # 记录时序感知指标
        self.metrics['temporal_loss'] = float(temporal_loss.item())
        self.metrics['periodic_strength'] = float(temporal_outputs['periodic_strength'].mean().item())
        
        # ------- 协调正则化更新 -------
        # 协调正则化网络前向传播
        coordination_outputs = self.coordination_network(states, actions)
        
        # 更新协调正则化网络
        self.coordination_opt.zero_grad()
        coordination_outputs['coordination_loss'].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.coordination_network.parameters(), self.max_grad_norm)
        self.coordination_opt.step()
        
        # 记录协调正则化指标
        self.metrics['coordination_loss'] = float(coordination_outputs['coordination_loss'].item())
        self.metrics['coordination_strength'] = float(coordination_outputs['coordination_strength'].item())
        self.metrics['conflict_probability'] = float(coordination_outputs['conflict_probabilities'].mean().item())

        # ------- 软更新目标网络（Polyak） -------
        self._train_steps += 1
        if self.tau is not None and self.tau > 0.0:
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            if hasattr(self, 'multiscale_value'):
                self._soft_update(self.multiscale_value, self.multiscale_value_target)
        elif self.target_interval is not None and self.target_interval > 0 and (self._train_steps % self.target_interval == 0):  # type: ignore
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            if hasattr(self, 'multiscale_value'):
                self.multiscale_value_target.load_state_dict(self.multiscale_value.state_dict())

        # 返回总损失和指标（包含约束损失、多尺度价值损失、多目标优化损失、时序感知损失和协调正则化损失）
        total_loss = float(q1_loss.item() + q2_loss.item() + pi_loss.item() + constraint_loss.item() +
                          multiscale_loss.item() + multi_obj_loss.item() +
                          temporal_loss.item() + coordination_outputs['coordination_loss'].item())  # type: ignore
        return {
            'total_loss': total_loss,
            'q1_loss': self.metrics['q1_loss'],
            'q2_loss': self.metrics['q2_loss'],
            'pi_loss': self.metrics['pi_loss'],
            'constraint_loss': self.metrics['constraint_loss'],
            'multiscale_value_loss': self.metrics['multiscale_value_loss'],
            'multi_obj_loss': self.metrics['multi_obj_loss'],
            'temporal_loss': self.metrics['temporal_loss'],
            'coordination_loss': self.metrics['coordination_loss'],
            'alpha_loss': self.metrics.get('alpha_loss', 0.0),
            'avg_alpha': self.temperature,
            'td_errors': td_errors.detach().cpu().numpy(),  # type: ignore  # 用于PER优先级更新
        }

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        tau = self.tau
        with torch.no_grad():
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.mul_(1.0 - tau)
                p_t.data.add_(tau * p_o.data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "multi_obj_network": self.multi_obj_network.state_dict(),
            "multiscale_value": self.multiscale_value.state_dict(),
            "multiscale_value_target": self.multiscale_value_target.state_dict(),
            "temporal_network": self.temporal_network.state_dict(),
            "coordination_network": self.coordination_network.state_dict(),
            "optimizers": {
                "policy": self.policy_opt.state_dict(),
                "q1": self.q1_opt.state_dict(),
                "q2": self.q2_opt.state_dict(),
                "multi_obj": self.multi_obj_opt.state_dict(),
                "multiscale": self.multiscale_opt.state_dict(),
                "temporal": self.temporal_opt.state_dict(),
                "coordination": self.coordination_opt.state_dict(),
                "alpha": self.alpha_opt.state_dict() if self.auto_alpha and self.alpha_opt is not None else None,
            },
            "alpha": self.temperature,
            "auto_alpha": self.auto_alpha,
            "log_alphas": self.log_alphas.detach().cpu().numpy().tolist() if self.auto_alpha and self.log_alphas is not None else None,  # type: ignore
            "target_entropies": self.target_entropies.cpu().numpy().tolist() if self.auto_alpha and self.target_entropies is not None else None,  # type: ignore
            "_train_steps": self._train_steps,
            "nvec": self.nvec,
            "meta": {
                "gamma": self.gamma,
                "tau": self.tau,
                "target_interval": self.target_interval,
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "actor_detach_q": self.detach_actor_q,
                "policy_delay": self.policy_delay,
                "dropout": self.dropout,
                "weight_decay": self.weight_decay,
            },
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            self.policy.load_state_dict(ckpt["policy"])
            self.q1.load_state_dict(ckpt.get("q1", self.q1.state_dict()))
            self.q2.load_state_dict(ckpt.get("q2", self.q2.state_dict()))
            if "q1_target" in ckpt:
                self.q1_target.load_state_dict(ckpt["q1_target"])
            else:
                self.q1_target.load_state_dict(self.q1.state_dict())
            if "q2_target" in ckpt:
                self.q2_target.load_state_dict(ckpt["q2_target"])
            else:
                self.q2_target.load_state_dict(self.q2.state_dict())

            opts = ckpt.get("optimizers")
            if isinstance(opts, dict):
                for name, opt in [("policy", self.policy_opt), ("q1", self.q1_opt), ("q2", self.q2_opt),
                                 ("multi_obj", self.multi_obj_opt), ("multiscale", self.multiscale_opt),
                                 ("temporal", self.temporal_opt), ("coordination", self.coordination_opt)]:
                    state = opts.get(name)
                    if state is not None:
                        try:
                            opt.load_state_dict(state)
                        except RuntimeError:
                            pass
                
                # 加载多目标优化网络
                if "multi_obj_network" in ckpt:
                    try:
                        self.multi_obj_network.load_state_dict(ckpt["multi_obj_network"])
                    except RuntimeError:
                        pass
                
                # 加载多尺度价值网络
                if "multiscale_value" in ckpt:
                    try:
                        self.multiscale_value.load_state_dict(ckpt["multiscale_value"])
                    except RuntimeError:
                        pass
                
                if "multiscale_value_target" in ckpt:
                    try:
                        self.multiscale_value_target.load_state_dict(ckpt["multiscale_value_target"])
                    except RuntimeError:
                        pass
                
                # 加载时序感知网络
                if "temporal_network" in ckpt:
                    try:
                        self.temporal_network.load_state_dict(ckpt["temporal_network"])
                    except RuntimeError:
                        pass
                
                # 加载协调正则化网络
                if "coordination_network" in ckpt:
                    try:
                        self.coordination_network.load_state_dict(ckpt["coordination_network"])
                    except RuntimeError:
                        pass

            if self.auto_alpha:
                if "log_alphas" in ckpt and ckpt["log_alphas"] is not None:
                    vals = ckpt["log_alphas"]
                    self.log_alphas.data = torch.tensor(vals, dtype=torch.float32, device=self.device)  # type: ignore
                elif "alpha" in ckpt:
                    a = float(ckpt["alpha"])
                    self.log_alphas.data = torch.full((self.num_heads,), np.log(max(a, 1e-8)), dtype=torch.float32, device=self.device)  # type: ignore
                
                if "target_entropies" in ckpt and ckpt["target_entropies"] is not None:
                    self.target_entropies = torch.tensor(ckpt["target_entropies"], dtype=torch.float32, device=self.device)
                
                if isinstance(opts, dict):
                    alpha_state = opts.get("alpha")
                    if alpha_state is not None and self.alpha_opt is not None:
                        try:
                            self.alpha_opt.load_state_dict(alpha_state)
                        except RuntimeError:
                            pass
            else:
                if "alpha" in ckpt:
                    self.alpha = float(ckpt["alpha"])

            self._train_steps = int(ckpt.get("_train_steps", self._train_steps))
            meta = ckpt.get("meta", {})
            if "policy_delay" in meta:
                self.policy_delay = int(meta["policy_delay"])
        else:
            try:
                self.policy.load_state_dict(ckpt)
            except Exception:
                pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取智能体完整状态，用于检查点保存
        
        Returns:
            dict: 包含所有网络状态、优化器状态和训练参数的字典
        """
        try:
            # 网络状态
            network_state = {
                'policy': self.policy.state_dict(),
                'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'q1_target': self.q1_target.state_dict(),
                'q2_target': self.q2_target.state_dict(),
                'multi_obj_network': self.multi_obj_network.state_dict(),
                'multiscale_value': self.multiscale_value.state_dict(),
                'multiscale_value_target': self.multiscale_value_target.state_dict(),
                'temporal_network': self.temporal_network.state_dict(),
                'coordination_network': self.coordination_network.state_dict(),
            }
            
            # 优化器状态
            optimizer_state = {
                'policy': self.policy_opt.state_dict(),
                'q1': self.q1_opt.state_dict(),
                'q2': self.q2_opt.state_dict(),
                'multi_obj': self.multi_obj_opt.state_dict(),
                'multiscale': self.multiscale_opt.state_dict(),
                'temporal': self.temporal_opt.state_dict(),
                'coordination': self.coordination_opt.state_dict(),
            }
            
            # Alpha优化器状态
            alpha_optimizer_state = None
            if self.auto_alpha and self.alpha_opt is not None:
                alpha_optimizer_state = self.alpha_opt.state_dict()
            
            # 训练参数和状态
            training_state = {
                'log_alphas': self.log_alphas.detach().cpu().numpy().tolist() if self.auto_alpha and self.log_alphas is not None else None,
                'target_entropies': self.target_entropies.cpu().numpy().tolist() if self.auto_alpha and self.target_entropies is not None else None,
                '_train_steps': self._train_steps,
                'alpha': self.alpha,
                'auto_alpha': self.auto_alpha,
                'alpha_min': self.alpha_min,
                'alpha_max': self.alpha_max,
                'gamma': self.gamma,
                'tau': self.tau,
                'target_interval': self.target_interval,
                'max_grad_norm': self.max_grad_norm,
                'detach_actor_q': self.detach_actor_q,
                'policy_delay': self.policy_delay,
                'dropout': self.dropout,
                'weight_decay': self.weight_decay,
                'obs_dim': self.obs_dim,
                'nvec': self.nvec,
                'num_heads': self.num_heads,
                'num_groups': self.num_groups,
                'max_delta_kph': self.max_delta_kph,
                'constraint_weight': self.constraint_weight,
                'speed_levels_mps': self.speed_levels_mps,
                'state_history': self.state_history[-self.max_history_len:] if self.state_history else [],
                'max_history_len': self.max_history_len,
                '_prev_actions': self._prev_actions.tolist() if self._prev_actions is not None else None,
            }
            
            # 合并所有状态
            complete_state = {
                'network_state': network_state,
                'optimizer_state': optimizer_state,
                'alpha_optimizer_state': alpha_optimizer_state,
                'training_state': training_state,
                'metrics': self.metrics.copy(),
            }
            
            return complete_state
            
        except Exception as e:
            print(f"[智能体状态] 获取状态失败: {e}")
            return {}
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载智能体完整状态，用于检查点恢复
        
        Args:
            state: 包含所有网络状态、优化器状态和训练参数的字典
        """
        try:
            # 恢复网络状态
            if 'network_state' in state:
                network_state = state['network_state']
                if 'policy' in network_state:
                    self.policy.load_state_dict(network_state['policy'])
                if 'q1' in network_state:
                    self.q1.load_state_dict(network_state['q1'])
                if 'q2' in network_state:
                    self.q2.load_state_dict(network_state['q2'])
                if 'q1_target' in network_state:
                    self.q1_target.load_state_dict(network_state['q1_target'])
                if 'q2_target' in network_state:
                    self.q2_target.load_state_dict(network_state['q2_target'])
                if 'multi_obj_network' in network_state:
                    self.multi_obj_network.load_state_dict(network_state['multi_obj_network'])
                if 'multiscale_value' in network_state:
                    self.multiscale_value.load_state_dict(network_state['multiscale_value'])
                if 'multiscale_value_target' in network_state:
                    self.multiscale_value_target.load_state_dict(network_state['multiscale_value_target'])
                if 'temporal_network' in network_state:
                    self.temporal_network.load_state_dict(network_state['temporal_network'])
                if 'coordination_network' in network_state:
                    self.coordination_network.load_state_dict(network_state['coordination_network'])
            
            # 恢复优化器状态
            if 'optimizer_state' in state:
                optimizer_state = state['optimizer_state']
                try:
                    if 'policy' in optimizer_state:
                        self.policy_opt.load_state_dict(optimizer_state['policy'])
                    if 'q1' in optimizer_state:
                        self.q1_opt.load_state_dict(optimizer_state['q1'])
                    if 'q2' in optimizer_state:
                        self.q2_opt.load_state_dict(optimizer_state['q2'])
                    if 'multi_obj' in optimizer_state:
                        self.multi_obj_opt.load_state_dict(optimizer_state['multi_obj'])
                    if 'multiscale' in optimizer_state:
                        self.multiscale_opt.load_state_dict(optimizer_state['multiscale'])
                    if 'temporal' in optimizer_state:
                        self.temporal_opt.load_state_dict(optimizer_state['temporal'])
                    if 'coordination' in optimizer_state:
                        self.coordination_opt.load_state_dict(optimizer_state['coordination'])
                except RuntimeError as e:
                    print(f"[智能体状态] 恢复优化器状态失败（可能由于版本不兼容）: {e}")
            
            # 恢复Alpha优化器状态
            if 'alpha_optimizer_state' in state and state['alpha_optimizer_state'] is not None:
                if self.auto_alpha and self.alpha_opt is not None:
                    try:
                        self.alpha_opt.load_state_dict(state['alpha_optimizer_state'])
                    except RuntimeError as e:
                        print(f"[智能体状态] 恢复Alpha优化器状态失败: {e}")
            
            # 恢复训练参数和状态
            if 'training_state' in state:
                training_state = state['training_state']
                
                # 恢复alpha相关参数
                if 'log_alphas' in training_state and training_state['log_alphas'] is not None:
                    if self.auto_alpha and self.log_alphas is not None:
                        self.log_alphas.data = torch.tensor(
                            training_state['log_alphas'],
                            dtype=torch.float32,
                            device=self.device
                        )
                
                if 'target_entropies' in training_state and training_state['target_entropies'] is not None:
                    if self.auto_alpha and self.target_entropies is not None:
                        self.target_entropies = torch.tensor(
                            training_state['target_entropies'],
                            dtype=torch.float32,
                            device=self.device
                        )
                
                # 恢复其他训练参数
                for key in ['_train_steps', 'alpha', 'auto_alpha', 'alpha_min', 'alpha_max',
                           'gamma', 'tau', 'target_interval', 'max_grad_norm',
                           'detach_actor_q', 'policy_delay', 'dropout', 'weight_decay',
                           'obs_dim', 'num_heads', 'num_groups', 'max_delta_kph',
                           'constraint_weight', 'speed_levels_mps', 'max_history_len']:
                    if key in training_state:
                        setattr(self, key, training_state[key])
                
                # 恢复nvec（特殊处理）
                if 'nvec' in training_state:
                    self.nvec = list(int(n) for n in training_state['nvec'])
                
                # 恢复状态历史
                if 'state_history' in training_state:
                    self.state_history = training_state['state_history'][-self.max_history_len:]
                
                # 恢复前一时刻动作
                if '_prev_actions' in training_state and training_state['_prev_actions'] is not None:
                    self._prev_actions = np.array(training_state['_prev_actions'])
            
            # 恢复指标
            if 'metrics' in state:
                self.metrics.update(state['metrics'])
            
            print(f"[智能体状态] 状态恢复成功，训练步数: {self._train_steps}")
            
        except Exception as e:
            print(f"[智能体状态] 加载状态失败: {e}")
