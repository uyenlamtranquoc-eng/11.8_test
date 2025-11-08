from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadPolicy(nn.Module):
    """
    最小策略网络：共享两层 MLP 干路 + 每头一个线性输出层。
    输出为每个头的 logits 列表（未归一化），用于 Categorical。
    """

    def __init__(self, obs_dim: int, nvec: List[int], hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.nvec = list(map(int, nvec))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.nvec])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(x)
        return [head(h) for head in self.heads]


class MultiHeadQ(nn.Module):
    """
    最小 Q 网络：共享两层 MLP 干路 + 每头一个线性输出层。
    输出为每个头的 Q 值列表（shape: [B, n_i]）。
    """

    def __init__(self, obs_dim: int, nvec: List[int], hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.nvec = list(map(int, nvec))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.nvec])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(x)
        return [head(h) for head in self.heads]


class MultiHeadSACAgent:
    """
    最简版离散多头 SAC（MultiDiscrete）
    - 无自适应温度（alpha 固定常数）
    - 无 multiscale/multi-objective/temporal/coordination 等增强模块
    - 仅保留双 Q + 策略更新 + 目标网络软更新
    """

    def __init__(
        self,
        obs_dim: int,
        nvec: List[int],
        lr: float = 3e-4,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        gamma: float = 0.99,
        target_update_interval: int = 1,
        device: str = "cpu",
        tau: float = 0.005,
        alpha: float = 0.2,
        hidden_size: int = 256,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.nvec = list(map(int, nvec))
        self.num_heads = len(self.nvec)

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.alpha = float(alpha)
        self._train_steps = 0
        self.max_grad_norm = max_grad_norm

        # 网络
        self.policy = MultiHeadPolicy(self.obs_dim, self.nvec, hidden_size).to(self.device)
        self.q1 = MultiHeadQ(self.obs_dim, self.nvec, hidden_size).to(self.device)
        self.q2 = MultiHeadQ(self.obs_dim, self.nvec, hidden_size).to(self.device)
        self.q1_target = MultiHeadQ(self.obs_dim, self.nvec, hidden_size).to(self.device)
        self.q2_target = MultiHeadQ(self.obs_dim, self.nvec, hidden_size).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

        # 优化器
        actor_lr = float(actor_lr) if actor_lr is not None else float(lr)
        critic_lr = float(critic_lr) if critic_lr is not None else float(lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        # 运行指标
        self.metrics: Dict[str, Any] = {}

    # ---------------------- 基本工具 ----------------------
    @property
    def temperature(self) -> float:
        return self.alpha

    def set_train(self) -> None:
        self.policy.train()
        self.q1.train()
        self.q2.train()
        self.q1_target.eval()
        self.q2_target.eval()

    def set_eval(self) -> None:
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()

    # ---------------------- 动作选择 ----------------------
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        多头离散动作选择：
        - deterministic=True 时，每头取 argmax(logits)
        - 否则按 Categorical 分布采样
        """
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits_list = self.policy(x)
            actions = []
            for logits in logits_list:
                if deterministic:
                    a = torch.argmax(logits, dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                actions.append(int(a.item()))
        return np.array(actions, dtype=np.int64)

    # ---------------------- 训练更新 ----------------------
    def _compute_policy_targets(self, next_states: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        计算 V 值相关项：
        - next_logits: 下一状态各头的 logits
        - next_logp: 下一状态各头的 log π(a|s)
        - min_q_next: 下一状态各头的 min(Q1, Q2)
        """
        next_logits = self.policy(next_states)
        next_logp = [F.log_softmax(l, dim=-1) for l in next_logits]
        next_pi = [torch.exp(lp) for lp in next_logp]
        q1_next = self.q1_target(next_states)
        q2_next = self.q2_target(next_states)
        min_q_next = [torch.minimum(q1_next[i], q2_next[i]) for i in range(self.num_heads)]

        return next_logits, next_logp, min_q_next

    def update(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        执行一次最小 SAC 更新。
        输入 batch: (states, actions, rewards, next_states, dones)
        - states: [B, obs_dim]
        - actions: [B, num_heads] (int64)
        - rewards: [B]
        - next_states: [B, obs_dim]
        - dones: [B] (bool or 0/1)
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).view(-1)
        dones = dones.to(self.device).view(-1)

        B = states.shape[0]

        # ---------- 1) 计算每头的目标值 ----------
        with torch.no_grad():
            _, next_logp_list, min_q_next_list = self._compute_policy_targets(next_states)
            next_pi_list = [torch.exp(lp) for lp in next_logp_list]
            # 每个头的 V_next_i = E_{a_i~pi_i}[ minQ_i - alpha*logpi_i ] -> [B]
            v_next_heads = [
                (next_pi_list[i] * (min_q_next_list[i] - self.alpha * next_logp_list[i])).sum(dim=-1)
                for i in range(self.num_heads)
            ]  # list of [B]
            # 每头的 TD 目标：target_q_per_head[i] 形状 [B]
            target_q_per_head = [
                rewards + (1.0 - dones) * self.gamma * v_next_heads[i]
                for i in range(self.num_heads)
            ]
            # 拼成 [B, H]
            target_q_per_head = torch.stack(target_q_per_head, dim=1)

        # ---------- 2) 更新 Q1/Q2 ----------
        q1_values_list = self.q1(states)
        q2_values_list = self.q2(states)
        # 为后续 actor 步骤复用，提前保存无梯度版本，避免重复前向
        q1_values_list_for_actor = [v.detach() for v in q1_values_list]
        q2_values_list_for_actor = [v.detach() for v in q2_values_list]

        # gather 当前动作的 Q 值，得到 [B, H]
        q1_a = []
        q2_a = []
        for i in range(self.num_heads):
            ai = actions[:, i].long()
            q1_sel = q1_values_list[i].gather(1, ai.view(-1, 1)).view(-1)
            q2_sel = q2_values_list[i].gather(1, ai.view(-1, 1)).view(-1)
            q1_a.append(q1_sel)
            q2_a.append(q2_sel)
        q1_a = torch.stack(q1_a, dim=1)  # [B, H]
        q2_a = torch.stack(q2_a, dim=1)  # [B, H]

        # 每样本 TD 误差（头维平均），用于内部监控（不返回给外部）
        td_errors_per_head = torch.abs(target_q_per_head - torch.minimum(q1_a, q2_a))  # [B, H]
        td_error_mean = td_errors_per_head.mean().item()

        # MSE 损失（也可替换为 Huber）；计算每样本、每头的误差并聚合
        q1_td = q1_a - target_q_per_head  # [B, H]
        q2_td = q2_a - target_q_per_head  # [B, H]
        q1_loss = torch.mean(q1_td ** 2)
        q2_loss = torch.mean(q2_td ** 2)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        # ---------- 3) 更新策略（最小化 J_pi = E[alpha*logpi - minQ]） ----------
        logits_list = self.policy(states)
        logp_list = [F.log_softmax(l, dim=-1) for l in logits_list]
        pi_list = [torch.exp(lp) for lp in logp_list]

        # 复用前面保存的 Q 值（detach 版本），避免重复前向
        min_q_list = [torch.minimum(q1_values_list_for_actor[i], q2_values_list_for_actor[i]) for i in range(self.num_heads)]

        actor_loss_terms = [
            (pi_list[i] * (self.alpha * logp_list[i] - min_q_list[i])).sum(dim=-1).mean()
            for i in range(self.num_heads)
        ]
        # 将各头损失取平均，避免随头数线性放大
        actor_loss = sum(actor_loss_terms) / float(self.num_heads)

        self.policy_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # ---------- 4) 软更新目标网络 ----------
        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self._soft_update(self.q1, self.q1_target, self.tau)
            self._soft_update(self.q2, self.q2_target, self.tau)

        # 记录指标：损失、温度、TD 误差统计、每头平均熵
        head_entropies = [
            float((-(pi_list[i] * logp_list[i]).sum(dim=-1)).mean().item())
            for i in range(self.num_heads)
        ]

        self.metrics = {
            'q1_loss': float(q1_loss.item()),
            'q2_loss': float(q2_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'alpha': float(self.alpha),
            'train_steps': int(self._train_steps),
            'td_error_mean': float(td_error_mean),
            'td_error_max': float(td_errors_per_head.max().item()),
            'td_error_std': float(td_errors_per_head.std().item()),
        }
        for i, h in enumerate(head_entropies):
            self.metrics[f'entropy_head_{i}'] = h

        out: Dict[str, Any] = dict(self.metrics)
        out['total_loss'] = self.metrics['q1_loss'] + self.metrics['q2_loss'] + self.metrics['actor_loss']
        return out

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_t, p_s in zip(target.parameters(), source.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p_s.data)

    # ---------------------- Checkpoint ----------------------
    def save(self, path: str) -> None:
        ckpt = {
            'obs_dim': self.obs_dim,
            'nvec': self.nvec,
            'gamma': self.gamma,
            'tau': self.tau,
            'target_update_interval': self.target_update_interval,
            'alpha': self.alpha,
            'policy_state': self.policy.state_dict(),
            'q1_state': self.q1.state_dict(),
            'q2_state': self.q2.state_dict(),
            'q1_target_state': self.q1_target.state_dict(),
            'q2_target_state': self.q2_target.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'q1_opt': self.q1_optimizer.state_dict(),
            'q2_opt': self.q2_optimizer.state_dict(),
            'train_steps': self._train_steps,
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)

        def pick(d: Dict[str, Any], primary: str, alts: List[str]) -> Optional[Any]:
            if primary in d:
                return d[primary]
            for k in alts:
                if k in d:
                    return d[k]
            for nest in ['networks', 'models', 'state', 'agent', 'agents']:
                v = d.get(nest)
                if isinstance(v, dict):
                    if primary in v:
                        return v[primary]
                    for k in alts:
                        if k in v:
                            return v[k]
            return None

        pol_sd = pick(ckpt, 'policy_state', ['policy', 'actor_state', 'policy_state_dict', 'pi_state'])
        q1_sd = pick(ckpt, 'q1_state', ['q1', 'critic1_state', 'critic_1_state'])
        q2_sd = pick(ckpt, 'q2_state', ['q2', 'critic2_state', 'critic_2_state'])
        q1_t_sd = pick(ckpt, 'q1_target_state', ['q1_target', 'target_q1_state'])
        q2_t_sd = pick(ckpt, 'q2_target_state', ['q2_target', 'target_q2_state'])
        pol_opt_sd = pick(ckpt, 'policy_opt', ['policy_optimizer', 'actor_opt', 'policy_optim_state'])
        q1_opt_sd = pick(ckpt, 'q1_opt', ['critic1_opt', 'q1_optimizer'])
        q2_opt_sd = pick(ckpt, 'q2_opt', ['critic2_opt', 'q2_optimizer'])

        if pol_sd is not None:
            self.policy.load_state_dict(pol_sd)
        if q1_sd is not None:
            self.q1.load_state_dict(q1_sd)
        if q2_sd is not None:
            self.q2.load_state_dict(q2_sd)
        if q1_t_sd is not None:
            self.q1_target.load_state_dict(q1_t_sd)
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if q2_t_sd is not None:
            self.q2_target.load_state_dict(q2_t_sd)
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())

        if pol_opt_sd is not None:
            self.policy_optimizer.load_state_dict(pol_opt_sd)
        if q1_opt_sd is not None:
            self.q1_optimizer.load_state_dict(q1_opt_sd)
        if q2_opt_sd is not None:
            self.q2_optimizer.load_state_dict(q2_opt_sd)

        self.alpha = float(ckpt.get('alpha', self.alpha))
        self._train_steps = int(ckpt.get('train_steps', ckpt.get('steps', 0)))

    # 兼容 run_train 的完整检查点保存/恢复
    def get_state(self) -> Dict[str, Any]:
        return {
            'obs_dim': self.obs_dim,
            'nvec': self.nvec,
            'gamma': self.gamma,
            'tau': self.tau,
            'target_update_interval': self.target_update_interval,
            'alpha': self.alpha,
            'policy_state': self.policy.state_dict(),
            'q1_state': self.q1.state_dict(),
            'q2_state': self.q2.state_dict(),
            'q1_target_state': self.q1_target.state_dict(),
            'q2_target_state': self.q2_target.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'q1_opt': self.q1_optimizer.state_dict(),
            'q2_opt': self.q2_optimizer.state_dict(),
            'train_steps': self._train_steps,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        def pick(d: Dict[str, Any], primary: str, alts: List[str]) -> Optional[Any]:
            if primary in d:
                return d[primary]
            for k in alts:
                if k in d:
                    return d[k]
            for nest in ['networks', 'models', 'state', 'agent', 'agents']:
                v = d.get(nest)
                if isinstance(v, dict):
                    if primary in v:
                        return v[primary]
                    for k in alts:
                        if k in v:
                            return v[k]
            return None

        pol_sd = pick(state, 'policy_state', ['policy', 'actor_state', 'policy_state_dict', 'pi_state'])
        q1_sd = pick(state, 'q1_state', ['q1', 'critic1_state', 'critic_1_state'])
        q2_sd = pick(state, 'q2_state', ['q2', 'critic2_state', 'critic_2_state'])
        q1_t_sd = pick(state, 'q1_target_state', ['q1_target', 'target_q1_state'])
        q2_t_sd = pick(state, 'q2_target_state', ['q2_target', 'target_q2_state'])
        pol_opt_sd = pick(state, 'policy_opt', ['policy_optimizer', 'actor_opt', 'policy_optim_state'])
        q1_opt_sd = pick(state, 'q1_opt', ['critic1_opt', 'q1_optimizer'])
        q2_opt_sd = pick(state, 'q2_opt', ['critic2_opt', 'q2_optimizer'])

        if pol_sd is not None:
            self.policy.load_state_dict(pol_sd)
        if q1_sd is not None:
            self.q1.load_state_dict(q1_sd)
        if q2_sd is not None:
            self.q2.load_state_dict(q2_sd)
        if q1_t_sd is not None:
            self.q1_target.load_state_dict(q1_t_sd)
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if q2_t_sd is not None:
            self.q2_target.load_state_dict(q2_t_sd)
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())

        if pol_opt_sd is not None:
            self.policy_optimizer.load_state_dict(pol_opt_sd)
        if q1_opt_sd is not None:
            self.q1_optimizer.load_state_dict(q1_opt_sd)
        if q2_opt_sd is not None:
            self.q2_optimizer.load_state_dict(q2_opt_sd)

        self.alpha = float(state.get('alpha', self.alpha))
        self._train_steps = int(state.get('train_steps', state.get('steps', 0)))