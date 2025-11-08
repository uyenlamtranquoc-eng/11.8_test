"""优先级经验回放（PER）实现

基于Sum Tree数据结构实现高效的O(log N)采样和更新。
支持n-step return计算。
"""

import numpy as np
import torch
from typing import Tuple, Optional


class SumTree:
    """Sum Tree数据结构用于高效优先级采样
    
    完全二叉树，叶子节点存储优先级，父节点存储子节点优先级之和。
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 完全二叉树：capacity叶子 + capacity-1内部节点
        self.data = np.zeros(capacity, dtype=object)  # 存储transition
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority: float, data: object):
        """添加数据到树中"""
        tree_idx = self.data_pointer + self.capacity - 1  # 叶子节点索引
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx: int, priority: float):
        """更新叶子节点优先级，并向上传播"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # 向上传播更新
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # 父节点索引
            self.tree[tree_idx] += change
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """根据累积优先级值s采样
        
        返回：(tree_idx, priority, data)
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果到达叶子节点
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # 选择左子树还是右子树
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_priority(self) -> float:
        """返回所有优先级之和"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区
    
    特性：
    1. 基于TD误差的优先级采样
    2. 重要性采样权重（IS weights）补偿偏差
    3. 支持n-step return计算
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-5,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """
        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数（0=均匀采样，1=完全优先级采样）
            beta_start: 重要性采样权重初始值
            beta_frames: beta线性退火到1.0的帧数
            epsilon: 添加到优先级的小常数，避免零优先级
            n_step: n步回报
            gamma: 折扣因子
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.n_step = n_step
        self.gamma = gamma
        self.frame = 0
        
        # n-step缓冲区
        self.n_step_buffer = []
    
    def _get_beta(self) -> float:
        """计算当前的beta值（线性退火）"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def _get_priority(self, error: float) -> float:
        """计算优先级：(|TD_error| + epsilon)^alpha"""
        return (abs(error) + self.epsilon) ** self.alpha
    
    def _compute_n_step_return(self) -> Tuple:
        """计算n步回报
        
        返回：(s_t, a_t, R^(n), s_{t+n}, done)
        其中 R^(n) = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1}
        """
        if len(self.n_step_buffer) < self.n_step:
            return None  # type: ignore
        
        # 计算累积奖励
        R = 0.0
        for i in range(self.n_step):
            s, a, r, s_next, done = self.n_step_buffer[i]
            R += (self.gamma ** i) * r
            if done:
                # 提前终止
                return (self.n_step_buffer[0][0], self.n_step_buffer[0][1], R, s_next, done)
        
        # 完整n步
        s_t, a_t, _, _, _ = self.n_step_buffer[0]
        _, _, _, s_tn, done_tn = self.n_step_buffer[-1]
        return (s_t, a_t, R, s_tn, done_tn)
    
    def add(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool, error: Optional[float] = None):  # type: ignore
        """添加transition到缓冲区
        
        Args:
            error: TD误差（可选）。如果为None，使用最大优先级
        """
        # 添加到n-step缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return  # 等待积累足够的步数
        
        # 计算n步回报
        transition = self._compute_n_step_return()
        if transition is None:
            return
        
        # 移除最旧的transition
        self.n_step_buffer.pop(0)
        
        # 计算优先级
        if error is None:
            # 使用当前最大优先级（确保新样本至少被采样一次）
            max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.size > 0 else 1.0
            priority = max_priority
        else:
            priority = self._get_priority(error)
        
        self.tree.add(float(priority), transition)  # type: ignore
        self.frame += 1
    
    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple:
        """采样一个batch
        
        返回：(states, actions, rewards, next_states, dones, is_weights, indices)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size
        
        beta = self._get_beta()
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # 计算重要性采样权重
        priorities = np.array(priorities)
        probs = priorities / self.tree.total_priority
        is_weights = np.power(self.tree.size * probs, -beta)
        is_weights /= is_weights.max()  # 归一化
        
        # 解包batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(device)
        is_weights = torch.FloatTensor(is_weights).to(device)
        
        return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: list, errors: np.ndarray):
        """更新采样样本的优先级"""
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        return self.tree.size
