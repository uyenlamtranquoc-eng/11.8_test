import random
from collections import deque
from typing import List, Tuple
import numpy as np
import torch


class ReplayBuffer:
    """Simple FIFO replay buffer for off-policy RL."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Push transition. Action can be scalar int or array-like (e.g., MultiDiscrete of shape [6])."""
        a = np.asarray(action)
        if a.ndim == 0:
            a = a.astype(np.int64)
        else:
            a = a.astype(np.int64).reshape(-1)
        self.buffer.append((state.astype(np.float32), a, float(reward), next_state.astype(np.float32), bool(done)))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.from_numpy(np.stack(states))
        # 统一规范动作到二维 [B, H]（H 可为 1）
        first = actions[0]
        first_arr = np.asarray(first)
        if first_arr.ndim >= 1:
            # 将每个动作压平为 1D，再 stack 成 [B, H]
            norm_actions = [np.asarray(a, dtype=np.int64).reshape(-1) for a in actions]
            actions_t = torch.from_numpy(np.stack(norm_actions)).long()
        else:
            # 纯标量动作：构造 [B, 1]
            actions_t = torch.from_numpy(np.array(actions, dtype=np.int64).reshape(-1, 1)).long()
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.from_numpy(np.stack(next_states))
        dones_t = torch.tensor(dones, dtype=torch.float32)
        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self) -> int:
        return len(self.buffer)


def _demo_main():
    buf = ReplayBuffer(capacity=1000, obs_dim=9)
    for _ in range(100):
        s = np.random.randn(9).astype(np.float32)
        a = np.random.randint(0, 64)
        r = float(np.random.randn())
        ns = np.random.randn(9).astype(np.float32)
        d = bool(np.random.rand() < 0.1)
        buf.push(s, a, r, ns, d)
    print(f"Buffer size: {len(buf)}")
    states, actions, rewards, next_states, dones = buf.sample(32)
    print("Sample shapes:", states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)


if __name__ == "__main__":
    _demo_main()