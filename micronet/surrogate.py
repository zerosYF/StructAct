import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time

class SurrogateNet(nn.Module):
    def __init__(self, input_dim, hidden=[64,32]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape (batch,)

class SurrogateModel:
    """
    Lightweight surrogate/regressor to predict expected reward given pool state + action.
    Simple API:
      - add_transition(features, action_id, reward)
      - predict(features, action_id) -> float
      - train_step(batch_size=32, epochs=1)
    """
    def __init__(self, input_dim, device=None, lr=1e-3, buffer_size=2000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SurrogateNet(input_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()
        self.last_train_ts = 0
        self.train_interval = 10.0  # seconds, safe default
        self.min_train_samples = 64

    def add_transition(self, features: np.ndarray, reward: float):
        # features: 1D numpy array
        self.replay.append((features.astype(np.float32), float(reward)))

    def predict(self, features: np.ndarray):
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)
            out = self.net(x).cpu().item()
        return float(out)

    def batch_predict(self, features_batch: np.ndarray):
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(features_batch.astype(np.float32)).to(self.device)
            out = self.net(x).cpu().numpy()
        return out

    def train_step(self, batch_size=64, epochs=1):
        if len(self.replay) < self.min_train_samples:
            return 0.0
        # sample batches
        losses = []
        for ep in range(epochs):
            batch = random.sample(self.replay, min(batch_size, len(self.replay)))
            xs = np.stack([b[0] for b in batch], axis=0)
            ys = np.array([b[1] for b in batch], dtype=np.float32)
            x_t = torch.from_numpy(xs).to(self.device)
            y_t = torch.from_numpy(ys).to(self.device)
            self.net.train()
            pred = self.net(x_t).squeeze(-1)
            loss = self.loss_fn(pred, y_t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0
        self.last_train_ts = time.time()
        return avg_loss


import numpy as np

def build_pool_action_features(pool_diag: dict, pool_obj, action_name: str, global_params: dict = None, action_id: int = 0, n_actions: int = 2):
    """
    构建特征向量（numpy 1d）：
    - pool_diag: compute_cpool() 返回的 diagnostics
    - pool_obj: ContinuousSamplePool 实例（可读取 var_unstable, max_size 等）
    - action_name: 'FailureDrivenAction'/'SuccessDrivenAction'...
    - global_params: 可选键值字典用于加入更多超参数（如 var_unstable 等）
    - action_id: int index of action (for one-hot embedding)
    - n_actions: total number of actions
    """
    total = float(pool_diag.get("total_samples", 0.0))
    easy = float(pool_diag.get("easy_ratio", 0.0))
    informative = float(pool_diag.get("informative_ratio", 0.0))
    hard = float(pool_diag.get("hard_ratio", 0.0))
    cpool = float(pool_diag.get("cpool", 0.0))
    # normalized global params
    max_size = float(getattr(pool_obj, "max_size", 1000)) / 1000.0
    var_unstable = float(getattr(pool_obj, "var_unstable", 0.05))
    if global_params is None:
        global_params = {}
    # action one-hot
    action_onehot = np.zeros(n_actions, dtype=np.float32)
    action_onehot[action_id] = 1.0
    base = np.array([total/1000.0, easy, informative, hard, cpool, max_size, var_unstable], dtype=np.float32)
    feat = np.concatenate([base, action_onehot], axis=0)
    return feat