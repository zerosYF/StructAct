import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from micronet.parameters import ParamBundle

class MultiHeadSurrogateNet(nn.Module):
    """
    多头网络，每个头对应 ParamBundle 中的一组可学习参数
    """
    def __init__(self, input_dim, hidden=[64, 32]):
        super().__init__()
        # 公共隐藏层
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared_net = nn.Sequential(*layers)
        
        # 多头输出
        self.info_head = nn.Linear(last, 3)   # informative_logits
        self.pool_head = nn.Linear(last, 3)   # pool_logits
        self.mcts_head = nn.Linear(last, 1)   # alpha

    def forward(self, x):
        h = self.shared_net(x)
        info = self.info_head(h)
        pool = self.pool_head(h)
        alpha = self.mcts_head(h).squeeze(-1)
        return info, pool, alpha

class MultiHeadSurrogateController:
    """
    多头控制器，强化训练 ParamBundle
    """
    def __init__(self, param_bundle: ParamBundle, lr=1e-3, device=None, buffer_size=2000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = param_bundle  # ParamBundle
        self.input_dim = len(param_bundle.to_tensor())
        self.net = MultiHeadSurrogateNet(self.input_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()
        self.min_train_samples = 16

    def get_features(self):
        """
        使用当前 ParamBundle 的参数作为网络输入
        """
        return self.bundle.to_tensor().detach().cpu().numpy().astype(np.float32)

    def _add_transition(self, reward: float):
        """
        用 reward 构建训练样本
        当前网络预测值 -> 目标为 reward + 当前参数的梯度引导
        """
        features = self.get_features()
        # target: reward作为强化信号，这里简单用 reward 叠加到当前参数向量
        target = features + reward  # 简单强化信号方案
        self.replay.append((features, target.astype(np.float32)))
    
    def _train_step(self, batch_size=16, epochs=1):
        if len(self.replay) < self.min_train_samples:
            return 0.0
        losses = []
        for _ in range(epochs):
            batch = random.sample(self.replay, min(batch_size, len(self.replay)))
            xs = np.stack([b[0] for b in batch], axis=0)
            ys = np.stack([b[1] for b in batch], axis=0)
            x_t = torch.from_numpy(xs).to(self.device)
            y_t = torch.from_numpy(ys).to(self.device)
            self.net.train()
            info_pred, pool_pred, alpha_pred = self.net(x_t)
            # 组合所有头的预测值为单个向量
            pred_concat = torch.cat([info_pred, pool_pred, alpha_pred.unsqueeze(-1)], dim=1)
            loss = self.loss_fn(pred_concat, y_t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def predict_and_apply(self):
        """
        预测当前 bundle 的参数更新，并直接写回 bundle
        """
        self.net.eval()
        with torch.no_grad():
            x = self.bundle.get_vector().unsqueeze(0).to(self.device)  # shape (1,8)
            info_pred, pool_pred, alpha_pred = self.net(x)
        
        # 写回 bundle
        self.bundle.informative_logits.data = info_pred.squeeze(0)
        self.bundle.pool_logits.data = pool_pred.squeeze(0)
        self.bundle.mcts_alpha.data = alpha_pred.squeeze(0)
    
        return self.bundle  # 返回 bundle，Node 可以直接使用
    
    def reforce(self, reward: float):
        """
        将 reward 作为强化信号，训练多头网络预测更优参数
        """
        features = self.bundle.get_vector().cpu().numpy()
        # 使用 reward 作为目标，这里简单做回归
        target_bundle = {
            'informative': features[:3],   # 或者某种目标策略
            'pool': features[3:6],
            'alpha': features[6]
        }
        self._add_transition(features, target_bundle, reward)
        return self._train_step(batch_size=16, epochs=1)