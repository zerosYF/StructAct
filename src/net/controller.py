import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from src.net.parameters import ParamBundle

class SurrogateNet(nn.Module):
    """
    Gaussian Policy Network
    输出每个参数的 Δθ 分布
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
        
        # 均值
        self.mu = nn.Linear(last, input_dim)
        # 方差
        self.log_std = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        h = self.shared_net(x)
        mu = self.mu(h)
        std = torch.exp(self.log_std)
        return mu, std

class SurrogateNetController:
    """
    控制器，强化训练 ParamBundle
    """
    def __init__(self, lr=1e-3, device=None, gamma=0.99):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = ParamBundle()
        self.input_dim = len(self.bundle.to_tensor())
        self.net = SurrogateNet(self.input_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.lr = lr
        
        self.gamma = gamma
        self.baseline = 0.0
        self.last_log_prob = None
        self.last_reward = None
    
    def act_and_apply(self):
        state = self.bundle.to_tensor().unsqueeze(0).to(self.device)  # shape (1, input_dim)
        mu, std = self.net(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        self.bundle.apply_delta(action.squeeze(0), lr=self.lr)
        self.last_log_prob = log_prob
        return self.bundle
    
    def reforce(self, reward: float):
        """
        将 reward 作为强化信号，训练多头网络预测更优参数
        """
        advantage = reward - self.baseline
        loss = - self.last_log_prob * advantage
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # 更新 baseline
        self.baseline = self.gamma * self.baseline + (1 - self.gamma) * reward