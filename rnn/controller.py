import torch
from torch.optim import Adam
from typing import List
from rnn.rnn import RNN
from logger import logger
from visualizer import Visualizer

class TemplateController:
    def __init__(self, search_space:List[int], hidden_dim: int = 128, lr=1e-3):
        self.model:RNN = RNN(search_space, hidden_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.baseline = 0.0
        self.baseline_alpha = 0.9
    
    def train_step(self):
        self.model.train()
        actions, log_prob, entropy = self.model()
        return actions, log_prob, entropy

    def reinforce(self, log_prob_sum, reward: float, entropy:torch.Tensor):
        """执行一次 REINFORCE 学习"""
        self.model.train()
        advantage = reward - self.baseline
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * reward
        entropy_weight = 0.01
        loss = -advantage * log_prob_sum - entropy_weight * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) #防止梯度爆炸
        self.optimizer.step()
        logger.info(f"📈 [RNNController] REINFORCE 更新完成 - reward={reward:.4f}, loss={loss.item():.4f}")
        Visualizer.log_train(loss.detach().item(), reward, entropy.detach().item())
    
