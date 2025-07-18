import torch
from torch.optim import Adam
from typing import List
from Experiment.rnn.rnn import RNN
from logger import logger
from visualizer import Visualizer
from Experiment.rnn.blocks import PromptBlock

class RNNController:
    def __init__(self, blocks:List[PromptBlock], hidden_dim: int = 128, lr=1e-3):
        self.blocks = blocks
        self.search_space = [dim for block in blocks for dim in block.get_search_space()]
        self.model:RNN = RNN(self.search_space, hidden_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.baseline = 0.0
        self.baseline_alpha = 0.9


    def decode(self, flat_params: List[int]) -> List[str]:
        """
        将 RNN 输出的超参数扁平向量转换为每个 Block 的渲染文本
        """
        idx = 0
        results = []
        for block in self.blocks:
            num_slots = block.get_num_slots()
            params = flat_params[idx:idx + num_slots]
            rendered = block.render(params)
            results.append(rendered)
            idx += num_slots
        return results
    
    def get_slot_dims(self) -> List[int]:
        return self.search_space

    def render_prompt(self, flat_params: List[int]) -> str:
        return "\n".join(self.decode(flat_params))
    
    def train_step(self):
        self.model.train()
        actions, log_prob, entropy = self.model()
        return actions, log_prob, entropy

    def reinforce(self, log_prob_sum, reward: float, entropy:torch.Tensor):
        """用前向生成出的动作序列和 reward，执行一次 REINFORCE 学习"""
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
    
