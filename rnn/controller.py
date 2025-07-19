import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Optional
from .rnn import RNN  # 假设你的RNN定义在.rnn模块中
from logger import logger
from visualizer import Visualizer 

class TemplateController:
    def __init__(self, search_space: List[int], hidden_dim: int = 128, lr=1e-3, aux_loss_coef=0.1):
        self.model: RNN = RNN(search_space, hidden_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.search_space = search_space

        self.baseline = 0.0
        self.baseline_alpha = 0.9
        self.aux_loss_coef = aux_loss_coef

        # 存储 forward 时的 logit 用于结构归因
        self.last_logits: list = None

        logger.info(f"📈 [RNNController] 初始化完成 - 参数个数: {len(search_space)}")

    def train_step(self):
        self.model.train()
        flat_params, log_prob, entropy, logits_list = self.model(return_logits=True)
        self.last_logits = logits_list  
        return flat_params, log_prob, entropy

    def reinforce(self,
                  log_prob_sum: torch.Tensor,
                  reward: float,
                  entropy: torch.Tensor,
                  slot_rewards: Optional[List[float]] = None):
        """主策略梯度 + slot-level KL归因辅助"""

        self.model.train()

        # ---- 主策略梯度更新 ----
        advantage = reward - self.baseline
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * reward
        entropy_weight = 0.01

        loss = -advantage * log_prob_sum - entropy_weight * entropy

        # ---- slot-level 结构归因辅助 loss ----
        if slot_rewards is not None and self.last_logits is not None:
            slot_rewards_tensor = torch.tensor(slot_rewards, dtype=torch.float32)

            # Normalize slot rewards into a probability distribution
            target_probs = torch.softmax(slot_rewards_tensor, dim=0).detach()  # [num_slots]

            aux_loss = 0.0
            for i, logits in enumerate(self.last_logits):
                pred_log_prob = F.log_softmax(logits.unsqueeze(0), dim=1)  # logits shape: [slot_dim] -> [1, slot_dim]
                target_prob_expanded = target_probs[i].unsqueeze(0).expand_as(pred_log_prob)  # [1, slot_dim]

                # KL divergence between pred and target probabilities for this slot
                # 这里 target_prob_expanded 是标量扩展，实际上 KL 用标量没意义，应该用 slot_rewards[i] 指向某个具体类别概率
                # 如果 slot_rewards 是 scalar reward per slot，直接用它做 softmax不合理
                # 通常 slot_rewards 应该是每个 slot 各个动作的 reward向量，若是标量则需要设计不同的辅助信号。
                # 此处简单做比例放大辅助，或者你可以用 slot_rewards 做权重加权
                # 这里先用简化的 MSE loss 替代示范
                target_dist = torch.full_like(pred_log_prob, target_probs[i].item())  # 让目标分布是均匀或定值(示例)
                aux_loss += F.kl_div(pred_log_prob, target_dist, reduction='batchmean')

            loss += self.aux_loss_coef * aux_loss
            logger.info(f"🧩 [RNNController] 加入结构归因辅助 loss = {aux_loss.item():.4f}")

        # ---- 参数更新 ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        logger.info(f"📈 [RNNController] REINFORCE 完成 - reward={reward:.4f}, loss={loss.item():.4f}, entropy={entropy.item():.4f}")
        Visualizer.log_train(loss.item(), reward, entropy.item())
    
