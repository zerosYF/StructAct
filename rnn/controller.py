import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Optional
from .rnn import RNN  # 假设你的RNN定义在.rnn模块中
from logger import logger
from visualizer import Visualizer 

class TemplateController:
    def __init__(self, search_space: List[int], hidden_dim: int = 128, lr=1e-3, aux_loss_coef=0.1, device='cuda'):
        self.model: RNN = RNN(search_space, hidden_dim).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.search_space = search_space
        self.device = device

        self.baseline = 0.0
        self.baseline_alpha = 0.9
        self.aux_loss_coef = aux_loss_coef

        # 存储 forward 时的 logit 用于结构归因
        self.last_logits: Optional[torch.Tensor] = None

        logger.info(f"📈 [RNNController] 初始化完成 - 参数个数: {len(search_space)}")

    def train_step(self):
        self.model.train()
        flat_params, log_prob, entropy, logits = self.model(return_logits=True)
        self.last_logits = logits.detach()  # 结构归因辅助时需要 detach
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
            slot_rewards_tensor = torch.tensor(slot_rewards, dtype=torch.float32, device=self.device)

            # Normalize slot rewards into a probability distribution
            target_probs = torch.softmax(slot_rewards_tensor, dim=0).detach()

            # 模型输出的 slot logits => 概率
            pred_log_probs = F.log_softmax(self.last_logits, dim=1)  # shape: [num_slots, num_choices]
            target_probs_expanded = target_probs.unsqueeze(1).expand_as(pred_log_probs)

            # 使用 KL 散度作为结构归因 loss
            aux_loss = F.kl_div(pred_log_probs, target_probs_expanded, reduction='batchmean')
            loss += self.aux_loss_coef * aux_loss

            logger.info(f"🧩 [RNNController] 加入结构归因 KL loss = {aux_loss.item():.4f}")

        # ---- 参数更新 ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        logger.info(f"📈 [RNNController] REINFORCE 完成 - reward={reward:.4f}, loss={loss.item():.4f}, entropy={entropy.item():.4f}")
        Visualizer.log_train(loss.item(), reward, entropy.item())
    
