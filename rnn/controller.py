import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Optional
from .rnn import RNN
from logger import logger
from visualizer import Visualizer 

class TemplateController:
    def __init__(self, search_space: List[int], 
                 hidden_dim: int = 128, 
                 lr:float=1e-3, 
                 reward_scale:int=10,
                 baseline:float=0.8, 
                 baseline_alpha:float=0.5,
                 min_entropy_weight:float=0.001, 
                 max_entropy_weight:float=0.5, 
                 entropy_decay_rate:float=0.95,
                 attribution_interval:int=10,
                 aux_loss_coef:float=1.5
                 ):
        self.model: RNN = RNN(search_space, hidden_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.search_space = search_space

        self.reward_scale = reward_scale
        self.baseline = baseline
        self.baseline_alpha = baseline_alpha
        self.aux_loss_coef = aux_loss_coef
        self.min_entropy_weight = min_entropy_weight
        self.max_entropy_weight = max_entropy_weight
        self.entropy_decay_rate = entropy_decay_rate
        self.attribution_interval = attribution_interval
        self.iter_count = 0

        self.last_logits: list = None

        logger.info(f"📈 [RNNController] Initialized - params counts: {len(search_space)}")
 
        self.rewards = [] 
    
    def get_slot_dim(self, slot_index: int) -> int:
        return self.search_space[slot_index]

    def train_step(self):
        self.iter_count += 1
        flat_params, log_prob, entropy, logits_list = self.model()
        self.last_logits = logits_list  
        return flat_params, log_prob, entropy

    def reinforce(self,
                  log_prob_sum: torch.Tensor,
                  reward: float,
                  entropy: torch.Tensor,
                  slot_rewards: Optional[List[float]] = None):
        self.model.train()
        # ---- policy ----
        advantage = (reward - self.baseline) * self.reward_scale
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * reward
        entropy_weight = max(self.min_entropy_weight, self.max_entropy_weight * (self.entropy_decay_rate ** self.iter_count))
        loss = -advantage * log_prob_sum - entropy_weight * entropy

        # ---- slot-level loss ----
        self._slot_level_atrribution(slot_rewards)

        # ---- param update ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.rewards.append(reward)
        reward_mean = sum(self.rewards) / len(self.rewards)
        logger.info(f"📈 [RNNController] REINFORCE finished - avg_reward={reward_mean:.4f}, loss={loss.item():.4f}, entropy={entropy.item():.4f}")
        Visualizer.log_train(reward_mean, entropy.item())
    
    def _slot_level_atrribution(self, slot_rewards=None):
        if slot_rewards is not None and self.last_logits is not None:
            losses = []
            for logits, reward in zip(self.last_logits, slot_rewards):
            # Normalize slot rewards into a probability distribution
                log_prob = F.log_softmax(logits, dim=-1)
                target_prob = torch.softmax(torch.tensor([reward], dtype=torch.float32), dim=-1)
                target_prob = target_prob.unsqueeze(0).expand(log_prob.size(0), -1)
                loss = F.kl_div(log_prob, target_prob, reduction='mean')
                losses.append(loss)
            aux_loss = torch.mean(torch.stack(losses))
            loss += self.aux_loss_coef * aux_loss
            logger.info(f"🧩 [RNNController] slot level loss = {aux_loss.item():.4f}")
