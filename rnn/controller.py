# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from typing import List, Optional
# from .rnn import RNN
# from logger import logger
# from visualizer import Visualizer 

# class TemplateController:
#     def __init__(self, search_space: List[int], 
#                  hidden_dim: int = 128, 
#                  lr:float=1e-3, 
#                  reward_scale:int=10,
#                  ):
#         self.model: RNN = RNN(search_space, hidden_dim)
#         self.optimizer = Adam(self.model.parameters(), lr=lr)
#         self.search_space = search_space

#         self.reward_scale = reward_scale
#         self.iter_count = 0

#         self.ppo_epochs = 4
#         self.ppo_clip = 0.2
#         self.value_coef = 0.5
#         self.entropy_coef = 0.01

#         self.last_logits: list = None
#         self.last_params: List[int] = None
#         self.last_log_prob_sum: torch.Tensor = None
#         self.last_entropy: torch.Tensor = None
#         self.last_value: torch.Tensor = None

#         logger.info(f"📈 [RNNController] Initialized - params counts: {len(search_space)}")
 
#         self.rewards = [] 
    
#     def get_slot_dim(self, slot_index: int) -> int:
#         return self.search_space[slot_index]

#     def train_step(self):
#         self.iter_count += 1
#         flat_params, log_prob_sum, entropy_mean, logits_list, value = self.model()
#         self.last_params = flat_params
#         self.last_log_prob_sum = log_prob_sum
#         self.last_entropy = entropy_mean
#         self.last_logits = logits_list  
#         self.last_value = value
#         logger.info(f"🤖 [RNNController] Sampled new structure parameters: {flat_params}")
#         return flat_params

#     def reinforce(self, reward: float):
#         self.model.train()
#         #PPO
#         scaled_reward = reward * self.reward_scale
#         old_value = self.last_value
#         returns = torch.tensor([[scaled_reward]], dtype=torch.float32)
#         advantage = (returns - old_value).detach()
#         old_log_prob = self.last_log_prob_sum.detach()
#         total_policy_loss = 0.0
#         total_value_loss = 0.0
#         total_entropy = 0.0
#         for _ in range(self.ppo_epochs):
#             new_log_prob, new_entropy, _, new_value = self.model.evaluate_params(self.last_params)
#             ratio = torch.exp(new_log_prob - old_log_prob)
#             adv = advantage.squeeze()
#             surr1 = ratio * adv
#             surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
#             policy_loss = -torch.min(surr1, surr2).mean()
#             value_loss = F.mse_loss(new_value, returns) * self.value_coef
#             entropy_loss = - self.entropy_coef * new_entropy
#             loss = policy_loss + value_loss + entropy_loss 

#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()

#             total_policy_loss += policy_loss.item()
#             total_value_loss += value_loss.item()
#             total_entropy += new_entropy.item()

#         self.rewards.append(reward)
#         reward_mean = sum(self.rewards) / len(self.rewards)
#         logger.info(
#             f"📈 [RNNController][PPO] epochs={self.ppo_epochs} avg_reward={reward_mean:.4f} "
#             f"policy_loss={total_policy_loss/self.ppo_epochs:.4f} value_loss={total_value_loss/self.ppo_epochs:.4f} "
#             f"entropy={total_entropy/self.ppo_epochs:.4f}"
#         )
#         Visualizer.log_train(reward_mean, self.last_entropy.item())

import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from .rnn import RNN
from logger import logger
from visualizer import Visualizer

class TemplateController:
    def __init__(self, search_space: List[int],
                 hidden_dim: int = 128,
                 lr: float = 1e-3,
                 reward_scale: int = 10):
        self.model: RNN = RNN(search_space, hidden_dim)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.search_space = search_space

        self.reward_scale = reward_scale
        self.iter_count = 0

        self.ppo_epochs = 4
        self.ppo_clip = 0.2
        self.value_coef = 1.0      # Increased from 0.5 to match policy loss scale
        self.entropy_coef_init = 0.05   # Initial entropy coefficient
        self.entropy_coef = self.entropy_coef_init
        self.entropy_coef_decay = 0.995  # Decay factor per epoch
        self.entropy_coef_min = 0.001    # Minimum entropy coefficient
        self.target_kl = 0.02       # 目标 KL，可被批量接口覆盖

        self.last_logits: list = None
        self.last_params: List[int] = None
        self.last_log_prob_sum: torch.Tensor = None
        self.last_entropy: torch.Tensor = None
        self.last_value: torch.Tensor = None

        logger.info(f"📈 [RNNController] Initialized - params counts: {len(search_space)}")
        self.rewards = []

    def get_slot_dim(self, slot_index: int) -> int:
        return self.search_space[slot_index]

    @torch.no_grad()
    def train_step(self):
        """采样一次（单步回合）。仅缓存旧策略统计量，不建图。"""
        self.iter_count += 1
        self.model.eval()
        flat_params, log_prob_sum, entropy_mean, logits_list, value = self.model()
        self.last_params = flat_params
        self.last_log_prob_sum = log_prob_sum
        self.last_entropy = entropy_mean
        self.last_logits = logits_list
        self.last_value = value
        logger.info(f"🤖 [RNNController] Sampled new structure parameters: {flat_params}")
        return flat_params

    def reinforce(self, reward: float):
        """单样本的 PPO-Clip 更新（保留以兼容旧流程）。"""
        self.model.train()
        scaled_reward = reward * self.reward_scale
        old_value = self.last_value
        returns = torch.tensor([[scaled_reward]], dtype=torch.float32, device=old_value.device)

        advantage = (returns - old_value).detach()
        old_log_prob = self.last_log_prob_sum.detach()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            new_log_prob, new_entropy, _, new_value = self.model.evaluate_params(self.last_params)

            ratio = torch.exp(new_log_prob - old_log_prob)
            adv = advantage.squeeze()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_value, returns) * self.value_coef
            entropy_loss = - self.entropy_coef * new_entropy  # ✅ 熵正则真正生效

            loss = policy_loss + value_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += new_entropy.item()

            approx_kl = (old_log_prob - new_log_prob).detach().item()
            if approx_kl > self.target_kl:
                break

        self.rewards.append(reward)
        reward_mean = sum(self.rewards) / len(self.rewards)
        
        # Decay entropy coefficient
        self.entropy_coef = max(self.entropy_coef * self.entropy_coef_decay, self.entropy_coef_min)
        
        logger.info(
            f"📈 [RNNController][PPO] epochs={self.ppo_epochs} avg_reward={reward_mean:.4f} "
            f"policy_loss={total_policy_loss/self.ppo_epochs:.4f} value_loss={total_value_loss/self.ppo_epochs:.4f} "
            f"entropy={total_entropy/self.ppo_epochs:.4f} entropy_coef={self.entropy_coef:.4f}"
        )
        # Remove duplicate log_train call - already called in SearchController
        # Visualizer.log_train(reward_mean, float(self.last_entropy))

    def reinforce_batch(self, batch_samples, minibatch_size: int = 128, target_kl: float = 0.02):
        """
        批量版 PPO-Clip（推荐）：多样本、多 epoch、小批、KL 早停。
        batch_samples: List[dict] with keys:
            - params: List[int]
            - old_logp: torch.Tensor (scalar)
            - old_value: torch.Tensor (scalar or [1])
            - reward: float
        """
        self.model.train()
        device = next(self.model.parameters()).device
        B = len(batch_samples)
        if B == 0:
            return

        old_logp = torch.stack([s["old_logp"].reshape(()) for s in batch_samples]).to(device)         # [B]
        old_value = torch.stack([s["old_value"].reshape(()) for s in batch_samples]).to(device)       # [B]
        returns = torch.tensor([s["reward"] * self.reward_scale for s in batch_samples],
                               dtype=torch.float32, device=device)                                     # [B]
        adv = (returns - old_value).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        params_list = [s["params"] for s in batch_samples]

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                mb = idx[start:start + minibatch_size].tolist()

                new_logp_list, new_value_list, entropy_list = [], [], []
                for i in mb:
                    nl, ne, _, nv = self.model.evaluate_params(params_list[i])
                    new_logp_list.append(nl.reshape(()))
                    new_value_list.append(nv.reshape(()))
                    entropy_list.append(ne.reshape(()))

                new_logp = torch.stack(new_logp_list).to(device)     # [b]
                new_value = torch.stack(new_value_list).to(device)   # [b]
                entropy = torch.stack(entropy_list).mean().to(device)

                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_value, returns[mb]) * self.value_coef
                entropy_loss = - self.entropy_coef * entropy

                loss = policy_loss + value_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            with torch.no_grad():
                approx_kl = (old_logp - new_logp).mean().item()
            if approx_kl > target_kl:
                break

        mean_reward = (returns / self.reward_scale).mean().item()
        # Store the batch entropy for visualization
        self.batch_entropy = float(entropy) if 'entropy' in locals() else 0.0
        
        # Update rewards history with batch rewards
        batch_rewards = (returns / self.reward_scale).tolist()
        self.rewards.extend(batch_rewards)
        avg_reward = sum(self.rewards) / len(self.rewards) if self.rewards else mean_reward
        
        # Decay entropy coefficient
        self.entropy_coef = max(self.entropy_coef * self.entropy_coef_decay, self.entropy_coef_min)
        
        logger.info(
            f"📈 [RNNController][PPO-Batch] epochs={self.ppo_epochs} "
            f"batch_mean={mean_reward:.4f} avg_reward={avg_reward:.4f} "
            f"entropy_coef={self.entropy_coef:.4f}"
        )
        
        # Remove duplicate log_train call - already called in SearchController
        # Visualizer.log_train(mean_reward, float(entropy))
