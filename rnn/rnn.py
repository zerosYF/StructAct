# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from typing import List

# class RNN(nn.Module):
#     def __init__(self, slot_dims: List[int], hidden_dim=128):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.slot_num = len(slot_dims)
#         self.slot_dims = slot_dims

#         # One embedding layer per slot
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(dim, hidden_dim) for dim in slot_dims
#         ])

#         # LSTM cell for step-by-step generation
#         self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)

#         # One output head per slot to predict logits
#         self.heads = nn.ModuleList([
#             nn.Linear(hidden_dim, dim) for dim in slot_dims
#         ])

#         self.value_head = nn.Linear(hidden_dim, 1)

#         # Learnable start token for the first input
#         self.start_token = nn.Parameter(torch.zeros(1, hidden_dim))

#     def forward(self):
#         # Initialize LSTM hidden and cell states
#         h = torch.zeros(1, self.hidden_dim, device=self.start_token.device)
#         c = torch.zeros_like(h)
#         input_emb = self.start_token

#         decisions = []
#         log_probs = []
#         entropies = []
#         logits_list = []

#         for i in range(self.slot_num):
#             # Step-by-step generation through the RNN
#             h, c = self.rnn(input_emb, (h, c))

#             # Predict logits for the current slot
#             logits = self.heads[i](h)  # shape: [1, slot_dim]
#             dist = Categorical(logits=logits.squeeze(0))

#             # Sample one parameter value from the distribution
#             choice = dist.sample()
#             log_prob = dist.log_prob(choice)
#             entropy = dist.entropy()

#             # Update the next-step input embedding
#             input_emb = self.embeddings[i](choice.unsqueeze(0))

#             # Record values
#             decisions.append(choice.item())
#             log_probs.append(log_prob)
#             entropies.append(entropy)
#             logits_list.append(logits.squeeze(0))  # shape: [slot_dim]

#         log_prob_sum = torch.stack(log_probs).sum()
#         entropy_mean = torch.stack(entropies).mean()
#         value = self.value_head(h)

#         return decisions, log_prob_sum, entropy_mean, logits_list, value
    
#     def evaluate_params(self, params: List[int]):
#         """
#         Given a *concrete* action sequence (list of ints length slot_num),
#         compute the log_prob_sum (under current policy) and value,
#         and also entropy mean and logits_list. This does NOT sample.
#         """
#         assert len(params) == self.slot_num
#         device = self.start_token.device
#         h = torch.zeros(1, self.hidden_dim, device=device)
#         c = torch.zeros_like(h)
#         input_emb = self.start_token

#         log_probs = []
#         entropies = []
#         logits_list = []

#         for i, act in enumerate(params):
#             h, c = self.rnn(input_emb, (h, c))
#             logits = self.heads[i](h)  # [1, slot_dim]
#             dist = Categorical(logits=logits.squeeze(0))
#             action_tensor = torch.tensor(int(act), device=device)
#             log_prob = dist.log_prob(action_tensor)
#             entropy = dist.entropy()
#             input_emb = self.embeddings[i](action_tensor.unsqueeze(0))
#             log_probs.append(log_prob)
#             entropies.append(entropy)
#             logits_list.append(logits.squeeze(0))

#         log_prob_sum = torch.stack(log_probs).sum()
#         entropy_mean = torch.stack(entropies).mean()
#         value = self.value_head(h)  # [1,1]

#         return log_prob_sum, entropy_mean, logits_list, value

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List

class RNN(nn.Module):
    def __init__(self, slot_dims: List[int], hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_num = len(slot_dims)
        self.slot_dims = slot_dims

        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in slot_dims
        ])
        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in slot_dims
        ])
        self.value_head = nn.Linear(hidden_dim, 1)
        self.start_token = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self):
        device = self.start_token.device
        h = torch.zeros(1, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        input_emb = self.start_token

        decisions, log_probs, entropies, logits_list = [], [], [], []

        for i in range(self.slot_num):
            h, c = self.rnn(input_emb, (h, c))
            logits = self.heads[i](h)                        # [1, slot_dim]
            dist = Categorical(logits=logits.squeeze(0))     # ✅ 直接用 logits，更稳

            choice = dist.sample()                           # []
            log_prob = dist.log_prob(choice)                 # []
            entropy = dist.entropy()                         # []

            input_emb = self.embeddings[i](choice.unsqueeze(0))
            decisions.append(choice.item())
            log_probs.append(log_prob)
            entropies.append(entropy)
            logits_list.append(logits.squeeze(0))            # [slot_dim]

        log_prob_sum = torch.stack(log_probs).sum()
        entropy_mean = torch.stack(entropies).mean()
        value = self.value_head(h)                           # [1,1]
        return decisions, log_prob_sum, entropy_mean, logits_list, value

    def evaluate_params(self, params: List[int]):
        """给定离散动作序列，计算当前策略下的联合 logp、平均熵、价值。"""
        assert len(params) == self.slot_num
        device = self.start_token.device

        h = torch.zeros(1, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        input_emb = self.start_token

        log_probs, entropies, logits_list = [], [], []

        for i, act in enumerate(params):
            h, c = self.rnn(input_emb, (h, c))
            logits = self.heads[i](h)                        # [1, slot_dim]
            dist = Categorical(logits=logits.squeeze(0))     # ✅ logits 版本
            action_tensor = torch.tensor(int(act), device=device)
            log_prob = dist.log_prob(action_tensor)
            entropy = dist.entropy()

            input_emb = self.embeddings[i](action_tensor.unsqueeze(0))
            log_probs.append(log_prob)
            entropies.append(entropy)
            logits_list.append(logits.squeeze(0))

        log_prob_sum = torch.stack(log_probs).sum()
        entropy_mean = torch.stack(entropies).mean()
        value = self.value_head(h)                           # [1,1]
        return log_prob_sum, entropy_mean, logits_list, value
