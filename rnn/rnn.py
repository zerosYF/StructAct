import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List

class RNN(nn.Module):
    def __init__(self, slot_dims:List[int], hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_num = len(slot_dims)
        self.slot_dims = slot_dims

        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in slot_dims])
        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in slot_dims])

        # 可学习的起始输入向量，作为 x₀（input_emb₀）
        self.start_token = nn.Parameter(torch.zeros(1, hidden_dim))


    def forward(self):
        h = torch.zeros(1, self.hidden_dim)   # ← 每次都初始化
        c = torch.zeros_like(h)
        input_emb = self.start_token

        decisions, log_probs, entropies = [], [], []

        for i in range(self.slot_num):
            h, c = self.rnn(input_emb, (h, c))   # ← 使用局部状态
            logits = self.heads[i](h)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            dist = Categorical(probs)

            choice = dist.sample()
            log_prob = dist.log_prob(choice)
            entropy = dist.entropy()

            input_emb = self.embeddings[i](choice.unsqueeze(0))

            decisions.append(choice.item())
            log_probs.append(log_prob)
            entropies.append(entropy)

        return decisions, torch.stack(log_probs).sum(), torch.stack(entropies).mean()