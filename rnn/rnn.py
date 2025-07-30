import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List

class RNN(nn.Module):
    def __init__(self, slot_dims: List[int], hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_num = len(slot_dims)
        self.slot_dims = slot_dims

        # One embedding layer per slot
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in slot_dims
        ])

        # LSTM cell for step-by-step generation
        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)

        # One output head per slot to predict logits
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in slot_dims
        ])

        # Learnable start token for the first input
        self.start_token = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self):
        # Initialize LSTM hidden and cell states
        h = torch.zeros(1, self.hidden_dim, device=self.start_token.device)
        c = torch.zeros_like(h)
        input_emb = self.start_token

        decisions = []
        log_probs = []
        entropies = []
        logits_list = []

        for i in range(self.slot_num):
            # Step-by-step generation through the RNN
            h, c = self.rnn(input_emb, (h, c))

            # Predict logits for the current slot
            logits = self.heads[i](h)  # shape: [1, slot_dim]
            probs = F.softmax(logits, dim=-1).squeeze(0)
            dist = Categorical(probs)

            # Sample one parameter value from the distribution
            choice = dist.sample()
            log_prob = dist.log_prob(choice)
            entropy = dist.entropy()

            # Update the next-step input embedding
            input_emb = self.embeddings[i](choice.unsqueeze(0))

            # Record values
            decisions.append(choice.item())
            log_probs.append(log_prob)
            entropies.append(entropy)
            logits_list.append(logits.squeeze(0))  # shape: [slot_dim]

        log_prob_sum = torch.stack(log_probs).sum()
        entropy_mean = torch.stack(entropies).mean()

        return decisions, log_prob_sum, entropy_mean, logits_list