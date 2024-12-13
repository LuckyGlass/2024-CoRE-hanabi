import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiCard
)
from torch import nn
from typing import List
from .utils import count_total_cards


class DiscardPileEncoder(nn.Module):
    def __init__(self, num_colors: int, num_ranks: int, dim_discard: int, device: str):
        super().__init__()
        self.rnn_encoder = nn.LSTM(num_colors * num_ranks, dim_discard, num_layers=1, batch_first=True, device=device)
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.device = device
        self.dim_discard = dim_discard
    
    def forward(self, discard_pile: List[HanabiCard]):
        # Get RNN Embedding
        if len(discard_pile) == 0:
            rnn_emb = torch.zeros(self.dim_discard, dtype=torch.float32, device=self.device)
        else:
            card_ids = [card.color() * self.num_ranks + card.rank() for card in discard_pile]
            card_ids = torch.tensor(card_ids, dtype=torch.long, device=self.device, requires_grad=False)
            rnn_input = nn.functional.one_hot(card_ids, num_classes=self.num_colors * self.num_ranks)
            rnn_emb = self.rnn_encoder(rnn_input.float())[0][-1]
        # Get Hard Embedding
        temp_count = [[0 for _ in range(self.num_ranks)] for _ in range(self.num_colors)]
        for card in discard_pile:
            temp_count[card.color()][card.rank()] += 1
        hard_emb = []
        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                if r == 0:
                    hard_emb += [1] * temp_count[c][r] + [0] * (3 - temp_count[c][r])
                elif r == 5:
                    hard_emb += [1] * temp_count[c][r] + [0] * (1 - temp_count[c][r])
                else:
                    hard_emb += [1] * temp_count[c][r] + [0] * (2 - temp_count[c][r])
        hard_emb = torch.tensor(hard_emb, dtype=torch.float32, requires_grad=False, device=self.device)
        return torch.concat((rnn_emb, hard_emb))
