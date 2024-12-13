import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiCard,
    HanabiObservation
)
from torch import nn
from typing import List
from .logic_solver import observation_solver


class CardKnowledgeEncoder(nn.Module):
    """
    Output a hard embedding of the given card knowledge.
    Shape: #players x max_hand_size x #colors x #ranks
    """
    def __init__(self, num_players: int, num_colors: int, num_ranks: int, hand_size: int, device: str):
        """
        Args:
            num_players (int):
            hand_size (int): the max number of cards in a player's hand
            device (str):
        """
        super().__init__()
        self.device = device
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
    
    def forward(self, observation: HanabiObservation):
        detailed_card_knowledge = observation_solver(observation, self.num_ranks, self.num_colors, self.num_players)
        knowledge_emb = []
        for player_knowledge in detailed_card_knowledge:
            for card_knowledge in player_knowledge:
                knowledge_emb.append(torch.tensor(card_knowledge.plausible, dtype=torch.float32, device=self.device, requires_grad=False))
            for _ in range(self.hand_size - len(player_knowledge)):
                knowledge_emb.append(torch.zeros((self.num_colors, self.num_ranks), dtype=torch.float32, device=self.device))
        knowledge_emb = torch.stack(knowledge_emb)
        return knowledge_emb.flatten()


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


class FireworkEncoder(nn.Module):
    """
    Output a hard embedding of the current fireworks.
    Shape: (#colors x #ranks, )
    """
    def __init__(self, num_ranks: int, device: str):
        """
        Args:
            num_ranks (int):
            device (str):
        """
        super().__init__()
        self.num_ranks = num_ranks
        self.device = device
    
    def forward(self, firework: List[int]):
        firework = torch.tensor(firework, dtype=torch.long, device=self.device, requires_grad=False)
        firework_emb = nn.functional.one_hot(firework, self.num_ranks + 1)
        return firework_emb[:, :-1].flatten().float()
