import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiCard,
    HanabiObservation,
    HanabiHistoryItem
)
from torch import nn
from typing import List
from .logic_solver import observation_solver
from .utils import count_total_moves, move2id, count_total_cards


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

    def dim(self):
        return self.num_players * self.hand_size * self.num_colors * self.num_ranks


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

    def dim(self):
        return self.dim_discard + count_total_cards(self.num_colors, self.num_ranks)


class FireworkEncoder(nn.Module):
    """
    Output a hard embedding of the current fireworks.
    Shape: (#colors x #ranks, )
    """
    def __init__(self, num_colors: int, num_ranks: int, device: str):
        """
        Args:
            num_colors (int):
            num_ranks (int):
            device (str):
        """
        super().__init__()
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.device = device
    
    def forward(self, firework: List[int]):
        firework = torch.tensor(firework, dtype=torch.long, device=self.device, requires_grad=False)
        firework_emb = nn.functional.one_hot(firework, self.num_ranks + 1)
        return firework_emb[:, :-1].flatten().float()

    def dim(self):
        return self.num_ranks * self.num_colors


class LastMovesEncoder(nn.Module):
    def __init__(self, num_players: int, hand_size: int, num_colors: int, num_ranks: int, device: str, dim_move: int):
        """
        Args:
            num_players (int):
            hand_size (int):
            num_colors (int):
            num_ranks (int):
            device (str):
            dim_moves (int): the dimension of the history movement dimmension.
        """
        super().__init__()
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.count_total_moves = count_total_moves(num_players, num_colors, num_ranks, hand_size)
        self.move_rnn = nn.LSTM(
            input_size=num_players * self.count_total_moves,
            hidden_size=dim_move,
            num_layers=1,
            batch_first=True,
            device=device
        )
        self.dim_move = dim_move
        self.device = device
        self.h0 = nn.Parameter(torch.randn((1, dim_move), device=device, requires_grad=True))
        self.c0 = nn.Parameter(torch.randn((1, dim_move), device=device, requires_grad=True))
    
    def forward(self, last_moves: List[HanabiHistoryItem], cur_player_offset: int):
        if len(last_moves) == 0:
            return self.h0
        move_ids = []
        for history in reversed(last_moves):
            move = history.move()
            move_id = move2id(move, self.num_players, self.num_colors, self.num_ranks, self.hand_size)
            player = history.player()
            if player >= cur_player_offset:
                player -= cur_player_offset
            else:
                player = self.num_players + player - cur_player_offset
            move_ids.append(player * self.count_total_moves + move_id)
        move_ids = torch.tensor(move_ids, dtype=torch.long, device=self.device, requires_grad=False)
        move_ids = nn.functional.one_hot(move_ids, num_classes=self.num_players * self.count_total_moves).float()
        return self.move_rnn.forward(move_ids, (self.h0, self.c0))[0][-1]

    def dim(self):
        return self.dim_move


class TokenEncoder(nn.Module):
    def __init__(self, max_tokens: int, device: str):
        """
        Args:
            max_tokens (int): the maximum number of information tokens
        """
        self.max_tokens = max_tokens
        self.device = device
    
    def forward(self, cur_token: int):
        if cur_token == 0:
            return torch.zeros(self.max_tokens, dtype=torch.float32, device=self.device)
        else:
            cur_token = torch.tensor([cur_token - 1], dtype=torch.long, device=self.device, requires_grad=False)
            return nn.functional.one_hot(cur_token, num_classes=self.max_tokens).float()

    def dim(self):
        return self.max_tokens
