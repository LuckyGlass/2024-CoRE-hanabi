import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiCard,
    HanabiObservation,
    HanabiHistoryItem,
    HanabiMoveType
)
from torch import nn
from typing import List
from .logic_solver import observation_solver
from .utils import count_total_moves, move2id, count_total_cards


class CardKnowledgeEncoder(nn.Module):
    def __init__(self, num_players: int=2, num_colors: int=4, num_ranks: int=5, hand_size: int=5, device: str='cuda', **_):
        """
        Args:
            num_players (int): The number of players
            num_colors (int): The number of colors
            num_ranks (int): The number of ranks
            hand_size (int): The number of cards in hand
            device (str): `cuda` or `cpu`
        """
        super().__init__()
        self.device = device
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
    
    def forward(self, observation: HanabiObservation):
        """Encode the card knowledge of the player. In detail, the knowledge include his knowledge of his hand and from his view, other players' knowledge of theirselves' hands.
        Args:
            observation (HanabiObservation): The observation of the given player.
        Returns:
            knowledge_emb (Tensor): The knowledge embedding (a flattened tensor).
        """
        detailed_card_knowledge = observation_solver(observation, self.num_ranks, self.num_colors, self.num_players)
        knowledge_emb = []
        for player_knowledge in detailed_card_knowledge:
            for card_knowledge in player_knowledge:
                knowledge_emb.append(torch.tensor(card_knowledge.plausible, dtype=torch.float32, device=self.device, requires_grad=False))
            for _ in range(self.hand_size - len(player_knowledge)):
                knowledge_emb.append(torch.zeros((self.num_colors, self.num_ranks), dtype=torch.float32, device=self.device))
        knowledge_emb = torch.stack(knowledge_emb).flatten()
        return knowledge_emb

    def dim(self):
        return self.num_players * self.hand_size * self.num_colors * self.num_ranks


class DiscardPileEncoder(nn.Module):
    def __init__(self, num_colors: int=4, num_ranks: int=5, device: str='cuda', **_):
        """
        Args:
            num_colors (int): The number of colors.
            num_ranks (int): The number of ranks.
            device (str): `cuda` or `cpu`.
        """
        super().__init__()
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.device = device
        self.emb_dim_discard = count_total_cards(num_colors, num_ranks)
    
    def forward(self, discard_pile: List[HanabiCard]):
        """Encode the discarded pile.
        Args:
            discard_pile (List[HanabiCard]): The discard pile.
        Returns:
            emb (Tensor): The discard pile embedding (a flattened tensor).
        """
        # Get Hard Embedding
        temp_count = [[0 for _ in range(self.num_ranks)] for _ in range(self.num_colors)]
        for card in discard_pile:
            temp_count[card.color()][card.rank()] += 1
        hard_emb = []
        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                if r == 0:
                    hard_emb += [1] * temp_count[c][r] + [0] * (3 - temp_count[c][r])
                elif r == 4:
                    hard_emb += [1] * temp_count[c][r] + [0] * (1 - temp_count[c][r])
                else:
                    hard_emb += [1] * temp_count[c][r] + [0] * (2 - temp_count[c][r])
        hard_emb = torch.tensor(hard_emb, dtype=torch.float32, requires_grad=False, device=self.device)
        return hard_emb

    def dim(self):
        return self.emb_dim_discard


class FireworkEncoder(nn.Module):
    def __init__(self, num_colors: int=4, num_ranks: int=5, device: str='cuda', **_):
        """
        Args:
            num_colors (int): The number of colors.
            num_ranks (int): The number of ranks.
            device (str): `cuda` or `cpu`.
        """
        super().__init__()
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.device = device
    
    def forward(self, firework: List[int]):
        """Encode the fireworks.
        Args:
            firework (List[int]): The fireworks; `firework[i]` means the top rank of color `i`.
        Returns:
            emb (Tensor): The firework embedding (a flattened tensor).
        """
        firework = torch.tensor(firework, dtype=torch.long, device=self.device, requires_grad=False)
        firework_emb = nn.functional.one_hot(firework, self.num_ranks + 1)
        return firework_emb[:, :-1].flatten().float()

    def dim(self):
        return self.num_ranks * self.num_colors


class LastMovesEncoder(nn.Module):
    def __init__(self, num_players: int=2, hand_size: int=5, num_colors: int=4, num_ranks: int=5, gamma_history: float=0.9, device: str='cuda', **_):
        """
        Args:
            num_players (int): The number of players
            hand_size (int): The number of cards in hand
            num_colors (int): The number of colors
            num_ranks (int): The number of ranks
            gamma_history (float): The coefficient of exponential average
            device (str): `cuda` or `cpu`
        """
        super().__init__()
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.num_moves = count_total_moves(self.num_players, self.num_colors, self.num_ranks, self.hand_size)
        self.device = device
        self.emb_dim_history = num_players * self.num_moves
        self.gamma = gamma_history
    
    def forward(self, last_moves: List[HanabiHistoryItem], cur_player_offset: int):
        """Encode the history actions.
        Args:
            last_moves (List[HanabiHistoryItem]): The history actions.
            cur_player_offset (int): The index of the given player; the player IDs of the history actions are relative to the current player.
        Returns:
            emb (Tensor): The history action embedding (a flattened tensor).
        """
        if len(last_moves) == 0:
            return torch.zeros(self.emb_dim_history, dtype=torch.float32, device=self.device)
        move_ids = []
        for history in last_moves:
            move = history.move()
            if move.type() == HanabiMoveType.DEAL:
                continue
            move_id = move2id(move, self.num_players, self.num_colors, self.num_ranks, self.hand_size)
            player = history.player()
            if player >= cur_player_offset:
                player -= cur_player_offset
            else:
                player = self.num_players + player - cur_player_offset
            move_ids.append(player * self.num_moves + move_id)
        move_ids = torch.tensor(move_ids, dtype=torch.long, device=self.device, requires_grad=False)
        move_ids = nn.functional.one_hot(move_ids, num_classes=self.emb_dim_history).float()
        factor = self.gamma ** torch.arange(move_ids.shape[0], device=self.device)
        return torch.sum(factor[:, None] * move_ids, dim=0)

    def dim(self):
        return self.emb_dim_history


class TokenEncoder(nn.Module):
    def __init__(self, max_tokens: int=8, device: str='cuda'):
        """
        Args:
            max_tokens (int): the maximum number of tokens
            device (str): `cuda` or `cpu`
        """
        super().__init__()
        self.max_tokens = max_tokens
        self.device = device
    
    def forward(self, cur_token: int):
        """Encode the number of the token.
        Args:
            cur_token (int): The current number of the tokens.
        Returns:
            emb (Tensor): The embedding (a flattened tensor).
        """
        emb = torch.zeros(self.max_tokens, dtype=torch.float32, device=self.device)
        if cur_token > 0:
            emb[cur_token - 1] = 1
        return emb

    def dim(self):
        return self.max_tokens
