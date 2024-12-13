import torch
from torch import nn
from typing import Tuple


class ActorModule(nn.Module):
    def __init__(self, dim_state: int, dim_belief: int, dim_action: int, hidden_dim: int, num_intention: int, device: str):
        """
        Args:
            dim_state (int): the dimension of the state (input Tensor).
            dim_belief (int): the dimension of the belief embedding.
            dim_action (int): the number of the categories of actions (output).
            hidden_dim (int): the dimension of the key-value of the attention module.
            num_intention (int): the number of types of intention.
            device (str):
        """
        self.input_trans = nn.Sequential(
            nn.Linear(dim_state + dim_belief, hidden_dim * 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim, device=device)
        )
        self.intention_mapping = nn.Linear(hidden_dim, num_intention, bias=False, device=device)
        self.intention_values = nn.Parameter(torch.randn(num_intention, hidden_dim, dtype=torch.float32, device=device, requires_grad=True))
        self.output_trans = nn.Linear(hidden_dim, dim_action)
    
    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_emb = torch.concat((state, belief), dim=1)
        intention_keys = self.input_trans(input_emb)
        logits = self.intention_mapping(intention_keys)
        dist = nn.functional.softmax(logits, dim=1)
        intention_output = dist @ self.intention_values
        return self.output_trans(intention_output), dist


class CriticModule(nn.Module):
    def __init__(self, dim_state: int, hidden_dim: int, device: str):
        """
        Args:
            dim_state (int): the dimension of the state (input Tensor).
            hidden_dim (int): it determines the width of the module.
            device (str):
        """
        self.critic = nn.Sequential(
            nn.Linear(dim_state, hidden_dim * 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1, device=device),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).flatten()


class BeliefUpdateModule(nn.Module):
    def __init__(self, dim_belief: int, dim_state: int, num_move: int, hidden_dim: int, device: str):
        """
        Args:
            dim_belief (int): the dimension of the belief embedding.
            dim_state (int): the dimension of the states.
            num_move (int): the number of the types of movements.
            hiddim_dim (int): the input dimension of the GRU cell.
            device (str):
        """
        self.input_trans = nn.Sequential(
            nn.Linear(num_move + 2 * dim_state, 4 * hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 4 * hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim, device=device)
        )
        self.update = nn.GRUCell(hidden_dim, dim_belief, device=device)
        self.num_move = num_move
    
    def forward(self, belief: torch.Tensor, origin_state: torch.Tensor, result_state: torch.Tensor, action: torch.LongTensor) -> torch.Tensor:
        action_emb = nn.functional.one_hot(action, num_classes=self.num_move)
        input_emb = torch.concat((action_emb, origin_state, result_state), dim=1)
        return self.update.forward(input_emb, belief)


class ToMModule(nn.Module):
    def __init__(self, dim_state: int, num_move: int, num_intention: int, dim_belief: int, hidden_dim: int, device: str):
        """
        Args:
            dim_state (int): the dimension of the states.
            num_move (int): the number of the types of movements.
            num_intention (int): the number of types of intention.
            dim_belief (int): the dimension of the belief embedding.
            hidden_dim (int): it determines the width of the module.
            device (str):
        """
        super().__init__()
        self.tom = nn.Sequential(
            nn.Linear(dim_state * 2 + dim_belief + num_move, hidden_dim * 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, num_intention + dim_belief, device=device)
        )
        self.num_intention = num_intention
        
    def forward(self, initial_state: torch.Tensor, result_state: torch.Tensor, belief: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_emb = torch.concat((initial_state, result_state, belief, action), dim=1)
        output = self.tom.forward(input_emb)
        intention, pred_belief = output[:, :self.num_intention], output[:, self.num_intention:]
        intention = nn.functional.softmax(intention, dim=1)
        return intention, pred_belief
