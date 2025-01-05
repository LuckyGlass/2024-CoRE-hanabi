import torch
from torch import nn
from typing import Tuple


class SharedTransformation(nn.Module):
    def __init__(self, emb_dim_state: int, emb_dim_belief: int, hidden_dim_shared: int, device: str, **_):
        """The shared transformation module of the Actor and the Critic. It transforms a pair of state and belief embedding into a mixed representation.
        Args:
            emb_dim_state (int): The dimension of state embedding.
            emb_dim_belief (int): The dimension of belief embedding.
            hidden_dim_shared (int): The output dimenison.
        """
        self.trans = nn.Sequential(
            self._construct(emb_dim_belief + emb_dim_state, 4 * hidden_dim_shared, device),
            self._construct(4 * hidden_dim_shared, 4 * hidden_dim_shared, device),
            self._construct(4 * hidden_dim_shared, 4 * hidden_dim_shared, device),
            self._construct(4 * hidden_dim_shared, hidden_dim_shared, device)
        )
    
    def _construct(self, din: int, dout: int, device: str):
        return nn.Sequential(
            nn.Linear(din, dout, device=device),
            nn.GELU(),
            nn.BatchNorm1d(dout, device=device)
        )
    
    def forward(self, states: torch.Tensor, beliefs: torch.Tensor):
        """
        Args:
            states (Tensor): Batched state embedding, [shape, emb_dim_state]
            beliefs (Tensor): Batched belief embedding, [shape, emb_dim_belief]
        """
        input_emb = torch.concat([states, beliefs], dim=1)
        return self.trans(input_emb)


class ActorModule(nn.Module):
    def __init__(self, hidden_dim_shared: int, num_moves: int, hidden_dim_actor: int, num_intention: int, device: str, **_):
        """
        Args:
            dim_in (int): The input dimension.
            num_moves (int): The number of types of actions.
            hidden_dim_actor (int): It decides the width of the Actor module.
            num_intention (int): The number of the types of intentions.
            device (str):
        """
        super().__init__()
        self.input_trans = nn.Sequential(
            nn.Linear(hidden_dim_shared, 4 * hidden_dim_actor, device=device),
            nn.GELU(),
            nn.BatchNorm1d(4 * hidden_dim_actor, device=device),
            nn.Linear(4 * hidden_dim_actor, 4 * hidden_dim_actor, device=device),
            nn.GELU(),
            nn.BatchNorm1d(4 * hidden_dim_actor),
            nn.Linear(4 * hidden_dim_actor, hidden_dim_actor, device=device),
            nn.GELU()
        )
        self.intention_mapping = nn.Linear(hidden_dim_actor, num_intention, bias=False, device=device)
        self.intention_values = nn.Parameter(torch.randn(num_intention, hidden_dim_actor, dtype=torch.float32, device=device, requires_grad=True))
        self.output_trans = nn.Linear(hidden_dim_actor, num_moves, device=device)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state (torch.Tensor): The embeddings of states in the shape of [Batch, Embed].
            belief (torch.Tensor): The embeddings of believes in the shape of [Batch, Embed].
        """
        intention_keys = self.input_trans(inputs)
        logits = self.intention_mapping(intention_keys)
        dist = nn.functional.softmax(logits, dim=1)
        intention_output = dist @ self.intention_values
        action_logits = self.output_trans(intention_output)
        return nn.functional.softmax(action_logits, dim=1), dist


class CriticModule(nn.Module):
    def __init__(self, hidden_dim_shared: int, num_players: int, hidden_dim_critic: int, device: str, **_):
        """
        Args:
            hidden_dim_shared (int): The dimension of the shared transformation.
            num_players (int): The number of the players.
            hidden_dim_critic (int): It decides the width of the Critic module.
            device (str):
        """
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim_shared * num_players, 2 * hidden_dim_critic, device=device),
            nn.GELU(),
            nn.BatchNorm1d(2 * hidden_dim_critic),
            nn.Linear(2 * hidden_dim_critic, hidden_dim_critic, device=device),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim_critic),
            nn.Linear(hidden_dim_critic, 1, device=device),
        )
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states (torch.Tensor): The embeddings of states in the shape of [Batch, Embed].
        """
        return self.critic(states).flatten()


class BeliefUpdateModule(nn.Module):
    def __init__(self, emb_dim_belief: int, emb_dim_state: int, num_moves: int, hidden_dim_update: int, device: str, **_):
        """
        Args:
            emb_dim_belief (int): The dimension of the embeddings of believes.
            emb_dim_state (int): The dimension of the embeddings of states.
            num_moves (int): The number of types of actions.
            hidden_dim_update (int): It decides the width of the belief-update module.
            device (str):
        """
        super().__init__()
        self.input_trans = nn.Sequential(
            nn.Linear(num_moves + 2 * emb_dim_state, 4 * hidden_dim_update, device=device),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim_update, 4 * hidden_dim_update, device=device),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim_update, hidden_dim_update, device=device)
        )
        self.update = nn.GRUCell(hidden_dim_update, emb_dim_belief, device=device)
        self.num_moves = num_moves
    
    def forward(self, beliefs: torch.Tensor, origin_states: torch.Tensor, result_states: torch.Tensor, actions: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            beliefs (torch.Tensor): The embeddings of beliefs in the shape of [Batch, Embed].
            origin_states (torch.Tensor): The embeddings of states before actions in the shape of [Batch, Embed].
            result_states (torch.Tensor): The embeddings of states after actions in the shape of [Batch, Embed].
            actions (torch.LongTensor): The indices of actions in the shape of [Batch].
        """
        action_emb = nn.functional.one_hot(actions, num_classes=self.num_moves)
        input_emb = torch.concat((action_emb, origin_states, result_states), dim=1)
        input_emb = self.input_trans.forward(input_emb)
        return self.update.forward(input_emb, beliefs)  # It accepts a batch.


class ToMModule(nn.Module):
    def __init__(self, emb_dim_state: int, emb_dim_belief: int, num_moves: int, num_intention: int, hidden_dim_tom: int, device: str, **_):
        """
        Args:
            emb_dim_state (int): The dimension of the embeddings of states.
            dim_belief (int): The dimension of the embeddings of believes.
            num_moves (int): The number of types of actions.
            num_intention (int): The number of the types of intentions.
            hidden_dim_tom (int): It decides the width of the ToM module.
            device (str):
        """
        super().__init__()
        self.tom = nn.Sequential(
            nn.Linear(emb_dim_state * 2 + emb_dim_belief + num_moves, hidden_dim_tom * 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim_tom * 4, num_intention + emb_dim_belief, device=device)
        )
        self.num_intention = num_intention
        
    def forward(self, initial_states: torch.Tensor, result_states: torch.Tensor, beliefs: torch.Tensor, action_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            initial_states (torch.Tensor): The embeddings of states before actions in the shape of [Batch, Embed].
            result_states (torch.Tensor): The embeddings of states after actions in the shape of [Batch, Embed].
            beliefs (torch.Tensor): The embeddings of beliefs in the shape of [Batch, Embed].
            action_emb (torch.LongTensor): The indices of actions in the shape of [Batch, Embed].
        """
        input_emb = torch.concat((initial_states, result_states, beliefs, action_emb), dim=1)
        output = self.tom.forward(input_emb)
        intention, pred_belief = output[:, :self.num_intention], output[:, self.num_intention:]
        intention = nn.functional.softmax(intention, dim=1)
        return intention, pred_belief
