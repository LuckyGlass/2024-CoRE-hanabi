import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiObservation,
    HanabiMove
)
from torch import nn
from typing import Tuple, Dict, Union, List
from .models import ActorModule, CriticModule
from .utils import move2id
from copy import deepcopy


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.believes = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.believes[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCriticModule(nn.Module):
    def __init__(self, num_players: int=2, num_colors: int=4, num_ranks: int=5, hand_size: int=5, **kwargs):
        """
        Args:
            emb_dim_state (int): The dimension of the embeddings of states.
            emb_dim_belief (int): The dimension of the embeddings of believes.
            num_colors (int): The number of colors.
            num_moves (int): The number of types of actions.
            num_intention (int): The number of the types of intentions.
            num_players (int): The number of players.
            num_ranks (int): The number of ranks.
            hand_size (int): The maximum number of cards in a player's hands.
            hidden_dim_actor (int): It decides the width of the Actor module.
            hidden_dim_critic (int): It decides the width of the Critic module.
            device (str):
        """
        super().__init__()
        self.actor = ActorModule(**kwargs)
        self.critic = CriticModule(**kwargs)
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
    
    def evaluate(self, states: torch.Tensor, beliefs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs, _ = self.actor.forward(states, beliefs)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic.forward(states)
        return action_logprobs, state_values, dist_entropy

    @torch.no_grad()
    def act(self, states: torch.Tensor, beliefs: torch.Tensor, valid_moves: List[List[HanabiMove]]) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            states (torch.Tensor): The embeddings of states in the shape of [Batch, Embed].
            beliefs (torch.Tensor): The embeddings of believes in the shape of [Batch, Embed].
            valid_moves (List[List[HanabiMove]]): The list of valid moves.
        """
        action_probs, intention_probs = self.actor.forward(states, beliefs)
        valid_mask = torch.zeros_like(action_probs, dtype=torch.bool)
        valid_move_rows = sum([[i] * len(v) for i, v in enumerate(valid_moves)], start=[])
        valid_move_columns = sum([[move2id(m, self.num_players, self.num_colors, self.num_ranks, self.hand_size) for m in v] for v in valid_moves], start=[])
        valid_mask[valid_move_rows, valid_move_columns] = 1
        action_probs[~valid_mask] = 0
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic.forward(states)
        return action, action_logprob, state_val, intention_probs


class PPOAgent:
    def __init__(self, discount_factor: float=0.95, clip_epsilon: float=0.2, num_training_epochs: int=4, learning_rate_actor: float=1e-4, learning_rate_critic: float=3e-4, **kwargs):
        """
        Args:
            discount_factor (float):
            clip_epsilon (float):
            num_training_epochs (int): The number of epochs to train the policy model per updating step.
            learning_rate_actor (float): The learning rate to train the Actor module.
            learning_rate_critic (float): The learning rate to train the Critic module.
            emb_dim_state (int): The dimension of the embeddings of states.
            emb_dim_belief (int): The dimension of the embeddings of believes.
            num_moves (int): The number of types of actions.
            num_intention (int): The number of the types of intentions.
            num_players (int): The number of players.
            num_ranks (int): The number of ranks.
            hand_size (int): The maximum number of cards in a player's hands.
            hidden_dim_actor (int): It decides the width of the Actor module.
            hidden_dim_critic (int): It decides the width of the Critic module.
            device (str):
        """
        self.buffer = RolloutBuffer()
        self.policy = ActorCriticModule(**kwargs)
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.device = kwargs['device']
        self.num_training_epochs = num_training_epochs
        self.optimizer = torch.optim.AdamW(self.trainable_params())
        self.policy_old = ActorCriticModule(**kwargs)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.num_players = kwargs['num_players']
        self.num_colors = kwargs['num_colors']
        self.num_ranks = kwargs['num_ranks']
        self.hand_size = kwargs['hand_size']
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
    
    def trainable_params(self):
        return [
            {'params': self.policy.actor.parameters(), 'lr': self.learning_rate_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.learning_rate_critic}
        ]
    
    @torch.no_grad()
    def select_action(self, states: torch.Tensor, beliefs: torch.Tensor, valid_moves: List[List[HanabiMove]]) -> Tuple[List[HanabiMove], torch.Tensor]:
        """
        Args:
            states (torch.Tensor): The embeddings of states in the shape of [Batch, Embed].
            beliefs (torch.Tensor): The embeddings of believes in the shape of [Batch, Embed].
            valid_moves (List[List[HanabiMove]]): The lists of valid moves.
        """
        states = states.to(self.device)
        beliefs = beliefs.to(self.device)
        action_ids, action_logprobs, state_vals, intention_probs = self.policy_old.act(states, beliefs, valid_moves)
        for state, belief, action_logprob, action_id, state_val in zip(states, beliefs, action_logprobs, action_ids, state_vals):
            self.buffer.states.append(state.clone())
            self.buffer.believes.append(belief.clone())
            self.buffer.logprobs.append(action_logprob.item())
            self.buffer.actions.append(action_id.item())
            self.buffer.state_values.append(state_val.item())
        actions = [[m for m in v if move2id(m, self.num_players, self.num_colors, self.num_ranks, self.hand_size) == action_id] for v, action_id in zip(valid_moves, action_ids)]
        actions = sum(actions, start=[])
        assert len(actions) == states.shape[0]
        return actions, intention_probs
    
    def update(self, num_parallel: int) -> float:
        rewards_to_go = deepcopy(self.buffer.rewards)
        for i, is_terminal in enumerate(self.buffer.is_terminals):
            if not is_terminal and i + num_parallel < len(self.buffer.is_terminals):
                rewards_to_go[i] += self.discount_factor * rewards_to_go[i + num_parallel]
        rewards = torch.tensor(rewards_to_go, dtype=torch.float, requires_grad=False, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.stack(self.buffer.states, dim=0).to(self.device)
        old_believes = torch.stack(self.buffer.believes, dim=0).to(self.device)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long, requires_grad=False, device=self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float, requires_grad=False, device=self.device)
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float, requires_grad=False, device=self.device)
        advantages = rewards - old_state_values
        mse_loss_fn = nn.MSELoss()
        
        # Optimize
        res = 0
        for epoch in range(self.num_training_epochs):
            # Compute new value functions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_believes, old_actions)
            # Compute r
            ratios = torch.exp(logprobs - old_logprobs)
            # Compute loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            # Clipped policy loss, value loss, and regularization
            loss = -torch.min(surr1, surr2) + 0.5 * mse_loss_fn(state_values, rewards) - 0.01 * dist_entropy
            # Update
            self.optimizer.zero_grad()
            res = loss.detach().mean().cpu().item()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
        
        # Clone policy & clear buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return res
    
    def save(self, checkpoint_path: str):
        """Save the Actor-Critic module (with `torch.save`).
        Args:
            checkpoint_path (str): the file to save the checkpoint.
        """
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path: str):
        """Load the Actor-Critic module from the torch checkpoint.
        Args:
            checkpoint_path (str): the path of the checkpoint file.
        """
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

