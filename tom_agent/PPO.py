import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiObservation,
    HanabiMove
)
from torch import nn
from typing import Tuple, Dict, Union, List
from .models import ActorModule, CriticModule
from .utils import move2id


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
    def __init__(self, num_players: int, num_colors: int, num_ranks: int, hand_size: int, **kwargs):
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
    
    def evaluate(self, states: torch.Tensor, believes: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs, _ = self.actor.forward(states, believes)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic.forward(states)
        return action_logprobs, state_values, dist_entropy

    @torch.no_grad()
    def act(self, state: torch.Tensor, belief: torch.Tensor, valid_moves: List[HanabiMove]) -> Tuple[int, float, float, torch.Tensor]:
        state = state.unsqueeze(0)
        belief = belief.unsqueeze(0)
        action_probs, intention_probs = self.actor.forward(state, belief)
        action_probs = action_probs.flatten()
        valid_mask = torch.zeros_like(action_probs, dtype=torch.bool)
        valid_move_ids = [move2id(m, self.num_players, self.num_colors, self.num_ranks, self.hand_size) for m in valid_moves]
        valid_mask[valid_move_ids] = 1
        action_probs = torch.where(valid_mask, action_probs, 0) / torch.sum(action_probs[valid_move_ids])
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic.forward(state)
        return action.item(), action_logprob.item(), state_val.item(), intention_probs


class PPOAgent:
    def __init__(self, discount_factor: float, clip_epsilon: float, num_training_epochs: int, learning_rate_actor: float, learning_rate_critic: float, **kwargs):
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
        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': learning_rate_actor},
            {'params': self.policy.critic.parameters(), 'lr': learning_rate_critic}
        ])
        self.policy_old = ActorCriticModule(**kwargs)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.num_players = kwargs['num_players']
        self.num_colors = kwargs['num_colors']
        self.num_ranks = kwargs['num_ranks']
        self.hand_size = kwargs['hand_size']
    
    @torch.no_grad()
    def select_action(self, state: torch.Tensor, belief: torch.Tensor, valid_moves: List[HanabiMove]) -> Tuple[HanabiMove, torch.Tensor]:
        state = state.to(self.device)
        action_id, action_logprob, state_val, intention_prob = self.policy_old.act(state, belief, valid_moves)
        self.buffer.states.append(state.clone())
        self.buffer.believes.append(belief.clone())
        self.buffer.actions.append(action_id)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        action = [m for m in valid_moves if move2id(m, self.num_players, self.num_colors, self.num_ranks, self.hand_size) == action_id]
        assert len(action) == 1
        return action[0], intention_prob
    
    def update(self) -> float:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.discount_factor * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
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

