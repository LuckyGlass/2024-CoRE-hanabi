import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiObservation,
    HanabiMove
)
from torch import nn
from typing import Tuple, Dict
from .models import ActorModule, CriticModule


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
    def __init__(self, dim_state: int, dim_belief: int, dim_action: int, num_intention: int, actor_hidden_dim: int, critic_hidden_dim: int, device: str):
        super().__init__()
        self.actor = ActorModule(dim_state, dim_belief, dim_action, actor_hidden_dim, num_intention, device)
        self.critic = CriticModule(dim_state, critic_hidden_dim, device)
    
    def evaluate(self, states: torch.Tensor, believes: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs, _ = self.actor.forward(states, believes)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic.forward(states)
        return action_logprobs, state_values, dist_entropy

    @torch.no_grad()
    def act(self, state: torch.Tensor, belief: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = state.unsqueeze(0)
        belief = belief.unsqueeze(0)
        action_probs, intention_probs = self.actor.forward(state, belief)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic.forward(state)
        return action, action_logprob, state_val, intention_probs


class PPOAgent:
    def __init__(self, model_config: Dict[str, int], discount_factor: float, clip_epsilon: float, learning_rate_actor: float, learning_rate_critic: float, device: str, num_epochs: int):
        """
        Args:
            model_config (Dict[str, int]): the config of the Actor-Critic module.
                - `dim_state`: the dimension of state embeddings.
                - `dim_belief`: the dimension of belief embeddings.
                - `dim_action`: the number of types of actions.
                - `num_intention`: the number of types of intentions.
                - `actor_hidden_dim`: it decides the model width of the Actor model.
                - `critic_hidden_dim`: it decides the model width of the Critic model.
            discount_factor (float):
            clip_epsilon (float):
            device (str):
            learning_rate_actor (float): the learning rate to train the actor module.
            learning_rate_critic (float): the learning rate to train the critic module.
            num_epochs (int): the number of epochs to train the policy model.
        """
        self.buffer = RolloutBuffer()
        self.policy = ActorCriticModule(**model_config)
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': learning_rate_actor},
            {'params': self.policy.critic.parameters(), 'lr': learning_rate_critic}
        ])
        self.policy_old = ActorCriticModule(**model_config)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    @torch.no_grad()
    def select_action(self, state: torch.Tensor, belief: torch.Tensor) -> Tuple[int, torch.Tensor]:
        state = state.to(self.device)
        action, action_logprob, state_val, intention_prob = self.policy_old.act(state, belief)
        self.buffer.states.append(state.detach())
        self.buffer.believes.append(belief.detach())
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item(), intention_prob
    
    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in reversed(zip(self.buffer.rewards, self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.discount_factor * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.Tensor(rewards, torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)
        old_believes = torch.stack(self.buffer.believes, dim=0).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(self.device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(self.device)
        advantages = rewards.detach() - old_state_values.detach()
        mse_loss_fn = nn.MSELoss()
        
        # Optimize
        for _ in range(self.num_epochs):
            # Compute new value functions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_believes, old_actions)
            state_values = None
            # Compute r
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Compute loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            # Clipped policy loss, value loss, and regularization
            loss = -torch.min(surr1, surr2) + 0.5 * mse_loss_fn(state_values, rewards) - 0.01 * dist_entropy
            # Update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Clone policy & clear buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
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

