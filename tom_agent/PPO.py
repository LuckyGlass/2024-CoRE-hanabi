import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiObservation,
    HanabiMove
)
from torch import nn
from typing import Tuple


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCriticModule(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic = critic
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        return action_logprobs, state_values, dist_entropy

    @torch.no_grad()
    def act(self, state: torch.Tensor):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action, action_logprob, state_val


class PPOAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module, discount_factor: float, clip_epsilon: float, learning_rate_actor: float, learning_rate_critic: float, device: str, num_epochs: int):
        """
        Args:
            actor (Module): the instance of the Actor module.
            critic (Module): the instance of the Critic module.
            discount_factor (float):
            clip_epsilon (float):
            device (str):
            learning_rate_actor (float): the learning rate to train the actor module.
            learning_rate_critic (float): the learning rate to train the critic module.
            num_epochs (int): the number of epochs to train the policy model.
        """
        self.buffer = RolloutBuffer()
        self.policy = ActorCriticModule(actor, critic)
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': learning_rate_actor},
            {'params': self.policy.critic.parameters(), 'lr': learning_rate_critic}
        ])
        self.policy_old = ActorCriticModule(actor, critic)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    @torch.no_grad()
    def select_action(self, state: torch.Tensor):
        state = state.to(self.device)
        action, action_logprob, state_val = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()
    
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
        
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        advantages = rewards.detach() - old_state_values.detach()
        mse_loss_fn = nn.MSELoss()
        
        # Optimize
        for _ in range(self.num_epochs):
            # Compute new value functions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
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

