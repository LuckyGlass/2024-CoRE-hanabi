import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiGame,
    HanabiState,
    CHANCE_PLAYER_ID
)
from typing import Optional, List
from .PPO import PPOAgent
from .reward import compute_reward
from .encoders import (
    CardKnowledgeEncoder,
    DiscardPileEncoder,
    FireworkEncoder,
    LastMovesEncoder,
    TokenEncoder
)
from .models import BeliefUpdateModule, ToMModule
from .utils import count_total_moves


class HanabiPPOAgentWrapper:
    def __init__(self, ppo_agent: PPOAgent, num_players: int, num_colors: int, num_ranks: int, hand_size: int, max_information_token: int, dim_discard: int, dim_move: int, dim_belief: int, dim_belief_update: int, dim_tom: int, num_intention: int, device: str, do_train: bool=True, lr_encoder: Optional[float]=None):
        """
        Args:
            ppo_agent (PPOAgent): an instance of `PPOAgent`.
            num_players (int):
            num_colors (int):
            num_ranks (int):
            hand_size (int):
            max_information_token (int):
            dim_discard (int): the dimension of the RNN-embedding of discard piles.
            dim_move (int): the dimension of the embedding of history movements.
            dim_belief (int): the dimension of the embedding of believes.
            dim_belief_update (int): the hidden dimension of the updater of believes.
            dim_tom (int): the hidden dimension of the ToM module.
            num_intention (int): the number of types of intention.
            device (str):
            do_train (Optional, bool): whether to train the module.
        """
        self.ppo_agent = ppo_agent
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.card_knowledge_encoder = CardKnowledgeEncoder(num_players, num_colors, num_ranks, hand_size, device)
        self.discard_pile_encoder = DiscardPileEncoder(num_colors, num_ranks, dim_discard, device)
        self.firework_encoder = FireworkEncoder(num_ranks, device)
        self.last_moves_encoder = LastMovesEncoder(num_players, hand_size, num_colors, num_ranks, device, dim_move)
        self.info_token_encoder = TokenEncoder(max_information_token, device)
        self.dim_state = self.card_knowledge_encoder.dim() + self.discard_pile_encoder.dim() + self.firework_encoder.dim() + self.last_moves_encoder.dim() + self.info_token_encoder.dim()
        self.num_move = count_total_moves(num_players, num_colors, num_ranks, hand_size)
        self.update_self_belief = BeliefUpdateModule(dim_belief, self.dim_state, self.num_move, dim_belief_update, device)
        self.update_other_belief = BeliefUpdateModule(dim_belief, self.dim_state, self.num_move, dim_belief_update, device)
        self.tom = ToMModule(self.dim_state, self.num_move, num_intention, dim_belief, dim_tom, device=device)
        if do_train:
            self.optimizer = torch.optim.AdamW([
                {'params': self.discard_pile_encoder.parameters(), 'lr': lr_encoder},
                {'params': self.last_moves_encoder.parameters(), 'lr': lr_encoder}
            ])
            self.optimizer.zero_grad()
        else:
            self.optimizer = None
        self.device = device
    
    def encode_state(self, state: HanabiState):
        observation = state.observation(state.cur_player())
        card_knowledge_emb = self.card_knowledge_encoder.forward(observation)
        discard_pile_emb = self.discard_pile_encoder.forward(observation.discard_pile())
        firework_emb = self.firework_encoder.forward(observation.fireworks())
        last_moves_emb = self.last_moves_encoder.forward(observation.last_moves(), observation.cur_player_offset())
        info_token_emb = self.info_token_encoder.forward(observation.information_tokens())
        return torch.concat((card_knowledge_emb, discard_pile_emb, firework_emb, last_moves_emb, info_token_emb), dim=0)
    
    def select_action(self, state: HanabiState, belief: torch.Tensor):
        state_emb = self.encode_state(state)
        return self.ppo_agent.select_action(state_emb, belief)
    
    def update(self):
        self.ppo_agent.update()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def update_believes(self, believes: List[torch.Tensor], origin_state: torch.Tensor, result_state: torch.Tensor, action: int) -> torch.Tensor:
        """Update the belief embeddings.
        Args:
            believes (List[Tensor]): the belief embeddings, relatively indexed.
            origin_state (Tensor): the embedding of the state before the action.
            result_state (Tensor): the embedding of the state after the action.
            action (int): the index of the action.
        """
        batch_size = believes.shape[0]
        origin_state = origin_state.repeat((batch_size, 1))
        result_state = result_state.repeat((batch_size, 1))
        action = torch.tensor([action] * batch_size, dtype=torch.long, device=self.device)
        return [self.update_self_belief.forward(believes[:1], origin_state[:1], result_state[:1], action[:1])] + [self.update_self_belief.forward(believes[1:], origin_state[1:], result_state[1:], action[1:])]
    
    def tom_supervise(self, initial_state: torch.Tensor, result_state: torch.Tensor, action: int, gt_intention: torch.Tensor, believes: torch.Tensor):
        """
        Args:
            initial_state (Tensor): the embedding of the state before the action.
            result_state (Tensor): the embedding of the state after the action.
            action (int): the index of the action.
            gt_intention (Tensor): the ground-truth of the intention distribution.
            believes (Tensor): the embeddings of the believes, indexed +1, +2, ..., +(N-1), +0
        """
        gt_belief = believes[-1]
        others_belief = believes[:-1]
        batch_size = others_belief.shape[0]
        gt_belief = gt_belief.repeat((batch_size, 1))
        initial_state = initial_state.repeat((batch_size, 1))
        result_state = result_state.repeat((batch_size, 1))
        gt_intention = torch.distributions.Categorical(gt_intention.repeat(batch_size, 1))
        action = torch.tensor([action] * batch_size, dtype=torch.long, device=self.device)
        action = torch.nn.functional.one_hot(action, num_classes=self.num_move).float()
        pred_intention, pred_belief = self.tom.forward(initial_state, result_state, others_belief, action)
        loss_belief_fn = torch.nn.MSELoss(reduction='sum')
        loss_belief = loss_belief_fn(pred_belief, gt_belief) / batch_size
        pred_intention = torch.distributions.Categorical(pred_intention)
        loss_intention = 0.5 * (torch.distributions.kl_divergence(gt_intention, pred_intention) + torch.distributions.kl_divergence(pred_intention, gt_intention))
        loss_intention = loss_intention.mean()
        loss_belief.backward()
        loss_intention.backward()


def train(game: HanabiGame, ppo_agent: PPOAgent, max_training_timesteps: int, max_episode_len: int, update_timesteps: int, dim_discard: int, dim_move: int):
    """PPO training.
    Args:
        game (HanabiGame): the game starter.
        ppo_agent (PPOAgent): the agent to train.
        max_training_timesteps (int): max number of actions taken.
        max_episode_len (int): max number of actions taken in one episode; -1 means unlimited.
        update_timesteps (int): the number of timesteps between update.
    """
    global_time_steps = 0
    hanabi_agent = HanabiPPOAgentWrapper(ppo_agent, game.num_players(), game.num_colors(), game.num_ranks(), game.hand_size(), game.max_information_tokens(), dim_discard, dim_move)
    while global_time_steps <= max_training_timesteps:
        state = game.new_initial_state()
        believes: torch.Tensor = torch.zeros((game.num_players(), None))  # TODO
        episode_time_steps = 0
        episode_total_reward = 0
        while max_episode_len == -1 or episode_time_steps < max_episode_len:
            if state.cur_player() == CHANCE_PLAYER_ID:
                state.deal_random_card()
                continue
            # Cache initial state
            initial_state_emb = hanabi_agent.encode_state(state)
            # Take an action
            action, intention_probs = hanabi_agent.select_action(state, believes[0])
            # Environment
            state.apply_move(action)
            reward = compute_reward()
            done = state.is_terminal()
            result_state_emb = hanabi_agent.encode_state(state)
            believes = hanabi_agent.update_believes(believes, initial_state_emb, result_state_emb, action)
            believes = believes.roll(-1, dim=0)
            hanabi_agent.tom_supervise(initial_state_emb, result_state_emb, action, intention_probs, believes)
            # Save `reward` and `done`
            hanabi_agent.buffer.rewards.append(reward)
            hanabi_agent.buffer.is_terminals.append(done)
            global_time_steps += 1
            episode_time_steps += 1
            episode_total_reward += reward
            # Update PPO agent
            if global_time_steps % update_timesteps == 0:
                hanabi_agent.update()
            # Terminal
            if done:
                break
