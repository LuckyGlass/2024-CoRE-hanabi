import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiGame,
    HanabiState,
    HanabiMove,
    CHANCE_PLAYER_ID
)
from typing import Optional, List, Dict, Union, Tuple
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
from .utils import move2id


class HanabiPPOAgentWrapper:
    def __init__(self, max_information_token: int, learning_rate_encoder: float, learning_rate_update: float, **kwargs):
        """
        Args:
            clip_epsilon (float):
            device (str):
            discount_factor (float):
            emb_dim_belief (int): The dimension of the embeddings of believes.
            emb_dim_discard (int): The dimension of the RNN-embeddings of discarded cards; it also uses a hard embedding of discard piles.
            emb_dim_history (int): The dimension of the embeddings of history movements.
            hand_size (int):
            hidden_dim_actor (int): It decides the width of the Actor module.
            hidden_dim_critic (int): It decides the width of the Critic module.
            hidden_dim_tom (int): It decides the width of the ToM module.
            hidden_dim_update (int): It decides the width of the belief-update module.
            learning_rate_actor (float): The learning rate to train the Actor module.
            learning_rate_critic (float): The learning rate to train the Critic module.
            learning_rate_encoder (float): The learning rate to train the DiscardPileEncoder and the LastMovesEncoder.
            learning_rate_update (float): The learning rate to train the BeliefUpdateModule.
            max_information_token (int):
            num_colors (int):
            num_intention (int): The number of the types of intentions.
            num_moves (int): The number of types of actions.
            num_players (int):
            num_ranks (int):
            num_training_epochs (int): The number of epochs to train the policy model per updating step.
        """
        self.card_knowledge_encoder = CardKnowledgeEncoder(**kwargs)
        self.discard_pile_encoder = DiscardPileEncoder(**kwargs)
        self.firework_encoder = FireworkEncoder(**kwargs)
        self.last_moves_encoder = LastMovesEncoder(**kwargs)
        self.info_token_encoder = TokenEncoder(max_information_token, kwargs['device'])
        emb_dim_state = self.card_knowledge_encoder.dim() + self.discard_pile_encoder.dim() + self.firework_encoder.dim() + self.last_moves_encoder.dim() + self.info_token_encoder.dim()
        self.ppo_agent = PPOAgent(emb_dim_state=emb_dim_state, **kwargs)
        self.update_self_belief = BeliefUpdateModule(emb_dim_state=emb_dim_state, **kwargs)
        self.update_other_belief = BeliefUpdateModule(emb_dim_state=emb_dim_state, **kwargs)
        self.tom = ToMModule(emb_dim_state=emb_dim_state, **kwargs)
        self.num_players = kwargs['num_players']
        self.num_colors = kwargs['num_colors']
        self.num_ranks = kwargs['num_ranks']
        self.num_moves = kwargs['num_moves']
        self.hand_size = kwargs['hand_size']
        self.optimizer = torch.optim.AdamW([
            {'params': self.discard_pile_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.last_moves_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.update_self_belief.parameters(), 'lr': learning_rate_update},
            {'params': self.update_other_belief.parameters(), 'lr': learning_rate_update},
        ])
        self.optimizer.zero_grad()
        self.device = kwargs['device']
    
    def encode_state(self, state: HanabiState, cur_player: Optional[int]=None):
        if cur_player is not None:
            observation = state.observation(cur_player)
        else:
            observation = state.observation(state.cur_player())
        card_knowledge_emb = self.card_knowledge_encoder.forward(observation)
        discard_pile_emb = self.discard_pile_encoder.forward(observation.discard_pile())
        firework_emb = self.firework_encoder.forward(observation.fireworks())
        last_moves_emb = self.last_moves_encoder.forward(observation.last_moves(), observation.cur_player_offset())
        info_token_emb = self.info_token_encoder.forward(observation.information_tokens())
        return torch.concat((card_knowledge_emb, discard_pile_emb, firework_emb, last_moves_emb, info_token_emb), dim=0)
    
    def select_action(self, state: HanabiState, belief: torch.Tensor) -> Tuple[HanabiMove, torch.Tensor]:
        state_emb = self.encode_state(state)
        valid_moves = state.observation(state.cur_player()).legal_moves()
        return self.ppo_agent.select_action(state_emb, belief, valid_moves)
    
    def update(self):
        self.ppo_agent.update()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def update_believes(self, believes: List[torch.Tensor], origin_state: torch.Tensor, result_state: torch.Tensor, action: HanabiMove) -> torch.Tensor:
        """Update the belief embeddings.
        Args:
            believes (List[Tensor]): the belief embeddings, indexed as +0, +1, ..., +(N-1).
            origin_state (Tensor): the embedding of the state before the action.
            result_state (Tensor): the embedding of the state after the action.
            action (HanabiMove): the action.
        """
        batch_size = believes.shape[0]
        origin_state = origin_state.repeat((batch_size, 1))
        result_state = result_state.repeat((batch_size, 1))
        action_id = move2id(action, self.num_players, self.num_colors, self.num_ranks, self.hand_size)
        action_id = torch.tensor([action_id] * batch_size, dtype=torch.long, device=self.device)
        return torch.concat([self.update_self_belief.forward(believes[:1], origin_state[:1], result_state[:1], action_id[:1]), self.update_self_belief.forward(believes[1:], origin_state[1:], result_state[1:], action_id[1:])], dim=0)
    
    def tom_supervise(self, initial_state: torch.Tensor, result_state: torch.Tensor, action: HanabiMove, gt_intention: torch.Tensor, believes: torch.Tensor):
        """
        Args:
            initial_state (Tensor): the embedding of the state before the action.
            result_state (Tensor): the embedding of the state after the action.
            action (HanabiMove): the action.
            gt_intention (Tensor): the ground-truth of the intention distribution.
            believes (Tensor): the embeddings of the believes, indexed +1, +2, ..., +(N-1), +0
        """
        gt_belief = believes[-1].detach()
        others_belief = believes[:-1].detach()
        batch_size = others_belief.shape[0]
        gt_belief = gt_belief.repeat((batch_size, 1))
        initial_state = initial_state.repeat((batch_size, 1)).detach()
        result_state = result_state.repeat((batch_size, 1)).detach()
        gt_intention = torch.distributions.Categorical(gt_intention.detach().repeat(batch_size, 1))
        action_id = move2id(action, self.num_players, self.num_colors, self.num_ranks, self.hand_size)
        action_id = torch.tensor([action_id] * batch_size, dtype=torch.long, device=self.device)
        action_id = torch.nn.functional.one_hot(action_id, num_classes=self.num_moves).float()
        pred_intention, pred_belief = self.tom.forward(initial_state, result_state, others_belief, action_id)
        loss_belief_fn = torch.nn.MSELoss(reduction='sum')
        loss_belief = loss_belief_fn(pred_belief, gt_belief) / batch_size
        pred_intention = torch.distributions.Categorical(pred_intention)
        loss_intention = 0.5 * (torch.distributions.kl_divergence(gt_intention, pred_intention) + torch.distributions.kl_divergence(pred_intention, gt_intention))
        loss_intention = loss_intention.mean()
        loss_intention.backward(retain_graph=True)
        loss_belief.backward()


def train(game: HanabiGame, clip_epsilon: float, device: str, discount_factor: float, emb_dim_belief: int, emb_dim_discard: int, emb_dim_history: int, hand_size: int, hidden_dim_actor: int, hidden_dim_critic: int, hidden_dim_tom: int, hidden_dim_update: int, learning_rate_actor: float, learning_rate_critic: float, learning_rate_encoder: float, learning_rate_update: float, max_episode_length: int, max_information_token: int, max_training_timesteps: int, num_colors: int, num_intention: int, num_moves: int, num_players: int, num_ranks: int, num_training_epochs: int, update_interval: int, **_):
    """
    Args:
        clip_epsilon (float):
        device (str):
        discount_factor (float):
        emb_dim_belief (int): The dimension of the embeddings of believes.
        emb_dim_discard (int): The dimension of the RNN-embeddings of discarded cards; it also uses a hard embedding of discard piles.
        emb_dim_history (int): The dimension of the embeddings of history movements.
        hand_size (int):
        hidden_dim_actor (int): It decides the width of the Actor module.
        hidden_dim_critic (int): It decides the width of the Critic module.
        hidden_dim_tom (int): It decides the width of the ToM module.
        hidden_dim_update (int): It decides the width of the belief-update module.
        learning_rate_actor (float): The learning rate to train the Actor module.
        learning_rate_critic (float): The learning rate to train the Critic module.
        learning_rate_encoder (float): The learning rate to train the DiscardPileEncoder and the LastMovesEncoder.
        learning_rate_update (float): The learning rate to train the BeliefUpdateModule.
        max_episode_length (int): The maximum length of an episode.
        max_information_token (int):
        max_training_timesteps (int): The maximum number of actions throughout the training process.
        num_colors (int):
        num_intention (int): The number of the types of intentions.
        num_moves (int): The number of types of actions.
        num_players (int):
        num_ranks (int):
        num_training_epochs (int): The number of epochs to train the policy model per updating step.
        update_interval (int): The interval (timesteps) between two updating steps.
    """
    global_time_steps = 0
    hanabi_agent = HanabiPPOAgentWrapper(
        clip_epsilon=clip_epsilon,
        device=device,
        discount_factor=discount_factor,
        emb_dim_belief=emb_dim_belief,
        emb_dim_discard=emb_dim_discard,
        emb_dim_history=emb_dim_history,
        hand_size=hand_size,
        hidden_dim_actor=hidden_dim_actor,
        hidden_dim_critic=hidden_dim_critic,
        hidden_dim_tom=hidden_dim_tom,
        hidden_dim_update=hidden_dim_update,
        learning_rate_actor=learning_rate_actor,
        learning_rate_critic=learning_rate_critic,
        learning_rate_encoder=learning_rate_encoder,
        learning_rate_update=learning_rate_update,
        max_information_token=max_information_token,
        num_colors=num_colors,
        num_intention=num_intention,
        num_moves=num_moves,
        num_players=num_players,
        num_ranks=num_ranks,
        num_training_epochs=num_training_epochs,
    )
    count_episode = 0
    while global_time_steps <= max_training_timesteps:
        state = game.new_initial_state()
        believes: torch.Tensor = torch.zeros((game.num_players(), emb_dim_belief), dtype=torch.float32, device=device, requires_grad=False)
        episode_time_steps = 0
        episode_total_reward = 0
        count_episode += 1
        while max_episode_length == -1 or episode_time_steps < max_episode_length:
            if state.cur_player() == CHANCE_PLAYER_ID:
                state.deal_random_card()
                continue
            cur_player = state.cur_player()
            # Cache initial state
            initial_state_emb = hanabi_agent.encode_state(state, cur_player)
            # Take an action
            action, intention_probs = hanabi_agent.select_action(state, believes[0])
            print(f"Episode {count_episode}, step {episode_time_steps}, player {cur_player}: {action}")
            # Environment
            state.apply_move(action)
            reward = compute_reward()
            done = state.is_terminal()
            result_state_emb = hanabi_agent.encode_state(state, cur_player)
            believes = hanabi_agent.update_believes(believes, initial_state_emb, result_state_emb, action)
            believes = believes.roll(-1, 0)
            hanabi_agent.tom_supervise(initial_state_emb, result_state_emb, action, intention_probs, believes)
            # Save `reward` and `done`
            hanabi_agent.ppo_agent.buffer.rewards.append(reward)
            hanabi_agent.ppo_agent.buffer.is_terminals.append(done)
            global_time_steps += 1
            episode_time_steps += 1
            episode_total_reward += reward
            # Update PPO agent
            if global_time_steps % update_interval == 0:
                hanabi_agent.update()
            # Terminal
            if done:
                break
