import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiGame,
    HanabiState,
    HanabiMove,
    CHANCE_PLAYER_ID
)
from tqdm import tqdm
from typing import Optional, List, Dict, Union, Tuple
import wandb
from .PPO import PPOAgent
from .reward import (
    vanilla_reward,
    reward_punish_at_last,
    reward_for_reveal,
    simplest_reward
)
from .encoders import (
    CardKnowledgeEncoder,
    DiscardPileEncoder,
    FireworkEncoder,
    LastMovesEncoder,
    TokenEncoder
)
from .models import BeliefUpdateModule, ToMModule
from .utils import move2id, count_total_cards, count_total_moves
import os


class HanabiPPOAgentWrapper:
    def __init__(self, max_information_token: int=8, learning_rate_encoder: float=1e-4, learning_rate_update: float=1e-4, learning_rate_tom: float=3e-4, alpha_tom_loss: float=0.1, **kwargs):
        """
        Args:
            clip_epsilon (float):
            device (str):
            discount_factor (float):
            emb_dim_belief (int): The dimension of the embeddings of believes.
            gamma_history (float): The hyperparameter of the exponential average in LastMovesEncoder.
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
        self.hand_size = kwargs['hand_size']
        self.num_moves = count_total_moves(self.num_players, self.num_colors, self.num_ranks, self.hand_size)
        self.alpha_tom_loss = alpha_tom_loss
        self.optimizer = torch.optim.AdamW([
            {'params': self.card_knowledge_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.discard_pile_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.firework_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.last_moves_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.info_token_encoder.parameters(), 'lr': learning_rate_encoder},
            {'params': self.update_self_belief.parameters(), 'lr': learning_rate_update},
            {'params': self.update_other_belief.parameters(), 'lr': learning_rate_update},
            {'params': self.tom.parameters(), 'lr': learning_rate_tom}
        ])
        self.optimizer.zero_grad()
        self.device = kwargs['device']
    
    def encode_state(self, state: HanabiState, cur_player: Optional[int]=None):
        """
        Encode a single state from the perspective of a player.
        Args:
            state (HanabiState): the state to encode.
            cur_player (int): the player to encode the state. If None, the current player is used.
        """
        if cur_player is None:
            cur_player = state.cur_player()
        observation = state.observation(cur_player)
        card_knowledge_emb = self.card_knowledge_encoder.forward(observation)
        assert card_knowledge_emb.shape[-1] == self.card_knowledge_encoder.dim()
        discard_pile_emb = self.discard_pile_encoder.forward(observation.discard_pile())
        assert discard_pile_emb.shape[-1] == self.discard_pile_encoder.dim()
        firework_emb = self.firework_encoder.forward(observation.fireworks())
        assert firework_emb.shape[-1] == self.firework_encoder.dim()
        last_moves_emb = self.last_moves_encoder.forward(observation.last_moves(), cur_player)
        assert last_moves_emb.shape[-1] == self.last_moves_encoder.dim()
        info_token_emb = self.info_token_encoder.forward(observation.information_tokens())
        assert info_token_emb.shape[-1] == self.info_token_encoder.dim()
        return torch.concat((card_knowledge_emb, discard_pile_emb, firework_emb, last_moves_emb, info_token_emb), dim=0)
    
    def select_action(self, states: List[HanabiState], beliefs: torch.Tensor) -> Tuple[List[HanabiMove], torch.Tensor]:
        """Batched action selection.
        Args:
            states (List[HanabiState]): the states to select actions.
            beliefs (Tensor): the belief embeddings of the current players.
        """
        state_emb = torch.stack([self.encode_state(state) for state in states])
        valid_moves = [state.observation(state.cur_player()).legal_moves() for state in states]
        return self.ppo_agent.select_action(state_emb, beliefs, valid_moves)
    
    def update(self):
        res = self.ppo_agent.update()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return res
    
    def update_believes(self, beliefs: torch.Tensor, origin_states: List[HanabiState], result_states: List[HanabiState], actions: List[HanabiMove]) -> torch.Tensor:
        """Update the belief embeddings.
        Args:
            beliefs (Tensor): the belief embeddings of multiple games, indexed as +0, +1, ..., +(N-1).
            origin_states (List[HanabiState]): the states before the actions.
            result_states (List[HanabiState]): the states after the actions.
            actions (List[HanabiMove]): the actions.
        """
        start_player_id = [state.cur_player() for state in origin_states]
        start_player_origin_state_emb = [self.encode_state(origin_state, i) for origin_state, i in zip(origin_states, start_player_id)]
        start_player_origin_state_emb = torch.stack(start_player_origin_state_emb)
        start_player_result_state_emb = [self.encode_state(result_state, i) for result_state, i in zip(result_states, start_player_id)]
        start_player_result_state_emb = torch.stack(start_player_result_state_emb)
        other_player_origin_state_emb = [[self.encode_state(origin_state, (i + j) % self.num_players) for j in range(1, self.num_players)] for origin_state, i in zip(origin_states, start_player_id)]
        other_player_origin_state_emb = torch.stack(sum(other_player_origin_state_emb, []))
        other_player_result_state_emb = [[self.encode_state(result_state, (i + j) % self.num_players) for j in range(1, self.num_players)] for result_state, i in zip(result_states, start_player_id)]
        other_player_result_state_emb = torch.stack(sum(other_player_result_state_emb, []))
        
        action_id = [move2id(action, self.num_players, self.num_colors, self.num_ranks, self.hand_size) for action in actions]  # TODO: Here we only support 2 players
        action_id = torch.tensor(action_id, dtype=torch.long, device=self.device)
        other_player_action_id = action_id.repeat_interleave(self.num_players - 1)
        start_player_beliefs = beliefs[:, 0, :]
        other_player_beliefs = beliefs[:, 1:, :].reshape(-1, beliefs.shape[-1])
        start_player_beliefs = self.update_self_belief.forward(start_player_beliefs, start_player_origin_state_emb, start_player_result_state_emb, action_id)
        start_player_beliefs = start_player_beliefs[:, None, :]
        other_player_beliefs = self.update_other_belief.forward(other_player_beliefs, other_player_origin_state_emb, other_player_result_state_emb, other_player_action_id)
        other_player_beliefs = other_player_beliefs.reshape(-1, self.num_players - 1, beliefs.shape[-1])
        return torch.cat((start_player_beliefs, other_player_beliefs), dim=1)
    
    def tom_supervise(self, origin_states: List[HanabiState], result_states: List[HanabiState], actions: List[HanabiMove], gt_intention: torch.Tensor, believes: torch.Tensor) -> Tuple[float, float]:
        """
        Args:
            origin_states (List[HanabiState]): the states before the actions.
            result_states (List[HanabiState]): the states after the actions.
            actions (List[HanabiMove]): the actions.
            gt_intention (Tensor): the ground-truth intentions.
            believes (Tensor): the belief embeddings of multiple games, indexed as +0, +1, ..., +(N-1).
        """
        gt_belief = believes[:, 0].detach()  # The target needn't consider whether others can predict it.
        gt_belief = gt_belief.repeat_interleave(self.num_players - 1, dim=0)
        others_belief = believes[:, 1:].reshape(-1, believes.shape[-1])
        start_player_id = [origin_state.cur_player() for origin_state in origin_states]
        other_player_origin_state_emb = [[self.encode_state(origin_state, (i + j) % self.num_players) for j in range(1, self.num_players)] for origin_state, i in zip(origin_states, start_player_id)]
        other_player_origin_state_emb = torch.stack(sum(other_player_origin_state_emb, []))
        other_player_result_state_emb = [[self.encode_state(result_state, (i + j) % self.num_players) for j in range(1, self.num_players)] for result_state, i in zip(result_states, start_player_id)]
        other_player_result_state_emb = torch.stack(sum(other_player_result_state_emb, []))
        
        gt_intention = torch.distributions.Categorical(gt_intention.detach().repeat_interleave(self.num_players - 1, dim=0))  # The target needn't consider whether others can predict it.
        action_id = [move2id(action, self.num_players, self.num_colors, self.num_ranks, self.hand_size) for action in actions]  # TODO: Here we only support 2 players
        action_id = torch.tensor(action_id, dtype=torch.long, device=self.device)
        other_player_action_id = action_id.repeat_interleave(self.num_players - 1)
        other_player_action_id = torch.nn.functional.one_hot(other_player_action_id, num_classes=self.num_moves).float()
        pred_intention, pred_belief = self.tom.forward(other_player_origin_state_emb, other_player_result_state_emb, others_belief, other_player_action_id)
        # use MSE loss
        loss_belief_fn = torch.nn.MSELoss(reduction='sum')
        loss_belief = loss_belief_fn(pred_belief, gt_belief) / (self.num_players - 1) / len(origin_states) * self.alpha_tom_loss
        # use KL divergence
        pred_intention = torch.distributions.Categorical(pred_intention)
        loss_intention = torch.distributions.kl_divergence(pred_intention, gt_intention)
        loss_intention = loss_intention.mean() * self.alpha_tom_loss
        loss_intention.backward(retain_graph=True)
        loss_belief.backward(retain_graph=True)
        return loss_intention.detach().cpu().item(), loss_belief.detach().cpu().item()

    def save(self, save_path: str):
        torch.save(dict(
            ppo_policy=self.ppo_agent.policy.state_dict(),
            card_knowledge_encoder=self.card_knowledge_encoder.state_dict(),
            discard_pile_encoder=self.discard_pile_encoder.state_dict(),
            firework_encoder=self.firework_encoder.state_dict(),
            last_moves_encoder=self.last_moves_encoder.state_dict(),
            info_token_encoder=self.info_token_encoder.state_dict(),
            update_self_belief=self.update_self_belief.state_dict(),
            update_other_belief=self.update_other_belief.state_dict(),
            tom=self.tom.state_dict(),
            optimizer=self.optimizer.state_dict(),
            ppo_optimizer=self.ppo_agent.optimizer.state_dict()
        ), save_path)
    
    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.ppo_agent.policy.load_state_dict(checkpoint['ppo_policy'])
        self.card_knowledge_encoder.load_state_dict(checkpoint['card_knowledge_encoder'])
        self.discard_pile_encoder.load_state_dict(checkpoint['discard_pile_encoder'])
        self.firework_encoder.load_state_dict(checkpoint['firework_encoder'])
        self.last_moves_encoder.load_state_dict(checkpoint['last_moves_encoder'])
        self.info_token_encoder.load_state_dict(checkpoint['info_token_encoder'])
        self.update_self_belief.load_state_dict(checkpoint['update_self_belief'])
        self.update_other_belief.load_state_dict(checkpoint['update_other_belief'])
        self.tom.load_state_dict(checkpoint['tom'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'ppo_optimizer' in checkpoint:
            self.ppo_agent.optimizer.load_state_dict(checkpoint['ppo_optimizer'])


def train(game: HanabiGame, clip_epsilon: float, device: str, discount_factor: float, alpha_tom_loss: float, emb_dim_belief: int, gamma_history: float, hand_size: int, hidden_dim_actor: int, hidden_dim_critic: int, hidden_dim_tom: int, hidden_dim_update: int, learning_rate_actor: float, learning_rate_critic: float, learning_rate_encoder: float, learning_rate_update: float, learning_rate_tom: float, max_episode_length: int, max_information_token: int, max_training_timesteps: int, num_colors: int, num_intention: int, num_moves: int, num_players: int, num_ranks: int, num_training_epochs: int, update_interval: int, saving_interval: int, saving_dir: str, reward_type: str, resume_from_checkpoint: Optional[str], num_parallel_games: int, **_):
    """
    Args:
        clip_epsilon (float):
        device (str):
        discount_factor (float):
        alpha_tom_loss (float): The factor multiplied to the ToM loss.
        emb_dim_belief (int): The dimension of the embeddings of believes.
        gamma_history (float): The hyperparameter of the exponential average in LastMovesEncoder.
        hand_size (int):
        hidden_dim_actor (int): It decides the width of the Actor module.
        hidden_dim_critic (int): It decides the width of the Critic module.
        hidden_dim_tom (int): It decides the width of the ToM module.
        hidden_dim_update (int): It decides the width of the belief-update module.
        learning_rate_actor (float): The learning rate to train the Actor module.
        learning_rate_critic (float): The learning rate to train the Critic module.
        learning_rate_encoder (float): The learning rate to train the DiscardPileEncoder and the LastMovesEncoder.
        learning_rate_update (float): The learning rate to train the BeliefUpdateModule.
        learning_rate_tom (float): The learning rate to train the ToMModule.
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
        saving_interval (int): The interval (updating steps) between two checkpoints.
        saving_dir (str): The dir to save the checkpoints.
        reward_type (str): The type of the reward function (valid values = `vanilla`, `punish_at_last`, `reward_for_reveal`).
        resume_from_checkpoint(str | None): the path of the checkpoint.
        num_parallel_games (int): The number of parallel games.
    """
    print('-' * 10, "Game settings", '-' * 10)
    print("#Players", num_players)
    print("#Colors", num_colors)
    print("#Ranks", num_ranks)
    print("Hand size", hand_size)
    print("#Info tokens", max_information_token)
    print("#Moves", num_moves)
    print("#Cards", count_total_cards(num_colors, num_ranks))
    print('-' * 35)
    os.makedirs(saving_dir, exist_ok=True)
    global_time_steps = 0
    hanabi_agent = HanabiPPOAgentWrapper(
        clip_epsilon=clip_epsilon,
        device=device,
        discount_factor=discount_factor,
        emb_dim_belief=emb_dim_belief,
        gamma_history=gamma_history,
        hand_size=hand_size,
        hidden_dim_actor=hidden_dim_actor,
        hidden_dim_critic=hidden_dim_critic,
        hidden_dim_tom=hidden_dim_tom,
        hidden_dim_update=hidden_dim_update,
        learning_rate_actor=learning_rate_actor,
        learning_rate_critic=learning_rate_critic,
        learning_rate_encoder=learning_rate_encoder,
        learning_rate_update=learning_rate_update,
        learning_rate_tom=learning_rate_tom,
        max_information_token=max_information_token,
        num_colors=num_colors,
        num_intention=num_intention,
        num_moves=num_moves,
        num_players=num_players,
        num_ranks=num_ranks,
        num_training_epochs=num_training_epochs,
        alpha_tom_loss=alpha_tom_loss,
    )
    if resume_from_checkpoint is not None:
        hanabi_agent.load(resume_from_checkpoint)
    count_episode = 0
    cache_loss_intention, cache_loss_belief = [], []
    count_update = 0

    states = [game.new_initial_state() for _ in range(num_parallel_games)]
    # TODO: better initialization
    beliefs: torch.Tensor = torch.zeros((num_parallel_games, game.num_players(), emb_dim_belief), dtype=torch.float32, device=device, requires_grad=False)
    episode_time_steps = [0 for _ in range(num_parallel_games)]
    episode_total_reward = [0 for _ in range(num_parallel_games)]
    episode_total_score = [0 for _ in range(num_parallel_games)]
    update_time_steps = 0

    while global_time_steps <= max_training_timesteps:
        for i, state in enumerate(states):
            while state.cur_player() == CHANCE_PLAYER_ID:
                state.deal_random_card()
            episode_time_steps[i] += 1
        initial_states = [state.copy() for state in states]  # Cache initial states
        actions, intention_probs = hanabi_agent.select_action(states, beliefs)
        for state, action in zip(states, actions):
            state.apply_move(action)
        result_states = [state.copy() for state in states]
        for i, state in enumerate(states):
            global_time_steps += 1
            update_time_steps += 1
            if state.move_history()[-1].scored():
                episode_total_score[i] += 1
            if reward_type == 'vanilla':
                reward = vanilla_reward(result_states[i], initial_states[i].life_tokens())
            elif reward_type == 'punish_at_last':
                reward = reward_punish_at_last(result_states[i])
            elif reward_type == 'reward_for_reveal':
                reward = reward_for_reveal(initial_states[i], result_states[i], num_ranks, num_colors, num_players, hand_size)
            elif reward_type == 'simplest':
                reward = simplest_reward(result_states[i])
            else:
                raise ValueError(f"Unknown reward_type = \"{reward_type}\"")
            done = state.is_terminal() or episode_time_steps[i] == max_episode_length
            episode_total_reward[i] += reward
            hanabi_agent.ppo_agent.buffer.rewards.append(reward)
            hanabi_agent.ppo_agent.buffer.is_terminals.append(done)
            if done:
                wandb.log(dict(total_reward=episode_total_reward[i], total_score=episode_total_score[i]), step=global_time_steps)
                state[i] = game.new_initial_state()
                episode_time_steps[i] = 0
                episode_total_reward[i] = 0
                episode_total_score[i] = 0
        beliefs = hanabi_agent.update_believes(beliefs, initial_states, result_states, actions)
        loss_intention, loss_belief = hanabi_agent.tom_supervise(initial_states, result_states, actions, intention_probs, beliefs)
        cache_loss_intention.append(loss_intention)
        cache_loss_belief.append(loss_belief)
        beliefs = beliefs.roll(0, -1, 0)
        if update_time_steps >= update_interval:
            count_update += 1
            loss_reward = hanabi_agent.update()
            believes = believes.detach()
            wandb.log(dict(count_update=count_update, loss_intention=sum(cache_loss_intention)/len(cache_loss_intention), loss_belief=sum(cache_loss_belief)/len(cache_loss_belief), loss_reward=loss_reward), step=global_time_steps)
            update_time_steps = 0
            del cache_loss_belief[:], cache_loss_intention[:]
            if count_update % saving_interval == 0:
                hanabi_agent.save(os.path.join(saving_dir, f"checkpoint_{global_time_steps}.ckp"))
    hanabi_agent.save(os.path.join(saving_dir, f"checkpoint_{global_time_steps}.ckp"))
