import torch
from hanabi_learning_environment.pyhanabi import (
    HanabiGame,
    HanabiState,
    CHANCE_PLAYER_ID
)
from .PPO import PPOAgent
from .reward import compute_reward
from .encoders import (
    CardKnowledgeEncoder,
    DiscardPileEncoder,
    FireworkEncoder,
    LastMovesEncoder,
    TokenEncoder
)


class HanabiPPOAgentWrapper:
    def __init__(self, ppo_agent: PPOAgent, num_players: int, num_colors: int, num_ranks: int, hand_size: int, max_information_token: int, dim_discard: int, dim_move: int, device: str):
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
            device (str):
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
    
    def select_action(self, state: HanabiState):
        observation = state.observation(state.cur_player())
        card_knowledge_emb = self.card_knowledge_encoder.forward(observation)
        discard_pile_emb = self.discard_pile_encoder.forward(observation.discard_pile())
        firework_emb = self.firework_encoder.forward(observation.fireworks())
        last_moves_emb = self.last_moves_encoder.forward(observation.last_moves(), observation.cur_player_offset())
        info_token_emb = self.info_token_encoder.forward(observation.information_tokens())
        state_emb = torch.concat((card_knowledge_emb, discard_pile_emb, firework_emb, last_moves_emb, info_token_emb), dim=0)
        return self.ppo_agent.select_action(state_emb)


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
        episode_time_steps = 0
        episode_total_reward = 0
        while max_episode_len == -1 or episode_time_steps < max_episode_len:
            if state.cur_player() == CHANCE_PLAYER_ID:
                state.deal_random_card()
                continue
            # Take an action
            action = hanabi_agent.select_action(state)
            state.apply_move(action)
            reward = compute_reward()
            done = state.is_terminal()
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
