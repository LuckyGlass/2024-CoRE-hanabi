from tom_agent.train import train
from tom_agent.args import ModelArguments, GameArguments, TrainingArguments
from hanabi_learning_environment import pyhanabi
from transformers import HfArgumentParser


if __name__ == "__main__":
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    parser = HfArgumentParser((ModelArguments, GameArguments, TrainingArguments))
    model_args, game_args, training_args = parser.parse_args_into_dataclasses()
    game_config = {
        'players': game_args.num_players,
        'colors': game_args.num_colors,
        'ranks': game_args.num_ranks,
        'hand_size': game_args.hand_size,
        'max_information_tokens': game_args.max_information_token,
        'seed': -1,
        'random_start_player': False,
    }
    model_config = {
        'dim_state': 10,
        'dim_belief': 10,
        'dim_action': 10,
        'num_intention': 2,
        'actor_hidden_dim': 10,
        'critic_hidden_dim': 10,
        'device': 'cuda'
    }
    ppo_config = {
        'discount_factor': 0.9,
        'clip_epsilon': 0.1,
        'device': 'cuda',
        'learning_rate_actor': 1e-4,
        'learning_rate_critic': 1e-3,
        'num_epochs': 1
    }
    game = pyhanabi.HanabiGame(game_config)
    train(game, **(vars(model_args) | vars(game_args) | vars(training_args)))
