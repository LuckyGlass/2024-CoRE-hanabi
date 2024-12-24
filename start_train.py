from tom_agent.train import train
from tom_agent.args import ModelArguments, GameArguments, TrainingArguments
from hanabi_learning_environment import pyhanabi
from transformers import HfArgumentParser
import os
import wandb


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
    game = pyhanabi.HanabiGame(game_config)
    wandb.login(key=os.environ['WANDB_LOG_KEY'])
    wandb.init(
        project="2024-CoRE-Hanabi",
        config=(vars(model_args) | vars(game_args) | vars(training_args))
    )
    train(game, **(vars(model_args) | vars(game_args) | vars(training_args)))
