from dataclasses import dataclass, field
from .utils import count_total_moves
from typing import Optional


@dataclass
class ModelArguments:
    belief_only: bool = field(default=False, metadata={'help': "Whether to take actions only based on beliefs."})
    device: str = field(default='cpu')
    emb_dim_belief: int = field(default=10, metadata={'help': "The dimension of the embeddings of believes."})
    emb_dim_private_belief: int = field(default=0, metadata={'help': "The dimension of private belief in belief embedding."})
    gamma_history: float = field(default=0.9, metadata={'help': "The hyperparameter of the exponential average in LastMovesEncoder."})
    num_intention: int = field(default=2, metadata={'help': "The number of the types of intentions."})
    hidden_dim_actor: int = field(default=10, metadata={'help': "It decides the width of the Actor module."})
    hidden_dim_critic: int = field(default=10, metadata={'help': "It decides the width of the Critic module."})
    hidden_dim_shared: int = field(default=10, metadata={'help': "The output dimension of the shared transformation."})
    hidden_dim_tom: int = field(default=10, metadata={'help': "It decides the width of the ToM module."})
    hidden_dim_update: int = field(default=10, metadata={'help': "It decides the width of the belief-update module."})


@dataclass
class GameArguments:
    num_colors: int
    num_ranks: int
    num_players: int
    hand_size: int
    max_information_token: int
    num_moves: int = field(default=0, metadata={'help': "The number of types of actions."})
    
    def __post_init__(self):
        self.num_moves = count_total_moves(self.num_players, self.num_colors, self.num_ranks, self.hand_size)


@dataclass
class TrainingArguments:
    discount_factor: float
    clip_epsilon: float
    learning_rate_actor: float = field(metadata={'help': "The learning rate to train the Actor module."})
    learning_rate_critic: float = field(metadata={'help': "The learning rate to train the Critic module."})
    learning_rate_shared: float = field(metadata={'help': "The learning rate to train the shared transformation."})
    learning_rate_update: float = field(metadata={'help': "The learning rate to train the BeliefUpdateModule."})
    learning_rate_tom: float = field(metadata={'help': "The learning rate to train the ToMModule."})
    num_training_epochs: int = field(metadata={'help': "The number of epochs to train the policy model per updating step."})
    max_training_timesteps: int = field(metadata={'help': "The maximum number of actions throughout the training process."})
    max_episode_length: int = field(metadata={'help': "The maximum length of an episode."})
    update_interval: int = field(metadata={'help': "The interval (timesteps) between two updating steps."})
    saving_interval: int = field(metadata={'help': "The interval (updating steps) between two checkpoints."})
    saving_dir: str = field(metadata={'help': "The dir to save the checkpoints."})
    run_name: str = field(metadata={'help': "The run name reported to W&B."})
    reward_type: str = field(default='vanilla', metadata={'help': "The type of the reward function (valid values = `vanilla`, `punish_at_last`, `reward_for_reveal`)."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={'help': "The path of the checkpoint."})
    alpha_tom_loss: float = field(default=1.0, metadata={'help': "The factor multiplied to the ToM loss."})
    test_interval: Optional[int] = field(default=None, metadata={'help': "The interval (updating steps) between two testings."})
    num_parallel_games: int = field(default=1, metadata={'help': "The number of parallel games."})
    deprecate_min: Optional[int] = field(default=None, metadata={'help': "Refer to deprecate_step."})
    deprecate_step: Optional[int] = field(default=None, metadata={'help': "The episodes whose total scores < threshold will be deprecated; threshold starts from deprecate_min and increases by 1 every deprecate_step updating steps."})
    
    def __post_init__(self):
        assert (self.deprecate_min is None) == (self.deprecate_step is None), "deprecate_min and deprecate_step should be assigned simultaneously."
