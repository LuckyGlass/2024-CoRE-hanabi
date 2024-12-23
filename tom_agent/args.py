from dataclasses import dataclass, field
from .utils import count_total_moves


@dataclass
class ModelArguments:
    device: str = field(default='cpu')
    emb_dim_belief: int = field(default=10, metadata={'help': "The dimension of the embeddings of believes."})
    gamma_history: float = field(default=0.9, metadata={'help': "The hyperparameter of the exponential average in LastMovesEncoder."})
    num_intention: int = field(default=2, metadata={'help': "The number of the types of intentions."})
    hidden_dim_actor: int = field(default=10, metadata={'help': "It decides the width of the Actor module."})
    hidden_dim_critic: int = field(default=10, metadata={'help': "It decides the width of the Critic module."})
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
    learning_rate_encoder: float = field(metadata={'help': "The learning rate to train the DiscardPileEncoder and the LastMovesEncoder."})
    learning_rate_update: float = field(metadata={'help': "The learning rate to train the BeliefUpdateModule."})
    num_training_epochs: int = field(metadata={'help': "The number of epochs to train the policy model per updating step."})
    max_training_timesteps: int = field(metadata={'help': "The maximum number of actions throughout the training process."})
    max_episode_length: int = field(metadata={'help': "The maximum length of an episode."})
    update_interval: int = field(metadata={'help': "The interval (timesteps) between two updating steps."})
