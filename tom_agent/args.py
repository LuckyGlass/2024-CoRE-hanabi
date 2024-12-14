from dataclasses import dataclass, field
from .utils import count_total_moves


@dataclass
class ModelArguments:
    device: str
    emb_dim_state: int = field(metadata={'help': "The dimension of the embeddings of states."})
    emb_dim_belief: int = field(metadata={'help': "The dimension of the embeddings of believes."})
    emb_dim_discard: int = field(metadata={'help': "The dimension of the RNN-embeddings of discarded cards; it also uses a hard embedding of discard piles."})
    emb_dim_history: int = field(metadata={'help': "The dimension of the embeddings of history movements."})
    num_intention: int = field(metadata={'help': "The number of the types of intentions."})
    hidden_dim_actor: int = field(metadata={'help': "It decides the width of the Actor module."})
    hidden_dim_critic: int = field(metadata={'help': "It decides the width of the Critic module."})
    hidden_dim_tom: int = field(metadata={'help': "It decides the width of the ToM module."})
    hidden_dim_update: int = field(metadata={'help': "It decides the width of the belief-update module."})


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
    num_training_epochs: int = field(metadata={'help': "The number of epochs to train the policy model per updating step."})
    max_training_timesteps: int = field(metadata={'help': "The maximum number of actions throughout the training process."})
    max_episode_length: int = field(metadata={'help': "The maximum length of an episode."})
    update_interval: int = field(metadata={'help': "The interval (timesteps) between two updating steps."})
