import numpy as np
import torch
from hanabi_learning_environment.pyhanabi import HanabiMove, HanabiMoveType, HanabiState
from typing import List, Any


ID2COLOR = "RYGWB"


def count_total_cards(num_colors: int, num_ranks: int):
    if num_ranks == 1:  # +3
        return 3 * num_colors
    elif num_ranks == 2:  # +2
        return 5 * num_colors
    elif num_ranks == 3:  # +2
        return 7 * num_colors
    elif num_ranks == 4:  # +2
        return 9 * num_colors
    else:  # +1
        return 10 * num_colors


def count_total_moves(num_players: int, num_colors: int, num_ranks: int, hand_size: int):
    return 2 * hand_size + (num_players - 1) * (num_colors + num_ranks)


def move2id(move: HanabiMove, num_players: int, num_colors: int, num_ranks: int, hand_size: int) -> int:
    if move.type() == HanabiMoveType.DISCARD:
        return move.card_index()
    elif move.type() == HanabiMoveType.PLAY:
        return hand_size + move.card_index()
    elif move.type() == HanabiMoveType.REVEAL_COLOR:
        offset = (move.target_offset() % num_players + num_players) % num_players
        assert offset > 0
        return 2 * hand_size + (offset - 1) * num_colors + move.color()
    elif move.type() == HanabiMoveType.REVEAL_RANK:
        offset = (move.target_offset() % num_players + num_players) % num_players
        assert offset > 0
        return 2 * hand_size + (num_players - 1) * num_colors + (offset - 1) * num_ranks + move.rank()
    else:
        return -1


def list_index(l: List[Any], index: List[Any]) -> List[Any]:
    return [l[i] for i in index]


def visualize_info(state: HanabiState, action: HanabiMove, beliefs: torch.Tensor, intention_probs: torch.Tensor, num_ranks: int, num_colors: int):
    def get_possible_num_and_color(card_knowledge_list):
        return_list = []
        for card_knowledges in card_knowledge_list:
            for card_knowledge in card_knowledges:
                possible_rank = []
                possible_color = []
                for i in range(num_ranks):
                    if card_knowledge.rank_plausible(i):
                        possible_rank.append(i)
                for i in range(num_colors):
                    if card_knowledge.color_plausible(i):
                        possible_color.append(i)
                return_list.append((possible_color, possible_rank))
        return return_list
    
    def handle_card_list(card_list):
        return [(card._color, card._rank) for card in card_list]
    
    card_observation_0 = state.observation(0)
    card_observation_1 = state.observation(1)

    card_konwledge_0 = card_observation_0.card_knowledge()
    card_konwledge_1 = card_observation_1.card_knowledge()

    possible_num_and_color_0 = get_possible_num_and_color(card_konwledge_0)
    possible_num_and_color_1 = get_possible_num_and_color(card_konwledge_1)

    return {
        'player_hands': [handle_card_list(hands) for hands in state.player_hands()],
        'observed_hands': [handle_card_list(hands) for hands in state.observation(0).observed_hands()],
        'observed_hands_1': [handle_card_list(hands) for hands in state.observation(1).observed_hands()],
        'possible_num_and_color_0': possible_num_and_color_0,
        'possible_num_and_color_1': possible_num_and_color_1,
        "action": action.__repr__(),
        "discard_pile": handle_card_list(state.discard_pile()),
        "deck_size": state.deck_size(),
        "fireworks": state.fireworks(),
        "life_tokens": state.life_tokens(),
        "score": state.score(),
        "belief_embedding": beliefs.detach().cpu().numpy().astype(np.float32).tolist(),
        "intention": intention_probs.cpu().numpy().astype(np.float32).tolist()
    }