from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.pyhanabi import (
    HanabiObservation,
    HanabiCard,
    HanabiCardKnowledge
)
from typing import List, Any
from copy import deepcopy
from .utils import ID2COLOR


MAX_NUM_CARDS = [0 for _ in range(1 << 5)]
for rankset in range(1 << 5):
    if rankset & 1:
        MAX_NUM_CARDS[rankset] += 3
    if rankset & 2:
        MAX_NUM_CARDS[rankset] += 2
    if rankset & 4:
        MAX_NUM_CARDS[rankset] += 2
    if rankset & 8:
        MAX_NUM_CARDS[rankset] += 2
    if rankset & 16:
        MAX_NUM_CARDS[rankset] += 1
MAX_NUM_COLORS = [0 for _ in range(1 << 5)]
for colorset in range(1, 1 << 5):
    MAX_NUM_COLORS[colorset] = MAX_NUM_COLORS[colorset ^ (colorset & (-colorset))] + 1


def get_bit_knowledge(knowledge: HanabiCardKnowledge, num_colors: int, num_ranks: int):
    colorset = sum(1 << color for color in range(num_colors) if knowledge.color_plausible(color))
    rankset = sum(1 << rank for rank in range(num_ranks) if knowledge.rank_plausible(rank))
    return colorset, rankset


def modify_superset(num_ele: int, base_set: int, delta: Any, array: List[Any]):
    for s in range(1 << num_ele):
        if (s & base_set) == base_set:
            array[s] += delta


def passdown(num_ele: int, array: List[Any]):
    for s in range((1 << num_ele) - 1, 0, -1):
        t = s
        while t:
            p = t & (-t)
            array[s ^ p] = min(array[s ^ p], array[s])
            t ^= p


class DetailedCardKnowledge:
    """The detailed knowledge of a card.
    `self.plausible[c][r]` is a boolean value indicating whether it's possible that the card is of color `c` and rank `r`.
    """
    def __init__(self, num_colors: int, num_ranks: int):
        self.plausible = [[True for _ in range(num_ranks)] for _ in range(num_colors)]
    
    @classmethod
    def deduce(cls, direct_knowledge: HanabiCardKnowledge, remaining_cards: List[int], num_colors: int, num_ranks: int):
        instance = cls(num_colors, num_ranks)
        colorset, rankset = get_bit_knowledge(direct_knowledge, num_colors, num_ranks)
        for color in range(num_colors):
            for rank in range(num_ranks):
                if (colorset >> color & 1) and (rankset >> rank & 1):
                    instance.plausible[color][rank] = remaining_cards[(1 << (color + num_ranks)) | (1 << rank)] > 0
                else:
                    instance.plausible[color][rank] = False
        return instance
    
    def __str__(self):
        return ''.join(f'{ID2COLOR[i]}:' + ''.join(str(j + 1) for j, b in enumerate(arr) if b) + '|' for i, arr in enumerate(self.plausible))


def observation_solver(observation: HanabiObservation, max_rank: int, num_colors: int, num_players: int) -> List[List[DetailedCardKnowledge]]:
    """The logic solver for the card information.
    Args:
        observation (HanabiObservation): The observation of the current player provided by the env.
        max_rank (int): The maximum rank of the cards.
        num_colors (int): The number of colors.
        num_players (int): The number of players.
    Returns:
        detailed_card_knowledge (List[List[DetailedCardKnowledge]]): List[0] is Player +0's knowledge of his cards; List[i] (i > 0) is Player +i's knowledge of Player +i's knowledge in Player +0's view.
    """
    fireworks: List[int] = observation.fireworks()
    discard_pile: List[HanabiCard] = observation.discard_pile()
    observed_hands: List[List[HanabiCard]] = observation.observed_hands()
    card_knowledge: List[List[HanabiCardKnowledge]] = observation.card_knowledge()
    public_remaining_cards = [0 for _ in range(1 << (num_colors + max_rank))]
    for colorset in range(1 << num_colors):
        for rankset in range(1 << max_rank):
            public_remaining_cards[(colorset << max_rank) | rankset] = MAX_NUM_CARDS[rankset] * MAX_NUM_COLORS[colorset]
    for color in range(num_colors):
        for rank in range(fireworks[color]):
            modify_superset(num_colors + max_rank, (1 << (color + max_rank)) | (1 << rank), -1, public_remaining_cards)
    for card in discard_pile:
        modify_superset(num_colors + max_rank, (1 << (card.color() + max_rank)) | (1 << card.rank()), -1, public_remaining_cards)
    detailed_knowledge = []
    for playeri in range(num_players):
        private_detailed_knowledge = []
        private_remaining_cards = deepcopy(public_remaining_cards)
        for playerj in range(1, num_players):  # ignore +0
            if playeri != playerj:
                for card in observed_hands[playerj]:
                    modify_superset(num_colors + max_rank, (1 << (card.color() + max_rank)) | (1 << card.rank()), -1, private_remaining_cards)
        for knowledge in card_knowledge[playeri]:
            colorset, rankset = get_bit_knowledge(knowledge, num_colors, max_rank)
            modify_superset(num_colors + max_rank, (colorset << max_rank) | rankset, -1, private_remaining_cards)
        for knowledge in card_knowledge[playeri]:
            card_remaining_cards = deepcopy(private_remaining_cards)
            colorset, rankset = get_bit_knowledge(knowledge, num_colors, max_rank)
            modify_superset(num_colors + max_rank, (colorset << max_rank) | rankset, 1, card_remaining_cards)
            passdown(num_colors + max_rank, card_remaining_cards)
            private_detailed_knowledge.append(DetailedCardKnowledge.deduce(knowledge, card_remaining_cards, num_colors, max_rank))
        detailed_knowledge.append(private_detailed_knowledge)
    return detailed_knowledge
