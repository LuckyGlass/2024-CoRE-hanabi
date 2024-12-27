from hanabi_learning_environment.pyhanabi import (
    HanabiState,
    HanabiHistoryItem,
    HanabiMoveType,
    HanabiCard
)
from math import log2
from typing import List
from .logic_solver import observation_solver


def vanilla_reward(updated_state: HanabiState, initial_life_tokens: int):
    """Compute the reward for the last move.
    If it scores, reward += 10
    If it loses life, reward -= 100
    """
    last_move: HanabiHistoryItem = updated_state.move_history()[-1]
    total_reward = 0.0
    if last_move.scored():
        total_reward += 10
    if updated_state.life_tokens() < initial_life_tokens:
        total_reward -= 100
    return total_reward


def reward_punish_at_last(updated_state: HanabiState):
    """Compute the reward for the last move. I recommend this reward func be used with indication of current life.
    If it scores, reward += 10
    If it loses all the life, reward -= 300
    """
    last_move = updated_state.move_history()[-1]
    total_reward = 0.0
    if last_move.scored():
        total_reward += 10
    if updated_state.is_terminal():
        total_reward -= 300
    return total_reward


def reward_for_reveal(origin_state: HanabiState, updated_state: HanabiState, num_ranks: int, num_colors: int, num_players: int, hand_size: int):
    """Compute the reward for the last move.
    If it scores, reward += 10
    If it loses life, reward -= 100
    For any move, reward += reduced_entropy / (log(#color x #rank) x hand_size)
    If it reveals the card to play, for every revealed card to play, reward += 2
    """
    last_move: HanabiHistoryItem = updated_state.move_history()[-1]
    total_reward = 0.0
    if last_move.scored():
        total_reward += 10
    if updated_state.life_tokens() < origin_state.life_tokens():
        total_reward -= 100
    normalization = log2(num_colors * num_ranks) * hand_size
    for player in range(num_players):
        origin_observation = observation_solver(origin_state.observation(player), num_ranks, num_colors, num_players, self_eval=True)[0]
        updated_observation = observation_solver(updated_state.observation(player), num_ranks, num_colors, num_players, self_eval=True)[0]
        origin_entropy = sum(map(lambda x: x.entropy(), origin_observation))
        updated_entropy = sum(map(lambda x: x.entropy(), updated_observation))
        total_reward += (origin_entropy - updated_entropy) / normalization
    move = last_move.move()
    if move.type() in [HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK]:
        target = (move.target_offset() + last_move.player() + num_players) % num_players
        target_cards: List[HanabiCard] = updated_state.player_hands()[target]
        firework = updated_state.fireworks()
        for i in last_move.card_info_newly_revealed():
            card: HanabiCard = target_cards[i]
            if card.rank() == firework[card.color()]:
                total_reward += 2
    return total_reward


def simplest_reward(updated_state: HanabiState):
    if updated_state.move_history()[-1].scored():
        return 1
    return 0
