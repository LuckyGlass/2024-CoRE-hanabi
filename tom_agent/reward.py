from hanabi_learning_environment.pyhanabi import (
    HanabiState,
    HanabiHistoryItem
)


def compute_reward(updated_state: HanabiState, initial_life_tokens: int):
    last_move: HanabiHistoryItem = updated_state.move_history()[-1]
    total_reward = 0.0
    if last_move.scored():
        total_reward += 10
    if updated_state.life_tokens() < initial_life_tokens:
        total_reward -= 100
    return total_reward
