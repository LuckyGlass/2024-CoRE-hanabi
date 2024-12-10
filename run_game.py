from hanabi_learning_environment import pyhanabi
from tom_agent.logic_solver import observation_solver
from tom_agent.utils import ID2COLOR


def run_game(players: int=2, colors: int=4, rank: int=5, hand_size: int=4, max_information_tokens: int=8, seed: int=-1, random_start_player: bool=False):
    """
    Play a game, selecting random actions.
    
    Args:
        players (Optional, int): The number of players (2~5).
        colors (Optional, int): The number of colors (1~5).
        rank (Optional, int): The max rank of the cards (1~5).
        hand_size (Optional, int): The maximum number of cards in player hand (>=1).
        max_information_tokens (Optional, int): The maximum and initial number of information tokens (>=1).
        seed (Optional, int): The random seed; default to -1 which means to use system random device to get seed.
        random_start_player (Optional, bool): Whether to start with a random player; otherwise start with player 0.
    """
    game_config = {
        'players': players,
        'colors': colors,
        'rank': rank,
        'hand_size': hand_size,
        'max_information_tokens': max_information_tokens,
        'seed': seed,
        'random_start_player': random_start_player
    }
    game = pyhanabi.HanabiGame(game_config)
    print(game.parameter_string(), end="")
    
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        import time
        st_time = time.time()
        observation = state.observation(state.cur_player())
        detailed_knowledge = observation_solver(observation, rank, colors, players)
        print("Timecost =", time.time() - st_time)
        print('-' * 20)
        print(f"Fireworks:", ' '.join(f'{ID2COLOR[i]}{c}' for i, c in enumerate(observation.fireworks())))
        print(f"Life: {observation.life_tokens()}    Hint: {observation.information_tokens()}")
        print('-' * 20)
        observed_hands = observation.observed_hands()
        for a, b in zip(observed_hands, detailed_knowledge):
            print(a)
            for p in b:
                print(p)
            print('-' * 20)
        legal_moves = state.legal_moves()
        for i, m in enumerate(legal_moves):
            print(i, m)
        move = int(input())
        state.apply_move(legal_moves[move])


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    run_game({"players": 2, 'colors': 4, 'ranks': 5, 'hand_size': 5, "random_start_player": True})
