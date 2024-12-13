from __future__ import print_function
from hanabi_learning_environment import pyhanabi
from tom_agent.logic_solver import observation_solver
from tom_agent.utils import ID2COLOR
from tom_agent.encoders import (
    DiscardPileEncoder,
    CardKnowledgeEncoder,
    FireworkEncoder,
    LastMovesEncoder
)


def run_game(players: int=2, colors: int=4, ranks: int=5, hand_size: int=4, max_information_tokens: int=8, seed: int=-1, random_start_player: bool=False):
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
        'ranks': ranks,
        'hand_size': hand_size,
        'max_information_tokens': max_information_tokens,
        'seed': seed,
        'random_start_player': random_start_player,
        'dddd': 0
    }
    game = pyhanabi.HanabiGame(game_config)
    print(game.parameter_string(), end="")
    
    state = game.new_initial_state()
    discard_encoder = DiscardPileEncoder(colors, ranks, 10, device='cuda')
    firework_encoder = FireworkEncoder(ranks, 'cuda')
    last_moves_encoder = LastMovesEncoder(players, hand_size, colors, ranks, 'cuda', 10)
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        import time
        st_time = time.time()
        observation = state.observation(state.cur_player())
        detailed_knowledge = observation_solver(observation, ranks, colors, players)
        print("Timecost =", time.time() - st_time)
        print('-' * 20)
        print(f"Fireworks:", ' '.join(f'{ID2COLOR[i]}{c}' for i, c in enumerate(observation.fireworks())))
        firework_emb = firework_encoder.forward(observation.fireworks())
        print(firework_emb)
        print(f"Discard:", ' '.join(f'{ID2COLOR[card.color()]}{card.rank()}' for card in observation.discard_pile()))
        discard_emb = discard_encoder.forward(observation.discard_pile())
        print(discard_emb)
        print(f"Life: {observation.life_tokens()}    Hint: {observation.information_tokens()}")
        print('-' * 20)
        print('\n'.join(map(str, observation.last_moves())))
        print(last_moves_encoder.forward(observation.last_moves(), observation.cur_player_offset()))
        print('-' * 20)
        observed_hands = observation.observed_hands()
        for a, b in zip(observed_hands, detailed_knowledge):
            print(a)
            for p in b:
                print(p)
            print('-' * 20)
        # print(card_knowledge_encoder.forward(observation))
        legal_moves = state.legal_moves()
        for i, m in enumerate(legal_moves):
            print(i, m)
        move = int(input())
        state.apply_move(legal_moves[move])


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    run_game(players=2, colors=4, ranks=2, hand_size=2, random_start_player=True)
