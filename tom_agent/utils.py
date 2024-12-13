from hanabi_learning_environment.pyhanabi import HanabiMove, HanabiMoveType


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
        return 2 * hand_size + move.color()
    elif move.type() == HanabiMoveType.REVEAL_RANK:
        return 2 * hand_size + num_colors + move.rank()
    else:
        return -1
