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
    