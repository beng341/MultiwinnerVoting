from collections import defaultdict


def calculate_borda_score(preference_orders):
    # Initialize a dictionary to store the scores
    borda_scores = defaultdict(int)

    # Number of alternatives
    num_alternatives = len(preference_orders[0])

    # Loop through each preference order
    for order in preference_orders:
        for rank, alternative in enumerate(order):
            # Borda score: num_alternatives - rank - 1
            borda_scores[alternative] += num_alternatives - rank - 1

    # Convert defaultdict to a list of tuples sorted by alternative
    sorted_scores = sorted(borda_scores.items())

    # Return just the scores, preserving the order of alternatives
    return [score for alt, score in sorted_scores]


def count_preferences_in_positions(preference_orders):
    # Initialize a dictionary where each key is a position and the value is another defaultdict counting occurrences
    position_counts = defaultdict(lambda: defaultdict(int))

    # Iterate through each preference order
    for order in preference_orders:
        # Iterate through the position and the alternative ranked in that position
        for position, alternative in enumerate(order):
            position_counts[position][alternative] += 1

    # Print the results
    for position, counts in position_counts.items():
        print(f"Position {position}:")
        for alternative, count in sorted(counts.items()):
            print(f"{alternative} was ranked in position {position} {count} times")
        print()  # For readability

if __name__ == "__main__":
    prefs = [(3, 4, 0, 2, 1),
             (1, 3, 4, 0, 2),
             (4, 1, 0, 2, 3),
             (3, 4, 0, 2, 1),
             (3, 4, 0, 2, 1),
             (1, 2, 4, 0, 3),
             (1, 4, 3, 0, 2),
             (2, 0, 4, 1, 3),
             (1, 2, 3, 4, 0),
             (1, 4, 3, 2, 0),
             (1, 3, 4, 0, 2),
             (2, 0, 4, 1, 3),
             (0, 3, 2, 1, 4),
             (2, 4, 0, 3, 1),
             (2, 4, 3, 1, 0),
             (1, 4, 2, 3, 0),
             (4, 1, 2, 3, 0),
             (2, 4, 3, 1, 0),
             (1, 4, 3, 0, 2),
             (4, 3, 2, 0, 1),
             (1, 2, 3, 4, 0),
             (2, 4, 3, 1, 0),
             (4, 3, 2, 0, 1),
             (1, 2, 4, 0, 3),
             (3, 4, 0, 2, 1),
             (4, 3, 2, 0, 1),
             (4, 3, 2, 0, 1),
             (1, 2, 3, 4, 0),
             (1, 3, 4, 0, 2),
             (1, 2, 3, 4, 0),
             (1, 2, 3, 4, 0),
             (1, 3, 4, 0, 2),
             (0, 1, 4, 3, 2),
             (4, 3, 2, 0, 1),
             (3, 4, 0, 2, 1),
             (1, 4, 3, 2, 0),
             (4, 3, 2, 0, 1),
             (2, 0, 4, 1, 3),
             (2, 3, 0, 1, 4),
             (1, 3, 4, 0, 2),
             (2, 0, 4, 1, 3),
             (0, 2, 3, 1, 4),
             (2, 0, 4, 1, 3),
             (4, 0, 3, 2, 1),
             (1, 3, 4, 0, 2),
             (2, 4, 0, 3, 1),
             (0, 1, 4, 3, 2),
             (1, 3, 4, 0, 2),
             (3, 4, 0, 2, 1),
             (4, 1, 2, 3, 0)]

    count_preferences_in_positions(preference_orders=prefs)

    # score = calculate_borda_score(prefs)
    # print(score)
