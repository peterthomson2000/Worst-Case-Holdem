import random
import csv
import argparse
from collections import Counter

# Card representation: (rank, suit)
# ranks: 2-14 (where 14 = Ace)
# suits: 0-3 (we don't care which is which, just equality)

RANKS = list(range(2, 15))
SUITS = list(range(4))
DECK = [(r, s) for r in RANKS for s in SUITS]

HAND_TYPES = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush",
]


def best_five_of_seven(cards):
    """Return best 5-card hand score and hand type index from 7 cards.

    Score is a tuple where higher is better and comparable between hands.
    """
    assert len(cards) == 7

    best_score = None
    best_type_idx = None

    # Choose all 5-card combinations from 7 (21 combos)
    from itertools import combinations

    for combo in combinations(cards, 5):
        score, type_idx = evaluate_5card_hand(combo)
        if best_score is None or score > best_score:
            best_score = score
            best_type_idx = type_idx

    return best_score, best_type_idx


def evaluate_5card_hand(cards):
    """Evaluate a 5-card hand, return (score_tuple, hand_type_idx).

    Score tuple scheme:
    (category, primary ranks..., kicker ranks...)
    where category 9 = Royal Flush (strongest), 0 = High Card (weakest).
    """
    assert len(cards) == 5
    ranks = sorted((r for r, _ in cards), reverse=True)
    suits = [s for _, s in cards]

    # Count ranks
    counts = Counter(ranks)
    # Sort by (count, rank) descending for grouping
    counts_sorted = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    is_flush = len(set(suits)) == 1

    # Straight detection including wheel A-2-3-4-5
    uniq_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high_in_straight = None

    if len(uniq_ranks) == 5 and uniq_ranks[0] - uniq_ranks[4] == 4:
        is_straight = True
        high_in_straight = uniq_ranks[0]
    else:
        # Wheel: A-2-3-4-5 as 5-high straight
        if set(uniq_ranks) == {14, 5, 4, 3, 2}:
            is_straight = True
            high_in_straight = 5

    # Determine hand category
    # counts_sorted structure examples:
    # Four of a kind: [(rank4, 4), (kicker,1)]
    # Full house: [(rank3,3),(rank2,2)]
    # Trips: [(rank3,3),(k1,1),(k2,1)]
    # Two pair: [(r2a,2),(r2b,2),(k,1)]
    # One pair: [(rp,2),(k1,1),(k2,1),(k3,1)]

    if is_straight and is_flush:
        if high_in_straight == 14:
            # Royal flush
            category = 9
            hand_type_idx = HAND_TYPES.index("Royal Flush")
        else:
            category = 8
            hand_type_idx = HAND_TYPES.index("Straight Flush")
        score = (category, high_in_straight)
        return score, hand_type_idx

    if counts_sorted[0][1] == 4:
        # Four of a kind
        four_rank = counts_sorted[0][0]
        kicker = counts_sorted[1][0]
        category = 7
        hand_type_idx = HAND_TYPES.index("Four of a Kind")
        score = (category, four_rank, kicker)
        return score, hand_type_idx

    if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
        # Full house
        trip_rank = counts_sorted[0][0]
        pair_rank = counts_sorted[1][0]
        category = 6
        hand_type_idx = HAND_TYPES.index("Full House")
        score = (category, trip_rank, pair_rank)
        return score, hand_type_idx

    if is_flush:
        # Flush
        category = 5
        hand_type_idx = HAND_TYPES.index("Flush")
        score = (category,) + tuple(sorted(ranks, reverse=True))
        return score, hand_type_idx

    if is_straight:
        # Straight
        category = 4
        hand_type_idx = HAND_TYPES.index("Straight")
        score = (category, high_in_straight)
        return score, hand_type_idx

    if counts_sorted[0][1] == 3:
        # Three of a kind
        trip_rank = counts_sorted[0][0]
        kickers = [r for r, c in counts_sorted[1:] if c == 1]
        kickers_sorted = sorted(kickers, reverse=True)
        category = 3
        hand_type_idx = HAND_TYPES.index("Three of a Kind")
        score = (category, trip_rank) + tuple(kickers_sorted)
        return score, hand_type_idx

    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        # Two pair
        high_pair = counts_sorted[0][0]
        low_pair = counts_sorted[1][0]
        kicker = counts_sorted[2][0]
        category = 2
        hand_type_idx = HAND_TYPES.index("Two Pair")
        score = (category, high_pair, low_pair, kicker)
        return score, hand_type_idx

    if counts_sorted[0][1] == 2:
        # One pair
        pair_rank = counts_sorted[0][0]
        kickers = [r for r, c in counts_sorted[1:] if c == 1]
        kickers_sorted = sorted(kickers, reverse=True)
        category = 1
        hand_type_idx = HAND_TYPES.index("One Pair")
        score = (category, pair_rank) + tuple(kickers_sorted)
        return score, hand_type_idx

    # High card
    category = 0
    hand_type_idx = HAND_TYPES.index("High Card")
    score = (category,) + tuple(sorted(ranks, reverse=True))
    return score, hand_type_idx


def simulate(num_players_list, num_trials_per_player_count=50000, csv_filename="holdem_sim_results.csv", md_filename=None):
    """Run simulations for given list of player counts and write CSV (and optional markdown) with results.

    For each number of players N and focal player (seat 0):
      - Estimate probability of each final hand type for hero
      - Estimate probability hero wins given that hand type
      - Estimate overall hero win probability
    """
    fieldnames = [
        "num_players",
        "hand_type",
        "hero_hand_probability",
        "hero_win_given_type_probability",
        "hero_hand_and_win_probability",
        "hero_overall_win_probability",  # same per num_players per row (repeated)
    ]

    # For markdown summary: collect per-player, per-hand stats in memory
    md_summary = {}

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for num_players in num_players_list:
            if num_players < 2 or num_players > 9:
                raise ValueError("num_players must be between 2 and 9 for this sim")

            hero_type_counts = Counter()
            hero_type_win_counts = Counter()
            hero_type_equity_win_counts = Counter()  # accounts for split pots
            hero_overall_equity_wins = 0.0

            for _ in range(num_trials_per_player_count):
                # Shuffle deck
                deck = DECK[:]
                random.shuffle(deck)

                # Deal 2 cards to each player
                hands = [deck[2 * i : 2 * i + 2] for i in range(num_players)]

                # Deal 5 community cards
                community = deck[2 * num_players : 2 * num_players + 5]

                # Evaluate each player's best hand
                scores = []
                type_indices = []
                for i in range(num_players):
                    seven_cards = hands[i] + community
                    score, type_idx = best_five_of_seven(seven_cards)
                    scores.append(score)
                    type_indices.append(type_idx)

                # Determine winner(s)
                best_score = max(scores)
                winners = [i for i, s in enumerate(scores) if s == best_score]

                hero_type = HAND_TYPES[type_indices[0]]
                hero_type_counts[hero_type] += 1

                # Hero equity for this deal (1 if sole winner, fractional if tie on top, 0 if loses)
                hero_equity = 0.0
                if 0 in winners:
                    hero_equity = 1.0 / len(winners)

                if hero_equity > 0:
                    hero_type_equity_win_counts[hero_type] += hero_equity

                if 0 in winners:
                    hero_type_win_counts[hero_type] += 1  # pure win count

                hero_overall_equity_wins += hero_equity

            total_deals = float(num_trials_per_player_count)
            hero_overall_win_prob = hero_overall_equity_wins / total_deals

            # Initialize markdown summary record for this player count
            md_summary[num_players] = {
                "hero_overall_win_probability": hero_overall_win_prob,
                "hands": {},
            }

            # Write a row per hand type
            for hand_type in HAND_TYPES:
                count = float(hero_type_counts[hand_type])
                if count > 0:
                    hand_prob = count / total_deals
                    win_given_type_prob = hero_type_equity_win_counts[hand_type] / count
                    hand_and_win_prob = hero_type_equity_win_counts[hand_type] / total_deals
                else:
                    hand_prob = 0.0
                    win_given_type_prob = 0.0
                    hand_and_win_prob = 0.0

                writer.writerow({
                    "num_players": num_players,
                    "hand_type": hand_type,
                    "hero_hand_probability": hand_prob,
                    "hero_win_given_type_probability": win_given_type_prob,
                    "hero_hand_and_win_probability": hand_and_win_prob,
                    "hero_overall_win_probability": hero_overall_win_prob,
                })

                md_summary[num_players]["hands"][hand_type] = {
                    "hand_prob": hand_prob,
                    "win_given_type_prob": win_given_type_prob,
                    "hand_and_win_prob": hand_and_win_prob,
                }

    # Optionally write markdown summary file
    if md_filename:
        with open(md_filename, "w", encoding="utf-8") as md:
            md.write("# Texas Hold'em Simulation Summary\n")
            md.write(f"\n- Player counts simulated: {', '.join(str(n) for n in sorted(num_players_list))}\n")
            md.write(f"- Trials per player count: {num_trials_per_player_count}\n\n")

            for num_players in sorted(md_summary.keys()):
                overall = md_summary[num_players]["hero_overall_win_probability"]
                md.write(f"## {num_players} Players\n\n")
                md.write(f"- Hero overall win probability: {overall * 100:.2f}%\n\n")

                md.write("| Hand Type | P(hand) | P(win | hand) | P(hand & win) |\n")
                md.write("|----------|---------|--------------|----------------|\n")

                for hand_type in HAND_TYPES:
                    stats = md_summary[num_players]["hands"][hand_type]
                    md.write(
                        f"| {hand_type} | {stats['hand_prob'] * 100:.3f}% | "
                        f"{stats['win_given_type_prob'] * 100:.2f}% | "
                        f"{stats['hand_and_win_prob'] * 100:.3f}% |\n"
                    )

                md.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Texas Hold'em hand odds vs number of players.")
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=50000,
        help="Number of simulated deals per player count (default: 50000)",
    )
    parser.add_argument(
        "-p",
        "--players",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9],
        help="List of player counts to simulate (e.g. -p 2 6 9)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="holdem_sim_results.csv",
        help="Output CSV filename (default: holdem_sim_results.csv)",
    )
    parser.add_argument(
        "--md",
        type=str,
        default=None,
        help="Output markdown summary filename (default: same name as CSV but with .md)",
    )

    args = parser.parse_args()

    md_filename = args.md
    if md_filename is None:
        if args.csv.lower().endswith(".csv"):
            md_filename = args.csv[:-4] + ".md"
        else:
            md_filename = args.csv + ".md"

    simulate(args.players, args.trials, args.csv, md_filename)
    print(f"Simulation complete. Results written to {args.csv} and {md_filename}")
