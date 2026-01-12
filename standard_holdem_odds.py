from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

from texas_holdem_sim import DECK, HAND_TYPES, evaluate_5card_hand


def enumerate_standard_frequencies() -> Counter:
    """Enumerate all 5-card hands and count standard Texas Hold'em categories."""

    counts: Counter = Counter()
    for combo in combinations(DECK, 5):
        _, type_idx = evaluate_5card_hand(combo)
        hand_type = HAND_TYPES[type_idx]
        counts[hand_type] += 1
    return counts


if __name__ == "__main__":
    total = comb(52, 5)
    freqs = enumerate_standard_frequencies()

    print(f"Total 5-card hands: {total}")
    for name in HAND_TYPES:
        count = freqs[name]
        prob = count / total
        print(f"{name:15s} {count:10d} {prob:8.6f}")

    # Also write markdown table for side-by-side comparison with Worst Case odds.
    md_path = "standard_holdem_odds.md"
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Standard Texas Hold'em – 5-Card Hand Probabilities\n")
        md.write(f"\nTotal distinct 5-card hands: {total}\n\n")
        md.write("| Rank | Hand Type        | Combinations | Probability |\n")
        md.write("|------|------------------|-------------:|------------:|\n")

        rows = []
        for rank, name in enumerate(reversed(HAND_TYPES), start=1):
            # Reverse so that Royal Flush gets highest rank number.
            count = freqs[name]
            prob = count / total
            rows.append((rank, name, count, prob))

        for rank, name, count, prob in rows:
            md.write(f"| {rank:2d} | {name:16s} | {count:11d} | {prob*100:10.4f}% |\n")

    print(f"Markdown odds written to {md_path}")
