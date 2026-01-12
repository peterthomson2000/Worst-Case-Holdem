from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import List, Sequence, Tuple

# Card representation matches texas_holdem_sim: (rank, suit)
# ranks: 2-14 (where 14 = Ace)
# suits: 0-3

RANKS = list(range(2, 15))
SUITS = list(range(4))
DECK: List[Tuple[int, int]] = [(r, s) for r in RANKS for s in SUITS]


class WorstCaseHandType(IntEnum):
    """Worst Case Hold'em 5-card hand rankings.

    Values increase with unluckiness / strength in this variant.

    Enum values 1..10 define the global Worst Case ranking from
    least unlucky (LOW_CARD) to most unlucky (PERFECT_MISDEAL).
    """

    LOW_CARD = 1
    BROKEN_PAIR = 2
    FAUX_FLUSH = 3
    MIRROR_HAND = 4
    COLOR_DISASSOCIATE = 5
    GAP = 6
    ALMOST_FULL_HOUSE = 7
    COLOR_CLASH = 8
    DEAD_ROYAL = 9
    PERFECT_MISDEAL = 10


@dataclass(frozen=True)
class WorstCaseHandEval:
    hand_type: WorstCaseHandType | None


def _suit_to_color(suit: int) -> int:
    """Map suits to colors: 0/1 = red, 2/3 = black (arbitrary but consistent)."""

    return 0 if suit in (0, 1) else 1


def _is_faux_flush(suits: Sequence[int]) -> bool:
    counts = Counter(suits)
    return sorted(counts.values()) == [1, 4]


def _is_low_card(ranks: Sequence[int]) -> bool:
    """Low Card: all cards 9 or below, no pairs/trips/quads.

    This is a "dead low" high-card hand: nothing but small cards.
    """

    if max(ranks) > 9:
        return False
    counts = Counter(ranks)
    return max(counts.values()) == 1


def _is_gap(unique_ranks: Sequence[int]) -> bool:
    # Five distinct ranks forming a +2 arithmetic progression
    if len(unique_ranks) != 5:
        return False
    return all(unique_ranks[i + 1] - unique_ranks[i] == 2 for i in range(4))


def _is_mirror_hand(rank_counts: Counter) -> bool:
    # Two ranks appearing twice and one single: counts 2,2,1 (like two pair)
    return sorted(rank_counts.values()) == [1, 2, 2]


def _is_color_disassociate(ranks: Sequence[int], suits: Sequence[int]) -> bool:
    """Color Disassociate: a "rainbow low" hand with all four suits.

    Requirements:
      - All four suits must appear at least once (true rainbow).
      - All five ranks must be distinct (no pairs / trips / quads).
      - All ranks are 9 or below (keeps it in the "low junk" space).

    This is intentionally much rarer than the old 2,1,1,1 suit-count rule,
    pushing it closer to a Straight/Flush tier instead of 25% of all hands.
    """

    suit_set = set(suits)
    if suit_set != set(SUITS):
        return False

    # Distinct ranks only.
    if len(set(ranks)) != 5:
        return False

    # All ranks 9 or below.
    return max(ranks) <= 9


def _is_color_clash(ranks: Sequence[int], suits: Sequence[int]) -> bool:
    # Faux flush + a pair whose two cards are split across the main suit and the off-suit.
    if not _is_faux_flush(suits):
        return False

    suit_counts = Counter(suits)
    main_suit, _ = max(suit_counts.items(), key=lambda x: x[1])
    off_suit = next(s for s in suit_counts if s != main_suit)

    # Map rank -> list of suits
    by_rank: dict[int, List[int]] = {}
    for r, s in zip(ranks, suits):
        by_rank.setdefault(r, []).append(s)

    for suitelist in by_rank.values():
        if len(suitelist) == 2:
            # Check that the pair is split between main and off suit
            if main_suit in suitelist and off_suit in suitelist:
                return True

    return False


def _is_broken_pair(ranks: Sequence[int]) -> bool:
    """Broken Pair: a true low pair (<= 9) and three distinct kickers.

    This makes Broken Pair behave more like "One Pair" in standard poker odds,
    but only for lower-ranked pairs. High pairs will tend to be absorbed by
    other patterns or by the LOW_CARD catch-all.
    """

    counts = Counter(ranks)
    # Exactly one pair and three singletons.
    pair_ranks = [r for r, c in counts.items() if c == 2]
    if len(pair_ranks) != 1:
        return False

    pair_rank = pair_ranks[0]
    # Restrict to "low-ish" pairs to keep the Worst Case flavor.
    if pair_rank > 9:
        return False

    # Ensure the remaining three cards are all different ranks.
    if sorted(counts.values(), reverse=True) != [2, 1, 1, 1]:
        return False

    return True


def _is_double_broken_pair(ranks: Sequence[int], rank_counts: Counter) -> bool:
    # Two or more distinct adjacent rank pairs and no real pair (all ranks distinct).
    if max(rank_counts.values()) > 1:
        return False
    uniq = sorted(set(ranks))
    adjacencies = sum(1 for i in range(len(uniq) - 1) if uniq[i + 1] - uniq[i] == 1)
    return adjacencies >= 2


def _is_dead_royal(ranks: Sequence[int], suits: Sequence[int]) -> bool:
    # A, K, Q, J, 10 with mixed suits (not a flush).
    if set(ranks) != {10, 11, 12, 13, 14}:
        return False
    return len(set(suits)) > 1 and len(set(suits)) != 1


def classify_worst_case_hand(cards: Sequence[Tuple[int, int]]) -> WorstCaseHandEval:
    """Classify a 5-card hand into *true* Worst Case Hold'em categories.

    This no longer delegates to the standard Texas Hold'em evaluator.
    Instead, it uses bespoke pattern logic for the ten unlucky hand types
    we've been designing (Gap, Color Associate, Broken Pair, etc.).

    The ranking ladder, from "most common / least unlucky" to
    "rarest / most unlucky" is:

        LOW_CARD < BROKEN_PAIR < FAUX_FLUSH < MIRROR_HAND < COLOR_DISASSOCIATE
        < GAP < ALMOST_FULL_HOUSE < COLOR_CLASH < DEAD_ROYAL < PERFECT_MISDEAL
    """

    if len(cards) != 5:
        raise ValueError("Worst Case evaluation expects exactly 5 cards")

    ranks = [r for r, _ in cards]
    suits = [s for _, s in cards]
    rank_counts = Counter(ranks)
    suits_counter = Counter(suits)

    # Helper: standard flush / royal detection for DEAD_ROYAL / PERFECT_MISDEAL.
    is_flush = len(suits_counter) == 1
    uniq_ranks_sorted = sorted(set(ranks))

    # PERFECT_MISDEAL: true Royal Flush (A,K,Q,J,10 all same suit) – the "peak" bad beat.
    if set(ranks) == {10, 11, 12, 13, 14} and is_flush:
        wc_type = WorstCaseHandType.PERFECT_MISDEAL
        return WorstCaseHandEval(wc_type)

    # DEAD_ROYAL: A,K,Q,J,10 with *mixed* suits (not a flush).
    if _is_dead_royal(ranks, suits):
        wc_type = WorstCaseHandType.DEAD_ROYAL
        return WorstCaseHandEval(wc_type)

    # COLOR_CLASH: faux-flush style pattern where the odd card clashes
    # with the main suit on rank.
    if _is_color_clash(ranks, suits):
        wc_type = WorstCaseHandType.COLOR_CLASH
        return WorstCaseHandEval(wc_type)

    # ALMOST_FULL_HOUSE: 3-of-a-kind plus two different kickers.
    # (i.e. rank multiplicities 3,1,1 and *not* an actual full house).
    if sorted(rank_counts.values(), reverse=True) == [3, 1, 1]:
        wc_type = WorstCaseHandType.ALMOST_FULL_HOUSE
        return WorstCaseHandEval(wc_type)

    # GAP: five distinct ranks forming a +2 arithmetic progression.
    if _is_gap(sorted(set(ranks))):
        wc_type = WorstCaseHandType.GAP
        return WorstCaseHandEval(wc_type)

    # COLOR_DISASSOCIATE: rainbow low hand meeting the stricter definition.
    if _is_color_disassociate(ranks, suits):
        wc_type = WorstCaseHandType.COLOR_DISASSOCIATE
        return WorstCaseHandEval(wc_type)

    # MIRROR_HAND: counts pattern 2,2,1 (our earlier helper) – a kind of
    # "fake" two-pair shape.
    if _is_mirror_hand(rank_counts):
        wc_type = WorstCaseHandType.MIRROR_HAND
        return WorstCaseHandEval(wc_type)

    # FAUX_FLUSH: four cards in one suit, one in another.
    if _is_faux_flush(suits):
        wc_type = WorstCaseHandType.FAUX_FLUSH
        return WorstCaseHandEval(wc_type)

    # BROKEN_PAIR: a true low pair, handled by _is_broken_pair.
    if _is_broken_pair(ranks):
        wc_type = WorstCaseHandType.BROKEN_PAIR
        return WorstCaseHandEval(wc_type)

    # LOW_CARD: catch-all for hands that don't satisfy any special Worst Case
    # pattern. This soaks up the majority of garbage hands, similar to High
    # Card in standard poker odds.
    wc_type = WorstCaseHandType.LOW_CARD
    return WorstCaseHandEval(wc_type)


def enumerate_worst_case_frequencies() -> Counter:
    """Enumerate all 5-card hands and count Worst Case categories.

    Returns a Counter mapping WorstCaseHandType (or None) to combination counts.
    """

    counts: Counter = Counter()
    for combo in combinations(DECK, 5):
        eval_result = classify_worst_case_hand(combo)
        counts[eval_result.hand_type] += 1
    return counts


if __name__ == "__main__":
    from math import comb

    total = comb(52, 5)
    freqs = enumerate_worst_case_frequencies()

    # Plain-text summary to stdout
    print(f"Total 5-card hands: {total}")
    for hand_type, count in sorted(
        freqs.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else -1)
    ):
        label = "None" if hand_type is None else hand_type.name
        prob = count / total
        print(f"{label:20s} {count:10d} {prob:8.6f}")

    # Also write a markdown table for easy tracking/comparison.
    md_path = "worst_case_holdem_odds.md"
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Worst Case Hold'em – 5-Card Hand Probabilities\n")
        md.write(f"\nTotal distinct 5-card hands: {total}\n\n")
        md.write("| Rank | Hand Type        | Combinations | Probability |\n")
        md.write("|------|------------------|-------------:|------------:|\n")

        # Sort by enum value (hand_type) so 1..10 ordering.
        rows = []
        for hand_type, count in freqs.items():
            if hand_type is None:
                continue
            prob = count / total
            rows.append((int(hand_type), hand_type.name.replace("_", " "), count, prob))

        rows.sort(key=lambda r: r[0], reverse=True)
        for rank, name, count, prob in rows:
            md.write(f"| {rank:2d} | {name:16s} | {count:11d} | {prob*100:10.4f}% |\n")

    print(f"Markdown odds written to {md_path}")
