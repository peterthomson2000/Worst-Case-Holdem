import csv
import argparse
from collections import defaultdict


SAFE_HAND_COL = {
    # Map hand types to safe column name prefixes
    "High Card": "High_Card",
    "One Pair": "One_Pair",
    "Two Pair": "Two_Pair",
    "Three of a Kind": "Three_of_a_Kind",
    "Straight": "Straight",
    "Flush": "Flush",
    "Full House": "Full_House",
    "Four of a Kind": "Four_of_a_Kind",
    "Straight Flush": "Straight_Flush",
    "Royal Flush": "Royal_Flush",
}


def build_trend_sheet(input_csv: str, output_csv: str) -> None:
    """Read the long-format simulation CSV and write a wide-format CSV.

    Output layout (one row per num_players):
      num_players,
      hero_overall_win_probability,
      <Hand>_freq,
      <Hand>_win_given,
      <Hand>_hand_and_win,
    for each of the 10 hand types.
    """
    by_players = defaultdict(dict)
    overall_win_by_players = {}

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["num_players"])
            hand_type = row["hand_type"]
            hero_hand_prob = float(row["hero_hand_probability"])
            hero_win_given = float(row["hero_win_given_type_probability"])
            hero_hand_and_win = float(row["hero_hand_and_win_probability"])
            hero_overall = float(row["hero_overall_win_probability"])

            overall_win_by_players[n] = hero_overall

            prefix = SAFE_HAND_COL.get(hand_type, hand_type.replace(" ", "_"))
            rec = by_players[n]
            rec["num_players"] = n
            rec[f"{prefix}_freq"] = hero_hand_prob
            rec[f"{prefix}_win_given"] = hero_win_given
            rec[f"{prefix}_hand_and_win"] = hero_hand_and_win

    # Build a sorted list of player counts
    player_counts = sorted(by_players.keys())

    # Collect all fieldnames in a stable order
    base_fields = ["num_players", "hero_overall_win_probability"]
    extra_fields = []

    # Determine hand-type derived fields from first record(s)
    for n in player_counts:
        rec = by_players[n]
        for key in sorted(rec.keys()):
            if key == "num_players":
                continue
            if key not in extra_fields:
                extra_fields.append(key)

    fieldnames = base_fields + extra_fields

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in player_counts:
            rec = dict(by_players[n])
            rec["hero_overall_win_probability"] = overall_win_by_players.get(n, 0.0)
            # Ensure all fields exist
            for k in fieldnames:
                rec.setdefault(k, 0.0 if k != "num_players" else n)
            writer.writerow(rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a wide-format trend CSV from holdem_sim_results.csv for easier charting."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="holdem_sim_results.csv",
        help="Input long-format CSV (default: holdem_sim_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="holdem_trends_by_players.csv",
        help="Output wide-format CSV (default: holdem_trends_by_players.csv)",
    )

    args = parser.parse_args()
    build_trend_sheet(args.input, args.output)
    print(f"Trend sheet written to {args.output}")
