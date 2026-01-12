import matplotlib.pyplot as plt
import numpy as np

# Toggle this to add/remove trendlines on all line plots
ADD_TRENDLINES = True


def add_trendline(x, y, ax, color, degree=1, linestyle="--", alpha=0.5):
    """Fit a polynomial trendline and draw it on the given axis."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = ~np.isnan(y_arr)
    if mask.sum() < degree + 1:
        return
    coeffs = np.polyfit(x_arr[mask], y_arr[mask], degree)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(x_arr), max(x_arr), 200)
    y_fit = poly(x_fit)
    ax.plot(x_fit, y_fit, linestyle=linestyle, color=color, alpha=alpha)


# ------------------------------
# 1. Basic setup: player counts
# ------------------------------
players = [2, 3, 4, 5, 6, 7, 8, 9]

# Hero overall win probabilities (%)
worst_overall = [49.94, 33.13, 24.90, 20.07, 16.75, 14.27, 12.70, 11.03]
standard_overall = [49.90, 33.54, 25.05, 19.84, 16.66, 14.41, 12.63, 11.02]

TRIALS_PER_PLAYER_COUNT = 50000

# ------------------------------------------
# 2. Worst Case Hold'em data
# ------------------------------------------
worst_hand_types = [
    "Low Card",
    "Broken Pair",
    "Faux Flush",
    "Mirror Hand",
    "Color Disassociate",
    "Gap",
    "Almost Full House",
    "Color Clash",
    "Dead Royal",
    "Perfect Misdeal",
]

# P(hand) (%), by hand type across 2–9 players
worst_p_hand = {
    "Low Card":          [25.194, 25.156, 25.028, 24.650, 25.322, 25.132, 24.732, 24.722],
    "Broken Pair":       [18.876, 19.314, 19.340, 18.946, 19.040, 18.756, 19.094, 19.228],
    "Faux Flush":        [6.758,  6.712,  6.634,  6.802,  6.568,  6.688,  6.818,  6.840],
    "Mirror Hand":       [19.192, 19.110, 18.870, 19.444, 19.186, 19.150, 19.036, 19.136],
    "Color Disassociate":[5.556,  5.428,  5.362,  5.410,  5.400,  5.532,  5.518,  5.548],
    "Gap":               [2.190,  2.306,  2.308,  2.314,  2.384,  2.234,  2.292,  2.288],
    "Almost Full House": [6.914,  6.496,  6.642,  6.644,  6.660,  6.800,  6.780,  6.746],
    "Color Clash":       [14.706, 14.876, 15.198, 15.166, 14.900, 15.138, 15.110, 14.922],
    "Dead Royal":        [0.608,  0.596,  0.612,  0.620,  0.540,  0.570,  0.616,  0.566],
    "Perfect Misdeal":   [0.006,  0.006,  0.006,  0.004,  0.000,  0.000,  0.004,  0.004],
}

# P(win | hand) (%)
worst_p_win_given_hand = {
    "Low Card":          [20.75,  6.15,  2.07,  0.76,  0.30,  0.12,  0.05,  0.01],
    "Broken Pair":       [36.63, 17.52,  9.73,  5.57,  3.58,  2.44,  1.55,  1.14],
    "Faux Flush":        [52.35, 29.79, 17.95, 11.51,  7.24,  5.27,  2.81,  2.29],
    "Mirror Hand":       [56.51, 35.36, 23.03, 16.29, 12.05,  8.59,  6.41,  4.97],
    "Color Disassociate":[72.26, 54.19, 41.47, 34.24, 28.40, 22.25, 18.32, 16.14],
    "Gap":               [77.21, 64.67, 53.46, 43.11, 37.45, 29.04, 27.78, 21.92],
    "Almost Full House": [70.75, 56.61, 47.95, 40.97, 36.53, 30.86, 29.36, 24.25],
    "Color Clash":       [83.37, 72.54, 64.46, 57.99, 53.05, 48.81, 45.45, 42.00],
    "Dead Royal":        [91.12, 90.04, 84.75, 82.08, 81.30, 73.63, 75.51, 69.41],
    "Perfect Misdeal":   [100.00, 55.56, 75.00, 100.00, 0.00,  0.00, 100.00, 100.00],
}

# P(hand & win) (% of all deals)
worst_p_hand_and_win = {
    "Low Card":          [5.228, 1.548, 0.518, 0.188, 0.076, 0.030, 0.013, 0.004],
    "Broken Pair":       [6.914, 3.383, 1.883, 1.056, 0.681, 0.458, 0.297, 0.219],
    "Faux Flush":        [3.538, 2.000, 1.191, 0.783, 0.476, 0.352, 0.192, 0.157],
    "Mirror Hand":       [10.846, 6.758, 4.346, 3.168, 2.313, 1.646, 1.221, 0.951],
    "Color Disassociate":[4.015, 2.941, 2.224, 1.852, 1.534, 1.231, 1.011, 0.895],
    "Gap":               [1.691, 1.491, 1.234, 0.998, 0.893, 0.649, 0.637, 0.502],
    "Almost Full House": [4.892, 3.677, 3.185, 2.722, 2.433, 2.099, 1.991, 1.636],
    "Color Clash":       [12.260, 10.791, 9.796, 8.795, 7.905, 7.389, 6.867, 6.267],
    "Dead Royal":        [0.554, 0.537, 0.519, 0.509, 0.439, 0.420, 0.465, 0.393],
    "Perfect Misdeal":   [0.006, 0.003, 0.005, 0.004, 0.000, 0.000, 0.004, 0.004],
}

# ------------------------------------------------
# 3. Standard Texas Hold'em simulation data
# ------------------------------------------------
standard_hand_types = [
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

standard_p_hand = {
    "High Card":        [17.628, 17.408, 17.400, 17.794, 17.778, 17.388, 17.096, 17.514],
    "One Pair":         [43.544, 43.602, 43.966, 43.618, 43.550, 44.120, 43.616, 43.708],
    "Two Pair":         [23.526, 23.726, 23.518, 23.578, 23.484, 23.286, 23.778, 23.482],
    "Three of a Kind":  [4.806,  4.880,  4.852,  4.914,  4.806,  4.906,  5.008,  4.908],
    "Straight":         [4.672,  4.706,  4.464,  4.528,  4.578,  4.704,  4.644,  4.572],
    "Flush":            [3.028,  2.970,  2.988,  2.886,  3.066,  2.920,  3.048,  2.960],
    "Full House":       [2.602,  2.530,  2.632,  2.504,  2.554,  2.498,  2.584,  2.662],
    "Four of a Kind":   [0.166,  0.150,  0.148,  0.140,  0.168,  0.162,  0.196,  0.158],
    "Straight Flush":   [0.026,  0.022,  0.032,  0.036,  0.008,  0.016,  0.022,  0.034],
    "Royal Flush":      [0.002,  0.006,  0.000,  0.002,  0.008,  0.000,  0.008,  0.002],
}

standard_p_win_given_hand = {
    "High Card":        [17.68,  3.58,  0.84,  0.16,  0.10,  0.01,  0.01,  0.00],
    "One Pair":         [42.00, 22.70, 14.06,  8.64,  6.34,  4.61,  3.10,  2.33],
    "Two Pair":         [66.82, 48.81, 36.16, 27.93, 21.86, 17.70, 14.26, 11.39],
    "Three of a Kind":  [75.18, 65.45, 55.72, 52.65, 48.54, 44.90, 40.59, 37.78],
    "Straight":         [85.96, 78.01, 71.52, 66.12, 60.51, 55.50, 51.65, 48.03],
    "Flush":            [86.43, 77.17, 69.08, 65.78, 56.75, 54.66, 51.67, 49.54],
    "Full House":       [90.35, 83.20, 79.48, 73.22, 68.32, 67.59, 64.86, 61.27],
    "Four of a Kind":   [93.98, 96.44, 87.84, 79.62, 85.91, 87.83, 90.94, 91.14],
    "Straight Flush":   [88.46,100.00,100.00, 80.00,100.00, 87.50, 90.91, 88.89],
    "Royal Flush":      [100.00,100.00,  0.00,100.00,100.00,  0.00,100.00,100.00],
}

standard_p_hand_and_win = {
    "High Card":        [3.116, 0.624, 0.145, 0.029, 0.017, 0.002, 0.002, 0.000],
    "One Pair":         [18.289, 9.897, 6.183, 3.770, 2.760, 2.034, 1.351, 1.018],
    "Two Pair":         [15.721,11.581, 8.504, 6.586, 5.134, 4.123, 3.390, 2.675],
    "Three of a Kind":  [3.613, 3.194, 2.703, 2.587, 2.333, 2.203, 2.033, 1.854],
    "Straight":         [4.016, 3.671, 3.193, 2.994, 2.770, 2.610, 2.399, 2.196],
    "Flush":            [2.617, 2.292, 2.064, 1.898, 1.740, 1.596, 1.575, 1.466],
    "Full House":       [2.351, 2.105, 2.092, 1.833, 1.745, 1.688, 1.676, 1.631],
    "Four of a Kind":   [0.156, 0.145, 0.130, 0.111, 0.144, 0.142, 0.178, 0.144],
    "Straight Flush":   [0.023, 0.022, 0.032, 0.029, 0.008, 0.014, 0.020, 0.030],
    "Royal Flush":      [0.002, 0.006, 0.000, 0.002, 0.008, 0.000, 0.008, 0.002],
}

# ---------------------------------------------------
# 4. Standard 5-card hand probabilities (analytic)
# ---------------------------------------------------
fivecard_hand_types = [
    "Royal Flush",
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "One Pair",
    "High Card",
]

fivecard_combos = [
    4,
    36,
    624,
    3744,
    5108,
    10200,
    54912,
    123552,
    1098240,
    1302540,
]

fivecard_probs_percent = [
    0.0002,
    0.0014,
    0.0240,
    0.1441,
    0.1965,
    0.3925,
    2.1128,
    4.7539,
    42.2569,
    50.1177,
]

TOTAL_5CARD_HANDS = 2598960

# --------------------------
# Plot 1: Overall win rates
# --------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
line1, = ax1.plot(players, worst_overall, marker="o", label="Worst Case Hold'em")
line2, = ax1.plot(players, standard_overall, marker="s", label="Standard Hold'em")

ax1.set_title(
    "Hero Overall Win Probability vs Number of Players\n"
    f"(Trials per player count: {TRIALS_PER_PLAYER_COUNT:,})"
)
ax1.set_xlabel("Number of Players")
ax1.set_ylabel("Win Probability (%)")
ax1.set_xticks(players)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()
fig1.tight_layout()
fig1.savefig("overall_win_rates.png", dpi=150, bbox_inches="tight")

if ADD_TRENDLINES:
    add_trendline(players, worst_overall, ax1, line1.get_color())
    add_trendline(players, standard_overall, ax1, line2.get_color())
    fig1.savefig("overall_win_rates_with_trend.png", dpi=150, bbox_inches="tight")

# -------------------------------------------------------
# Plot 2: Worst Case – P(win | hand) vs players (all)
# -------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
for hand in worst_hand_types:
    y = worst_p_win_given_hand[hand]
    line, = ax2.plot(players, y, marker="o", label=hand)
    if ADD_TRENDLINES:
        add_trendline(players, y, ax2, line.get_color())

ax2.set_title("Worst Case Hold'em: P(win | hand type) vs Number of Players")
ax2.set_xlabel("Number of Players")
ax2.set_ylabel("P(win | hand) (%)")
ax2.set_xticks(players)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=8, ncol=2)
fig2.tight_layout()
fig2.savefig("worst_pwin_given_hand_vs_players.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 3: Standard – P(win | hand) vs players (all)
# -----------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))
for hand in standard_hand_types:
    y = standard_p_win_given_hand[hand]
    line, = ax3.plot(players, y, marker="o", label=hand)
    if ADD_TRENDLINES:
        add_trendline(players, y, ax3, line.get_color())

ax3.set_title("Standard Hold'em: P(win | hand type) vs Number of Players")
ax3.set_xlabel("Number of Players")
ax3.set_ylabel("P(win | hand) (%)")
ax3.set_xticks(players)
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(fontsize=8, ncol=2)
fig3.tight_layout()
fig3.savefig("standard_pwin_given_hand_vs_players.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 4: Worst Case – P(hand & win) vs players (all)
# -----------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 6))
for hand in worst_hand_types:
    y = worst_p_hand_and_win[hand]
    line, = ax4.plot(players, y, marker="o", label=hand)
    if ADD_TRENDLINES:
        add_trendline(players, y, ax4, line.get_color())

ax4.set_title("Worst Case Hold'em: P(hand & win) vs Number of Players")
ax4.set_xlabel("Number of Players")
ax4.set_ylabel("P(hand and win) (% of all deals)")
ax4.set_xticks(players)
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend(fontsize=8, ncol=2)
fig4.tight_layout()
fig4.savefig("worst_phand_and_win_vs_players.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 5: Standard – P(hand & win) vs players (all)
# -----------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 6))
for hand in standard_hand_types:
    y = standard_p_hand_and_win[hand]
    line, = ax5.plot(players, y, marker="o", label=hand)
    if ADD_TRENDLINES:
        add_trendline(players, y, ax5, line.get_color())

ax5.set_title("Standard Hold'em: P(hand & win) vs Number of Players")
ax5.set_xlabel("Number of Players")
ax5.set_ylabel("P(hand and win) (% of all deals)")
ax5.set_xticks(players)
ax5.grid(True, linestyle="--", alpha=0.5)
ax5.legend(fontsize=8, ncol=2)
fig5.tight_layout()
fig5.savefig("standard_phand_and_win_vs_players.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 6: Worst Case – P(hand) vs players (all)
# -----------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(10, 6))
for hand in worst_hand_types:
    y = worst_p_hand[hand]
    line, = ax6.plot(players, y, marker="o", label=hand)
    if ADD_TRENDLINES:
        add_trendline(players, y, ax6, line.get_color())

ax6.set_title("Worst Case Hold'em: P(hand) vs Number of Players")
ax6.set_xlabel("Number of Players")
ax6.set_ylabel("P(hand) (%)")
ax6.set_xticks(players)
ax6.grid(True, linestyle="--", alpha=0.5)
ax6.legend(fontsize=8, ncol=2)
fig6.tight_layout()
fig6.savefig("worst_phand_vs_players.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 7: Standard – P(hand) vs players (all)
# -----------------------------------------------------------
fig7, ax7 = plt.subplots(figsize=(10, 6))
for hand in standard_hand_types:
    y = standard_p_hand[hand]
    line, = ax7.plot(players, y, marker="o", label=hand)
    if ADD_TRENDLINES:
        add_trendline(players, y, ax7, line.get_color())

ax7.set_title("Standard Hold'em: P(hand) vs Number of Players")
ax7.set_xlabel("Number of Players")
ax7.set_ylabel("P(hand) (%)")
ax7.set_xticks(players)
ax7.grid(True, linestyle="--", alpha=0.5)
ax7.legend(fontsize=8, ncol=2)
fig7.tight_layout()
fig7.savefig("standard_phand_vs_players.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 8: Standard 5-card hand probabilities & combinations
# -----------------------------------------------------------
fig8, ax8_1 = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(fivecard_hand_types))

bars = ax8_1.bar(x_pos, fivecard_combos, color="tab:blue", alpha=0.6, label="Combinations")
ax8_1.set_yscale("log")
ax8_1.set_ylabel("Number of Combinations (log scale)")
ax8_1.set_xticks(x_pos)
ax8_1.set_xticklabels(fivecard_hand_types, rotation=45, ha="right")

ax8_2 = ax8_1.twinx()
line_probs, = ax8_2.plot(
    x_pos,
    fivecard_probs_percent,
    color="tab:red",
    marker="o",
    label="Probability (%)"
)
ax8_2.set_ylabel("Probability (%)")

title_8 = (
    "Standard 5-Card Hand Probabilities & Combinations\n"
    f"(Total distinct hands: {TOTAL_5CARD_HANDS:,})"
)
ax8_1.set_title(title_8)

handles_1, labels_1 = ax8_1.get_legend_handles_labels()
handles_2, labels_2 = ax8_2.get_legend_handles_labels()
ax8_1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

fig8.tight_layout()
fig8.savefig("fivecard_probs_and_combos.png", dpi=150, bbox_inches="tight")

# -----------------------------------------------------------
# Plot 9: Worst Case 5-card hand probabilities & combinations
# -----------------------------------------------------------
worst_fivecard_hand_types = [
    "Perfect Misdeal",
    "Dead Royal",
    "Color Clash",
    "Almost Full House",
    "Gap",
    "Color Disassociate",
    "Mirror Hand",
    "Faux Flush",
    "Broken Pair",
    "Low Card",
]

worst_fivecard_combos = [
    4,        # Perfect Misdeal
    1020,     # Dead Royal
    34320,    # Color Clash
    54912,    # Almost Full House
    5120,     # Gap
    13440,    # Color Disassociate
    123552,   # Mirror Hand
    76860,    # Faux Flush
    654720,   # Broken Pair
    1635012,  # Low Card
]

worst_fivecard_probs_percent = [
    c / TOTAL_5CARD_HANDS * 100 for c in worst_fivecard_combos
]

fig9, ax9_1 = plt.subplots(figsize=(10, 5))
x_pos_w = np.arange(len(worst_fivecard_hand_types))

bars_w = ax9_1.bar(x_pos_w, worst_fivecard_combos, color="tab:purple", alpha=0.6, label="Combinations")
ax9_1.set_yscale("log")
ax9_1.set_ylabel("Number of Combinations (log scale)")
ax9_1.set_xticks(x_pos_w)
ax9_1.set_xticklabels(worst_fivecard_hand_types, rotation=45, ha="right")

ax9_2 = ax9_1.twinx()
line_probs_w, = ax9_2.plot(
    x_pos_w,
    worst_fivecard_probs_percent,
    color="tab:red",
    marker="o",
    label="Probability (%)"
)
ax9_2.set_ylabel("Probability (%)")

title_9 = (
    "Worst Case Hold'em 5-Card Hand Probabilities & Combinations\n"
    f"(Total distinct hands: {TOTAL_5CARD_HANDS:,})"
)
ax9_1.set_title(title_9)

handles_w1, labels_w1 = ax9_1.get_legend_handles_labels()
handles_w2, labels_w2 = ax9_2.get_legend_handles_labels()
ax9_1.legend(handles_w1 + handles_w2, labels_w1 + labels_w2, loc="upper right")

fig9.tight_layout()
fig9.savefig("worst_fivecard_probs_and_combos.png", dpi=150, bbox_inches="tight")

# ----------------------------------------------------------------
# 9. Per-player bar charts:
#    P(win | hand) for each player count, worst vs standard
# ----------------------------------------------------------------
for i, n_players in enumerate(players):
    fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Worst Case
    worst_vals = [worst_p_win_given_hand[h][i] for h in worst_hand_types]
    x_w = np.arange(len(worst_hand_types))
    ax_w.bar(x_w, worst_vals, color="tab:orange")
    ax_w.set_xticks(x_w)
    ax_w.set_xticklabels(worst_hand_types, rotation=45, ha="right")
    ax_w.set_ylabel("P(win | hand) (%)")
    ax_w.set_title(f"Worst Case – {n_players} Players")

    # Standard
    standard_vals = [standard_p_win_given_hand[h][i] for h in standard_hand_types]
    x_s = np.arange(len(standard_hand_types))
    ax_s.bar(x_s, standard_vals, color="tab:green")
    ax_s.set_xticks(x_s)
    ax_s.set_xticklabels(standard_hand_types, rotation=45, ha="right")
    ax_s.set_title(f"Standard – {n_players} Players")

    fig.suptitle(f"P(win | hand) by Hand Type – {n_players} Players", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"per_player_{n_players}_pwin_given_hand_bars.png", dpi=150, bbox_inches="tight")

# Show all figures interactively
plt.show()
