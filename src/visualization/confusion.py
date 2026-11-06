import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import BUY_HOLD, DAY, NIGHT, STOCK


def plot_pairs_by_length(counts):
    """Bar chart of the number of one-letter pairs by length of the shorter ticker.

    Args:
        counts: Output of pairs_by_min_length.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values, color=STOCK)

    # Write the number of pairs and its share above each bar
    total = counts.sum()
    for letters, n in counts.items():
        share = 100 * n / total
        ax.text(letters, n, f"{n:,}\n({share:.0f} %)", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Letters in the shorter ticker of the pair")
    ax.set_ylabel("Number of pairs")
    ax.set_title("One-letter pairs of tickers")
    ax.set_xticks(counts.index)
    ax.set_ylim(0, 1.2 * counts.max())  # leave room for the text above the bars
    return fig


def plot_overnight_event_study(result, usual=None, title=None):
    """Mean overnight return of a ticker around the spikes of another ticker.

    Args:
        result: Output of event_study (only the overnight rows are used).
        usual: Usual overnight return of the ticker, drawn as a dashed line (0.01 = 1 %).
        title: Plot title.

    Returns:
        The figure.
    """
    night = result[result["return"] == "r_night"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(night["day"], night["mean"] * 100, yerr=night["ci95"] * 100, marker="o", capsize=3, color=NIGHT)

    # Dashed line at the usual overnight return, to see whether the nights after a spike are above it
    if usual is not None:
        ax.axhline(usual * 100, color=DAY, ls="--", lw=1.5, label=f"usual overnight return ({usual * 100:.2f} %)")
        ax.legend()

    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Trading days relative to the spike (day 0)")
    ax.set_ylabel("Mean overnight return (%)")
    ax.set_xticks(night["day"])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pair_density(case, buzz, receiver):
    """Each day as a point (abnormal volume of the well-known company, abnormal volume of the receiver).

    Many days are drawn as a density map; the spike days of the well-known company are drawn on top as dots.

    Args:
        case: Output of load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        The figure.
    """
    # One row per date, one column per ticker; keep the days where both values are positive (log scale)
    ratios = case.pivot(index="date", columns="ticker", values="abn_volume")[[buzz, receiver]]
    ratios = ratios[(ratios[buzz] > 0) & (ratios[receiver] > 0)]

    # Spike days of the well-known company, on the same dates
    buzz_spikes = case[case["ticker"] == buzz].set_index("date")["spike"]
    is_spike = buzz_spikes.reindex(ratios.index) == True
    spike_days = ratios[is_spike]
    other_days = ratios[~is_spike]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Density map: the colour counts the days in each small hexagon
    cells = ax.hexbin(ratios[buzz], ratios[receiver], gridsize=45, xscale="log", yscale="log", bins="log",
                      cmap="Blues", mincnt=1, linewidths=0.2)
    fig.colorbar(cells, ax=ax, label="Number of days (log scale)")

    # Spike days of the well-known company
    ax.scatter(spike_days[buzz], spike_days[receiver], s=26, color=DAY, edgecolor="black", lw=0.5,
               label=f"{len(spike_days)} spike days of {buzz}")

    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.axvline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel(buzz + ": abnormal volume")
    ax.set_ylabel(receiver + ": abnormal volume")
    ax.set_title(f"Does {receiver} trade more when {buzz} does?")
    ax.legend(loc="lower right")

    # Under the plot: median abnormal volume of the receiver on spike days and on other days
    text = (f"median {receiver} abnormal volume: {spike_days[receiver].median():.2f} on spike days, "
            f"{other_days[receiver].median():.2f} on other days")
    ax.text(0.5, -0.2, text, transform=ax.transAxes, ha="center", fontsize=10)
    return fig


def plot_placebo_scores(placebo_scores, score, p, title=None):
    """Histogram of the scores of the placebo companies, with the observed score as a vertical line.

    Args:
        placebo_scores: Score of each placebo company (0.01 = one percentage point).
        score: Observed score.
        p: One-sided p-value of the observed score.
        title: Plot title.

    Returns:
        The figure.
    """
    # Remove the missing scores and convert to percentage points
    scores = np.array(placebo_scores, dtype=float)
    scores = scores[~np.isnan(scores)] * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(scores, bins=20, color=BUY_HOLD, edgecolor="white", label="placebo companies")

    # If the real score is far to the right of the placebos, the effect is not due to chance
    ax.axvline(score * 100, color=DAY, lw=2.5, label=f"observed score: {score * 100:.2f} points (p = {p:.2f})")

    ax.set_xlabel("Score: excess overnight return (percentage points)")
    ax.set_ylabel("Number of placebo companies")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig