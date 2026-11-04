import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plots import BUY_HOLD, DAY, NIGHT, STOCK


def plot_pairs_by_length(counts: pd.Series) -> plt.Figure:
    """Number of one-letter pairs of tickers by length of the shorter ticker.

    Args:
        counts: Series returned by pairs_by_min_length.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.to_numpy(), color=STOCK)
    for letters, n in counts.items():
        ax.text(letters, n, f"{n:,}\n({100 * n / counts.sum():.0f} %)", ha="center", va="bottom", fontsize=10)
    ax.set(xlabel="Letters in the shorter ticker of the pair", ylabel="Number of pairs",
           title="One-letter pairs of tickers", xticks=counts.index, ylim=(0, 1.2 * counts.max()))
    return fig


def plot_overnight_event_study(result: pd.DataFrame, usual: float | None = None, title: str | None = None) -> plt.Figure:
    """Mean overnight return of a ticker around the spikes of another one.

    Args:
        result: Table returned by event_study (the rows of the overnight return are used).
        usual: Usual overnight return of the ticker, drawn as a dashed line (a return: 0.01 is 1 %).
        title: Optional title.

    Returns:
        The figure.
    """
    night = result[result["return"] == "r_night"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(night["day"], night["mean"] * 100, yerr=night["ci95"] * 100, marker="o", capsize=3, color=NIGHT)
    if usual is not None:
        ax.axhline(usual * 100, color=DAY, ls="--", lw=1.5, label=f"usual overnight return ({usual * 100:.2f} %)")
        ax.legend()
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Trading days relative to the spike (day 0)", ylabel="Mean overnight return (%)",
           xticks=night["day"])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pair_density(case: pd.DataFrame, buzz: str, receiver: str) -> plt.Figure:
    """Density of the days in the plane (abnormal volume of the well-known company, abnormal volume of the receiver).

    The colour counts the days in each cell (log scale); the spike days of the well-known company are drawn on
    top as points.

    Args:
        case: Table returned by load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        The figure.
    """
    ratios = case.pivot(index="date", columns="ticker", values="abn_volume")[[buzz, receiver]]
    spikes = case[case["ticker"] == buzz].set_index("date")["spike"]
    ratios = ratios[(ratios > 0).all(axis=1)]
    spike = spikes.reindex(ratios.index, fill_value=False).astype(bool)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cells = ax.hexbin(ratios[buzz], ratios[receiver], gridsize=45, xscale="log", yscale="log", bins="log",
                      cmap="Blues", mincnt=1, linewidths=0.2)
    fig.colorbar(cells, ax=ax, label="Number of days (log scale)")
    ax.scatter(ratios.loc[spike, buzz], ratios.loc[spike, receiver], s=26, color=DAY, edgecolor="black", lw=0.5,
               label=f"{spike.sum()} spike days of {buzz}")
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.axvline(1, color="black", lw=0.8, ls="--")
    medians = (f"median {receiver} abnormal volume: {ratios.loc[spike, receiver].median():.2f} on spike days, "
               f"{ratios.loc[~spike, receiver].median():.2f} on other days")
    ax.set(xlabel=f"{buzz}: abnormal volume", ylabel=f"{receiver}: abnormal volume",
           title=f"Does {receiver} trade more when {buzz} does?")
    ax.text(0.5, -0.2, medians, transform=ax.transAxes, ha="center", fontsize=10)
    ax.legend(loc="lower right")
    return fig


def plot_placebo_scores(placebo_scores: np.ndarray, score: float, p: float, title: str | None = None) -> plt.Figure:
    """Scores obtained with each placebo company, and the observed score.

    Args:
        placebo_scores: Score obtained with each placebo company (a return: 0.01 is one percentage point).
        score: Observed score.
        p: One-sided p-value of the observed score.
        title: Optional title.

    Returns:
        The figure.
    """
    scores = np.asarray(placebo_scores, dtype="float64")
    scores = scores[~np.isnan(scores)] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(scores, bins=20, color=BUY_HOLD, edgecolor="white", label="placebo companies")
    ax.axvline(score * 100, color=DAY, lw=2.5, label=f"observed score: {score * 100:.2f} points (p = {p:.2f})")
    ax.set(xlabel="Score: excess overnight return (percentage points)", ylabel="Number of placebo companies")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
