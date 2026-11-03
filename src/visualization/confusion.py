import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.plots import BUY_HOLD, DAY, STOCK


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


def plot_pair_abnormal_volume(case: pd.DataFrame, buzz: str, receiver: str) -> plt.Figure:
    """Abnormal volume of the well-known company (top) and of the ticker that may be bought by mistake (bottom).

    The spike days of the well-known company are marked in both panels.

    Args:
        case: Table returned by load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    top = case[case["ticker"] == buzz].set_index("date")
    bottom = case[case["ticker"] == receiver].set_index("date")["abn_volume"].where(lambda s: s > 0)
    spikes = top.loc[top["spike"], "abn_volume"]
    axes[0].plot(top.index, top["abn_volume"].where(top["abn_volume"] > 0), color=STOCK, lw=0.6)
    axes[0].scatter(spikes.index, spikes, color=DAY, s=14, zorder=3, label="attention spike")
    axes[1].plot(bottom.index, bottom, color=BUY_HOLD, lw=0.6)
    for when in spikes.index:
        axes[1].axvline(when, color=DAY, alpha=0.35, lw=0.8)
    for ax, ticker in zip(axes, (buzz, receiver)):
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set(yscale="log", ylabel=f"{ticker}: abnormal volume")
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("Date")
    fig.suptitle(f"Abnormal volume of {buzz} and {receiver}")
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


def plot_pair_episode(
    case: pd.DataFrame, buzz: str, receiver: str, start: str, end: str, events: dict[str, str],
) -> plt.Figure:
    """Daily trading volume of the two tickers over a short period, with dated events marked.

    Args:
        case: Table returned by load_pair.
        buzz: Ticker of the well-known company (volume in millions of shares).
        receiver: Ticker that may be bought by mistake (volume in thousands of shares).
        start: First day shown.
        end: Last day shown.
        events: Date (YYYY-MM-DD) -> short description of a documented event, drawn as a vertical line.

    Returns:
        The figure.
    """
    window = case[(case["date"] >= start) & (case["date"] <= end)]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 5.5))
    for ax, ticker, color, scale, unit in zip(axes, (buzz, receiver), (STOCK, BUY_HOLD), (1e6, 1e3), ("millions", "thousands")):
        rows = window[window["ticker"] == ticker]
        ax.bar(rows["date"], rows["volume"].astype("float64") / scale, color=color, width=0.8)
        ax.set_ylabel(f"{ticker}: shares traded\n({unit})")
    for height, (when, label) in zip((0.92, 0.74, 0.56), events.items()):
        for ax in axes:
            ax.axvline(pd.Timestamp(when), color=DAY, ls="--", lw=1.2)
        axes[0].text(pd.Timestamp(when), height, f" {label}", transform=axes[0].get_xaxis_transform(),
                     ha="left", va="top", fontsize=10, color=DAY)
    axes[1].set_xlabel("Date")
    fig.suptitle(f"Trading volume of {buzz} and {receiver} around a documented episode")
    return fig


def plot_pvalues(table: pd.DataFrame) -> plt.Figure:
    """Histogram of the placebo p-values of the pairs.

    Without any effect the p-values are spread evenly between 0 and 1; a real effect piles them up near 0.

    Args:
        table: First table returned by pooled_placebo_test.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(table["p"], bins=10, range=(0, 1), color=BUY_HOLD, edgecolor="white")
    ax.axhline(len(table) / 10, color=DAY, ls="--", lw=1.5, label="expected without any effect")
    ax.set(xlabel="Placebo p-value of a pair", ylabel="Number of pairs",
           title=f"p-values of the {len(table)} tested pairs")
    ax.legend()
    return fig
