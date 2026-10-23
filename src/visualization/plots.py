import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NIGHT = "#1f3a68"
DAY = "#f28e2b"
BUY_HOLD = "#8c8c8c"
STOCK = "#2a9d8f"
ETF = "#9b5de5"


def set_style() -> None:
    """Apply the common figure style of the project (call once at the top of the notebook)."""
    plt.rcParams.update({
        "figure.figsize": (9, 4.5),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })


def plot_tickers_per_year(counts: pd.DataFrame) -> plt.Figure:
    """Figure D1: number of tickers available per year, stocks vs ETFs.

    Args:
        counts: Table returned by tickers_per_year.

    Returns:
        The figure.
    """
    counts = counts.where(counts > 0)  # zeros cannot be drawn on a log scale
    fig, ax = plt.subplots()
    ax.plot(counts.index, counts["stock"], color=STOCK, label="Stocks")
    ax.plot(counts.index, counts["etf"], color=ETF, label="ETFs")
    ax.axvspan(counts.index[-1] - 0.5, counts.index[-1] + 0.5, color=BUY_HOLD, alpha=0.2, label="2020: until 1 April")
    ax.set(yscale="log", title="Tickers available per year", xlabel="Year", ylabel="Number of tickers (log scale)")
    ax.legend()
    return fig


def plot_suspicious_share(share: pd.DataFrame, etf_start: int = 1993) -> plt.Figure:
    """Figure G3: share of suspicious observations per year, stocks vs ETFs.

    Args:
        share: Table returned by suspicious_share_by_year.
        etf_start: First year shown for ETFs (earlier years contain only one or two ETFs).

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 4.5))
    for ax, asset, title in zip(axes, ["stock", "etf"], ["Stocks", "ETFs"]):
        data = 100 * share.loc[asset]
        data = data[data.index >= (etf_start if asset == "etf" else data.index.min())]
        for flag in data.columns.drop("stale_open"):
            ax.plot(data.index, data[flag], lw=1.2, label=flag)
        ax.plot(data.index, data["stale_open"], color="black", lw=2.5, label="stale_open (any of the 3 open flags)")
        ax.axvline(2001, color=BUY_HOLD, ls="--", lw=1)
        ax.text(2001.5, 95, "2001 decimalisation", color=BUY_HOLD, fontsize=10)
        ax.set(title=title, xlabel="Year", ylim=(0, 100))
    axes[0].set_ylabel("Share of observations (%)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Suspicious observations per year")
    return fig


def plot_return_distributions(df: pd.DataFrame, limit: float = 20) -> plt.Figure:
    """Figure D2: distributions of overnight vs intraday returns, stocks and ETFs.

    Args:
        df: Table with asset_type, r_night and r_day.
        limit: Returns are shown between -limit % and +limit %.

    Returns:
        The figure.
    """
    bins = np.linspace(-limit, limit, 201)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 4.5))
    for ax, asset, title in zip(axes, ["stock", "etf"], ["Stocks", "ETFs"]):
        subset = df[df["asset_type"] == asset]
        for column, label, color in [("r_night", "Overnight", NIGHT), ("r_day", "Intraday", DAY)]:
            r = 100 * subset[column].dropna()
            ax.hist(r, bins=bins, weights=np.full(len(r), 100 / len(r)), histtype="step", lw=1.5, color=color, label=label)
        ax.set(title=title, yscale="log", xlabel="Daily return (%)")
    axes[0].set_ylabel("Share of observations (%, log scale)")
    axes[1].legend()
    fig.suptitle("Distributions of overnight and intraday returns")
    return fig


def plot_volume_distribution(df: pd.DataFrame) -> plt.Figure:
    """Figure D3: distribution of daily dollar volume (log scale), stocks vs ETFs.

    Args:
        df: Table with asset_type and dollar_volume.

    Returns:
        The figure.
    """
    bins = np.logspace(2, 12, 81)
    fig, ax = plt.subplots()
    for asset, label, color in [("stock", "Stocks", STOCK), ("etf", "ETFs", ETF)]:
        dollar = df.loc[df["asset_type"] == asset, "dollar_volume"].to_numpy(dtype="float64")
        zero_share = 100 * np.mean(dollar == 0)
        positive = dollar[dollar > 0]
        ax.hist(positive, bins=bins, weights=np.full(len(positive), 100 / len(dollar)), histtype="step", lw=1.5,
                color=color, label=f"{label}: {zero_share:.1f} % zero-volume days (not shown)")
    ax.set(xscale="log", title="Distribution of daily dollar volume",
           xlabel="Dollar volume (close × shares, log scale)", ylabel="Share of observations (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)
    return fig


CRISES = (
    (pd.Timestamp("2000-03-24"), "2000 dot-com"),
    (pd.Timestamp("2008-09-15"), "2008 crisis"),
    (pd.Timestamp("2020-02-19"), "2020 COVID"),
)


def plot_wealth(wealth: pd.DataFrame, title: str) -> plt.Figure:
    """Figure G1: value of $1 invested overnight only, intraday only, or buy-and-hold.

    Args:
        wealth: Table returned by ticker_wealth, with date, night, day and buy_hold.
        title: Figure title (includes the ticker and the period).

    Returns:
        The figure.
    """
    fig, ax = plt.subplots()
    ax.plot(wealth["date"], wealth["night"], color=NIGHT, label="Overnight only")
    ax.plot(wealth["date"], wealth["day"], color=DAY, label="Intraday only")
    ax.plot(wealth["date"], wealth["buy_hold"], color=BUY_HOLD, label="Buy and hold")
    ax.set(yscale="log", title=title, xlabel="Date", ylabel="Value of $1 (log scale)")
    for when, label in CRISES:
        if wealth["date"].min() <= when <= wealth["date"].max():
            ax.axvline(when, color=BUY_HOLD, ls="--", lw=1)
            if when.year == 2020:
                ax.text(when, 0.32, f"{label}  ", transform=ax.get_xaxis_transform(),
                        ha="right", va="top", fontsize=9, color=BUY_HOLD)
            else:
                ax.text(when, 0.94, f"  {label}", transform=ax.get_xaxis_transform(),
                        ha="left", va="top", fontsize=9, color=BUY_HOLD)
    ax.legend(loc="lower right")
    return fig


def plot_wealth_grid(series: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Overnight / intraday / buy-and-hold wealth for several portfolios.

    Args:
        series: Panel title -> table returned by strategy_paths.
        title: Super-title.

    Returns:
        The figure.
    """
    n = len(series)
    fig, axes = plt.subplots(1, n, sharey=True, figsize=(4.2 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, wealth) in zip(axes, series.items()):
        ax.plot(wealth["date"], wealth["night"], color=NIGHT, label="Overnight")
        ax.plot(wealth["date"], wealth["day"], color=DAY, label="Intraday")
        ax.plot(wealth["date"], wealth["buy_hold"], color=BUY_HOLD, label="Buy and hold")
        ax.set(yscale="log", title=name, xlabel="Date")
    axes[0].set_ylabel("Value of $1 (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.12)
    return fig


def plot_gap_histogram(gaps: pd.DataFrame) -> plt.Figure:
    """Figure G2: annualised night − day gap across tickers.

    Args:
        gaps: Table returned by night_day_gap_by_ticker.

    Returns:
        The figure.
    """
    lo, hi = -100, 200
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 4.5))
    for ax, asset, title, color in zip(axes, ["stock", "etf"], ["Stocks", "ETFs"], [STOCK, ETF]):
        values = gaps.loc[gaps["asset_type"] == asset, "gap"]
        outside = int(((values < lo) | (values > hi)).sum())
        shown = values[(values >= lo) & (values <= hi)]
        ax.hist(shown, bins=60, color=color, weights=np.full(len(shown), 100 / len(values)))
        ax.axvline(0, color=BUY_HOLD, ls="--", lw=1)
        ax.axvline(values.median(), color="black", lw=1.5, label=f"median ({values.median():.1f} %)")
        ax.set(title=f"{title} ({outside} tickers outside {lo} to {hi} %)", xlabel="Annualised night − day gap (%)",
               xlim=(lo, hi))
        ax.legend(loc="upper right", fontsize=10)
    axes[0].set_ylabel("Share of tickers (%)")
    fig.suptitle("Does the night beat the day? Gap across tickers")
    return fig


def plot_gap_heatmap(table: pd.DataFrame) -> plt.Figure:
    """Figure G5: night − day gap by year and group.

    Args:
        table: Table returned by gap_heatmap_table.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    values = table.to_numpy(dtype="float64")
    limit = np.nanpercentile(np.abs(values), 95)
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set(
        xticks=range(table.shape[1]), xticklabels=table.columns,
        yticks=range(0, table.shape[0], 2), yticklabels=table.index[::2],
        xlabel="Group", ylabel="Year", title="Annualised night − day gap (%)",
    )
    fig.colorbar(image, ax=ax, label="Night − day (annualised, %)")
    return fig


def plot_filter_comparison(table: pd.DataFrame) -> plt.Figure:
    """Figure G4: annualised night − day gap of SPY and equal-weighted stocks, by filter.

    Args:
        table: Table returned by filter_comparison.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots()
    x = np.arange(len(table))
    width = 0.35
    ax.bar(x - width / 2, table["SPY gap"], width, color=BUY_HOLD, label="SPY")
    ax.bar(x + width / 2, table["EW stocks gap"], width, color=STOCK, label="Equal-weighted stocks")
    ax.axhline(0, color=BUY_HOLD, lw=1)
    ax.set(xticks=x, xticklabels=table.index, ylabel="Annualised night − day gap (%)",
           title="Does the gap survive the quality filters?")
    ax.legend()
    return fig
