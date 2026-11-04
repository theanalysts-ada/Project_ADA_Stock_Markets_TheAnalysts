import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.plots import BUY_HOLD, DAY, NIGHT


def plot_filing_times(profile: pd.DataFrame) -> plt.Figure:
    """Figure N1: time of the day at which 8-Ks are filed (New York time).

    Args:
        profile: Table returned by filing_time_profile.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots()
    ax.axvspan(9.5, 16, color=DAY, alpha=0.12, label="Trading session (9:30–16:00)")
    slots = profile.index
    ax.bar(slots + 0.14, profile["earnings"], width=0.22, color=NIGHT, label="Earnings 8-Ks (item 2.02)")
    ax.bar(slots + 0.36, profile["other 8-K"], width=0.22, color=BUY_HOLD, label="Other 8-Ks")
    ax.set(xlim=(0, 24), xticks=range(0, 25, 2), title="When do firms file their news?",
           xlabel="Filing time, New York (30-minute slots)", ylabel="Share of filings (%)")
    ax.legend(loc="upper left")
    return fig


def plot_news_wealth(paths: pd.DataFrame, title: str) -> plt.Figure:
    """Figure N2: value of $1 invested overnight only or intraday only, all and quiet stock-days.

    Args:
        paths: Table returned by news_paths.
        title: Figure title.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots()
    all_days = paths[paths["group"] == "all"]
    quiet = paths[paths["group"] == "quiet"]
    ax.plot(all_days["date"], all_days["night"], color=NIGHT, label="Overnight, all covered stock-days")
    ax.plot(all_days["date"], all_days["day"], color=DAY, label="Intraday, all covered stock-days")
    ax.plot(quiet["date"], quiet["night"], color=NIGHT, ls="--", label="Overnight, quiet stock-days")
    ax.plot(quiet["date"], quiet["day"], color=DAY, ls="--", label="Intraday, quiet stock-days")
    ax.set(yscale="log", title=title, xlabel="Date", ylabel="Value of $1 (log scale)")
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_gap_tests(tests: pd.DataFrame) -> plt.Figure:
    """Figure N3: night − day gap and its 95 % CI (clustered by date) for each news portfolio.

    Args:
        tests: Table returned by quiet_gap_tests.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = list(range(len(tests)))[::-1]
    gap = tests["gap annualised (%)"]
    errors = [gap - tests["CI low (clustered)"], tests["CI high (clustered)"] - gap]
    ax.errorbar(gap, y, xerr=errors, fmt="o", color=NIGHT, capsize=4)
    ax.axvline(0, color=BUY_HOLD, ls="--", lw=1)
    ax.set(yticks=y, yticklabels=tests.index, title="Does the gap survive without news?",
           xlabel="Night − day gap, annualised (%), 95 % CI clustered by date")
    ax.grid(axis="y", visible=False)
    return fig


def plot_macro_paths(paths: dict[str, pd.DataFrame]) -> plt.Figure:
    """Figure N4: value of $1 overnight on all nights, overnight without jobs / CPI nights, and intraday.

    Args:
        paths: Panel title -> table returned by macro_paths.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, len(paths), sharey=True, figsize=(5.5 * len(paths), 4.5), squeeze=False)
    for ax, (name, p) in zip(axes[0], paths.items()):
        ax.plot(p["date"], p["night"], color=NIGHT, label="Overnight, all nights")
        label = "Overnight, cash on jobs / CPI nights"
        ax.plot(p["date"], p["night without releases"], color=NIGHT, ls="--", label=label)
        ax.plot(p["date"], p["day"], color=DAY, label="Intraday")
        ax.set(yscale="log", title=name, xlabel="Date")
    axes[0][0].set_ylabel("Value of $1 (log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=10)
    return fig
