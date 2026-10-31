import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.plots import BUY_HOLD, DAY, NIGHT


def plot_filing_times(profile: pd.DataFrame) -> plt.Figure:
    """Figure N1: when 8-K filings are accepted, over the New York clock.

    Args:
        profile: Table returned by filing_time_profile.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots()
    ax.axvspan(9.5, 16, color=DAY, alpha=0.12, label="Trading session (9:30–16:00)")
    width = 0.45
    ax.bar(profile.index + 0.25 - width / 4, profile["earnings"], width=width / 2, color=NIGHT, label="Earnings 8-Ks (item 2.02)")
    ax.bar(profile.index + 0.25 + width / 4, profile["other 8-K"], width=width / 2, color=BUY_HOLD, label="Other 8-Ks")
    ax.set(xlim=(0, 24), xticks=range(0, 25, 2), xlabel="Acceptance time, New York (30-minute bins)",
           ylabel="Share of filings of each kind (%)", title="When do firms file their news?")
    ax.legend(loc="upper left")
    return fig


def plot_news_wealth(paths: pd.DataFrame, title: str) -> plt.Figure:
    """Figure N2: $1 invested overnight or intraday, in all covered stock-days and in quiet ones.

    Args:
        paths: Table returned by news_paths.
        title: Figure title.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots()
    styles = {"all": ("-", "all covered stock-days"), "quiet": ("--", "quiet stock-days (no 8-K within ±2 sessions)")}
    for group, (ls, label) in styles.items():
        p = paths[paths["group"] == group]
        ax.plot(p["date"], p["night"], color=NIGHT, ls=ls, label=f"Overnight, {label}")
        ax.plot(p["date"], p["day"], color=DAY, ls=ls, label=f"Intraday, {label}")
    ax.set(yscale="log", title=title, xlabel="Date", ylabel="Value of $1 (log scale)")
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_gap_tests(tests: pd.DataFrame) -> plt.Figure:
    """Figure N3: annualised night − day gap with its clustered 95 % CI, by news status.

    Args:
        tests: Table returned by quiet_gap_tests.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = range(len(tests))[::-1]
    gap = tests["gap annualised (%)"]
    err = [gap - tests["CI low (clustered)"], tests["CI high (clustered)"] - gap]
    ax.errorbar(gap, y, xerr=err, fmt="o", color=NIGHT, capsize=4)
    ax.axvline(0, color=BUY_HOLD, ls="--", lw=1)
    ax.set(yticks=list(y), yticklabels=tests.index, xlabel="Night − day gap, annualised (%), 95 % CI clustered by date",
           title="Does the gap survive without news?")
    ax.grid(axis="y", visible=False)
    return fig


def plot_macro_paths(paths: dict[str, pd.DataFrame]) -> plt.Figure:
    """Figure N4: $1 invested overnight on all nights or skipping 8:30 release nights, and intraday.

    Args:
        paths: Panel title -> table returned by macro_paths.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, len(paths), sharey=True, figsize=(5.5 * len(paths), 4.5))
    for ax, (name, p) in zip(np.atleast_1d(axes), paths.items()):
        ax.plot(p["date"], p["night"], color=NIGHT, label="Overnight, all nights")
        ax.plot(p["date"], p["night without releases"], color=NIGHT, ls="--", label="Overnight, cash on jobs / CPI nights")
        ax.plot(p["date"], p["day"], color=DAY, label="Intraday")
        ax.set(yscale="log", title=name, xlabel="Date")
    np.atleast_1d(axes)[0].set_ylabel("Value of $1 (log scale)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=10)
    return fig
