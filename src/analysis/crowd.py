import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import STUDY_START_YEAR


def event_study(
    df: pd.DataFrame, window: int = 5, start_year: int = STUDY_START_YEAR, drop_flags: list[str] | None = None,
) -> pd.DataFrame:
    """Average overnight and intraday returns around attention spikes.

    Day 0 is the spike day. Day k is the k-th trading day of the same ticker after
    (k > 0) or before (k < 0) the spike. Under the hangover hypothesis, the night
    return of day +1 is positive and the day return of day +1 is negative.

    Args:
        df: Table with ticker, date, r_night, r_day and spike
            (load_prices, add_returns, add_dollar_volume, flag_suspicious, then abnormal_volume),
            plus the flag columns named in drop_flags.
        window: Number of trading days kept on each side of the spike.
        start_year: Spikes and returns before this year are ignored.
        drop_flags: Quality flags (from flag_suspicious, e.g. FILTERS["liquidity"]). A flagged row
            cannot be an event and its returns are ignored; rows are blanked, not removed, so that
            day k stays the k-th trading day of the ticker.

    Returns:
        One row per (day, return): mean, median, half-width of the 95% confidence
        interval of the mean, and number of events used.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    keep = df["date"].dt.year >= start_year
    if drop_flags:
        keep &= ~df[drop_flags].any(axis=1)
    returns = df[["r_night", "r_day"]].where(keep)
    events = df["spike"] & keep
    grouped = returns.groupby(df["ticker"], observed=True)
    rows = []
    for k in range(-window, window + 1):
        for col in ["r_night", "r_day"]:
            values = grouped[col].shift(-k)[events].dropna().astype("float64")
            rows.append({
                "day": k,
                "return": col,
                "mean": values.mean(),
                "median": values.median(),
                "ci95": 1.96 * values.std() / np.sqrt(len(values)),
                "n": len(values),
            })
    return pd.DataFrame(rows)


def plot_event_study(result: pd.DataFrame, stat: str = "mean", title: str | None = None) -> plt.Figure:
    """Plot night and day returns around spikes, from the output of event_study.

    Args:
        result: Output of event_study.
        stat: "mean" (with 95% confidence intervals) or "median".
        title: Optional plot title.

    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [("r_night", "Night (close to open)"), ("r_day", "Day (open to close)")]:
        part = result[result["return"] == col]
        yerr = part["ci95"] * 100 if stat == "mean" else None
        ax.errorbar(part["day"], part[stat] * 100, yerr=yerr, marker="o", capsize=3, label=label)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Trading days relative to the volume spike (day 0)")
    ax.set_ylabel(f"{stat.capitalize()} return (%)")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig