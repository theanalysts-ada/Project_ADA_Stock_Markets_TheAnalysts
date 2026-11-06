import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import STUDY_START_YEAR


# Five sessions on either side show roughly one trading week before and after a spike.
# Return units: x100 for %, x10000 for bp; rounding only controls displayed precision.
def event_study(df, window=5, start_year=STUDY_START_YEAR, drop_flags=None):
    """Average night and day returns in the days around each volume spike.

    Args:
        df: Prices with ticker, date, r_night, r_day, spike and the flag columns.
        window: Number of days to look at before and after each spike.
        start_year: Data before this year is ignored.
        drop_flags: List of flag columns; returns on flagged rows are ignored.

    Returns:
        A table with, for each day and each return type: mean, median, 95% CI and number of events.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Rows we don't trust: before start_year, or with a quality flag
    bad = df["date"].dt.year < start_year
    if drop_flags is not None:
        for flag in drop_flags:
            bad = bad | df[flag]

    # Erase their returns but keep the rows, so that "day +1" is still the next trading day
    df.loc[bad, "r_night"] = np.nan
    df.loc[bad, "r_day"] = np.nan

    # The events are the spike days we trust
    is_event = df["spike"] & (bad == False)

    results = []
    for k in range(-window, window + 1):
        for col in ["r_night", "r_day"]:
            # Return k days after each row (k days before if k < 0), within the same ticker
            shifted = df.groupby("ticker", observed=True)[col].shift(-k)
            values = shifted[is_event].dropna()
            results.append({
                "day": k,
                "return": col,
                "mean": values.mean(),
                "median": values.median(),
                # 1.96 is the standard-normal cutoff for an approximate two-sided 95% interval.
                "ci95": 1.96 * values.std() / np.sqrt(len(values)),
                "n": len(values),
            })
    return pd.DataFrame(results)


def plot_event_study(res, stat="mean", title=None):
    """Plot the night and day returns around spikes.

    Args:
        res: Output of event_study.
        stat: "mean" (with 95% confidence intervals) or "median".
        title: Plot title.

    Returns:
        The figure.
    """
    night = res[res["return"] == "r_night"]
    day = res[res["return"] == "r_day"]

    fig, ax = plt.subplots(figsize=(8, 4))
    if stat == "mean":
        ax.errorbar(night["day"], night["mean"] * 100, yerr=night["ci95"] * 100, marker="o", capsize=3, label="Night")
        ax.errorbar(day["day"], day["mean"] * 100, yerr=day["ci95"] * 100, marker="o", capsize=3, label="Day")
    else:
        ax.plot(night["day"], night["median"] * 100, marker="o", label="Night")
        ax.plot(day["day"], day["median"] * 100, marker="o", label="Day")

    ax.axhline(0, color="grey")
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_xlabel("Days relative to the volume spike")
    ax.set_ylabel(stat + " return (%)")
    ax.set_title(title)
    ax.legend()
    return fig