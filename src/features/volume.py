import numpy as np
import pandas as pd


# 20 sessions approximate a trading month; 3x is a heuristic attention-spike cutoff.
def abnormal_volume(df, window=20, threshold=3.0):
    """Each day's volume divided by the average volume of the previous days, for each ticker.

    Args:
        df: Prices with ticker, date and volume.
        window: Number of previous days in the average.
        threshold: Above this ratio, the day is a spike.

    Returns:
        df with two new columns: abn_volume (the ratio) and spike (True or False).
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    volume = df["volume"].astype(float)

    def mean_of_previous_days(v):
        # shift(1): each day looks at the day before, so today is not in its own average
        return v.shift(1).rolling(window).mean()

    # Average of the previous days, computed separately for each ticker
    baseline = volume.groupby(df["ticker"], observed=True).transform(mean_of_previous_days)

    df["abn_volume"] = volume / baseline
    # If the average is 0, we would divide by zero: no value
    df.loc[baseline == 0, "abn_volume"] = np.nan
    df["spike"] = df["abn_volume"] > threshold
    return df