import pandas as pd


def abnormal_volume(df: pd.DataFrame, window: int = 20, threshold: float = 3.0) -> pd.DataFrame:
    """Add abnormal volume and attention-spike flags, computed within each ticker.

    abn_volume(t) = volume(t) / mean(volume over the previous `window` rows of the ticker).
    Today is excluded from the mean, so a spike does not inflate its own baseline.
    The first `window` rows of each ticker, and days whose baseline is zero, get NaN.

    Args:
        df: Table from load_prices, with ticker, date and volume.
        window: Number of previous trading days in the baseline.
        threshold: A day is a spike when abn_volume is strictly above this value.

    Returns:
        The table sorted by ticker and date, with abn_volume (float64) and spike (bool) added.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    volume = df["volume"].astype("float64")
    prev = volume.groupby(df["ticker"], observed=True).shift(1)
    baseline = prev.groupby(df["ticker"], observed=True).transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    ratio = (volume / baseline).where(baseline > 0)
    return df.assign(abn_volume=ratio, spike=ratio.gt(threshold))
