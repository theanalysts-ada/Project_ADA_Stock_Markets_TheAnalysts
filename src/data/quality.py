import numpy as np
import pandas as pd

# Report data anomalies separately so filters can be compared.
FLAGS = [
    "open_missing", "open_equals_prev_close", "open_equals_close", "flat_day", "zero_volume", "extreme_return",
    "spliced_history",
]
# Missing opens or exact price equalities identify potentially stale opens.
STALE_OPEN_FLAGS = ["open_missing", "open_equals_prev_close", "open_equals_close"]
# Main liquidity screen, with stale-open and light alternatives for robustness.
FILTERS = {
    "liquidity": ["open_missing", "illiquid", "extreme_return", "spliced_history"],
    "stale": STALE_OPEN_FLAGS + ["extreme_return", "spliced_history"],
    "light": ["open_missing", "extreme_return", "spliced_history"],
}


def _prev_dollar_volume_rank(df: pd.DataFrame) -> pd.Series:
    """Percentile rank (0-1] of the previous day's dollar volume of each row, among all rows of the same year."""
    prev = df.groupby("ticker", observed=True)["dollar_volume"].shift(1)
    return prev.groupby(df["date"].dt.year).rank(pct=True)


def flag_suspicious(
    df: pd.DataFrame,
    extreme_threshold: float = 0.5,  # Heuristic: flag half-day moves above 50%.
    splice_factor: float = 10,  # Tenfold price jumps flag possible joined histories.
    min_price: float = 5,  # Exclude low-priced stocks from the main sample.
    min_dollar_volume_rank: float = 0.2,  # Drop the least-traded 20% within each year.
) -> pd.DataFrame:
    """Add one boolean column per type of suspicious observation, plus the illiquid flag, without removing rows.

    open_missing: the raw Open was 0 or negative and was blanked by clean_prices.
    open_equals_prev_close: Open(t) == Close(t-1) of the same ticker (overnight return exactly 0).
    open_equals_close: Open(t) == Close(t) (intraday return exactly 0).
    flat_day: open, high, low and close are all equal.
    zero_volume: no share traded.
    extreme_return: |r_night| or |r_day| above extreme_threshold.
    spliced_history: the row is before the ticker's last jump of the raw close by a factor of at least
        splice_factor from one row to the next (raw prices already include splits, so such a jump means
        the history of another security was attached to the ticker).
    illiquid (not in FLAGS): on the previous day, the close was below min_price or the dollar volume was in the
        bottom min_dollar_volume_rank of its year. Previous-day values are known before the night starts.

    Args:
        df: Table from add_returns and add_dollar_volume, sorted by ticker and date.
        extreme_threshold: Absolute simple return above which a return is called extreme.
        splice_factor: Price ratio between two consecutive rows above which the earlier history is flagged.
        min_price: Minimum previous close ($) of a liquid observation.
        min_dollar_volume_rank: Minimum yearly percentile rank of the previous dollar volume of a liquid observation.

    Returns:
        A copy of the table with one column per flag in FLAGS and the illiquid column.
    """
    df = df.copy()
    prev_close = df.groupby("ticker", observed=True)["close"].shift(1)
    df["open_missing"] = df["open"].isna()
    df["open_equals_prev_close"] = df["open"].eq(prev_close)
    df["open_equals_close"] = df["open"].eq(df["close"])
    df["flat_day"] = df["open_adj"].eq(df["high_adj"]) & df["high_adj"].eq(df["low_adj"]) & df["low_adj"].eq(df["close_adj"])
    df["zero_volume"] = df["volume"].eq(0)
    df["extreme_return"] = (df["r_night"].abs() > extreme_threshold) | (df["r_day"].abs() > extreme_threshold)
    ratio = df["close"] / prev_close
    jump_date = df["date"].where((ratio >= splice_factor) | (ratio <= 1 / splice_factor))
    last_jump = jump_date.groupby(df["ticker"], observed=True).transform("max")
    df["spliced_history"] = df["date"] < last_jump
    df["illiquid"] = (prev_close < min_price) | (_prev_dollar_volume_rank(df) <= min_dollar_volume_rank)
    return df


def suspicious_share_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Share of flagged observations per year and asset type (figure G3).

    Args:
        df: Table from flag_suspicious.

    Returns:
        Index (asset_type, year), one column per flag plus stale_open (union of STALE_OPEN_FLAGS).
    """
    flags = df[FLAGS].assign(stale_open=df[STALE_OPEN_FLAGS].any(axis=1))
    year = df["date"].dt.year.rename("year")
    return flags.groupby([df["asset_type"], year], observed=True).mean()


def flag_count_by_year(df: pd.DataFrame, flag: str) -> pd.Series:
    """Number of observations raising one flag, per year.

    Args:
        df: Table from flag_suspicious.
        flag: Name of the flag column.

    Returns:
        Number of flagged observations, indexed by year.
    """
    return df[flag].groupby(df["date"].dt.year.rename("year")).sum()


def stale_share_by_liquidity(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Share of stale opens (%) by quintile of the previous day's dollar volume and by previous close.

    Quintiles are computed within each year over all tickers (1 = least traded), so they do not drift with
    the growth of the market.

    Args:
        df: Table from flag_suspicious, with dollar_volume.
        start_year: First year included.

    Returns:
        One row per asset type; columns Q1 to Q5, then price < $5 and price >= $5 (previous close).
    """
    keep = df["date"].dt.year >= start_year
    stale = df.loc[keep, STALE_OPEN_FLAGS].any(axis=1)
    asset = df.loc[keep, "asset_type"]
    # Five equal-frequency liquidity groups, from least to most traded.
    quintile = np.ceil(5 * _prev_dollar_volume_rank(df)[keep])
    prev_close = df.groupby("ticker", observed=True)["close"].shift(1)[keep]
    # Use the same $5 price split as the default liquidity screen.
    price = (prev_close >= 5).map({False: "price < $5", True: "price >= $5"}).where(prev_close.notna())
    by_quintile = stale.groupby([asset, quintile], observed=True).mean().unstack()
    by_quintile.columns = [f"Q{q:.0f}" for q in by_quintile.columns]
    by_price = stale.groupby([asset, price], observed=True).mean().unstack()
    return (100 * pd.concat([by_quintile, by_price], axis=1)).round(1).rename_axis(index="asset_type")


def splice_examples(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Describe the last price jump that defines the spliced history of some tickers.

    Args:
        df: Table from flag_suspicious.
        tickers: Tickers to describe.

    Returns:
        One row per ticker: first date, date of the jump, close before and after it, and number of flagged rows.
    """
    rows = {}
    for ticker in tickers:
        t = df[df["ticker"] == ticker]
        jump = t["date"][t["spliced_history"]].max()
        after = t[t["date"] > jump].iloc[0]
        rows[ticker] = {
            "first date": t["date"].min().date(), "jump date": after["date"].date(),
            "close before": float(t.loc[t["date"] == jump, "close"].iloc[0]), "close after": float(after["close"]),
            "flagged rows": int(t["spliced_history"].sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def filter_summary(df: pd.DataFrame, start_year: int, filters: dict[str, list[str]] = FILTERS) -> pd.DataFrame:
    """Rows kept by each quality filter.

    Args:
        df: Table from flag_suspicious.
        start_year: First year of the study period.
        filters: Filter name -> flags whose observations are removed.

    Returns:
        One row per filter: rows kept, and their share (%) of all rows and of the rows of the study period.
    """
    in_period = (df["date"].dt.year >= start_year).sum()
    summary = {}
    for name, flags in filters.items():
        kept = _keep_mask(df, start_year, flags).sum()
        summary[name] = {"rows kept": kept, "% of all rows": 100 * kept / len(df), "% of study period": 100 * kept / in_period}
    return pd.DataFrame.from_dict(summary, orient="index").round(1)


def apply_quality_filter(df: pd.DataFrame, start_year: int, drop_flags: list[str]) -> pd.DataFrame:
    """Keep observations from start_year onwards that raise none of drop_flags.

    Args:
        df: Table from flag_suspicious.
        start_year: First year of the study period.
        drop_flags: Flags whose observations are removed.

    Returns:
        The filtered table.
    """
    return df[_keep_mask(df, start_year, drop_flags)]


def _keep_mask(df: pd.DataFrame, start_year: int, drop_flags: list[str]) -> pd.Series:
    """True for observations from start_year onwards that raise none of drop_flags."""
    return (df["date"].dt.year >= start_year) & ~df[drop_flags].any(axis=1)
