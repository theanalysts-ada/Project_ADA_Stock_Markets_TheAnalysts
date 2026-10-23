import pandas as pd
import numpy as np


def clean_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove broken rows and blank out impossible values.

    Removed rows: missing values, Close <= 0, negative Adj Close.
    Kept rows with blanked values: Open <= 0 (open set to NaN), High < Low (both set to NaN).

    Args:
        df: Raw long table returned by read_all_csv.

    Returns:
        (clean, dropped): the kept rows, and the removed rows with a "reason" column.
    """
    missing = df[["open", "close", "adj_close", "volume"]].isna().any(axis=1)
    close_zero = df["close"] <= 0
    negative_adj = df["adj_close"] < 0
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[negative_adj] = "negative_adj"
    reason[close_zero] = "close_zero"
    reason[missing] = "missing"
    to_drop = reason.notna()
    dropped = df[to_drop].assign(reason=reason[to_drop])
    clean = df[~to_drop].copy()
    clean.loc[clean["open"] <= 0, "open"] = np.nan

    bad_range = clean["high"] < clean["low"]
    clean.loc[bad_range, ["high", "low"]] = np.nan
    return clean, dropped
