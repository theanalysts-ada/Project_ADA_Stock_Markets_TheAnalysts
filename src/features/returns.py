import numpy as np
import pandas as pd


# Annualisation convention: approximately 252 US trading sessions per year.
# Source: https://www.cmegroup.com/education/files/Volatility_Update_0710.pdf
TRADING_DAYS_PER_YEAR = 252


def add_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Add split- and dividend-adjusted open, high, low and close prices.

    The adjustment factor Adj Close / Close of each day is applied to Open, High and Low.

    Args:
        df: Clean table with open, high, low, close and adj_close columns.

    Returns:
        A copy of the table with open_adj, high_adj, low_adj and close_adj added.
    """
    df = df.copy()
    factor = df["adj_close"] / df["close"]
    df["open_adj"] = factor * (df["open"])
    df["high_adj"] = factor * (df["high"])
    df["low_adj"] = factor * (df["low"])
    df["close_adj"] = df["adj_close"]
    return df


def add_dollar_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Add the dollar volume, close x volume, the value traded in dollars.

    Raw close and volume are both split-adjusted in the source data, so their product is consistent over time.

    Args:
        df: Table with close and volume.

    Returns:
        A copy of the table with dollar_volume added (float64).
    """
    return df.assign(dollar_volume=df["close"].astype("float64") * df["volume"].astype("float64"))


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add overnight and intraday simple returns, computed within each ticker.

    r_night(t) = open_adj(t) / close_adj(t-1) - 1, set to NaN when the ticker's previous
    row is not the previous trading day of the market (first row, or a removed row in between).
    r_day(t) = close_adj(t) / open_adj(t) - 1.

    Args:
        df: Table from load_prices, with ticker, date, open_adj and close_adj.

    Returns:
        The table sorted by ticker and date, with r_night and r_day added.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    prev = df.groupby("ticker", observed=True)[["date", "close_adj"]].shift(1)
    trading_days = np.sort(df["date"].unique())
    prev_trading_day = pd.Series(trading_days[:-1], index=trading_days[1:])
    consecutive = prev["date"].eq(df["date"].map(prev_trading_day))
    df["r_night"] = (df["open_adj"] / prev["close_adj"] - 1).where(consecutive)
    df["r_day"] = df["close_adj"] / df["open_adj"] - 1
    return df


def add_price_night(df: pd.DataFrame) -> pd.DataFrame:
    """Add the overnight return without dividends, from raw prices.

    r_night_price(t) = open(t) / close(t-1) - 1 on raw prices, which are adjusted for splits but
    not for dividends. On an ex-dividend day the price drop is then counted as a loss, so
    r_night - r_night_price is the dividend. Missing where r_night is missing.

    Args:
        df: Table from add_returns (sorted by ticker and date), with raw open and close.

    Returns:
        A copy of the table with r_night_price added.
    """
    prev_close = df.groupby("ticker", observed=True)["close"].shift(1)
    return df.assign(r_night_price=(df["open"] / prev_close - 1).where(df["r_night"].notna()))


# 20 sessions approximate a trading month and smooth one-day volume spikes.
def lagged_mean_dollar_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Mean dollar volume of the previous `window` trading days of each ticker (excludes today)."""
    prev = df.groupby("ticker", observed=True)["dollar_volume"].shift(1)
    return prev.groupby(df["ticker"], observed=True).transform(lambda s: s.rolling(window, min_periods=window).mean())


def portfolio_returns(df: pd.DataFrame, weighting: str = "equal") -> pd.DataFrame:
    """Daily portfolio overnight and intraday returns.

    equal: cross-sectional mean of the tickers available that day.
    dollar_volume: weighted by the mean dollar volume of the previous 20 trading days
    (known before the night; a stable liquidity weighting, not a market-cap weighting,
    and not a one-day attention spike).

    Args:
        df: Table with date, ticker, r_night, r_day and, for dollar_volume, dollar_volume.
        weighting: "equal" or "dollar_volume".

    Returns:
        One row per date, columns r_night and r_day.
    """
    cols = ["r_night", "r_day"]
    if weighting == "equal":
        return df.groupby("date")[cols].mean()
    if weighting == "dollar_volume":
        weight = lagged_mean_dollar_volume(df)
        work = df.assign(_w=weight)
        work = work[work["_w"].gt(0)]
        out = {}
        for col in cols:
            tmp = work.dropna(subset=[col])
            weighted_returns = tmp[col] * tmp["_w"]
            total_returns = weighted_returns.groupby(tmp["date"]).sum()
            total_weights = tmp["_w"].groupby(tmp["date"]).sum()
            out[col] = total_returns / total_weights
        return pd.DataFrame(out)
    raise ValueError(f"Unknown weighting: {weighting}")


def cumulative_wealth(returns: pd.Series) -> pd.Series:
    """Value of $1 invested along a return series: V_t = prod_{s<=t} (1 + r_s).

    Args:
        returns: Simple returns, in order.

    Returns:
        Cumulative wealth, starting at 1 + first return after the first observation.
    """
    return (1 + returns.astype("float64")).cumprod()
