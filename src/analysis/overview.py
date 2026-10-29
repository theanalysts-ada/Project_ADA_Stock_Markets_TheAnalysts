import random
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from src.config import DATA_DIR, DROPPED_FILE, PRICES_FILE
from src.data.load import load_prices, read_one_csv


def tickers_per_year(prices: pd.DataFrame) -> pd.DataFrame:
    """Count the distinct tickers with at least one observation in each year.

    Args:
        prices: Table from load_prices, with ticker, date and asset_type.

    Returns:
        One row per year, columns stock and etf.
    """
    year = prices["date"].dt.year.rename("year")
    counts = prices.groupby([year, "asset_type"], observed=True)["ticker"].nunique().unstack(fill_value=0)
    return counts[["stock", "etf"]].rename_axis(columns=None)


def first_dates(prices: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find the dates on which the most tickers have their first observation.

    Many tickers starting on the same day reveal a change in data coverage, not real listings.

    Args:
        prices: Table from load_prices, with ticker and date.
        n: Number of dates to return.

    Returns:
        The n most frequent first dates, with the number of tickers starting on each.
    """
    first = prices.groupby("ticker", observed=True)["date"].min()
    return first.value_counts().head(n).rename_axis("first_date").rename("n_tickers").to_frame()


def removed_rows_summary(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Count the rows removed by clean_prices, by reason.

    Args:
        data_dir: Folder containing the Parquet files.

    Returns:
        One row per reason plus a total row, with the number of rows, their share of the raw data (%)
        and the number of tickers concerned.
    """
    dropped = pd.read_parquet(data_dir / DROPPED_FILE, columns=["ticker", "reason"])
    raw_rows = len(dropped) + pq.read_metadata(data_dir / PRICES_FILE).num_rows
    summary = dropped["reason"].value_counts().rename("rows").to_frame()
    summary.loc["total"] = summary["rows"].sum()
    summary["share (%)"] = (100 * summary["rows"] / raw_rows).round(3)
    tickers = dropped.groupby("reason")["ticker"].nunique()
    summary["tickers"] = tickers.reindex(summary.index).fillna(dropped["ticker"].nunique()).astype(int)
    return summary.rename_axis("reason")


def adjustment_example(prices: pd.DataFrame, ticker: str = "AAPL", date: str = "2020-02-07") -> pd.DataFrame:
    """Compare the raw and the adjusted overnight return of a ticker on one day (e.g. an ex-dividend date).

    Args:
        prices: Table from load_prices, with raw (open, close) and adjusted (open_adj, close_adj) prices.
        ticker: Ticker of the example.
        date: Day of the example.

    Returns:
        Rows raw and adjusted, with the previous close, the open and the overnight return (%).
    """
    rows = prices[prices["ticker"] == ticker].set_index("date").sort_index()
    i = rows.index.get_loc(pd.Timestamp(date))
    prev, day = rows.iloc[i - 1], rows.iloc[i]
    previous_close = [float(prev["close"]), float(prev["close_adj"])]
    open_ = [float(day["open"]), float(day["open_adj"])]
    overnight = [100 * (o / c - 1) for o, c in zip(open_, previous_close)]
    table = {"previous close": previous_close, "open": open_, "overnight return (%)": overnight}
    return pd.DataFrame(table, index=["raw", "adjusted"]).round(2)


RETURNS = {"r_night": "night", "r_day": "day"}
ASSETS = ["stock", "etf"]


def missing_values(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Count the missing values per column.

    Args:
        df: Any table.
        columns: Columns to inspect (default: all).

    Returns:
        One row per column, with the number and the share (%) of missing values.
    """
    missing = df[columns or df.columns].isna()
    return pd.DataFrame({"missing": missing.sum(), "share (%)": (100 * missing.mean()).round(2)})


def return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the distribution of overnight and intraday returns, for stocks and ETFs.

    Args:
        df: Table with asset_type, r_night and r_day.

    Returns:
        One row per (asset_type, return): number of observations, mean, median, standard deviation,
        1st and 99th percentiles (all in %), skewness and excess kurtosis.
    """
    stats = {}
    for asset in ASSETS:
        subset = df[df["asset_type"] == asset]
        for column, name in RETURNS.items():
            r = 100 * subset[column].dropna().astype("float64")
            stats[(asset, name)] = {
                "n": len(r), "mean (%)": r.mean(), "median (%)": r.median(), "std (%)": r.std(),
                "p1 (%)": r.quantile(0.01), "p99 (%)": r.quantile(0.99),
                "skewness": r.skew(), "excess kurtosis": r.kurt(),
            }
    table = pd.DataFrame.from_dict(stats, orient="index").rename_axis(["asset_type", "return"])
    return table.round(3)


def night_day_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the overnight and the intraday return of the same day, for stocks and ETFs.

    Args:
        df: Table with asset_type, r_night and r_day.

    Returns:
        One row per asset type, with the Pearson and the Spearman correlation.
    """
    correlations = {}
    for asset in ASSETS:
        pair = df.loc[df["asset_type"] == asset, ["r_night", "r_day"]].dropna()
        correlations[asset] = {
            "pearson": pair["r_night"].corr(pair["r_day"]),
            "spearman": pair["r_night"].corr(pair["r_day"], method="spearman"),
        }
    return pd.DataFrame.from_dict(correlations, orient="index").rename_axis("asset_type").round(3)


def volume_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise daily dollar volume (close × volume), for stocks and ETFs.

    Args:
        df: Table with asset_type and dollar_volume.

    Returns:
        One row per asset type: share of zero-volume days (%) and, over days with a positive volume,
        the 1st percentile, median and 99th percentile ($).
    """
    summary = {}
    for asset in ASSETS:
        dollar = df.loc[df["asset_type"] == asset, "dollar_volume"].to_numpy(dtype="float64")
        positive = dollar[dollar > 0]
        summary[asset] = {
            "zero-volume days (%)": round(100 * np.mean(dollar == 0), 1),
            "p1 ($)": int(np.percentile(positive, 1)), "median ($)": int(np.median(positive)),
            "p99 ($)": int(np.percentile(positive, 99)),
        }
    return pd.DataFrame.from_dict(summary, orient="index").rename_axis("asset_type")


def stale_among_zero_volume(df: pd.DataFrame) -> pd.Series:
    """Share of zero-volume days that also have a stale open.

    Args:
        df: Table from flag_suspicious.

    Returns:
        Number of zero-volume days and the share (%) that have a stale open.
    """
    from src.data.quality import STALE_OPEN_FLAGS
    zero = df["zero_volume"]
    stale = df.loc[zero, STALE_OPEN_FLAGS].any(axis=1)
    return pd.Series({"zero-volume days": int(zero.sum()), "of which stale open (%)": round(100 * stale.mean(), 1)})


def _read_sample(paths: list[Path], optimised: bool) -> tuple[pd.DataFrame, float]:
    """Read CSVs with default or optimised dtypes and return the table and the elapsed time (s)."""
    start = time.perf_counter()
    if optimised:
        df = pd.concat([read_one_csv(p, p.parent.name.rstrip("s")) for p in paths], ignore_index=True)
        df[["ticker", "asset_type"]] = df[["ticker", "asset_type"]].astype("category")
    else:
        df = pd.concat([pd.read_csv(p).assign(ticker=p.stem, asset_type=p.parent.name) for p in paths], ignore_index=True)
    return df, time.perf_counter() - start


def size_report(data_dir: Path = DATA_DIR, n_sample: int = 500, seed: int = 0) -> pd.DataFrame:
    """Compare rows, memory, loading time and disk size of the raw CSVs and of the Parquet file.

    CSV memory and loading time are measured on a random sample of files and scaled to all files;
    row counts, disk sizes and Parquet figures are exact.

    Args:
        data_dir: Folder containing stocks/, etfs/ and the Parquet files.
        n_sample: Number of CSV files in the sample.
        seed: Random seed of the sample.

    Returns:
        One row per format, one column per metric.
    """
    paths = sorted((data_dir / "stocks").glob("*.csv")) + sorted((data_dir / "etfs").glob("*.csv"))
    sample = random.Random(seed).sample(paths, n_sample)
    scale = len(paths) / n_sample

    start = time.perf_counter()
    prices = load_prices(data_dir=data_dir)
    parquet_time = time.perf_counter() - start
    raw_rows = len(prices) + len(pd.read_parquet(data_dir / DROPPED_FILE, columns=["date"]))
    csv_disk = sum(p.stat().st_size for p in paths)

    columns = {"CSV, default dtypes": False, "CSV, optimised dtypes": True}
    report = {}
    for name, optimised in columns.items():
        df, elapsed = _read_sample(sample, optimised)
        memory = df.memory_usage(deep=True).sum() / len(df) * raw_rows
        report[name] = [raw_rows, memory / 1e9, elapsed * scale, csv_disk / 1e9]
    parquet_disk = (data_dir / PRICES_FILE).stat().st_size
    report["Parquet"] = [len(prices), prices.memory_usage(deep=True).sum() / 1e9, parquet_time, parquet_disk / 1e9]
    metrics = ["rows", "memory (GB)", "loading time (s)", "disk size (GB)"]
    return pd.DataFrame.from_dict(report, orient="index", columns=metrics).round(2)


def missing_sessions_by_ticker(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Internal missing sessions between each ticker's first/last observation.

    Use the unfiltered price table restricted to the study period: liquidity-screen
    exclusions are sample choices, not missing source data. No pre-IPO/post-exit
    dates are counted. Gaps are not automatically provider errors or suspensions.
    """
    sessions = pd.DatetimeIndex(calendar["date"].drop_duplicates().sort_values())
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date observations")
    records = []
    for ticker, frame in prices.groupby("ticker", observed=True):
        observed = pd.DatetimeIndex(frame["date"].unique()).sort_values()
        expected = sessions[(sessions >= observed.min()) & (sessions <= observed.max())]
        missing = expected.difference(observed)
        records.append({"ticker": ticker, "first_date": observed.min(), "last_date": observed.max(),
                        "expected_sessions": len(expected), "missing_sessions": len(missing),
                        "missing_share": len(missing) / len(expected) if len(expected) else np.nan,
                        "non_session_dates": len(observed.difference(sessions))})
    return pd.DataFrame(records).sort_values("missing_sessions", ascending=False).reset_index(drop=True)
