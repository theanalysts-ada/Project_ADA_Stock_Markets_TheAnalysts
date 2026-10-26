import numpy as np
import pandas as pd

from src.features.returns import TRADING_DAYS_PER_YEAR, cumulative_wealth


def ticker_wealth(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Night-only, day-only and buy-and-hold wealth of $1 for one ticker.

    The three paths use the same dates: a day is kept only when both r_night and r_day are observed.
    Buy-and-hold compounds (1 + r_night) × (1 + r_day) each day.

    Args:
        df: Table with ticker, date, r_night and r_day.
        ticker: Ticker to keep.

    Returns:
        One row per date, columns night, day and buy_hold.
    """
    pair = df.loc[df["ticker"] == ticker, ["date", "r_night", "r_day"]].dropna()
    return pd.DataFrame({
        "date": pair["date"].to_numpy(),
        "night": cumulative_wealth(pair["r_night"]).to_numpy(),
        "day": cumulative_wealth(pair["r_day"]).to_numpy(),
        "buy_hold": cumulative_wealth((1 + pair["r_night"]) * (1 + pair["r_day"]) - 1).to_numpy(),
    })


def crime_headline(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Headline numbers of the night / day / buy-and-hold strategies for one ticker.

    Args:
        df: Table with ticker, date, r_night and r_day.
        ticker: Ticker to keep.

    Returns:
        One row per strategy: first and last date, number of days, final value of $1,
        annualised mean return (252 × daily mean, %) and share of the total log return (%).
    """
    pair = df.loc[df["ticker"] == ticker, ["date", "r_night", "r_day"]].dropna().astype({"r_night": "float64", "r_day": "float64"})
    night, day = pair["r_night"], pair["r_day"]
    hold = (1 + night) * (1 + day) - 1
    lr_night, lr_day = np.log1p(night), np.log1p(day)
    total_log = lr_night.sum() + lr_day.sum()
    strategies = {"overnight": night, "intraday": day, "buy-and-hold": hold}
    shares = {"overnight": lr_night.sum() / total_log, "intraday": lr_day.sum() / total_log, "buy-and-hold": 1.0}
    rows = {}
    for name, r in strategies.items():
        rows[name] = {
            "first date": pair["date"].iloc[0].date(),
            "last date": pair["date"].iloc[-1].date(),
            "days": len(r),
            "value of $1": (1 + r).prod(),
            "annualised mean (%)": 100 * TRADING_DAYS_PER_YEAR * r.mean(),
            "share of log return (%)": 100 * shares[name],
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def strategy_paths(pair: pd.DataFrame) -> pd.DataFrame:
    """Night-only, day-only and buy-and-hold wealth of $1 from a daily return table.

    Args:
        pair: Table with date, r_night and r_day (e.g. portfolio_returns reset_index).

    Returns:
        One row per date, columns night, day and buy_hold.
    """
    pair = pair[["date", "r_night", "r_day"]].dropna()
    dummy = pair.assign(ticker="portfolio")
    return ticker_wealth(dummy, "portfolio")


def strategy_headline(pair: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers from a daily return table (same columns as crime_headline)."""
    pair = pair[["date", "r_night", "r_day"]].dropna()
    dummy = pair.assign(ticker="portfolio")
    return crime_headline(dummy, "portfolio")


MIN_OBS = 252


def study_portfolios(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Equal-weighted and dollar-volume-weighted overnight/intraday returns, stocks and ETFs.

    Args:
        df: Study sample (special ETFs already excluded).

    Returns:
        Four daily return tables, keyed by a short panel title.
    """
    from src.features.returns import portfolio_returns

    stocks, etfs = df["asset_type"] == "stock", df["asset_type"] == "etf"
    return {
        "Stocks, equal-weighted": portfolio_returns(df[stocks], "equal").reset_index(),
        "Stocks, dollar-volume": portfolio_returns(df[stocks], "dollar_volume").reset_index(),
        "ETFs, equal-weighted": portfolio_returns(df[etfs], "equal").reset_index(),
        "ETFs, dollar-volume": portfolio_returns(df[etfs], "dollar_volume").reset_index(),
    }


def portfolio_paths(portfolios: dict[str, pd.DataFrame], names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cumulative wealth of $1 for a subset of study_portfolios."""
    return {name: strategy_paths(portfolios[name]) for name in (names or portfolios)}


def portfolio_headlines(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack strategy_headline for each portfolio."""
    parts = []
    for name, pair in portfolios.items():
        table = strategy_headline(pair)
        table.index = pd.MultiIndex.from_product([[name], table.index], names=["portfolio", "strategy"])
        parts.append(table)
    return pd.concat(parts)


def night_day_gap_by_ticker(df: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Annualised night − day gap of each ticker with at least min_obs paired returns.

    Args:
        df: Table with ticker, asset_type, r_night and r_day.
        min_obs: Minimum number of days with both returns.

    Returns:
        One row per ticker: n, asset_type, and gap (annualised, %).
    """
    pair = df[["ticker", "asset_type", "r_night", "r_day"]].dropna().astype({"r_night": "float64", "r_day": "float64"})
    stats = pair.groupby("ticker", observed=True).agg(
        n=("r_night", "size"),
        asset_type=("asset_type", "first"),
        night=("r_night", "mean"),
        day=("r_day", "mean"),
    )
    stats = stats[stats["n"] >= min_obs]
    stats["gap"] = 100 * TRADING_DAYS_PER_YEAR * (stats["night"] - stats["day"])
    return stats


def gap_share_above_zero(gaps: pd.DataFrame) -> pd.DataFrame:
    """Share of tickers whose annualised night − day gap is positive.

    Args:
        gaps: Table returned by night_day_gap_by_ticker.

    Returns:
        One row per asset type: number of tickers and the share (%) with gap > 0.
    """
    rows = {}
    for asset in gaps["asset_type"].unique():
        subset = gaps[gaps["asset_type"] == asset]
        rows[asset] = {
            "tickers": len(subset),
            "night > day (%)": 100 * (subset["gap"] > 0).mean(),
            "median gap (%)": subset["gap"].median(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(1).rename_axis("asset_type")


def gap_heatmap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised night − day gap (%) by year and group (stocks, ETFs, dollar-volume quintiles).

    Quintiles use the 20-day lagged mean dollar volume, ranked within each year (Q1 = least traded).
    2020 is omitted: the file ends on 1 April, so an annualised 2020 gap would be a Q1 crash, not a year.

    Args:
        df: Table with date, asset_type, r_night, r_day, dollar_volume and ticker.

    Returns:
        Rows = years, columns = Stocks, ETFs, Q1–Q5.
    """
    keep = df["r_night"].notna() & df["r_day"].notna()
    year = df.loc[keep, "date"].dt.year
    night = df.loc[keep, "r_night"].astype("float64")
    day = df.loc[keep, "r_day"].astype("float64")
    asset = df.loc[keep, "asset_type"]
    from src.features.returns import lagged_mean_dollar_volume
    weight = lagged_mean_dollar_volume(df).loc[keep]
    quintile = np.ceil(5 * weight.groupby(year).rank(pct=True))
    scale = 100 * TRADING_DAYS_PER_YEAR

    def gap(n: pd.Series, d: pd.Series, by) -> pd.Series:
        return scale * (n.groupby(by, observed=True).mean() - d.groupby(by, observed=True).mean())

    table = pd.DataFrame({
        "Stocks": gap(night[asset == "stock"], day[asset == "stock"], year[asset == "stock"]),
        "ETFs": gap(night[asset == "etf"], day[asset == "etf"], year[asset == "etf"]),
    })
    by_q = gap(night, day, [year, quintile]).unstack()
    by_q.columns = [f"Q{int(c)}" for c in by_q.columns]
    table = pd.concat([table, by_q], axis=1).rename_axis("year")
    return table.loc[table.index < 2020].round(1)


def filter_comparison(prices: pd.DataFrame, start_year: int, filters: dict[str, list[str]]) -> pd.DataFrame:
    """Annualised night and day returns of SPY and of equal-weighted stocks, under each filter.

    Args:
        prices: Full flagged table (all years).
        start_year: First year of the study period.
        filters: Name -> drop_flags, as in quality.FILTERS.

    Returns:
        One row per filter: SPY and equal-weighted stock night / day / gap (annualised, %).
    """
    from src.data.quality import apply_quality_filter
    from src.features.returns import portfolio_returns

    slim = prices[["ticker", "date", "asset_type", "r_night", "r_day", *{f for flags in filters.values() for f in flags}]]
    rows = {}
    for name, flags in filters.items():
        sample = apply_quality_filter(slim, start_year, flags)
        spy = sample.loc[sample["ticker"] == "SPY", ["r_night", "r_day"]].dropna()
        ew = portfolio_returns(sample.loc[sample["asset_type"] == "stock"], "equal")
        scale = 100 * TRADING_DAYS_PER_YEAR
        rows[name] = {
            "SPY night": scale * spy["r_night"].mean(),
            "SPY day": scale * spy["r_day"].mean(),
            "SPY gap": scale * (spy["r_night"] - spy["r_day"]).mean(),
            "EW stocks night": scale * ew["r_night"].mean(),
            "EW stocks day": scale * ew["r_day"].mean(),
            "EW stocks gap": scale * (ew["r_night"] - ew["r_day"]).mean(),
        }
        del sample
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def night_indicator_regression(df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Regress stacked night and day returns on a night indicator (lecture 05).

    r_{t,s} = α + β 1[s = night] + ε. Standard errors are clustered by date.

    Args:
        df: Table with date, r_night and r_day (one ticker or a portfolio).
        log: If True, use log(1 + r) instead of simple returns.

    Returns:
        One row: beta (daily and annualised, %), clustered SE, 95 % CI and p-value.
    """
    import statsmodels.api as sm

    night = df[["date", "r_night"]].dropna().rename(columns={"r_night": "r"})
    day = df[["date", "r_day"]].dropna().rename(columns={"r_day": "r"})
    night["night"] = 1
    day["night"] = 0
    stacked = pd.concat([night, day], ignore_index=True)
    stacked["r"] = stacked["r"].astype("float64")
    if log:
        stacked["r"] = np.log1p(stacked["r"])
    fit = sm.OLS(stacked["r"], sm.add_constant(stacked["night"])).fit(
        cov_type="cluster", cov_kwds={"groups": stacked["date"]},
    )
    beta, se = fit.params["night"], fit.bse["night"]
    lo, hi = fit.conf_int().loc["night"]
    return pd.DataFrame([{
        "beta daily (%)": 100 * beta,
        "beta annualised (%)": 100 * TRADING_DAYS_PER_YEAR * beta,
        "SE daily (%)": 100 * se,
        "CI 95% low": 100 * TRADING_DAYS_PER_YEAR * lo,
        "CI 95% high": 100 * TRADING_DAYS_PER_YEAR * hi,
        "p-value": fit.pvalues["night"],
        "n days": stacked["date"].nunique(),
    }]).round(4)


def block_bootstrap_ci(diff: pd.Series, block: int = 21, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Block-bootstrap 95 % CI for the mean of a daily night − day series.

    Args:
        diff: Daily series d_t = r_night − r_day.
        block: Length of each contiguous block (trading days).
        n: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Mean (daily and annualised, %) and the 95 % percentile interval (annualised, %).
    """
    rng = np.random.default_rng(seed)
    x = diff.dropna().astype("float64").to_numpy()
    n_obs = len(x)
    n_blocks = int(np.ceil(n_obs / block))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, n_obs - block + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n_obs]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return pd.DataFrame([{
        "block": block,
        "resamples": n,
        "mean daily (%)": 100 * x.mean(),
        "mean annualised (%)": 100 * TRADING_DAYS_PER_YEAR * x.mean(),
        "CI 95% low": 100 * TRADING_DAYS_PER_YEAR * lo,
        "CI 95% high": 100 * TRADING_DAYS_PER_YEAR * hi,
    }]).round(2)


def gap_tests_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """Night-indicator regression on simple and log returns, full sample and around 2008.

    Args:
        df: Table with date, r_night and r_day.

    Returns:
        One row per (sample, return type), with period labels from the input dates.
        The 2008 split is included when both sides are present.
    """
    df = df.dropna(subset=["r_night", "r_day"])
    first_year, last_year = df["date"].dt.year.min(), df["date"].dt.year.max()
    samples = {f"{first_year}–{last_year}": df}
    if first_year < 2008 <= last_year:
        samples[f"{first_year}–2007"] = df[df["date"].dt.year <= 2007]
        samples[f"2008–{last_year}"] = df[df["date"].dt.year >= 2008]
    rows = []
    for sample, subset in samples.items():
        for log, kind in ((False, "simple"), (True, "log")):
            row = night_indicator_regression(subset, log=log).iloc[0]
            rows.append({"sample": sample, "return": kind, **row})
    return pd.DataFrame(rows).set_index(["sample", "return"])
