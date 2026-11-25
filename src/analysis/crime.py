import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.quality import apply_quality_filter
from src.features.returns import (
    TRADING_DAYS_PER_YEAR,
    cumulative_wealth,
    lagged_mean_dollar_volume,
    portfolio_returns,
)


# Return units: x100 for %, x10000 for bp; rounding only controls displayed precision.
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


# Require about one trading year before reporting a ticker-level mean.
MIN_OBS = 252


def asset_portfolios(df: pd.DataFrame, asset_type: str) -> dict[str, pd.DataFrame]:
    """Equal-weighted and dollar-volume-weighted overnight/intraday returns of one asset type.

    Args:
        df: Study sample, or a subset of it.
        asset_type: "stock" or "etf".

    Returns:
        Two daily return tables, keyed "Stocks, ..." or "ETFs, ...".
    """
    rows = df[df["asset_type"].eq(asset_type)]
    label = "Stocks" if asset_type == "stock" else "ETFs"
    return {
        f"{label}, equal-weighted": portfolio_returns(rows, "equal").reset_index(),
        f"{label}, dollar-volume": portfolio_returns(rows, "dollar_volume").reset_index(),
    }


def study_portfolios(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Equal-weighted and dollar-volume-weighted overnight/intraday returns, stocks and ETFs.

    Args:
        df: Study sample (special ETFs already excluded).

    Returns:
        Four daily return tables, keyed by a short panel title.
    """
    return {**asset_portfolios(df, "stock"), **asset_portfolios(df, "etf")}


def make_books(df: pd.DataFrame, portfolios: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """SPY and the four portfolios of study_portfolios, the books tested in Part II.

    Args:
        df: Study sample, or a variant of it with the same columns.
        portfolios: Output of study_portfolios on df, computed if not given.

    Returns:
        Book name -> daily table with date, r_night and r_day.
    """
    spy = df.loc[df["ticker"].eq("SPY"), ["date", "r_night", "r_day"]].dropna()
    return {"SPY": spy, **(portfolios or study_portfolios(df))}


def portfolio_paths(portfolios: dict[str, pd.DataFrame], names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cumulative wealth of $1 for a subset of study_portfolios."""
    return {name: strategy_paths(portfolios[name]) for name in (names or portfolios)}


def portfolio_headlines(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack strategy_headline for each portfolio."""
    tables = {}
    for name, pair in portfolios.items():
        tables[name] = strategy_headline(pair)
    return pd.concat(tables, names=["portfolio", "strategy"])


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


def filter_comparison(prices: pd.DataFrame, start_year: int, filters: dict[str, list[str]]) -> pd.DataFrame:
    """Annualised night and day returns of SPY and of equal-weighted stocks, under each filter.

    Args:
        prices: Full flagged table (all years).
        start_year: First year of the study period.
        filters: Name -> drop_flags, as in quality.FILTERS.

    Returns:
        One row per filter: SPY and equal-weighted stock night / day / gap (annualised, %).
    """
    flags_used = sorted({flag for flags in filters.values() for flag in flags})
    slim = prices[["ticker", "date", "asset_type", "r_night", "r_day", *flags_used]]
    scale = 100 * TRADING_DAYS_PER_YEAR
    rows = {}
    for name, flags in filters.items():
        sample = apply_quality_filter(slim, start_year, flags)
        spy = sample.loc[sample["ticker"] == "SPY", ["r_night", "r_day"]].dropna()
        ew = portfolio_returns(sample.loc[sample["asset_type"] == "stock"], "equal")
        rows[name] = {
            "SPY night": scale * spy["r_night"].mean(),
            "SPY day": scale * spy["r_day"].mean(),
            "SPY gap": scale * (spy["r_night"] - spy["r_day"]).mean(),
            "EW stocks night": scale * ew["r_night"].mean(),
            "EW stocks day": scale * ew["r_day"].mean(),
            "EW stocks gap": scale * (ew["r_night"] - ew["r_day"]).mean(),
        }
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


# 21 sessions ~ one trading month; 1000 draws balance precision and runtime; seed=0 fixes the draws.
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
    # Leave 2.5% in each tail for a central 95% bootstrap interval.
    lo, hi = np.percentile(means, [2.5, 97.5])
    return pd.DataFrame([{
        "block": block,
        "resamples": n,
        "mean daily (%)": 100 * x.mean(),
        "mean annualised (%)": 100 * TRADING_DAYS_PER_YEAR * x.mean(),
        "CI 95% low": 100 * TRADING_DAYS_PER_YEAR * lo,
        "CI 95% high": 100 * TRADING_DAYS_PER_YEAR * hi,
    }]).round(2)


def gap_bootstrap(pairs: dict[str, pd.DataFrame], block: int = 21, winsorise: float = 0.0) -> pd.DataFrame:
    """Block-bootstrap 95 % CI of the night − day gap for each portfolio.

    Args:
        pairs: Portfolio name -> table with date, r_night and r_day.
        block: Length of each block of consecutive trading days.
        winsorise: If above 0, daily gaps below this quantile or above 1 − this quantile
            are set to the quantile before the bootstrap.

    Returns:
        One row per portfolio, from block_bootstrap_ci, with the sample dates.
    """
    rows = []
    for name, pair in pairs.items():
        diff = pair["r_night"] - pair["r_day"]
        if winsorise > 0:
            diff = diff.clip(diff.quantile(winsorise), diff.quantile(1 - winsorise))
        rows.append(label_sample(block_bootstrap_ci(diff, block=block), name, pair))
    return pd.concat(rows, ignore_index=True)


def gap_regression(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Date-clustered night-indicator regression for each portfolio.

    Args:
        pairs: Portfolio name -> table with date, r_night and r_day.

    Returns:
        One row per portfolio, from night_indicator_regression, with the sample dates.
    """
    rows = [
        label_sample(night_indicator_regression(pair), name, pair)
        for name, pair in pairs.items()
    ]
    return pd.concat(rows, ignore_index=True)


def rolling_annualised(pair: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling annualised mean of overnight and intraday returns.

    Args:
        pair: Table with date, r_night and r_day, one row per date.
        window: Length of the window in trading days.

    Returns:
        One row per complete window: date, night and day, in annualised percent.
    """
    ordered = pair.sort_values("date")
    scale = 100 * TRADING_DAYS_PER_YEAR
    rolled = ordered[["r_night", "r_day"]].rolling(window, min_periods=window).mean()
    out = pd.DataFrame({
        "date": ordered["date"].to_numpy(),
        "night": scale * rolled["r_night"].to_numpy(),
        "day": scale * rolled["r_day"].to_numpy(),
    })
    return out.dropna()


def night_ahead_share(rolled: dict[str, pd.DataFrame]) -> pd.Series:
    """Share (%) of rolling windows in which the overnight mean is above the intraday mean.

    Args:
        rolled: Portfolio name -> table returned by rolling_annualised.

    Returns:
        One value per portfolio.
    """
    shares = {name: 100 * (frame["night"] > frame["day"]).mean() for name, frame in rolled.items()}
    return pd.Series(shares, name="windows with night > day (%)").round(1)


def outside_crises(pair: pd.DataFrame) -> pd.DataFrame:
    """Drop the two crises of the sample from a daily return table.

    The crises are the whole of 2008 (the financial crisis) and every session from
    February 2020 (the COVID crash; the data end on 1 April 2020).

    Args:
        pair: Table with a date column.

    Returns:
        The same columns, without those sessions.
    """
    date = pair["date"]
    crisis = date.dt.year.eq(2008) | date.ge("2020-02-01")
    return pair.loc[~crisis]


def gap_robustness(
    books: dict[str, pd.DataFrame], study: pd.DataFrame, excluded_etfs: pd.Series, unlinked: pd.Series,
) -> pd.DataFrame:
    """Annualised night − day gap and its 95 % block-bootstrap interval under other choices.

    Checks on every book: the main test (21-day blocks), 5- and 63-day blocks, daily gaps
    winsorised at 1 %, without the crises (outside_crises), and the night without dividends
    (r_night_price). On the stock books only: without the stocks not linked to the SEC.
    On the ETF books only: without excluded_etfs.

    Args:
        books: Output of make_books on study.
        study: Study sample, with r_night_price.
        excluded_etfs: Tickers of the ETFs removed in the last check.
        unlinked: Boolean mask of study, True for the stocks not linked to the SEC.

    Returns:
        One row per book, one column per check; each cell reads "gap [low, high]" in %.
    """
    columns = ["ticker", "date", "asset_type", "r_night_price", "r_day", "dollar_volume"]
    no_dividend = study[columns].rename(columns={"r_night_price": "r_night"})
    checks = {
        "main (21-day blocks)": gap_bootstrap(books),
        "5-day blocks": gap_bootstrap(books, block=5),
        "63-day blocks": gap_bootstrap(books, block=63),
        "winsorised at 1 %": gap_bootstrap(books, winsorise=0.01),
        "without 2008 and 2020": gap_bootstrap({name: outside_crises(pair) for name, pair in books.items()}),
        "night without dividends": gap_bootstrap(make_books(no_dividend)),
        "stocks linked to the SEC only": gap_bootstrap(asset_portfolios(study[~unlinked], "stock")),
        "without volatility and commodity ETFs": gap_bootstrap(
            asset_portfolios(study[~study["ticker"].isin(excluded_etfs)], "etf"),
        ),
    }
    cells = {}
    for check, table in checks.items():
        text = table.set_index("portfolio")[["mean annualised (%)", "CI 95% low", "CI 95% high"]].round(1).astype(str)
        cells[check] = text["mean annualised (%)"] + " [" + text["CI 95% low"] + ", " + text["CI 95% high"] + "]"
    return pd.DataFrame(cells).reindex(list(books)).fillna("–")


def liquidity_quintiles(df: pd.DataFrame, asset_type: str) -> dict[str, pd.DataFrame]:
    """Equal-weighted books of one asset type by lagged dollar-volume quintile.

    Quintiles use the 20-day lagged mean dollar volume, ranked among the tickers of the same
    asset type on each date (Q1 = least traded), so they use no later information. The first
    20 sessions of a ticker have no lagged mean and are left out.

    Args:
        df: Study sample with date, ticker, asset_type, r_night, r_day and dollar_volume.
        asset_type: "stock" or "etf".

    Returns:
        Q1–Q5, each a daily table with date, r_night and r_day.
    """
    rows = df.loc[df["asset_type"].eq(asset_type)].dropna(subset=["r_night", "r_day"])
    weight = lagged_mean_dollar_volume(df).loc[rows.index]
    quintile = np.ceil(5 * weight.groupby(rows["date"]).rank(pct=True)).rename("quintile")
    means = rows.groupby([quintile, "date"])[["r_night", "r_day"]].mean().reset_index()
    return {f"Q{int(q)}": part.drop(columns="quintile") for q, part in means.groupby("quintile")}


def etf_kind_books(df: pd.DataFrame, kinds: pd.Series) -> dict[str, pd.DataFrame]:
    """Equal-weighted ETF books by kind of fund.

    Args:
        df: Study sample.
        kinds: Kind of each ETF, indexed by ticker (from etf_kinds).

    Returns:
        Kind -> daily table with date, r_night and r_day.
    """
    etfs = df[df["asset_type"].eq("etf")]
    kind = etfs["ticker"].astype(str).map(kinds).rename("kind")
    means = etfs.groupby([kind, "date"])[["r_night", "r_day"]].mean().reset_index()
    return {name: part.drop(columns="kind") for name, part in means.groupby("kind")}


def label_sample(table: pd.DataFrame, portfolio: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add the portfolio name and the first and last date of the paired returns.

    Args:
        table: One-row result, as returned by night_indicator_regression or block_bootstrap_ci.
        portfolio: Name shown in the table.
        df: Table with date, r_night and r_day. Dates are taken from rows where both returns exist.

    Returns:
        A copy of table with portfolio, first date and last date as the first columns.
    """
    pair = df.dropna(subset=["r_night", "r_day"])
    out = table.copy()
    out.insert(0, "portfolio", portfolio)
    out.insert(1, "first date", pair["date"].min().date())
    out.insert(2, "last date", pair["date"].max().date())
    return out
