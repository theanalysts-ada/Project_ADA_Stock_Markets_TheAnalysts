"""Export the Part II results used by the data story website (docs/).

Runs the same pipeline as the notebook and writes:
- docs/assets/data/crime/*.json: the data of the charts;
- docs/_data/crime.json: the numbers quoted in the text, read by Jekyll at build time.

Usage, from the repository root:
    python -m scripts.export_story_data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.crime import (
    crime_headline,
    etf_kind_books,
    gap_bootstrap,
    liquidity_quintiles,
    make_books,
    night_ahead_share,
    night_day_gap_by_ticker,
    portfolio_headlines,
    portfolio_paths,
    rolling_annualised,
    study_portfolios,
    ticker_wealth,
    without_dividends,
)
from src.config import ROOT_DIR
from src.data.load import load_prices
from src.data.metadata import etf_kinds, load_metadata, tag_special_etfs
from src.data.quality import flag_suspicious
from src.data.sample import exclusion_waterfall, study_sample
from src.features.returns import TRADING_DAYS_PER_YEAR, add_dollar_volume, add_price_night, add_returns

# Date on which the main test was fixed (Part II decision).
MAIN_TEST_DATE = "25 Nov 2026"
# Books shown in Exhibit D, in display order: notebook name -> site label.
INTERVAL_BOOKS = {
    "SPY": "SPY",
    "Stocks, equal-weighted": "Stocks, equal-weighted",
    "Stocks, dollar-volume": "Stocks, weighted by traded value",
    "ETFs, equal-weighted": "ETFs, equal-weighted",
    "ETFs, dollar-volume": "ETFs, weighted by traded value",
}
# Histogram of ticker gaps: same range and 5-point bins as figure G2 of the notebook.
GAP_EDGES = np.arange(-100, 205, 5)


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Last row of each calendar week, to keep long daily series small."""
    return df.groupby(df["date"].dt.to_period("W")).tail(1)


def series(df: pd.DataFrame, columns: list[str], digits: int = 4) -> dict:
    """Columns of a dated table as JSON lists, dates in ISO format."""
    out = {"dates": df["date"].dt.strftime("%Y-%m-%d").tolist()}
    for column in columns:
        out[column] = df[column].astype("float64").round(digits).tolist()
    return out


def interval(row: pd.Series) -> dict:
    """Annualised gap and 95 % interval of one gap_bootstrap row, in %, rounded like the notebook tables."""
    columns = {"est": "mean annualised (%)", "lo": "CI 95% low", "hi": "CI 95% high"}
    values = row[list(columns.values())].astype("float64").round(1)
    return {key: float(values[col]) for key, col in columns.items()}


def write_json(path: Path, data) -> None:
    """Write compact JSON and report its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"{path.relative_to(ROOT_DIR)}: {path.stat().st_size / 1024:.0f} kB")


def main(docs: Path) -> None:
    prices = flag_suspicious(add_dollar_volume(add_price_night(add_returns(load_prices()))))
    meta = tag_special_etfs(load_metadata())
    study = study_sample(prices, meta)
    ports = study_portfolios(study)
    books = make_books(study, ports)
    charts = docs / "assets" / "data" / "crime"

    # Exhibit A: value of $1 in SPY.
    spy = crime_headline(study, "SPY")
    write_json(charts / "spy_value.json", series(weekly(ticker_wealth(study, "SPY")), ["night", "day", "buy_hold"]))

    # Exhibit B: the four cases, the ticker histogram and the ticker lookup.
    values = portfolio_headlines(ports)["value of $1"].unstack()
    cases = [{"name": "SPY (the market)", "night": float(spy.loc["overnight", "value of $1"]),
              "day": float(spy.loc["intraday", "value of $1"])}]
    for name, label in (("Stocks, dollar-volume", "Stocks, weighted by traded value"),
                        ("Stocks, equal-weighted", "Stocks, equal-weighted"),
                        ("ETFs, equal-weighted", "ETFs, equal-weighted")):
        cases.append({"name": label, "night": float(values.loc[name, "overnight"]), "day": float(values.loc[name, "intraday"])})
    write_json(charts / "portfolios_final.json", cases)

    gaps = night_day_gap_by_ticker(study)
    hist = {"edges": GAP_EDGES.tolist()}
    tickers_count, share_ahead = {}, {}
    for asset, key in (("stock", "stocks"), ("etf", "etfs")):
        values_gap = gaps.loc[gaps["asset_type"] == asset, "gap"]
        counts, _ = np.histogram(values_gap.clip(GAP_EDGES[0], GAP_EDGES[-1]), bins=GAP_EDGES)
        hist[key] = counts.tolist()
        tickers_count[key] = len(values_gap)
        share_ahead[key] = round(100 * float((values_gap > 0).mean()))
    write_json(charts / "ticker_gaps.json", hist)

    names = meta.set_index("ticker")["security_name"]
    scale = 100 * TRADING_DAYS_PER_YEAR
    lookup = pd.DataFrame({
        "name": names.reindex(gaps.index.astype(str)).fillna("").str[:70].to_numpy(),
        "type": gaps["asset_type"].astype(str).to_numpy(),
        "night": (scale * gaps["night"]).round(1).to_numpy(),
        "day": (scale * gaps["day"]).round(1).to_numpy(),
        "gap": gaps["gap"].round(1).to_numpy(),
        "n_days": gaps["n"].astype(int).to_numpy(),
    }, index=gaps.index.astype(str))
    write_json(charts / "tickers.json", lookup.to_dict(orient="index"))

    # Exhibit C: the COVID crash, daily.
    covid = portfolio_paths({"SPY": books["SPY"].loc[books["SPY"]["date"].ge("2020-02-01")]})["SPY"]
    write_json(charts / "covid_crash.json", series(covid, ["night", "day"]))

    # Exhibit D: gap intervals with and without dividends.
    shown = {name: books[name] for name in INTERVAL_BOOKS}
    with_div = gap_bootstrap(shown).set_index("portfolio")
    no_div_books = make_books(without_dividends(study))
    no_div = gap_bootstrap({name: no_div_books[name] for name in INTERVAL_BOOKS}).set_index("portfolio")
    intervals = [{"name": label, "with": interval(with_div.loc[name]), "without": interval(no_div.loc[name])}
                 for name, label in INTERVAL_BOOKS.items()]
    write_json(charts / "gap_intervals.json", intervals)

    # Section 7: three-year rolling means.
    rolled = {key: rolling_annualised(books[name], 756) for key, name in
              (("spy", "SPY"), ("stocks_dv", "Stocks, dollar-volume"), ("stocks_ew", "Stocks, equal-weighted"))}
    write_json(charts / "rolling_3y.json", {key: series(weekly(frame), ["night", "day"], 2) for key, frame in rolled.items()})
    windows = night_ahead_share(rolled)

    # Numbers quoted in the text.
    q5 = interval(gap_bootstrap(liquidity_quintiles(study, "stock")).set_index("portfolio").loc["Q5"])
    bond = interval(gap_bootstrap(etf_kind_books(study, etf_kinds(meta))).set_index("portfolio").loc["bond / income"])
    waterfall = exclusion_waterfall(prices, meta)
    by_name = {row["name"]: row for row in intervals}
    dividend_share = {key: round(100 * (1 - by_name[label]["without"]["est"] / by_name[label]["with"]["est"]))
                      for key, label in (("spy", "SPY"), ("stocks_ew", "Stocks, equal-weighted"))}
    facts = {
        "main_test_date": MAIN_TEST_DATE,
        "first_date": str(spy.loc["overnight", "first date"]),
        "last_date": str(spy.loc["overnight", "last date"]),
        "spy": {"night": f"{spy.loc['overnight', 'value of $1']:.2f}", "day": f"{spy.loc['intraday', 'value of $1']:.2f}",
                "hold": f"{spy.loc['buy-and-hold', 'value of $1']:.2f}",
                "night_share": f"{spy.loc['overnight', 'share of log return (%)']:.0f}"},
        "cases": [{"name": c["name"], "night": f"{c['night']:.2f}", "day": f"{c['day']:.2f}"} for c in cases],
        "tickers": {"stocks": f"{tickers_count['stocks']:,}", "etfs": f"{tickers_count['etfs']:,}",
                    "stocks_ahead": share_ahead["stocks"], "etfs_ahead": share_ahead["etfs"]},
        "covid": {"night": f"{covid['night'].iloc[-1]:.2f}", "day": f"{covid['day'].iloc[-1]:.2f}"},
        "intervals": intervals,
        "dividend_share": dividend_share,
        "q5": q5,
        "bond_etfs": bond,
        "windows": {"spy": f"{windows['spy']:.0f}", "stocks_dv": f"{windows['stocks_dv']:.0f}"},
        "sample": {"tickers": f"{int(waterfall.iloc[0][['stock tickers', 'ETF tickers']].sum()):,}",
                   "stocks": f"{int(waterfall.iloc[-1]['stock tickers']):,}",
                   "etfs": f"{int(waterfall.iloc[-1]['ETF tickers']):,}"},
    }
    write_json(docs / "_data" / "crime.json", facts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=Path, default=ROOT_DIR / "docs", help="Folder of the website.")
    main(parser.parse_args().docs)
