import pandas as pd

from src.config import STUDY_START_YEAR
from src.data.edgar import EDGAR_DIR
from src.data.metadata import tag_special_etfs
from src.data.quality import FILTERS, keep_mask


def foreign_stocks() -> pd.Series:
    """Tickers linked to a CIK that filed a 6-K, 20-F or 40-F (forms of foreign companies).

    Returns:
        Ticker series. Stocks with no CIK are not in it.
    """
    mapping = pd.read_parquet(EDGAR_DIR / "mapping.parquet", columns=["ticker", "cik"])
    issuers = pd.read_parquet(EDGAR_DIR / "issuers.parquet", columns=["cik", "foreign_filer"])
    linked = mapping.merge(issuers, on="cik")
    return linked.loc[linked["foreign_filer"].eq(True), "ticker"]


def unlinked_stocks(df: pd.DataFrame) -> pd.Series:
    """True for the stock rows whose ticker we could not link to an SEC identifier (CIK).

    Args:
        df: Table with ticker and asset_type.

    Returns:
        Boolean mask aligned with df.
    """
    mapping = pd.read_parquet(EDGAR_DIR / "mapping.parquet", columns=["ticker", "cik"])
    linked = mapping.loc[mapping["cik"].notna(), "ticker"]
    return df["asset_type"].eq("stock") & ~df["ticker"].isin(linked)


def sample_steps(prices: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Rows kept after each step of the main sample, as cumulative boolean masks.

    Steps: study period from STUDY_START_YEAR, liquidity and quality filter, without foreign and
    leveraged / inverse ETFs, without foreign stocks.

    Args:
        prices: Table from flag_suspicious.
        meta: Table returned by load_metadata.

    Returns:
        Step name -> mask aligned with prices; the last mask is the main sample.
    """
    tagged = tag_special_etfs(meta)
    special_etfs = tagged.loc[tagged["foreign_etf"] | tagged["leveraged_etf"], "ticker"]
    period = prices["date"].dt.year >= STUDY_START_YEAR
    filtered = keep_mask(prices, STUDY_START_YEAR, FILTERS["liquidity"])
    us_etfs = filtered & ~prices["ticker"].isin(special_etfs)
    return {
        "all rows": pd.Series(True, index=prices.index),
        f"from {STUDY_START_YEAR}": period,
        "liquidity and quality filter": filtered,
        "without foreign and leveraged ETFs": us_etfs,
        "without foreign stocks": us_etfs & ~prices["ticker"].isin(foreign_stocks()),
    }


def study_sample(prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Main sample shared by every part: the rows kept by the last step of sample_steps.

    Args:
        prices: Table from flag_suspicious.
        meta: Table returned by load_metadata.

    Returns:
        One row per remaining ticker-date.
    """
    return prices[list(sample_steps(prices, meta).values())[-1]]


def exclusion_waterfall(prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Stock tickers, ETF tickers and rows left after each step of the main sample.

    Args:
        prices: Table from flag_suspicious.
        meta: Table returned by load_metadata.

    Returns:
        One row per step of sample_steps, with the share of all rows (%).
    """
    rows = {}
    for step, mask in sample_steps(prices, meta).items():
        tickers = prices.loc[mask, ["ticker", "asset_type"]].drop_duplicates("ticker")["asset_type"]
        rows[step] = {
            "stock tickers": int(tickers.eq("stock").sum()),
            "ETF tickers": int(tickers.eq("etf").sum()),
            "rows": int(mask.sum()),
            "% of all rows": round(100 * mask.mean(), 1),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
