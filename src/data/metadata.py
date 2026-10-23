import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, META_FILE

EXCHANGES = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Q": "Nasdaq", "Z": "Cboe BZX"}
MARKET_CATEGORIES = {"Q": "Global Select", "G": "Global Market", "S": "Capital Market"}
FINANCIAL_STATUS = {"N": "Normal", "D": "Deficient", "E": "Delinquent", "H": "Deficient and delinquent"}


def load_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the ticker metadata with readable column names and values.

    Args:
        data_dir: Folder containing symbols_valid_meta.csv.

    Returns:
        One row per ticker: ticker (matches the CSV file names and prices.ticker),
        security_name, is_etf, listing_exchange, market_category, financial_status.
        Market category and financial status only exist for Nasdaq listings (NaN otherwise).
    """
    meta = pd.read_csv(data_dir / META_FILE)
    return pd.DataFrame({
        "ticker": meta["NASDAQ Symbol"],
        "security_name": meta["Security Name"],
        "is_etf": meta["ETF"].eq("Y"),
        "listing_exchange": meta["Listing Exchange"].map(EXCHANGES).astype("category"),
        "market_category": meta["Market Category"].str.strip().map(MARKET_CATEGORIES).astype("category"),
        "financial_status": meta["Financial Status"].map(FINANCIAL_STATUS).astype("category"),
    })


def exchange_breakdown(meta: pd.DataFrame) -> pd.DataFrame:
    """Count stocks and ETFs per listing exchange.

    Args:
        meta: Table returned by load_metadata.

    Returns:
        One row per exchange (plus a total row), columns stock, etf and total.
    """
    asset_type = meta["is_etf"].map({False: "stock", True: "etf"})
    table = pd.crosstab(meta["listing_exchange"], asset_type, margins=True, margins_name="total")[["stock", "etf", "total"]]
    order = table.drop(index="total").sort_values("total", ascending=False).index.tolist() + ["total"]
    return table.loc[order].rename_axis(index="exchange", columns=None)


FOREIGN_KEYWORDS = (
    "japan", "europe", "eurozone", "emerging", "china", "chinese", "india", "brazil",
    "korea", "mexico", "australia", "hong kong", "taiwan", "russia", "latin", "eafe",
    "frontier", "international", "germany", "united kingdom", "britain", "asia",
    "pacific", "africa",
)
LEVERAGED_KEYWORDS = (
    "ultrashort", "ultrapro", "ultra ", "inverse", " 2x", " 3x", "2x ", "3x ",
    "bear 1x", "bear 2x", "bear 3x", "bull 2x", "bull 3x", "leveraged", "proshares short",
)


def tag_special_etfs(meta: pd.DataFrame) -> pd.DataFrame:
    """Flag ETFs that track foreign markets or that are leveraged / inverse.

    Matching is case-insensitive on security_name. Stocks are never flagged.

    Args:
        meta: Table returned by load_metadata.

    Returns:
        The metadata with boolean columns foreign_etf and leveraged_etf.
    """
    name = meta["security_name"].fillna("").str.lower()
    is_etf = meta["is_etf"]
    foreign = is_etf & name.apply(lambda s: any(k in s for k in FOREIGN_KEYWORDS))
    leveraged = is_etf & name.apply(lambda s: any(k in s for k in LEVERAGED_KEYWORDS))
    return meta.assign(foreign_etf=foreign, leveraged_etf=leveraged)


def etf_exclusion_counts(meta: pd.DataFrame) -> pd.DataFrame:
    """Count ETFs excluded as foreign, leveraged/inverse, or both.

    Args:
        meta: Table returned by tag_special_etfs.

    Returns:
        One row per category, with the number of ETFs.
    """
    etf = meta[meta["is_etf"]]
    both = etf["foreign_etf"] & etf["leveraged_etf"]
    return pd.Series({
        "all ETFs": len(etf),
        "foreign": int(etf["foreign_etf"].sum()),
        "leveraged / inverse": int(etf["leveraged_etf"].sum()),
        "both": int(both.sum()),
        "kept in the main analysis": int((~etf["foreign_etf"] & ~etf["leveraged_etf"]).sum()),
    }).rename("ETFs").to_frame()


def exclude_special_etfs(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Drop rows of ETFs tagged as foreign or leveraged / inverse.

    Args:
        df: Price table with a ticker column.
        meta: Table returned by tag_special_etfs.

    Returns:
        The table without those ETFs. Stocks are unchanged.
    """
    special = meta.loc[meta["foreign_etf"] | meta["leveraged_etf"], "ticker"]
    return df[~df["ticker"].isin(special)]
