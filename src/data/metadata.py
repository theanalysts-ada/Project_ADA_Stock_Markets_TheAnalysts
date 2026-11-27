import re
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, META_FILE

# Decode the listing codes supplied in the Nasdaq metadata file.
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


# Name screen for ETFs exposed to markets outside the US: regions, countries,
# "outside the US" wording, global funds and foreign currencies.
FOREIGN_KEYWORDS = (
    "japan", "nikkei", "jpx", "europe", "euro", "nordic", "china", "chinese", "india", "brazil",
    "korea", "mexico", "australia", "hong kong", "taiwan", "russia", "latin", "asia", "pacific",
    "africa", "germany", "united kingdom", "britain", "canada", "canadian", "switzerland", "swiss",
    "france", "spain", "italy", "sweden", "norway", "denmark", "finland", "belgium", "austria",
    "netherlands", "ireland", "israel", "chile", "peru", "colombia", "argentina", "turkey",
    "vietnam", "indonesia", "philippines", "thailand", "malaysia", "singapore", "poland", "greece",
    "new zealand", "egypt", "nigeria", "pakistan", "saudi", "qatar", "uae", "gulf", "middle east",
    "emerging", " em ", "bric", "frontier", "eafe", "international", "intl", "foreign", "non-us",
    "ex-us", "ex us", "ex-u.s.", "ex u.s.", "ex-north america", "developed", "acwi", "all country",
    "world", "global", "stoxx select", "vstoxx", "currency", "yen", "sterling", "franc", "yuan",
)
# Name markers of bond and income funds, used to split ETFs by kind.
BOND_KEYWORDS = (
    "bond", "treasury", "muni", "aggregate", "tips", "duration", "credit", "loan", "mortgage",
    "fixed income", "high yield", "government", "floating rate", "maturity",
)
# Issuer names that contain "global" without saying where the fund invests.
FOREIGN_ISSUER_NAMES = ("global x", "u.s. global", "global markets holdings", "global beta")
# Name markers for leveraged or inverse products with mechanical returns.
LEVERAGED_KEYWORDS = (
    "ultrashort", "ultrapro", "ultra ", "inverse", " 2x", " 3x", "2x ", "3x ",
    "bear 1x", "bear 2x", "bear 3x", "bull 2x", "bull 3x", "leveraged", "proshares short",
)
# Volatility futures and commodity products. Equity sectors and miners are not in this list.
VOL_COMMODITY_KEYWORDS = (
    "vix", "commodity", "crude", "natural gas", "gold", "silver", "copper", "wheat", "corn", "oil",
)
VOL_COMMODITY_EXCEPTIONS = (
    "goldman", "golden", "miners", "explorers", "refiner", "cornerstone",
    "oil & gas", "oil services", "oil equipment", "first trust natural gas",
)


def tag_special_etfs(meta: pd.DataFrame) -> pd.DataFrame:
    """Flag ETFs that track foreign markets or that are leveraged / inverse.

    Matching is case-insensitive on security_name, padded with spaces so that " em " only
    matches the word. Issuer names in FOREIGN_ISSUER_NAMES are removed before the foreign screen.
    Stocks are never flagged.

    Args:
        meta: Table returned by load_metadata.

    Returns:
        The metadata with boolean columns foreign_etf and leveraged_etf.
    """
    name = " " + meta["security_name"].fillna("").str.lower() + " "
    fund_name = name.str.replace(_any_of(FOREIGN_ISSUER_NAMES), "", regex=True)
    foreign = meta["is_etf"] & fund_name.str.contains(_any_of(FOREIGN_KEYWORDS))
    leveraged = meta["is_etf"] & name.str.contains(_any_of(LEVERAGED_KEYWORDS))
    return meta.assign(foreign_etf=foreign, leveraged_etf=leveraged)


def etf_kinds(meta: pd.DataFrame) -> pd.Series:
    """Kind of each ETF: "volatility / commodity", "bond / income" or "equity / other".

    Args:
        meta: Table returned by load_metadata.

    Returns:
        Kind, indexed by ticker, for ETFs only.
    """
    etfs = meta[meta["is_etf"]]
    name = " " + etfs["security_name"].fillna("").str.lower() + " "
    kind = pd.Series("equity / other", index=etfs.index)
    kind[name.str.contains(_any_of(BOND_KEYWORDS))] = "bond / income"
    kind[etfs["ticker"].isin(commodity_vol_etfs(meta))] = "volatility / commodity"
    return kind.set_axis(etfs["ticker"])


def _any_of(keywords: tuple[str, ...]) -> str:
    """Regular expression that matches any of the keywords as plain text."""
    return "|".join(re.escape(k) for k in keywords)


def commodity_vol_etfs(meta: pd.DataFrame) -> pd.Series:
    """Tickers of volatility-futures and commodity ETFs.

    Args:
        meta: Table returned by load_metadata.

    Returns:
        Ticker series. Equity names that only share a substring, such as Goldman or miners, are kept.
    """
    name = meta["security_name"].fillna("").str.lower()
    hit = meta["is_etf"] & name.str.contains("|".join(VOL_COMMODITY_KEYWORDS))
    skip = name.str.contains("|".join(VOL_COMMODITY_EXCEPTIONS))
    return meta.loc[hit & ~skip, "ticker"]


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
