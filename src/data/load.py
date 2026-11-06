import kagglehub
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, DROPPED_FILE, META_FILE, PRICES_FILE
from src.data.clean import clean_prices
from src.features.returns import add_adjusted_prices

# Shared Parquet schema: raw prices for quality checks, adjusted prices for returns.
COLUMNS = ["ticker", "date", "open", "close", "open_adj", "high_adj", "low_adj", "close_adj", "volume", "asset_type"]
# Kaggle identifier of the project price dataset.
DATASET = "jacksoncrow/stock-market-dataset"


def download_data(data_dir: Path = DATA_DIR) -> None:
    """Download and unzip the Kaggle dataset into data_dir, unless it is already there.

    Args:
        data_dir: Folder that receives stocks/, etfs/ and symbols_valid_meta.csv.
    """
    if (data_dir / META_FILE).exists():
        return
    kagglehub.dataset_download(DATASET, output_dir=str(data_dir))


def build_prices(data_dir: Path = DATA_DIR, force: bool = False) -> None:
    """Run the full pipeline (read, clean, adjust) and write the Parquet files.

    Writes prices.parquet (one row per ticker and date, columns in COLUMNS)
    and dropped_rows.parquet (removed rows with their reason).

    Args:
        data_dir: Folder containing the raw CSVs, where the Parquet files are written.
        force: Rebuild even if prices.parquet already exists.
    """
    if (data_dir / PRICES_FILE).exists() and not force:
        return
    df = read_all_csv(data_dir)
    clean, dropped = clean_prices(df)
    clean = add_adjusted_prices(clean)
    clean = clean[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)
    clean.to_parquet(data_dir / PRICES_FILE, index=False)
    dropped.to_parquet(data_dir / DROPPED_FILE, index=False)

def load_prices(tickers: list[str] | None = None, columns: list[str] | None = None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the processed price table shared by the whole team.

    Args:
        tickers: Tickers to load; all tickers if None.
        columns: Columns to load; all columns if None.
        data_dir: Folder containing prices.parquet.

    Returns:
        Long table with one row per (ticker, date).
    """
    filters = [("ticker", "in", tickers)] if tickers else None
    return pd.read_parquet(data_dir / PRICES_FILE, columns=columns, filters=filters)


def read_one_csv(path: Path, asset_type: str) -> pd.DataFrame:
    """Read one raw CSV with optimised dtypes and add its ticker and asset type.

    Args:
        path: Path to the CSV; the file name (without .csv) is the ticker.
        asset_type: "stock" or "etf".

    Returns:
        Table with lower-case columns plus ticker and asset_type.
    """
    # float32 limits memory use; nullable Int64 preserves missing share counts.
    df = pd.read_csv(path, parse_dates=["Date"], dtype={"Volume": "Int64", "Open": "float32", "High": "float32", "Low": "float32", "Close": "float32", "Adj Close": "float32"})
    df["asset_type"] = asset_type
    df["ticker"] = path.stem
    df.rename(columns={"Date": "date", "Adj Close": "adj_close", "Volume": "volume","Open": "open", "High": "high", "Low": "low", "Close": "close"}, inplace=True)
    return df


def read_all_csv(data_dir: Path) -> pd.DataFrame:
    """Read every CSV of stocks/ and etfs/ into a single long table.

    Args:
        data_dir: Folder containing stocks/ and etfs/.

    Returns:
        Concatenated table, with ticker and asset_type stored as categories.
    """
    frames = []
    for path in (data_dir/"stocks").glob("*.csv"):
        df = read_one_csv(path, "stock")
        frames.append(df)
    for path in (data_dir/"etfs").glob("*.csv"):
        df = read_one_csv(path, "etf")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype("category")
    df["asset_type"] = df["asset_type"].astype("category")
    return df
