from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
META_FILE = "symbols_valid_meta.csv"
PRICES_FILE = "prices.parquet"
DROPPED_FILE = "dropped_rows.parquet"

# First year with no missing opening prices in this dataset.
STUDY_START_YEAR = 1998
