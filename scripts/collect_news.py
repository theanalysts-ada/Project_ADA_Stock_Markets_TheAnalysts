"""Download the 8-Ks of every stock into data/edgar/ (about 20 minutes; a stopped run can be restarted).

    SEC_USER_AGENT="Your Name your.email@epfl.ch" python -m scripts.collect_news
"""
from src.data.edgar import EDGAR_DIR, collect_filings, map_names, map_tickers, sec_name_table, sec_ticker_table
from src.data.metadata import load_metadata

START, END = "1997-12-01", "2020-04-30"


def main() -> None:
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    mapping = map_tickers(load_metadata(), sec_ticker_table())
    mapping = map_names(mapping, sec_name_table())
    mapping.to_parquet(EDGAR_DIR / "mapping.parquet", index=False)
    collect_filings(mapping["cik"].dropna().astype(int).tolist(), START, END)


if __name__ == "__main__":
    main()
