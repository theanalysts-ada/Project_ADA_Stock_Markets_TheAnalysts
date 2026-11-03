"""Download the SEC 8-K filings of every stock of the dataset (about 20 minutes).

Run from the repository root, with your name and email (required by the SEC):

    SEC_USER_AGENT="Your Name your.email@epfl.ch" python -m scripts.collect_news

Tickers are matched with today's SEC ticker file, then with all the names registered on EDGAR.
Writes data/edgar/mapping.parquet, issuers.parquet and filings_8k.parquet. If the run stops,
running it again only downloads the missing companies.
"""
from src.data.edgar import EDGAR_DIR, collect_filings, map_names, map_tickers, sec_name_table, sec_ticker_table
from src.data.metadata import load_metadata

START, END = "1997-12-01", "2020-04-30"


def main() -> None:
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    mapping = map_tickers(load_metadata(), sec_ticker_table())
    mapping = map_names(mapping, sec_name_table())
    mapping.to_parquet(EDGAR_DIR / "mapping.parquet", index=False)
    print(mapping.drop_duplicates("ticker")["match"].value_counts(dropna=False))
    collect_filings(mapping["cik"].dropna().astype(int).tolist(), START, END)


if __name__ == "__main__":
    main()
