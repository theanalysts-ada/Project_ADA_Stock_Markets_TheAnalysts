"""Download the SEC 8-K filings of every stock in the dataset (about 30 minutes, resumable).

Run from the repository root:

    SEC_USER_AGENT="Your Name your.email@epfl.ch" python -m scripts.collect_news

Matches tickers through today's SEC ticker file, then through every name ever registered on EDGAR.
Writes data/edgar/mapping.parquet, issuers.parquet and filings_8k.parquet. Re-running only
downloads the issuers that are missing or failed.
"""
from src.data.edgar import EDGAR_DIR, collect_filings, map_names, map_tickers, sec_name_table, sec_ticker_table
from src.data.metadata import load_metadata

START, END = "1997-12-01", "2020-04-30"


def main() -> None:
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    mapping = map_names(map_tickers(load_metadata(), sec_ticker_table()), sec_name_table())
    mapping.to_parquet(EDGAR_DIR / "mapping.parquet", index=False)
    ciks = mapping["cik"].dropna().astype(int).tolist()
    print(mapping.drop_duplicates("ticker")["match"].value_counts(dropna=False).to_string(), flush=True)
    print(f"{len(set(ciks))} CIKs to download", flush=True)
    collect_filings(ciks, START, END)
    print("done", flush=True)


if __name__ == "__main__":
    main()
