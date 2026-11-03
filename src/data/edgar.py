import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.config import DATA_DIR

EDGAR_DIR = DATA_DIR / "edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NAMES_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"

# The SEC allows at most 10 requests per second: https://www.sec.gov/os/accessing-edgar-data
MIN_INTERVAL = 0.12
FOREIGN_FORMS = ["6-K", "20-F", "40-F"]
NAME_STOPWORDS = [
    "inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "lp", "llc", "sa", "nv", "ag",
    "holdings", "holding", "group", "the", "class", "common", "stock", "shares", "ordinary", "of", "and", "com",
]

_lock = threading.Lock()
_last_request = 0.0


def sec_headers() -> dict:
    """HTTP headers for the SEC, which requires a name and an email (SEC_USER_AGENT variable)."""
    return {"User-Agent": os.environ["SEC_USER_AGENT"]}


def get_json(url: str, retries: int = 5) -> dict:
    """Download a JSON file from the SEC, waiting between requests and retrying on errors.

    Args:
        url: SEC URL.
        retries: Number of attempts before the error is raised.

    Returns:
        The decoded JSON.
    """
    global _last_request
    for attempt in range(retries):
        with _lock:
            wait = MIN_INTERVAL - (time.monotonic() - _last_request)
            if wait > 0:
                time.sleep(wait)
            _last_request = time.monotonic()
        try:
            with urlopen(Request(url, headers=sec_headers()), timeout=60) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def normalise_name(name: str) -> str:
    """Lower-case company name without punctuation, share class or legal suffix.

    For example "Apple Inc. - Common Stock" becomes "apple" and "APPLE COMPUTER INC" becomes "apple computer".
    """
    name = str(name).lower().split(" - ")[0]
    name = name.replace("&", " and ")
    name = re.sub(r"['’.]", "", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return " ".join(word for word in name.split() if word not in NAME_STOPWORDS)


def name_similarity(a: str, b: str) -> float:
    """Similarity between two company names, from 0 to 1.

    We take the larger of two scores: the difflib ratio of the two strings, and the share of the
    words of the shorter name that appear in the longer one ("apple" in "apple computer").
    """
    a, b = normalise_name(a), normalise_name(b)
    if a == "" or b == "":
        return 0.0
    words_a, words_b = set(a.split()), set(b.split())
    common_words = len(words_a & words_b) / min(len(words_a), len(words_b))
    return max(SequenceMatcher(None, a, b).ratio(), common_words)


def sec_ticker_table() -> pd.DataFrame:
    """Today's SEC table of tickers and CIKs.

    Returns:
        One row per ticker: ticker, cik and sec_name.
    """
    raw = get_json(TICKERS_URL)
    sec = pd.DataFrame(list(raw.values()))
    sec = sec.rename(columns={"cik_str": "cik", "title": "sec_name"})
    sec["ticker"] = sec["ticker"].str.upper()
    return sec.drop_duplicates("ticker")[["ticker", "cik", "sec_name"]]


def map_tickers(meta: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Match each stock of the dataset to a CIK through its ticker.

    The SEC table only has today's tickers, so a match is checked later by verify_identity.
    The dataset writes share classes as "LGF.B" and when-issued shares as "CARR#", while the SEC
    writes "LGF-B" and "CARR".

    Args:
        meta: Table returned by load_metadata.
        sec: Table returned by sec_ticker_table.

    Returns:
        One row per stock: ticker, security_name, cik (missing if no match), sec_name and match.
    """
    stocks = meta.loc[~meta["is_etf"], ["ticker", "security_name"]].copy()
    stocks["sec_ticker"] = stocks["ticker"].str.upper().str.replace(".", "-").str.rstrip("#")
    sec = sec.rename(columns={"ticker": "sec_ticker"})
    out = stocks.merge(sec, on="sec_ticker", how="left").drop(columns="sec_ticker")
    out["cik"] = out["cik"].astype("Int64")
    out["match"] = None
    out.loc[out["cik"].notna(), "match"] = "ticker"
    return out


def sec_name_table(path: Path = EDGAR_DIR / "cik-lookup-data.txt") -> pd.DataFrame:
    """Every company name ever registered on EDGAR, with its CIK (about one million names).

    The file is downloaded once. Each line reads "NAME:CIK:".

    Returns:
        Columns sec_name, cik and key (the normalised name).
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(Request(NAMES_URL, headers=sec_headers()), timeout=300) as response:
            path.write_bytes(response.read())
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.endswith(":") and line.count(":") >= 2:
            name, cik, _ = line.rsplit(":", 2)
            if name != "" and cik.isdigit():
                rows.append((name, int(cik)))
    names = pd.DataFrame(rows, columns=["sec_name", "cik"])
    names["key"] = names["sec_name"].map(normalise_name)
    names = names[names["key"] != ""]
    return names.drop_duplicates(["key", "cik"])


def map_names(mapping: pd.DataFrame, names: pd.DataFrame, max_candidates: int = 3) -> pd.DataFrame:
    """Match the stocks that map_tickers could not match, through their company name.

    These are mostly companies acquired, bankrupt or renamed since 2020: their ticker is no
    longer in the SEC table, but their name is still on EDGAR. If a name has several CIKs, we keep
    them all (up to max_candidates) and verify_identity picks one.

    Args:
        mapping: Table returned by map_tickers.
        names: Table returned by sec_name_table.
        max_candidates: Maximum number of CIKs for one name.

    Returns:
        The mapping with the name matches added (match = "name"). A ticker can have several rows.
    """
    missing = mapping[mapping["cik"].isna()].drop(columns=["cik", "sec_name", "match"])
    missing["key"] = missing["security_name"].map(normalise_name)
    found = missing.merge(names, on="key").drop(columns="key")
    n_candidates = found.groupby("ticker")["cik"].transform("nunique")
    found = found[n_candidates <= max_candidates]
    found["cik"] = found["cik"].astype("Int64")
    found["match"] = "name"
    rest = mapping[~mapping["ticker"].isin(found["ticker"])]
    return pd.concat([rest, found], ignore_index=True).sort_values("ticker").reset_index(drop=True)


def read_filings(block: dict) -> pd.DataFrame:
    """Table of filings from one block of the SEC submissions JSON."""
    columns = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return pd.DataFrame({col: pd.Series(block.get(col, []), dtype="object") for col in columns})


def fetch_issuer(cik: int, start: str, end: str) -> tuple[dict, pd.DataFrame]:
    """Download the filing history of one company and keep its 8-Ks between start and end.

    The SEC gives the most recent filings in the main file and older ones in extra pages.

    Args:
        cik: SEC identifier of the company.
        start: First filing date kept (YYYY-MM-DD).
        end: Last filing date kept (YYYY-MM-DD).

    Returns:
        A dictionary with facts about the company, and one row per 8-K or 8-K/A.
    """
    root = get_json(f"{SUBMISSIONS_URL}CIK{int(cik):010d}.json")
    frames = [read_filings(root["filings"]["recent"])]
    for page in root["filings"].get("files", []):
        if page["filingFrom"] <= end and page["filingTo"] >= start:
            frames.append(read_filings(get_json(SUBMISSIONS_URL + page["name"])))
    filings = pd.concat(frames, ignore_index=True).drop_duplicates("accessionNumber")
    in_period = filings[filings["filingDate"].between(start, end)]

    issuer = {
        "cik": int(cik),
        "sec_name": root.get("name"),
        "former_names": "|".join(f["name"] for f in root.get("formerNames", [])),
        "entity_type": root.get("entityType"),
        "first_filing": filings["filingDate"].min(),
        "n_filings_period": len(in_period),
        "n_8k_period": int((in_period["form"] == "8-K").sum()),
        "foreign_filer": bool(in_period["form"].isin(FOREIGN_FORMS).any()),
    }
    news = in_period[in_period["form"].isin(["8-K", "8-K/A"])].copy()
    news.insert(0, "cik", int(cik))
    return issuer, news.reset_index(drop=True)


def download_issuer(cik: int, start: str, end: str, cache: Path) -> dict:
    """Download one company into the cache folder, unless it is already there.

    Returns:
        The company facts with status "ok", or status "failed: <error>" if the download failed.
    """
    issuer_path = cache / f"{cik}.json"
    if not issuer_path.exists():
        try:
            issuer, filings = fetch_issuer(cik, start, end)
        except Exception as error:
            return {"cik": cik, "status": f"failed: {type(error).__name__}"}
        filings.to_parquet(cache / f"{cik}.parquet", index=False)
        issuer_path.write_text(json.dumps(issuer))
    issuer = json.loads(issuer_path.read_text())
    issuer["status"] = "ok"
    return issuer


def collect_filings(ciks: list[int], start: str, end: str, out_dir: Path = EDGAR_DIR, workers: int = 4) -> None:
    """Download the 8-Ks of every company and save them in two Parquet files.

    Each company is saved in out_dir/cache as soon as it is downloaded, so a run that stops can
    be restarted. Downloads run in a few threads; get_json keeps the total rate under the SEC limit.
    Writes out_dir/issuers.parquet (one row per company) and out_dir/filings_8k.parquet (one row per filing).

    Args:
        ciks: CIKs to download.
        start: First filing date kept (YYYY-MM-DD).
        end: Last filing date kept (YYYY-MM-DD).
        out_dir: Output folder.
        workers: Number of downloads in parallel.
    """
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    ciks = sorted(set(int(c) for c in ciks))
    with ThreadPoolExecutor(workers) as pool:
        issuers = list(pool.map(lambda cik: download_issuer(cik, start, end, cache), ciks))
    issuers = pd.DataFrame(issuers)
    issuers.to_parquet(out_dir / "issuers.parquet", index=False)
    ok = issuers.loc[issuers["status"] == "ok", "cik"]
    filings = pd.concat([pd.read_parquet(cache / f"{cik}.parquet") for cik in ok], ignore_index=True)
    filings.to_parquet(out_dir / "filings_8k.parquet", index=False)
    print(f"{len(ok)} of {len(ciks)} companies downloaded")


def verify_identity(mapping: pd.DataFrame, issuers: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Check that each matched CIK is really the company of the dataset.

    A ticker can be reused by another company after 2020, so we compare the security name of the
    dataset with the current and former SEC names of the company and keep the best score.
    Each ticker then gets a status, in this order of priority:
    unmatched, download_failed, name_mismatch, foreign_filer (files 6-Ks, not 8-Ks), no_8k, verified.
    When a ticker has several candidate CIKs, we keep a verified one with the most 8-Ks.

    Args:
        mapping: Table returned by map_names.
        issuers: issuers.parquet written by collect_filings.
        threshold: Minimum name similarity.

    Returns:
        One row per ticker, with the company facts, name_score and status.
    """
    out = mapping.merge(issuers.drop(columns="sec_name"), on="cik", how="left")
    scores = []
    for row in out.itertuples():
        if not isinstance(row.sec_name, str):  # no SEC company for this ticker
            scores.append(0.0)
            continue
        candidates = [row.sec_name] + [name for name in row.former_names.split("|") if name]
        scores.append(max(name_similarity(row.security_name, name) for name in candidates))
    out["name_score"] = scores

    status = pd.Series("verified", index=out.index)
    status[out["n_8k_period"].fillna(0) == 0] = "no_8k"
    status[out["foreign_filer"].eq(True)] = "foreign_filer"
    status[out["name_score"] < threshold] = "name_mismatch"
    status[out["cik"].notna() & (out["status"] != "ok")] = "download_failed"
    status[out["cik"].isna()] = "unmatched"
    out["status"] = status

    out["is_verified"] = out["status"] == "verified"
    out = out.sort_values(["ticker", "is_verified", "n_8k_period", "cik"], ascending=[True, False, False, True])
    return out.drop_duplicates("ticker").drop(columns="is_verified").reset_index(drop=True)


def load_filings(out_dir: Path = EDGAR_DIR) -> pd.DataFrame:
    """Load the 8-K filings with their acceptance time.

    acceptanceDateTime is in UTC: an Apple earnings 8-K reads 20:30 in summer and 21:30 in winter,
    which is 16:30 in New York. Before 2002 EDGAR only kept the date, and the time then reads
    00:00:00 in New York: we mark these filings with time_known = False.
    Earnings announcements are item 2.02 (item 12 in 2003-2004).

    Returns:
        One row per filing: cik, accession, form, filing_date, accepted (UTC), accepted_ny,
        time_known, items and earnings.
    """
    raw = pd.read_parquet(out_dir / "filings_8k.parquet")
    filings = pd.DataFrame({
        "cik": raw["cik"].astype("int64"),
        "accession": raw["accessionNumber"],
        "form": raw["form"],
        "filing_date": pd.to_datetime(raw["filingDate"]),
        "accepted": pd.to_datetime(raw["acceptanceDateTime"], utc=True, errors="coerce"),
        "items": raw["items"].fillna("").astype(str),
    })
    filings["accepted_ny"] = filings["accepted"].dt.tz_convert("America/New_York")
    midnight = filings["accepted_ny"].dt.strftime("%H:%M:%S") == "00:00:00"
    filings["time_known"] = filings["accepted"].notna() & ~midnight
    item_lists = filings["items"].str.split(",")
    filings["earnings"] = item_lists.apply(lambda items: "2.02" in items or "12" in items)
    return filings
