import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.config import DATA_DIR

EDGAR_DIR = DATA_DIR / "edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NAMES_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
# The SEC asks automated clients to identify themselves and to stay under 10 requests per second.
# Source: https://www.sec.gov/os/accessing-edgar-data
MIN_INTERVAL = 0.12
NEWS_FORMS = ["8-K", "8-K/A"]
# Issuers filing these forms report as foreign private issuers (6-K instead of 8-K): no 8-K never means no news.
FOREIGN_FORMS = {"6-K", "20-F", "40-F"}
NAME_STOPWORDS = {
    "inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "lp", "llc", "sa", "nv", "ag",
    "holdings", "holding", "group", "the", "class", "common", "stock", "shares", "ordinary", "of", "and", "com",
}

_last_request = 0.0
_lock = threading.Lock()


def user_agent() -> str:
    """User-Agent sent to the SEC: the SEC_USER_AGENT environment variable, e.g. "Name email@epfl.ch"."""
    return os.environ.get("SEC_USER_AGENT", "EPFL ADA student project (academic research)")


def get_json(url: str, retries: int = 5) -> dict:
    """Fetch one SEC JSON document, respecting the rate limit and retrying transient errors.

    Args:
        url: SEC URL.
        retries: Attempts before the error is raised (a failed download is never read as "no filings").

    Returns:
        The decoded JSON.
    """
    global _last_request
    for attempt in range(retries):
        with _lock:  # requests from all threads are spaced by MIN_INTERVAL
            time.sleep(max(0.0, MIN_INTERVAL - (time.monotonic() - _last_request)))
            _last_request = time.monotonic()
        try:
            with urlopen(Request(url, headers={"User-Agent": user_agent()}), timeout=60) as response:
                return json.loads(response.read())
        except HTTPError as error:
            if error.code == 404 or attempt == retries - 1:
                raise
        except (URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def normalise_name(name: str) -> str:
    """Lower-case issuer name without punctuation, share-class description or legal suffixes.

    "Apple Inc. - Common Stock" and "APPLE COMPUTER INC" become "apple" and "apple computer".
    """
    name = str(name).lower().split(" - ")[0]
    name = re.sub(r"['’.]", "", name.replace("&", " and "))  # "Conn's" and "CONNS" agree
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return " ".join(w for w in name.split() if w not in NAME_STOPWORDS)


def name_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two issuer names after normalise_name.

    The larger of the character similarity (difflib ratio, robust to spelling variants) and the
    share of the shorter name's words found in the longer one ("apple" in "apple computer").
    """
    a, b = normalise_name(a), normalise_name(b)
    if not a or not b:
        return 0.0
    words_a, words_b = set(a.split()), set(b.split())
    containment = len(words_a & words_b) / min(len(words_a), len(words_b))
    return max(SequenceMatcher(None, a, b).ratio(), containment)


def sec_ticker_table() -> pd.DataFrame:
    """Current SEC ticker -> CIK table (company_tickers.json), one row per ticker.

    Returns:
        Columns ticker (SEC spelling, class shares with "-"), cik and sec_name.
    """
    raw = get_json(TICKERS_URL)
    sec = pd.DataFrame(raw.values()).rename(columns={"cik_str": "cik", "title": "sec_name"})
    sec["ticker"] = sec["ticker"].str.upper()
    return sec.drop_duplicates("ticker", keep="first")[["ticker", "cik", "sec_name"]]


def map_tickers(meta: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Match the dataset's stock tickers to SEC CIKs by ticker.

    company_tickers.json lists today's tickers, not those of April 2020, so a match is only a
    candidate: verify_identity() later checks it against the issuer's current and former names.

    Args:
        meta: Table returned by load_metadata.
        sec: Table returned by sec_ticker_table.

    Returns:
        One row per stock: ticker, security_name, cik (Int64, missing if unmatched), sec_name, match ("ticker").
    """
    stocks = meta.loc[~meta["is_etf"], ["ticker", "security_name"]].copy()
    # The dataset writes share classes as AGM-A or LGF.B and when-issued shares as CARR#; the SEC uses LGF-B and CARR.
    stocks["key"] = stocks["ticker"].str.upper().str.replace(".", "-", regex=False).str.rstrip("#")
    out = stocks.merge(sec.rename(columns={"ticker": "key"}), on="key", how="left", validate="many_to_one")
    out["cik"] = out["cik"].astype("Int64")
    out["match"] = out["cik"].notna().map({True: "ticker", False: pd.NA})
    return out.drop(columns="key")


def sec_name_table(path: Path = EDGAR_DIR / "cik-lookup-data.txt") -> pd.DataFrame:
    """Every name ever registered on EDGAR, with its CIK (about one million rows).

    Downloaded once from NAMES_URL to path. Lines read NAME:CIK:, and names may contain colons.

    Returns:
        Columns cik and key (normalise_name of the EDGAR name), one row per distinct pair.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(Request(NAMES_URL, headers={"User-Agent": user_agent()}), timeout=300) as response:
            path.write_bytes(response.read())
    lines = path.read_text(encoding="latin-1").splitlines()
    rows = [(line[:-12], int(line[-11:-1])) for line in lines if len(line) > 12 and line.endswith(":")]
    names = pd.DataFrame(rows, columns=["sec_name", "cik"])
    names["key"] = names["sec_name"].map(normalise_name)
    return names[names["key"].ne("")].drop_duplicates(["key", "cik"])


def map_names(mapping: pd.DataFrame, names: pd.DataFrame, max_candidates: int = 3) -> pd.DataFrame:
    """Match tickers left unmatched by map_tickers through their exact normalised issuer name.

    This recovers companies that were listed in 2020 but have since been acquired, gone bankrupt or
    changed ticker: they are absent from today's ticker file but remain on EDGAR. When a name has
    several CIKs (up to max_candidates), all are kept as candidates; verify_identity keeps the one
    with the most 8-Ks in the period. Names with more candidates are left unmatched.

    Args:
        mapping: Table returned by map_tickers.
        names: Table returned by sec_name_table.
        max_candidates: Largest number of CIKs accepted for one name.

    Returns:
        mapping with name matches added (match = "name"); a ticker can appear on several rows.
    """
    todo = mapping[mapping["cik"].isna()].drop(columns=["cik", "sec_name", "match"])
    todo["key"] = todo["security_name"].map(normalise_name)
    found = todo.merge(names, on="key", how="inner")
    n = found.groupby("ticker")["cik"].transform("nunique")
    found = found[n <= max_candidates].drop(columns="key").assign(match="name")
    found["cik"] = found["cik"].astype("Int64")
    rest = mapping[~mapping["ticker"].isin(found["ticker"])]
    return pd.concat([rest, found], ignore_index=True).sort_values("ticker").reset_index(drop=True)


def _filings_frame(block: dict) -> pd.DataFrame:
    keep = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return pd.DataFrame({k: pd.Series(block.get(k, []), dtype="object") for k in keep})


def fetch_issuer(cik: int, start: str, end: str) -> tuple[dict, pd.DataFrame]:
    """Download one issuer's filing history and keep its 8-K filings in [start, end].

    Reads the "recent" block and every older page that overlaps the period.

    Args:
        cik: SEC Central Index Key.
        start: First filing date kept (YYYY-MM-DD).
        end: Last filing date kept (YYYY-MM-DD).

    Returns:
        (issuer, filings): issuer facts (cik, name, former names, number of filings of each
        relevant kind, first filing date) and one row per 8-K / 8-K/A.
    """
    root = get_json(f"{SUBMISSIONS_URL}CIK{int(cik):010d}.json")
    frames = [_filings_frame(root["filings"]["recent"])]
    for page in root["filings"].get("files", []):
        if page["filingFrom"] <= end and page["filingTo"] >= start:
            frames.append(_filings_frame(get_json(SUBMISSIONS_URL + page["name"])))
    filings = pd.concat(frames, ignore_index=True).drop_duplicates("accessionNumber")
    in_period = filings["filingDate"].between(start, end)
    issuer = {
        "cik": int(cik),
        "sec_name": root.get("name"),
        "former_names": "|".join(f["name"] for f in root.get("formerNames", [])),
        "entity_type": root.get("entityType"),
        "first_filing": filings["filingDate"].min(),
        "n_filings_period": int(in_period.sum()),
        "n_8k_period": int((in_period & filings["form"].eq("8-K")).sum()),
        "foreign_filer": bool(filings.loc[in_period, "form"].isin(FOREIGN_FORMS).any()),
    }
    news = filings[in_period & filings["form"].isin(NEWS_FORMS)].copy()
    news.insert(0, "cik", int(cik))
    return issuer, news.reset_index(drop=True)


def _collect_one(cik: int, cache: Path, start: str, end: str) -> dict:
    """Download one issuer into the cache unless it is already there; return its facts."""
    issuer_path, filings_path = cache / f"{cik}.json", cache / f"{cik}.parquet"
    if not issuer_path.exists():
        try:
            issuer, filings = fetch_issuer(cik, start, end)
        except Exception as error:  # recorded and retried on the next run
            return {"cik": cik, "status": f"failed: {type(error).__name__}"}
        filings.to_parquet(filings_path, index=False)
        issuer_path.write_text(json.dumps(issuer))
    return json.loads(issuer_path.read_text()) | {"status": "ok"}


def collect_filings(ciks: list[int], start: str, end: str, out_dir: Path = EDGAR_DIR, workers: int = 4) -> None:
    """Download the 8-K history of every CIK, resumably, and write two Parquet files.

    Each issuer is cached in out_dir/cache/<cik>.parquet as soon as it is downloaded, so an
    interrupted run restarts where it stopped. Failures are recorded, never treated as "no news".
    Several threads download in parallel; get_json keeps the overall rate below the SEC limit.
    Writes out_dir/issuers.parquet (one row per CIK, with a status column) and
    out_dir/filings_8k.parquet (one row per filing).

    Args:
        ciks: CIKs to download.
        start: First filing date kept (YYYY-MM-DD).
        end: Last filing date kept (YYYY-MM-DD).
        out_dir: Output folder.
        workers: Number of parallel downloads.
    """
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    todo = sorted(set(int(c) for c in ciks))
    issuers = []
    with ThreadPoolExecutor(workers) as pool:
        for i, issuer in enumerate(pool.map(lambda c: _collect_one(c, cache, start, end), todo)):
            issuers.append(issuer)
            if i % 250 == 0:
                print(f"{i} of {len(todo)} issuers done", flush=True)
    table = pd.DataFrame(issuers)
    table.to_parquet(out_dir / "issuers.parquet", index=False)
    parts = [pd.read_parquet(cache / f"{c}.parquet") for c in table.loc[table["status"].eq("ok"), "cik"]]
    pd.concat(parts, ignore_index=True).to_parquet(out_dir / "filings_8k.parquet", index=False)


def verify_identity(mapping: pd.DataFrame, issuers: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Check that each ticker match is the same company as in 2020, using names.

    A ticker can be reused by another company after 2020, so the dataset's security name is
    compared with the SEC's current name and every former name; the best score is kept.
    When map_names left several candidate CIKs for a ticker, the one with the most 8-Ks wins.

    Args:
        mapping: Table returned by map_tickers.
        issuers: issuers.parquet written by collect_filings.
        threshold: Minimum name similarity accepted.

    Returns:
        mapping with the issuer facts, name_score and a status column: unmatched, download_failed,
        name_mismatch, foreign_filer, no_8k (no 8-K at all in the period) or verified.
    """
    out = mapping.merge(issuers.drop(columns="sec_name"), on="cik", how="left", validate="many_to_one")
    names = out["sec_name"].fillna("") + "|" + out["former_names"].fillna("")
    out["name_score"] = [
        max((name_similarity(s, n) for n in candidates.split("|") if n), default=0.0)
        for s, candidates in zip(out["security_name"], names)
    ]
    status = pd.Series("verified", index=out.index)
    status[out["n_8k_period"].fillna(0).eq(0)] = "no_8k"
    status[out["foreign_filer"].astype("boolean").fillna(False).to_numpy(dtype=bool)] = "foreign_filer"
    status[out["name_score"] < threshold] = "name_mismatch"
    status[out["status"].ne("ok") & out["cik"].notna()] = "download_failed"
    status[out["cik"].isna()] = "unmatched"
    out = out.drop(columns="status").assign(status=status)
    rank = out["status"].eq("verified").astype(int) * 1e6 + out["n_8k_period"].fillna(0)
    out = out.assign(_rank=rank).sort_values("_rank", ascending=False).drop_duplicates("ticker")
    return out.drop(columns="_rank").sort_values("ticker").reset_index(drop=True)


def load_filings(out_dir: Path = EDGAR_DIR) -> pd.DataFrame:
    """Load the 8-K filings with parsed timestamps.

    acceptanceDateTime is UTC (an Apple earnings 8-K reads 20:30Z in summer and 21:30Z in winter,
    i.e. 16:30 in New York all year). Before about 2001, EDGAR stored only the date: the time
    then reads exactly 00:00:00 in New York, and such filings are marked time_known = False.

    Returns:
        One row per filing: cik, accession, form, filing_date, accepted (UTC), accepted_ny,
        time_known, items and earnings (item 2.02 from Aug 2004, item 12 in 2003-2004).
    """
    raw = pd.read_parquet(out_dir / "filings_8k.parquet")
    accepted = pd.to_datetime(raw["acceptanceDateTime"], utc=True, errors="coerce")
    accepted_ny = accepted.dt.tz_convert("America/New_York")
    items = raw["items"].fillna("").astype(str)
    item_list = items.str.split(",")
    return pd.DataFrame({
        "cik": raw["cik"].astype("int64"),
        "accession": raw["accessionNumber"],
        "form": raw["form"],
        "filing_date": pd.to_datetime(raw["filingDate"]),
        "accepted": accepted,
        "accepted_ny": accepted_ny,
        "time_known": accepted.notna() & (accepted_ny.dt.hour + accepted_ny.dt.minute + accepted_ny.dt.second).gt(0),
        "items": items,
        "earnings": item_list.apply(lambda xs: "2.02" in xs or "12" in xs),
    })
