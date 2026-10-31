import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.config import DATA_DIR

MACRO_DIR = DATA_DIR / "macro"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FRED_API = "https://api.stlouisfed.org/fred/"
# Jobs report: FRED release 50 (Employment Situation). CPI: vintage dates of the unadjusted CPI
# series, which change only at the monthly release (FRED's CPI release dates also list seasonal
# factor revisions). Both are published at 8:30 New York time, before the open.
# Source: https://fred.stlouisfed.org/releases and https://www.bls.gov/schedule/
FRED_SOURCES = {"jobs report": "release/dates?release_id=50", "CPI": "series/vintagedates?series_id=CPIAUCNS"}
BLS_RELEASE_TIME = "08:30"
MONTHS = {m[:3]: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}


def _get(url: str) -> str:
    request = Request(url, headers={"User-Agent": "EPFL ADA student project (academic research)"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_fomc_headers(headers: list[str]) -> pd.DataFrame:
    """Turn the meeting headers of a Fed history page into announcement dates.

    "January 29-30 Meeting - 2019", "Jan/Feb 31-1 Meeting - 2017" and "June 30-July 1 Meeting -
    1998" give the last day of the meeting (the last month and day named), when the statement is
    released. Conference calls, notation votes and meetings marked unscheduled or cancelled are
    dropped: their timing is irregular.

    Args:
        headers: Header texts of one fomchistorical<year> page.

    Returns:
        One row per scheduled meeting: date (announcement day).
    """
    dates = []
    for text in headers:
        if "Meeting" not in text or re.search(r"unscheduled|cancelled", text, re.I):
            continue
        days_part, year = text.split("Meeting")[0], int(re.findall(r"\d{4}", text)[-1])
        month = [MONTHS[w.lower()[:3]] for w in re.findall(r"[A-Za-z]+", days_part) if w.lower()[:3] in MONTHS][-1]
        day = int(re.findall(r"\d+", days_part)[-1])
        dates.append(pd.Timestamp(year, month, day))
    return pd.DataFrame({"date": dates})


def fomc_dates(start_year: int, end_year: int, out_dir: Path = MACRO_DIR) -> pd.DataFrame:
    """Announcement days of scheduled FOMC meetings, from the Fed's historical pages (cached).

    Statements are released during the trading session: at 14:15 New York time until 2012,
    at 14:00 from 2013 (12:30 at the 2011-2012 meetings with a press conference).

    Returns:
        Columns event ("FOMC"), date and release (New York time, timezone-aware).
    """
    path = out_dir / "fomc.csv"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for year in range(start_year, end_year + 1):
            html = _get(FOMC_URL.format(year=year))
            headers = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<h5[^>]*>(.*?)</h5>", html, re.S)]
            frames.append(parse_fomc_headers(headers))
        pd.concat(frames).to_csv(path, index=False)
    dates = pd.read_csv(path, parse_dates=["date"])["date"].sort_values().reset_index(drop=True)
    # Two meetings on consecutive days (15 and 16 September 2003) end with a single statement.
    dates = dates[~(dates.shift(-1) - dates).le(pd.Timedelta(days=1))].reset_index(drop=True)
    clock = pd.Series("14:15", index=dates.index).where(dates.dt.year < 2013, "14:00")
    release = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + clock).dt.tz_localize("America/New_York")
    return pd.DataFrame({"event": "FOMC", "date": dates, "release": release})


def bls_release_dates(start: str, end: str, api_key: str | None = None, out_dir: Path = MACRO_DIR) -> pd.DataFrame:
    """Release days of the jobs report and the CPI, from the FRED API (cached).

    FRED also lists a few correction dates a few days after a release; only the first date of
    each month is kept, which leaves exactly one release per month. Needs a free FRED API key
    (argument or FRED_API_KEY environment variable) on the first run.

    Returns:
        Columns event, date and release (8:30 New York time, timezone-aware).
    """
    path = out_dir / "bls_releases.csv"
    if not path.exists():
        key = api_key or os.environ["FRED_API_KEY"]
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for event, query in FRED_SOURCES.items():
            url = (f"{FRED_API}{query}&api_key={key}&file_type=json"
                   f"&realtime_start={start}&realtime_end={end}&sort_order=asc&limit=10000")
            raw = json.loads(_get(url))
            dates = [d["date"] for d in raw["release_dates"]] if "release_dates" in raw else raw["vintage_dates"]
            frames.append(pd.DataFrame({"event": event, "date": dates}))
        pd.concat(frames).to_csv(path, index=False)
    table = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    table = table[~table.assign(month=table["date"].dt.to_period("M")).duplicated(["event", "month"])].reset_index(drop=True)
    table["release"] = pd.to_datetime(table["date"].dt.strftime("%Y-%m-%d") + " " + BLS_RELEASE_TIME).dt.tz_localize("America/New_York")
    return table
