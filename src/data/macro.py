import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.config import ROOT_DIR

MACRO_DIR = ROOT_DIR / "external" / "macro"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FRED_URL = "https://api.stlouisfed.org/fred/"
# CPI release dates on FRED include seasonal revisions, so we use the unadjusted series.
# FRED identifiers for employment release dates and CPI vintage dates.
FRED_QUERIES = {"jobs report": "release/dates?release_id=50", "CPI": "series/vintagedates?series_id=CPIAUCNS"}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def download_text(url: str) -> str:
    """Download a web page as text."""
    request = Request(url, headers={"User-Agent": "EPFL ADA student project"})
    # 60 seconds bounds the wait for each remote page or API response.
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_fomc_headers(headers: list[str]) -> pd.DataFrame:
    """Statement dates from the meeting titles of a Fed history page.

    The statement comes on the last day written in the title ("Jan/Feb 31-1 Meeting - 2017").
    Unscheduled or cancelled meetings and conference calls are dropped.
    """
    dates = []
    for title in headers:
        if "Meeting" not in title or "unscheduled" in title or "cancelled" in title:
            continue
        days, year = title.split("Meeting")
        # Three-letter month prefixes match the MONTHS lookup table.
        words = [w[:3].lower() for w in re.findall(r"[A-Za-z]+", days)]
        month = [MONTHS.index(w) + 1 for w in words if w in MONTHS][-1]
        day = int(re.findall(r"\d+", days)[-1])
        dates.append(pd.Timestamp(int(re.findall(r"\d{4}", year)[0]), month, day))
    return pd.DataFrame({"date": dates})


def fomc_dates(start_year: int, end_year: int, out_dir: Path = MACRO_DIR) -> pd.DataFrame:
    """Scheduled FOMC statements, released at 14:15 New York time until 2012 and at 14:00 after."""
    path = out_dir / "fomc.csv"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for year in range(start_year, end_year + 1):
            html = download_text(FOMC_URL.format(year=year))
            titles = re.findall(r"<h5[^>]*>(.*?)</h5>", html, re.S)
            titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]
            frames.append(parse_fomc_headers(titles))
        pd.concat(frames).to_csv(path, index=False)

    dates = pd.read_csv(path, parse_dates=["date"])["date"].sort_values().reset_index(drop=True)
    # Sept 2003: two meetings listed for one statement.
    next_day_meeting = dates.shift(-1) - dates == pd.Timedelta(days=1)
    dates = dates[~next_day_meeting].reset_index(drop=True)
    # Historical statement-time convention used here: 14:15 before 2013, 14:00 after.
    time = dates.dt.year.map(lambda year: "14:15" if year < 2013 else "14:00")
    release = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + time).dt.tz_localize("America/New_York")
    return pd.DataFrame({"event": "FOMC", "date": dates, "release": release})


def bls_release_dates(start: str, end: str, out_dir: Path = MACRO_DIR) -> pd.DataFrame:
    """Jobs report and CPI release dates from the FRED API, released at 8:30 New York time.

    We keep the first date of each month, as FRED also lists later corrections.
    The dates are saved in the repository; a FRED_API_KEY is only needed to download them again.
    """
    path = out_dir / "bls_releases.csv"
    if not path.exists():
        key = os.environ["FRED_API_KEY"]
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for event, query in FRED_QUERIES.items():
            url = f"{FRED_URL}{query}&api_key={key}&file_type=json&realtime_start={start}&realtime_end={end}"
            answer = json.loads(download_text(url))
            if "release_dates" in answer:
                dates = [d["date"] for d in answer["release_dates"]]
            else:
                dates = answer["vintage_dates"]
            frames.append(pd.DataFrame({"event": event, "date": dates}))
        pd.concat(frames).to_csv(path, index=False)

    table = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    table["month"] = table["date"].dt.to_period("M")
    table = table.drop_duplicates(["event", "month"]).drop(columns="month").reset_index(drop=True)
    # 08:30 New York time places these jobs/CPI releases before the market open.
    release = table["date"].dt.strftime("%Y-%m-%d") + " 08:30"
    table["release"] = pd.to_datetime(release).dt.tz_localize("America/New_York")
    return table
