import time

import pandas as pd
from pytrends.request import TrendReq


# Google returns daily values only for short windows (longer windows come back weekly).
# Kept conservative so every request stays daily.
MAX_DAILY_DAYS = 250


def fetch_trends(term: str, start: str, end: str, geo: str = "US", client: TrendReq | None = None) -> pd.Series:
    """Daily Google search interest for one term over a short window.

    Values are scaled 0-100 within the requested window only, so they can be
    compared across days of the same window, not across windows or terms.

    Args:
        term: Search term, e.g. "Apple stock".
        start: First day, "YYYY-MM-DD".
        end: Last day, "YYYY-MM-DD".
        geo: Country code; "US" by default.
        client: Existing TrendReq connection to reuse; a new one is created if None.

    Returns:
        Daily series indexed by date, named after the term; empty if Google has no data.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if (end - start).days > MAX_DAILY_DAYS:
        raise ValueError(f"Window longer than {MAX_DAILY_DAYS} days: Google would return weekly data.")
    client = client or TrendReq(hl="en-US", tz=300)
    client.build_payload([term], timeframe=f"{start:%Y-%m-%d} {end:%Y-%m-%d}", geo=geo)
    df = client.interest_over_time()
    if df.empty:
        return pd.Series(dtype="float64", name=term)
    return df[term].astype("float64").rename(term)


def fetch_many(terms: list[str], start: str, end: str, geo: str = "US", pause: float = 2.0) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch several terms one by one, pausing between requests to avoid being blocked.

    Each term is requested alone, so each column has its own 0-100 scale.

    Args:
        terms: Search terms, e.g. ["Apple stock", "Tesla stock"].
        start: First day, "YYYY-MM-DD".
        end: Last day, "YYYY-MM-DD".
        geo: Country code; "US" by default.
        pause: Seconds to wait between requests.

    Returns:
        (data, failures): one column per successful term, and the error message of each failed term.
    """
    client = TrendReq(hl="en-US", tz=300)
    series, failures = [], {}
    for term in terms:
        try:
            s = fetch_trends(term, start, end, geo=geo, client=client)
            if s.empty:
                failures[term] = "empty"
            else:
                series.append(s)
        except Exception as e:
            failures[term] = str(e)[:100]
        time.sleep(pause)
    data = pd.concat(series, axis=1) if series else pd.DataFrame()
    return data, failures
