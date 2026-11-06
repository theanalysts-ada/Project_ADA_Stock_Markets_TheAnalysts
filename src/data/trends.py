import time

import pandas as pd
from pytrends.request import TrendReq

# Above about 250 days, Google gives one value per week instead of one per day
MAX_DAYS = 250


def fetch_trends(term, start, end, connection=None):
    """Daily Google search interest in the US for one search term.

    Args:
        term: Search term, e.g. "Apple stock".
        start: First day, "YYYY-MM-DD".
        end: Last day, "YYYY-MM-DD".
        connection: Google Trends connection to reuse. A new one is created if None.

    Returns:
        One value per day, between 0 and 100. Empty if Google has no data.
    """
    # Check the window is short enough to get daily data
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    if n_days > MAX_DAYS:
        raise ValueError("Window too long: Google would return weekly data")

    if connection is None:
        connection = TrendReq(hl="en-US", tz=300)

    # Ask Google for this term over this period, in the US
    period = start + " " + end
    connection.build_payload([term], timeframe=period, geo="US")
    df = connection.interest_over_time()

    if df.empty:
        return pd.Series(dtype=float)
    return df[term]


def fetch_many(terms, start, end, pause=2):
    """Daily Google search interest for several terms, one request per term.

    Args:
        terms: List of search terms.
        start: First day, "YYYY-MM-DD".
        end: Last day, "YYYY-MM-DD".
        pause: Seconds to wait between two requests, to avoid being blocked.

    Returns:
        A table with one column per term, and a dictionary with the error of each term that failed.
    """
    connection = TrendReq(hl="en-US", tz=300)
    data = {}
    errors = {}

    for term in terms:
        try:
            data[term] = fetch_trends(term, start, end, connection)
        except Exception as e:
            errors[term] = str(e)
        time.sleep(pause)

    return pd.DataFrame(data), errors