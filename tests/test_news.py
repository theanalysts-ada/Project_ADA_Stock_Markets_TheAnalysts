import numpy as np
import pandas as pd

from src.analysis.news import assign_sessions, build_news_panel, news_regression, session_windows, timing_validation
from src.data.edgar import name_similarity, normalise_name


def make_filings(times, filing_dates=None, known=None, earnings=None, form="8-K"):
    """Small table of filings with the columns of load_filings."""
    accepted = pd.to_datetime(pd.Series(times), utc=True)
    accepted_ny = accepted.dt.tz_convert("America/New_York")
    if filing_dates is None:
        filing_dates = accepted_ny.dt.date
    n = len(times)
    return pd.DataFrame({
        "cik": 1,
        "accession": [f"a{i}" for i in range(n)],
        "form": form,
        "filing_date": pd.to_datetime(filing_dates),
        "accepted": accepted,
        "accepted_ny": accepted_ny,
        "time_known": known if known is not None else [True] * n,
        "items": "",
        "earnings": earnings if earnings is not None else [False] * n,
    })


def test_night_and_day_around_holiday_and_early_close():
    windows = session_windows("2019-07-01", "2019-07-09")
    # 3 July 2019 closes at 13:00 New York (17:00 UTC); 4 July is a holiday.
    filings = make_filings([
        "2019-07-03T16:59:00Z",  # 12:59 New York, before the early close: day of 3 July
        "2019-07-03T17:01:00Z",  # after the early close: night of 5 July
        "2019-07-04T15:00:00Z",  # holiday: night of 5 July
        "2019-07-05T13:29:00Z",  # 09:29: night of 5 July
        "2019-07-05T13:30:00Z",  # 09:30, the open: day of 5 July
        "2019-07-05T20:30:00Z",  # 16:30 Friday: night of Monday 8 July
    ])
    out = assign_sessions(filings, windows)
    assert out["timing"].tolist() == ["day", "night", "night", "night", "day", "night"]
    assert out["date"].dt.strftime("%m-%d").tolist() == ["07-03", "07-05", "07-05", "07-05", "07-05", "07-08"]


def test_unknown_time_goes_to_filing_date_session():
    windows = session_windows("2019-07-01", "2019-07-09")
    filings = make_filings(["2019-07-06T04:00:00Z"], filing_dates=["2019-07-06"], known=[False])
    out = assign_sessions(filings, windows)
    assert out.loc[0, "timing"] == "unknown"
    assert out.loc[0, "date"] == pd.Timestamp("2019-07-08")


def test_panel_flags_and_quiet_window():
    windows = session_windows("2019-06-01", "2019-08-30")
    dates = windows.loc[windows["date"].between("2019-07-01", "2019-07-31"), "date"]
    stocks = pd.DataFrame({"ticker": "A", "date": dates, "r_night": 0.0, "r_day": 0.0})
    identity = pd.DataFrame({"ticker": ["A"], "cik": [1], "first_filing": ["1996-01-01"], "status": ["verified"]})
    filings = make_filings(["2019-07-15T20:30:00Z"], earnings=[True])  # Monday 16:30: night of Tuesday 16 July
    panel = build_news_panel(stocks, identity, assign_sessions(filings, windows), windows, start_year=2019)
    row = panel.set_index("date")
    assert row.loc["2019-07-16", "news_night"] and row.loc["2019-07-16", "earnings_night"]
    assert not row.loc["2019-07-15", "news_night"] and not row.loc["2019-07-16", "news_day"]
    noisy = row.index[~row["quiet"]].strftime("%m-%d").tolist()
    assert noisy == ["07-12", "07-15", "07-16", "07-17", "07-18"]


def test_amendment_is_not_news_but_breaks_quiet():
    windows = session_windows("2019-06-01", "2019-08-30")
    dates = windows.loc[windows["date"].between("2019-07-01", "2019-07-31"), "date"]
    stocks = pd.DataFrame({"ticker": "A", "date": dates, "r_night": 0.0, "r_day": 0.0})
    identity = pd.DataFrame({"ticker": ["A"], "cik": [1], "first_filing": ["1996-01-01"], "status": ["verified"]})
    filings = make_filings(["2019-07-15T20:30:00Z"], form="8-K/A")
    panel = build_news_panel(stocks, identity, assign_sessions(filings, windows), windows, start_year=2019)
    assert not panel["news_night"].any()
    assert (~panel["quiet"]).sum() == 5


def test_name_matching():
    assert normalise_name("Apple Inc. - Common Stock") == "apple"
    assert name_similarity("Apple Inc. - Common Stock", "APPLE COMPUTER INC") > 0.6
    assert name_similarity("Facebook, Inc. - Class A Common Stock", "Meta Platforms, Inc.") < 0.6


def test_timing_validation_and_regression_recover_planted_effect():
    rng = np.random.default_rng(0)
    n_t, n_d = 40, 60
    dates = pd.date_range("2010-01-01", periods=n_d)
    panel = pd.DataFrame({"ticker": np.repeat([f"T{i}" for i in range(n_t)], n_d), "date": np.tile(dates, n_t)})
    panel["news_night"] = rng.random(len(panel)) < 0.1
    panel["news_day"] = ~panel["news_night"] & (rng.random(len(panel)) < 0.1)
    panel["earnings_night"] = panel["news_night"] & (rng.random(len(panel)) < 0.5)
    panel["earnings_day"] = False
    panel["unknown"] = False
    panel["timed"] = True
    market = np.tile(rng.normal(0, 1e-2, n_d), n_t)
    panel["r_night"] = market + rng.normal(0, 1e-3, len(panel)) + 50e-4 * panel["news_night"]
    panel["r_day"] = rng.normal(0, 1e-3, len(panel)) + 0.05 * rng.normal(size=len(panel)) * panel["news_day"]
    check = timing_validation(panel)
    assert check.loc["day 8-K only", "|r_day| (bp)"] > 5 * check.loc["no 8-K", "|r_day| (bp)"]
    assert check.loc["day 8-K only", "|r_night| (bp)"] < 2 * check.loc["no 8-K", "|r_night| (bp)"]
    reg = news_regression(panel, "night")
    for regressor in ["earnings 8-K", "other 8-K"]:
        row = reg.loc[("r_night", regressor)]
        assert row["CI low"] < 50 < row["CI high"]
