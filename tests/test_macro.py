import numpy as np
import pandas as pd

from src.analysis.macro import macro_flags, macro_free_gap, macro_regression, place_events
from src.analysis.news import session_windows
from src.data.macro import parse_fomc_headers


def make_event(name, date, time):
    return {"event": name, "date": pd.Timestamp(date), "release": pd.Timestamp(f"{date} {time}", tz="America/New_York")}


def test_parse_fomc_headers():
    headers = ["January 29-30 Meeting - 2019", "Jan/Feb 31-1 Meeting - 2017", "June 30-July 1 Meeting - 1998",
               "October 4 (unscheduled) - 2019", "March 2 (unscheduled) Meeting - 2020",
               "March 17-18 (cancelled) Meeting - 2020", "January 9 Conference Call - 2008"]
    dates = parse_fomc_headers(headers)["date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2019-01-30", "2017-02-01", "1998-07-01"]


def test_releases_are_placed_in_the_right_half():
    windows = session_windows("2015-03-25", "2015-04-15")
    events = pd.DataFrame([
        make_event("jobs report", "2015-04-03", "08:30"),
        make_event("CPI", "2015-04-14", "08:30"),
        make_event("FOMC", "2015-03-18", "14:00"),
        make_event("FOMC", "2015-04-14", "14:00"),
    ])
    placed = place_events(events, windows)
    got = placed.assign(date=placed["date"].dt.strftime("%m-%d"))[["event", "date", "timing"]].values.tolist()
    assert got == [["jobs report", "04-06", "night"], ["CPI", "04-14", "night"], ["FOMC", "04-14", "day"]]
    flags = macro_flags(placed, windows).set_index("date")
    assert flags.loc["2015-04-14", "CPI night"] and flags.loc["2015-04-14", "FOMC day"]
    assert flags.loc["2015-04-06", "BLS night"] and flags.loc["2015-04-06", "macro"]
    assert flags["macro"].sum() == 2


def test_regression_and_gap_recover_planted_release_effect():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-01", periods=2000)
    flags = pd.DataFrame({"date": dates})
    for col in ["jobs report night", "CPI night", "FOMC day"]:
        flags[col] = rng.random(len(dates)) < 0.05
    flags["BLS night"] = flags[["jobs report night", "CPI night"]].any(axis=1)
    flags["macro"] = flags["BLS night"] | flags["FOMC day"]
    daily = pd.DataFrame({"date": dates, "r_night": rng.normal(0, 1e-3, len(dates)) + 40e-4 * flags["CPI night"],
                          "r_day": rng.normal(0, 1e-3, len(dates))})
    reg = macro_regression(daily, flags)
    assert reg.loc[("night", "CPI night"), "CI low"] < 40 < reg.loc[("night", "CPI night"), "CI high"]
    assert reg.loc[("night", "jobs report night"), "CI low"] < 0 < reg.loc[("night", "jobs report night"), "CI high"]
    gap = macro_free_gap({"p": daily}, flags)
    assert gap.loc[("p", "no macro release"), "gap annualised (%)"] < gap.loc[("p", "all sessions"), "gap annualised (%)"]
