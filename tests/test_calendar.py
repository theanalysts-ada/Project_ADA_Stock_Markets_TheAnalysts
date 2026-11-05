import pandas as pd
import pytest

from src.features.calendar import build_night_calendar, study_night_calendar


@pytest.fixture(scope="module")
def calendar():
    return build_night_calendar("2020-01-01", "2020-12-01").set_index("date")


def test_weekday_weekend_and_long_weekend_use_elapsed_hours(calendar):
    for date, hours, kind in [
        ("2020-01-07", 17.5, "Weekday"),
        ("2020-01-06", 65.5, "Weekend"),
        ("2020-01-21", 89.5, "Long weekend"),
    ]:
        row = calendar.loc[date]
        assert row["night_hours"] == pytest.approx(hours)
        assert row["night_type"] == kind


def test_us_clock_changes_shorten_and_lengthen_the_weekend(calendar):
    spring, autumn = calendar.loc["2020-03-09"], calendar.loc["2020-11-02"]
    assert spring["night_hours"] == pytest.approx(64.5)
    assert autumn["night_hours"] == pytest.approx(66.5)
    assert spring["dst_change"] == "spring"
    assert autumn["dst_change"] == "autumn"
    assert spring["dst_monday"] and autumn["dst_monday"]
    assert not calendar.loc["2020-03-30", "dst_monday"]


def test_early_close_extends_the_following_night(calendar):
    row = calendar.loc["2020-11-30"]
    assert row["previous_early_close"]
    assert row["previous_close"].tz_convert("America/New_York").hour == 13
    assert row["night_hours"] == pytest.approx(68.5)
    assert row["night_type"] == "Weekend"


def test_midweek_holiday_keeps_the_actual_early_close():
    row = build_night_calendar("2012-07-05", "2012-07-05").iloc[0]
    assert row["night_type"] == "Midweek holiday"
    assert row["night_hours"] == pytest.approx(44.5)
    assert row["previous_early_close"]


def test_september_2001_is_an_exceptional_closure():
    calendar = build_night_calendar("2001-09-10", "2001-09-17").set_index("date")
    assert len(calendar) == 2
    row = calendar.loc["2001-09-17"]
    assert row["previous_session"] == pd.Timestamp("2001-09-10")
    # The calendar records the 17 September reopening at 09:33 New York time.
    assert row["market_open"].tz_convert("America/New_York").strftime("%H:%M") == "09:33"
    assert row["night_hours"] == pytest.approx(161.55)
    assert row["night_type"] == "Exceptional closure"


def test_first_study_night_has_a_previous_session_outside_the_study():
    row = build_night_calendar("1998-01-01", "1998-01-02").iloc[0]
    assert row["previous_session"] == pd.Timestamp("1997-12-31")
    assert row["night_hours"] == pytest.approx(41.5)


def test_study_calendar_uses_the_requested_start_year():
    prices = pd.DataFrame({"date": pd.to_datetime(["2019-01-02", "2020-01-08"])})
    calendar = study_night_calendar(prices, start_year=2020)
    assert calendar["date"].min() == pd.Timestamp("2020-01-02")
    assert calendar["date"].max() == pd.Timestamp("2020-01-08")
    assert calendar.iloc[0]["previous_session"] == pd.Timestamp("2019-12-31")
