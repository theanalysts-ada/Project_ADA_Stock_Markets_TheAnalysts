import pandas as pd
import pandas_market_calendars as mcal


# Separate routine closures from shutdowns outside the regular holiday schedule.
NIGHT_TYPES = ["Weekday", "Weekend", "Long weekend", "Midweek holiday", "Exceptional closure"]


def build_night_calendar(start_date, end_date) -> pd.DataFrame:
    """Measure each NYSE close-to-open interval, including holidays and early closes.

    Session dates are New York calendar dates without a timezone. Opening and closing
    timestamps are in UTC, so night_hours measures elapsed time across US clock changes.
    A 30-day buffer supplies the previous session at the start of the requested period.

    Args:
        start_date: First session date to include (YYYY-MM-DD).
        end_date: Last session date to include (YYYY-MM-DD).

    Returns:
        One row per session: date, previous_session, previous_close, market_open,
        night_hours, night_type, previous_early_close, dst_change and dst_monday.
        Exceptional closures are identified from exceptional full-day holidays in the calendar.
    """
    # Convert the requested dates and check that the period is valid.
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    # Load NYSE sessions, including 30 earlier days to find the previous close.
    market = mcal.get_calendar("NYSE")
    schedule = market.schedule(start_date=start - pd.Timedelta(days=30), end_date=end)
    schedule.index.name = "date"
    schedule = schedule.reset_index()

    # Pair each opening with the previous session and close, then keep the requested dates.
    calendar = pd.DataFrame({
        "date": schedule["date"],
        "previous_session": schedule["date"].shift(1),
        "previous_close": schedule["market_close"].shift(1),
        "market_open": schedule["market_open"],
    })
    calendar = calendar[calendar["date"].between(start, end)].reset_index(drop=True)

    # Measure each closure in elapsed hours using UTC timestamps (3600 seconds per hour).
    duration = calendar["market_open"] - calendar["previous_close"]
    calendar["night_hours"] = duration.dt.total_seconds() / 3600

    # Classify routine closures from the number of days and the opening/closing weekdays.
    gap = (calendar["date"] - calendar["previous_session"]).dt.days
    previous_day = calendar["previous_session"].dt.day_name()
    day = calendar["date"].dt.day_name()
    # Holiday weekends span at least four days: Thursday/Friday to Monday/Tuesday.
    long_weekend = (gap >= 4) & previous_day.isin(["Thursday", "Friday"]) & day.isin(["Monday", "Tuesday"])
    # A regular Friday-to-Monday closure spans three calendar days.
    weekend = (gap == 3) & (previous_day == "Friday") & (day == "Monday")
    calendar["night_type"] = "Midweek holiday"
    calendar.loc[gap == 1, "night_type"] = "Weekday"
    calendar.loc[weekend, "night_type"] = "Weekend"
    calendar.loc[long_weekend, "night_type"] = "Long weekend"

    # Mark closures spanning exceptional weekday shutdowns and order the closure categories.
    for holiday in market.adhoc_holidays:
        holiday = pd.Timestamp(holiday).tz_localize(None)
        if holiday.dayofweek >= 5:  # Saturday/Sunday do not add a closed weekday.
            continue
        spans_holiday = (calendar["previous_session"] < holiday) & (holiday < calendar["date"])
        calendar.loc[spans_holiday, "night_type"] = "Exceptional closure"
    calendar["night_type"] = pd.Categorical(calendar["night_type"], categories=NIGHT_TYPES, ordered=True)

    # Flag early closes by comparing the local closing hour with the regular 16:00 close.
    local_close = calendar["previous_close"].dt.tz_convert("America/New_York")
    local_open = calendar["market_open"].dt.tz_convert("America/New_York")
    calendar["previous_early_close"] = local_close.dt.hour < 16

    # Detect spring/autumn clock changes from UTC offsets and flag the affected Mondays.
    close_offset = local_close.map(lambda time: time.utcoffset())
    open_offset = local_open.map(lambda time: time.utcoffset())
    calendar["dst_change"] = "none"
    calendar.loc[open_offset > close_offset, "dst_change"] = "spring"
    calendar.loc[open_offset < close_offset, "dst_change"] = "autumn"
    calendar["dst_monday"] = (day == "Monday") & (open_offset != close_offset)
    return calendar


def study_night_calendar(prices: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Build the study calendar up to the last date in the shared price table.

    Args:
        prices: Shared price table with a date column.
        start_year: First year of the study period.

    Returns:
        Table from build_night_calendar, ending on the last available price date.
    """
    end_date = prices["date"].max()
    return build_night_calendar(f"{start_year}-01-01", end_date)


def night_calendar_summary(calendar: pd.DataFrame) -> pd.DataFrame:
    """Describe the number and elapsed duration of each closure type.

    Args:
        calendar: Table returned by build_night_calendar.

    Returns:
        One row per observed type: number of nights, mean/minimum/maximum hours,
        nights following an early close and Mondays following a US clock change.
    """
    return calendar.groupby("night_type", observed=True).agg(
        nights=("date", "size"), mean_hours=("night_hours", "mean"),
        min_hours=("night_hours", "min"), max_hours=("night_hours", "max"),
        previous_early_close=("previous_early_close", "sum"),
        dst_mondays=("dst_monday", "sum"),
    ).round(2)
