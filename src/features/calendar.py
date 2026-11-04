import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


NIGHT_TYPES = ["Weekday", "Weekend", "Long weekend", "Midweek holiday", "Exceptional closure"]


def build_night_calendar(start_date: str, end_date: str) -> pd.DataFrame:
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
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    market = mcal.get_calendar("NYSE")
    schedule = market.schedule(start_date=start - pd.Timedelta(days=30), end_date=end)
    dates = pd.Series(schedule.index, index=schedule.index)
    calendar = pd.DataFrame({
        "date": dates,
        "previous_session": dates.shift(1),
        "previous_close": schedule["market_close"].shift(1),
        "market_open": schedule["market_open"],
    })
    calendar = calendar[calendar["date"].between(start, end)].reset_index(drop=True)
    calendar["night_hours"] = (calendar["market_open"] - calendar["previous_close"]).dt.total_seconds() / 3600

    gap = (calendar["date"] - calendar["previous_session"]).dt.days
    previous_day = calendar["previous_session"].dt.dayofweek
    day = calendar["date"].dt.dayofweek
    long_weekend = (gap >= 4) & previous_day.isin([3, 4]) & day.isin([0, 1])
    weekend = (gap == 3) & (previous_day == 4) & (day == 0)
    kind = pd.Series(
        np.select(
            [long_weekend, weekend, gap == 1],
            ["Long weekend", "Weekend", "Weekday"],
            default="Midweek holiday",
        ),
        index=calendar.index,
        dtype="object",
    )

    adhoc = pd.DatetimeIndex(market.adhoc_holidays).tz_localize(None)
    adhoc = adhoc[adhoc.dayofweek < 5].sort_values().unique()
    before_open = adhoc.searchsorted(calendar["date"], side="left")
    after_close = adhoc.searchsorted(calendar["previous_session"], side="right")
    kind[before_open > after_close] = "Exceptional closure"
    calendar["night_type"] = pd.Categorical(kind, categories=NIGHT_TYPES, ordered=True)

    local_close = calendar["previous_close"].dt.tz_convert("America/New_York")
    local_open = calendar["market_open"].dt.tz_convert("America/New_York")
    calendar["previous_early_close"] = local_close.dt.hour < 16
    # Compare local wall time with UTC wall time to get each timestamp's UTC
    # offset as a vectorized timedelta, including historical DST transitions.
    close_offset = local_close.dt.tz_localize(None) - calendar["previous_close"].dt.tz_localize(None)
    open_offset = local_open.dt.tz_localize(None) - calendar["market_open"].dt.tz_localize(None)
    change = (open_offset - close_offset).dt.total_seconds() / 3600
    calendar["dst_change"] = np.select([change > 0, change < 0], ["spring", "autumn"], default="none")
    calendar["dst_monday"] = (day == 0) & (change != 0)
    return calendar


def study_night_calendar(prices: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Build the study calendar up to the last date in the shared price table.

    Args:
        prices: Shared price table with a date column.
        start_year: First year of the study period.

    Returns:
        Table from build_night_calendar, ending on the last available price date.
    """
    end_date = prices["date"].max().strftime("%Y-%m-%d")
    return build_night_calendar(f"{start_year}-01-01", end_date)


def calendar_coverage(df: pd.DataFrame, calendar: pd.DataFrame, start_year: int | None = None) -> pd.Series:
    """Compare price dates with the NYSE sessions of the same requested period.

    Args:
        df: Price table with a date column.
        calendar: Table returned by build_night_calendar for that period.
        start_year: Exclude earlier price dates; None if df is already restricted.

    Returns:
        Number of market sessions, price dates, missing sessions and non-session dates.
    """
    dates = pd.DatetimeIndex(df["date"].unique())
    if start_year is not None:
        dates = dates[dates.year >= start_year]
    sessions = pd.DatetimeIndex(calendar["date"])
    return pd.Series({
        "market sessions": len(sessions), "price dates": len(dates),
        "sessions absent from prices": len(sessions.difference(dates)),
        "price dates outside the calendar": len(dates.difference(sessions)),
    })


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


def night_calendar_examples(calendar: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Show the previous close, current open and elapsed duration of selected nights.

    Args:
        calendar: Table returned by build_night_calendar.
        dates: Session dates to display (YYYY-MM-DD).

    Returns:
        One row per requested date, with UTC timestamps, duration, type and clock change.
    """
    columns = ["previous_close", "market_open", "night_hours", "night_type", "dst_change"]
    return calendar.set_index("date").loc[dates, columns]


def attach_night_calendar(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach the market closure ending on each price date without recalculating returns.

    Args:
        df: Price or daily portfolio table with a date column.
        calendar: Table returned by build_night_calendar.

    Returns:
        The input rows with calendar columns added. Dates must be covered, and the
        calendar must have one row per date; missing returns remain missing.
    """
    result = df.merge(calendar, on="date", how="left", validate="many_to_one")
    missing = result.loc[result["night_hours"].isna(), "date"].drop_duplicates()
    if len(missing):
        raise ValueError(f"Dates absent from the night calendar: {missing.dt.strftime('%Y-%m-%d').tolist()[:5]}")
    return result
