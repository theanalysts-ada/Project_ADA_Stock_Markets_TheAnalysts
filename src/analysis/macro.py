import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.news import assign_sessions, gap_test
from src.features.returns import cumulative_wealth

EVENTS = ["jobs report", "CPI", "FOMC"]


def place_events(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Put each macroeconomic release in the night or in the trading day of a session.

    We use the same rule as for 8-Ks (assign_sessions). An 8:30 release is in the night of that
    day's session, or of the next session if the market is closed (for example a jobs report on
    Good Friday). An FOMC statement at 14:00 is in the day of its session.

    Args:
        events: Tables returned by bls_release_dates and fomc_dates, concatenated.
        windows: Table returned by session_windows.

    Returns:
        One row per release: event, release, date (session) and timing ("night" or "day").
    """
    releases = pd.DataFrame({
        "event": events["event"].to_numpy(),
        "accepted": events["release"].dt.tz_convert("UTC").to_numpy(),
        "filing_date": events["date"].to_numpy(),
        "time_known": True,
    })
    placed = assign_sessions(releases, windows)
    return placed[["event", "accepted", "date", "timing"]].rename(columns={"accepted": "release"})


def macro_flags(placed: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """One row per session, saying which releases fall in its night or its day.

    Returns:
        Columns date, "jobs report night", "CPI night", "FOMC day", "BLS night" (jobs report or CPI)
        and "macro" (any of the three).
    """
    flags = pd.DataFrame({"date": windows["date"]})
    for event, timing in [("jobs report", "night"), ("CPI", "night"), ("FOMC", "day")]:
        dates = placed.loc[(placed["event"] == event) & (placed["timing"] == timing), "date"]
        flags[f"{event} {timing}"] = flags["date"].isin(dates)
    flags["BLS night"] = flags["jobs report night"] | flags["CPI night"]
    flags["macro"] = flags["BLS night"] | flags["FOMC day"]
    return flags


def event_counts(placed: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Number of releases of each kind in the night and in the day, between start and end."""
    placed = placed[placed["date"].between(start, end)]
    return pd.crosstab(placed["event"], placed["timing"]).reindex(EVENTS).fillna(0).astype(int)


def macro_regression(daily: pd.DataFrame, flags: pd.DataFrame, maxlags: int = 5) -> pd.DataFrame:
    """Do release sessions earn more than the other sessions?

    Overnight return on jobs-report and CPI nights, and intraday return on FOMC days:
    r_t = a + b * release_t + e_t, in bp, with HAC standard errors (5 lags, as in Part III).
    The constant a is the mean of the other sessions, b the extra return of a release session.

    Args:
        daily: Daily portfolio with date, r_night and r_day.
        flags: Table returned by macro_flags.
        maxlags: Number of lags of the HAC standard errors.

    Returns:
        One row per (half, regressor): coefficient (bp), 95 % CI, p-value and number of sessions.
    """
    data = daily.merge(flags, on="date")
    models = {"night": ["jobs report night", "CPI night"], "day": ["FOMC day"]}
    rows = []
    for half, regressors in models.items():
        sub = data.dropna(subset=[f"r_{half}"])
        X = sm.add_constant(sub[regressors].astype(float))
        y = 10000 * sub[f"r_{half}"].astype(float)
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        ci = fit.conf_int()
        for name in ["const"] + regressors:
            rows.append({
                "half": half,
                "regressor": "other sessions (mean)" if name == "const" else name,
                "coef (bp)": fit.params[name],
                "CI low": ci.loc[name, 0],
                "CI high": ci.loc[name, 1],
                "p-value": fit.pvalues[name],
                "sessions": len(sub) if name == "const" else int(sub[name].sum()),
            })
    return pd.DataFrame(rows).set_index(["half", "regressor"]).round(3)


def macro_decomposition(daily: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Share of the summed overnight log return earned on nights with a jobs report or a CPI release.

    Returns:
        One row for release nights and one for the other nights: nights, share of nights (%),
        mean (bp) and share of the summed log return (%).
    """
    data = daily.merge(flags, on="date").dropna(subset=["r_night"])
    r_night = data["r_night"].astype(float)
    log_night = np.log1p(r_night)
    release = data["BLS night"]
    rows = {}
    for name, mask in {"release nights (8:30)": release, "other nights": ~release}.items():
        rows[name] = {
            "nights": int(mask.sum()),
            "share of nights (%)": 100 * mask.mean(),
            "mean (bp)": 10000 * r_night[mask].mean(),
            "share of summed log return (%)": 100 * log_night[mask].sum() / log_night.sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def macro_free_gap(portfolios: dict[str, pd.DataFrame], flags: pd.DataFrame) -> pd.DataFrame:
    """Night - day gap of each portfolio with and without the sessions that have a release.

    A session is dropped (night and day) when a jobs report, a CPI release or an FOMC statement
    falls in it. Same regression as Part II.

    Returns:
        One row per (portfolio, sample): gap (%), CI, p-value and number of dates.
    """
    no_release = flags.loc[~flags["macro"], "date"]
    rows = {}
    for name, daily in portfolios.items():
        rows[(name, "all sessions")] = gap_test(daily)
        rows[(name, "no macro release")] = gap_test(daily[daily["date"].isin(no_release)])
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["portfolio", "sample"]).round(3)


def macro_paths(daily: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Value of $1 invested overnight on all nights, overnight without release nights, and intraday.

    On a skipped night the investor stays in cash.
    """
    data = daily.merge(flags[["date", "BLS night"]], on="date").dropna(subset=["r_night", "r_day"])
    night_without_releases = data["r_night"].where(~data["BLS night"], 0)
    return pd.DataFrame({
        "date": data["date"].to_numpy(),
        "night": cumulative_wealth(data["r_night"]).to_numpy(),
        "night without releases": cumulative_wealth(night_without_releases).to_numpy(),
        "day": cumulative_wealth(data["r_day"]).to_numpy(),
    })
