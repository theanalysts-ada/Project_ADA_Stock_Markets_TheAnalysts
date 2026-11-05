import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.news import assign_sessions, gap_test
from src.features.returns import cumulative_wealth

EVENTS = ["jobs report", "CPI", "FOMC"]


def place_events(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Put each macro release in the night or the day of a session, as for the 8-Ks."""
    releases = pd.DataFrame({
        "event": events["event"].to_numpy(),
        "accepted": events["release"].dt.tz_convert("UTC").to_numpy(),
        "filing_date": events["date"].to_numpy(),
        "time_known": True,
    })
    placed = assign_sessions(releases, windows)
    return placed[["event", "accepted", "date", "timing"]].rename(columns={"accepted": "release"})


def macro_flags(placed: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Flags per session: "jobs report night", "CPI night", "FOMC day", "BLS night" and "macro" (any)."""
    flags = pd.DataFrame({"date": windows["date"]})
    for event, timing in [("jobs report", "night"), ("CPI", "night"), ("FOMC", "day")]:
        dates = placed.loc[(placed["event"] == event) & (placed["timing"] == timing), "date"]
        flags[f"{event} {timing}"] = flags["date"].isin(dates)
    flags["BLS night"] = flags["jobs report night"] | flags["CPI night"]
    flags["macro"] = flags["BLS night"] | flags["FOMC day"]
    return flags


def event_counts(placed: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Number of releases of each kind in the night and in the day."""
    placed = placed[placed["date"].between(start, end)]
    return pd.crosstab(placed["event"], placed["timing"]).reindex(EVENTS).fillna(0).astype(int)


def macro_regression(daily: pd.DataFrame, flags: pd.DataFrame, maxlags: int = 5) -> pd.DataFrame:
    """Extra return of release sessions, in bp, with HAC errors (5 lags, as in Part III).

    Overnight return on jobs report and CPI nights, intraday return on FOMC days.
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


def macro_regression_table(portfolios: dict[str, pd.DataFrame], flags: pd.DataFrame) -> pd.DataFrame:
    """Extra return (bp) and p-value of each release, one row per portfolio."""
    rows = {}
    for name, daily in portfolios.items():
        reg = macro_regression(daily, flags)
        rows[name] = {}
        for half, release in [("night", "jobs report night"), ("night", "CPI night"), ("day", "FOMC day")]:
            rows[name][(release, "bp")] = reg.loc[(half, release), "coef (bp)"]
            rows[name][(release, "p")] = reg.loc[(half, release), "p-value"]
        rows[name][("other days", "bp")] = reg.loc[("day", "other sessions (mean)"), "coef (bp)"]
    return pd.DataFrame.from_dict(rows, orient="index").round(3)


def macro_decomposition(daily: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Share of the summed overnight log return earned on jobs report and CPI nights."""
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
    """Night - day gap with all sessions and without the sessions that have a release."""
    no_release = flags.loc[~flags["macro"], "date"]
    rows = {}
    for name, daily in portfolios.items():
        rows[(name, "all sessions")] = gap_test(daily)
        rows[(name, "no macro release")] = gap_test(daily[daily["date"].isin(no_release)])
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["portfolio", "sample"]).round(3)


def macro_paths(daily: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Value of $1 overnight, overnight in cash on release nights, and intraday."""
    data = daily.merge(flags[["date", "BLS night"]], on="date").dropna(subset=["r_night", "r_day"])
    night_without_releases = data["r_night"].where(~data["BLS night"], 0)
    return pd.DataFrame({
        "date": data["date"].to_numpy(),
        "night": cumulative_wealth(data["r_night"]).to_numpy(),
        "night without releases": cumulative_wealth(night_without_releases).to_numpy(),
        "day": cumulative_wealth(data["r_day"]).to_numpy(),
    })
