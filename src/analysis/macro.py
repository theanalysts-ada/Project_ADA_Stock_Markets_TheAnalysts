import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.crime import night_indicator_regression
from src.analysis.news import assign_sessions
from src.features.returns import cumulative_wealth

BLS_EVENTS = ["jobs report", "CPI"]


def place_events(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Place each macroeconomic release in the night or the trading day of a session.

    Same rule as for 8-K filings (assign_sessions): an 8:30 release belongs to the night of that
    day's session, or of the next session when the market is closed (e.g. a jobs report on Good
    Friday); a 14:00 FOMC statement belongs to the day of its session.

    Args:
        events: Tables returned by fomc_dates and bls_release_dates, concatenated.
        windows: Table returned by session_windows.

    Returns:
        One row per release: event, release, session date and timing ("night" or "day").
    """
    as_filings = pd.DataFrame({
        "event": events["event"].to_numpy(),
        "accepted": events["release"].dt.tz_convert("UTC").to_numpy(),
        "filing_date": events["date"].to_numpy(),
        "time_known": True,
    })
    placed = assign_sessions(as_filings, windows)
    return placed[["event", "accepted", "date", "timing"]].rename(columns={"accepted": "release"})


def macro_flags(placed: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """One row per session with a boolean column per (event, half), e.g. "CPI night", "FOMC day".

    Also adds "BLS night" (any 8:30 release in the night) and "macro" (any release in the
    session's night or day).
    """
    keys = placed["event"] + " " + placed["timing"]
    flags = pd.crosstab(placed["date"], keys).gt(0)
    flags = flags.reindex(windows["date"], fill_value=False).rename_axis("date")
    for event in BLS_EVENTS + ["FOMC"]:
        for half in ["night", "day"]:
            if f"{event} {half}" not in flags:
                flags[f"{event} {half}"] = False
    flags["BLS night"] = flags[[f"{e} night" for e in BLS_EVENTS]].any(axis=1)
    flags["macro"] = flags.any(axis=1)
    return flags.reset_index()


def event_counts(placed: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Number of releases of each event in the night and in the trading day, over [start, end]."""
    p = placed[placed["date"].between(start, end)]
    return pd.crosstab(p["event"], p["timing"]).reindex(BLS_EVENTS + ["FOMC"]).fillna(0).astype(int)


def macro_regression(daily: pd.DataFrame, flags: pd.DataFrame, maxlags: int = 5) -> pd.DataFrame:
    """Mean overnight and intraday return on release sessions versus other sessions.

    r_t = a + sum_k b_k 1[event k in that half of session t] + e_t, for the overnight half
    (jobs report and CPI nights) and the intraday half (FOMC days), in bp, with HAC standard
    errors (maxlags, as in Part III). b_k is the extra return of a release session.

    Args:
        daily: Daily portfolio with date, r_night and r_day.
        flags: Table returned by macro_flags.

    Returns:
        One row per (half, regressor): coefficient (bp), 95 % CI, p-value, number of release
        sessions; plus the constant (mean on other sessions).
    """
    data = daily.merge(flags, on="date", how="inner")
    rows = []
    for half, regressors in {"night": [f"{e} night" for e in BLS_EVENTS], "day": ["FOMC day"]}.items():
        sub = data.dropna(subset=[f"r_{half}"])
        design = sm.add_constant(sub[regressors].astype(float))
        fit = sm.OLS(1e4 * sub[f"r_{half}"].astype(float), design).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        ci = fit.conf_int()
        for name in ["const"] + regressors:
            rows.append({"half": half, "regressor": "other sessions (mean)" if name == "const" else name,
                         "coef (bp)": fit.params[name], "CI low": ci.loc[name, 0], "CI high": ci.loc[name, 1],
                         "p-value": fit.pvalues[name],
                         "sessions": int(len(sub) if name == "const" else sub[name].sum())})
    return pd.DataFrame(rows).set_index(["half", "regressor"]).round(3)


def macro_decomposition(daily: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Share of sessions and of the summed overnight log return carried by 8:30 release nights.

    Returns:
        Two rows (nights with a jobs report or CPI release, other nights): nights, share of
        nights (%), mean (bp), share of the summed overnight log return (%).
    """
    data = daily.merge(flags, on="date", how="inner").dropna(subset=["r_night"])
    log_night = np.log1p(data["r_night"].astype(float))
    rows = {}
    for name, mask in {"release nights (8:30)": data["BLS night"], "other nights": ~data["BLS night"]}.items():
        rows[name] = {"nights": int(mask.sum()), "share of nights (%)": 100 * mask.mean(),
                      "mean (bp)": 1e4 * data.loc[mask, "r_night"].astype(float).mean(),
                      "share of summed log return (%)": 100 * log_night[mask].sum() / log_night.sum()}
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def macro_free_gap(portfolios: dict[str, pd.DataFrame], flags: pd.DataFrame) -> pd.DataFrame:
    """Night - day gap of each portfolio on all sessions and on sessions without any release.

    A session is dropped (both halves) when a jobs report, CPI or FOMC statement falls in
    its night or its day. Same regression as Part II, standard errors clustered by date.

    Returns:
        One row per (portfolio, sample): annualised gap (%), 95 % CI, p-value, number of dates.
    """
    quiet_dates = flags.loc[~flags["macro"], "date"]
    rows = {}
    for name, daily in portfolios.items():
        for sample, sub in {"all sessions": daily, "no macro release": daily[daily["date"].isin(quiet_dates)]}.items():
            reg = night_indicator_regression(sub).iloc[0]
            rows[(name, sample)] = {"gap annualised (%)": reg["beta annualised (%)"], "CI low": reg["CI 95% low"],
                                    "CI high": reg["CI 95% high"], "p-value": reg["p-value"], "dates": int(reg["n days"])}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["portfolio", "sample"]).round(3)


def macro_paths(daily: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """$1 invested overnight on all nights, on nights without an 8:30 release, and intraday.

    A skipped night earns nothing (the investor stays in cash), so the gap between the first two
    paths is the part of the overnight wealth earned on release nights.
    """
    data = daily.merge(flags[["date", "BLS night"]], on="date", how="inner").dropna(subset=["r_night", "r_day"])
    return pd.DataFrame({
        "date": data["date"].to_numpy(),
        "night": cumulative_wealth(data["r_night"]).to_numpy(),
        "night without releases": cumulative_wealth(data["r_night"].where(~data["BLS night"], 0)).to_numpy(),
        "day": cumulative_wealth(data["r_day"]).to_numpy(),
    })

