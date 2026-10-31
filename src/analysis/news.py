import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.crime import block_bootstrap_ci, night_indicator_regression
from src.config import STUDY_START_YEAR
from src.features.calendar import build_night_calendar
from src.features.returns import TRADING_DAYS_PER_YEAR, cumulative_wealth

# First year in which EDGAR records the acceptance time of every 8-K (see filing_time_share_by_year).
# Before it, filings are only dated: enough for the quiet test, not for the night / day split.
TIMED_START_YEAR = 2003
QUIET_WINDOW = 2
STATUS_ORDER = ["verified", "unmatched", "name_mismatch", "foreign_filer", "no_8k", "download_failed"]


def session_windows(start_date: str, end_date: str) -> pd.DataFrame:
    """NYSE sessions with their night (previous close -> open) and day (open -> close), in UTC.

    Args:
        start_date: First session (YYYY-MM-DD).
        end_date: Last session (YYYY-MM-DD).

    Returns:
        One row per session: date, previous_close, market_open, market_close and session index i.
    """
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    cal = build_night_calendar(start_date, end)[["date", "previous_close", "market_open"]]
    cal["market_close"] = cal["previous_close"].shift(-1)
    cal = cal[cal["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    cal["i"] = np.arange(len(cal))
    return cal


def assign_sessions(filings: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Place each filing in the night or the trading day of a session.

    A filing accepted after the close of session t-1 and before the open of t belongs to the
    night of t; one accepted between the open and the close of t belongs to the day of t.
    Filings without a recorded time are kept with timing "unknown", on the first session on
    or after their filing date: they mark the date as ambiguous, never as quiet.

    Args:
        filings: Table returned by load_filings.
        windows: Table returned by session_windows.

    Returns:
        filings restricted to the calendar, with session date, session index i and timing
        ("night", "day" or "unknown").
    """
    closes = windows["market_close"].to_numpy()
    opens = windows["market_open"].to_numpy()
    out = filings.copy()
    known = out["time_known"].to_numpy()
    # First session whose close is strictly after the timestamp.
    j = np.searchsorted(closes, out["accepted"].to_numpy(), side="right")
    j_unknown = np.searchsorted(windows["date"].to_numpy(), out["filing_date"].to_numpy(), side="left")
    j = np.where(known, j, j_unknown)
    # Keep filings after the close that opens the first night of the calendar.
    first_close = windows["previous_close"].iloc[0]
    after_start = np.where(known, out["accepted"] > first_close,
                           out["filing_date"] > first_close.tz_convert("America/New_York").tz_localize(None).normalize())
    inside = (j < len(windows)) & after_start
    out, j, known = out[inside].copy(), j[inside], known[inside]
    out["i"] = j
    out["date"] = windows["date"].to_numpy()[j]
    is_day = out["accepted"].to_numpy() >= opens[j]
    out["timing"] = np.where(~known, "unknown", np.where(is_day, "day", "night"))
    return out.reset_index(drop=True)


def filing_time_share_by_year(filings: pd.DataFrame) -> pd.DataFrame:
    """Share of 8-K filings with a recorded acceptance time, and number of filings, per year.

    Args:
        filings: Table returned by load_filings (form 8-K only is counted).

    Returns:
        One row per year: filings, share_time_known (%).
    """
    f = filings[filings["form"].eq("8-K")]
    year = f["filing_date"].dt.year
    return pd.DataFrame({
        "filings": f.groupby(year).size(),
        "share_time_known (%)": 100 * f.groupby(year)["time_known"].mean(),
    }).rename_axis("year").round(1)


def filing_time_profile(assigned: pd.DataFrame, start_year: int = TIMED_START_YEAR) -> pd.DataFrame:
    """Distribution of 8-K acceptance times over the New York clock, in 30-minute bins.

    Args:
        assigned: Table returned by assign_sessions.
        start_year: Filings before this year are ignored.

    Returns:
        One row per bin (hour as a decimal, e.g. 16.5 = 16:30): share of filings (%) for
        earnings and other 8-Ks.
    """
    f = assigned[assigned["form"].eq("8-K") & assigned["time_known"] & (assigned["date"].dt.year >= start_year)]
    t = f["accepted_ny"]
    bin_start = t.dt.hour + (t.dt.minute // 30) / 2
    kind = np.where(f["earnings"], "earnings", "other 8-K")
    counts = pd.crosstab(bin_start, kind)
    return (100 * counts / counts.sum()).reindex(np.arange(0, 24, 0.5), fill_value=0).rename_axis("hour")


def timing_summary(assigned: pd.DataFrame, start_year: int = TIMED_START_YEAR) -> pd.DataFrame:
    """Where 8-K filings fall: share in the night, during the trading day, or unknown.

    Args:
        assigned: Table returned by assign_sessions.
        start_year: Filings before this year are ignored.

    Returns:
        One row per group (all 8-Ks, earnings, other), columns night, day, unknown (%), and n.
    """
    f = assigned[assigned["form"].eq("8-K") & (assigned["date"].dt.year >= start_year)]
    groups = {"all 8-Ks": f, "earnings (item 2.02)": f[f["earnings"]], "other 8-Ks": f[~f["earnings"]]}
    rows = {}
    for name, g in groups.items():
        share = g["timing"].value_counts(normalize=True).reindex(["night", "day", "unknown"], fill_value=0)
        rows[name] = (100 * share).round(1).to_dict() | {"n": len(g)}
    return pd.DataFrame.from_dict(rows, orient="index")


def coverage_table(identity: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """How many stocks, and how much of the study sample, can be linked to their 8-Ks.

    Args:
        identity: Table returned by verify_identity.
        stocks: Study sample restricted to stocks (rows of ticker-dates).

    Returns:
        One row per identity status: tickers, share of stock tickers (%), share of stock rows (%).
    """
    rows = stocks.groupby("ticker", observed=True).size().rename("rows")
    t = identity.merge(rows, left_on="ticker", right_index=True, how="inner")
    table = t.groupby("status").agg(tickers=("ticker", "size"), rows=("rows", "sum"))
    table["share of tickers (%)"] = 100 * table["tickers"] / table["tickers"].sum()
    table["share of rows (%)"] = 100 * table["rows"] / table["rows"].sum()
    return table.reindex([s for s in STATUS_ORDER if s in table.index]).drop(columns="rows").round(1)


def name_check_examples(identity: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Rejected ticker matches with the highest name score, to check the threshold by eye."""
    rejected = identity[identity["status"].eq("name_mismatch")].sort_values("name_score", ascending=False)
    return rejected[["ticker", "security_name", "sec_name", "former_names", "name_score"]].head(n).round(2)


def build_news_panel(
    stocks: pd.DataFrame, identity: pd.DataFrame, assigned: pd.DataFrame, windows: pd.DataFrame,
    start_year: int = STUDY_START_YEAR, quiet_window: int = QUIET_WINDOW,
) -> pd.DataFrame:
    """Attach 8-K news flags to every stock-day of verified issuers.

    Only rows of verified tickers, from start_year and from the issuer's first EDGAR filing, are
    kept ("covered"): for other rows, the absence of an 8-K does not mean the absence of news.
    Amendments (8-K/A) never count as news, but they do break a quiet window.
    The quiet flag only needs filing dates, so it is valid over the whole study period; the
    night / day flags need acceptance times, and timed is True from TIMED_START_YEAR.

    Args:
        stocks: Study sample restricted to stocks, with ticker, date, r_night and r_day.
        identity: Table returned by verify_identity.
        assigned: Table returned by assign_sessions.
        windows: Table returned by session_windows.
        start_year: First year of the news analysis.
        quiet_window: A row is quiet when its issuer has no filing of any timing in the
            sessions t - quiet_window to t + quiet_window.

    Returns:
        Covered rows with cik, timed, news_night, news_day, earnings_night, earnings_day,
        unknown (an 8-K without a recorded time on that session) and quiet.
    """
    verified = identity.loc[identity["status"].eq("verified"), ["ticker", "cik", "first_filing"]]
    panel = stocks.merge(verified, on="ticker", how="inner")
    panel = panel[(panel["date"].dt.year >= start_year) & (panel["date"] >= pd.to_datetime(panel["first_filing"]))]
    panel = panel.merge(windows[["date", "i"]], on="date", how="left", validate="many_to_one")

    f = assigned[assigned["cik"].isin(verified["cik"])]
    news = f[f["form"].eq("8-K")]
    flags = pd.DataFrame({
        "news_night": news["timing"].eq("night"),
        "news_day": news["timing"].eq("day"),
        "earnings_night": news["timing"].eq("night") & news["earnings"],
        "earnings_day": news["timing"].eq("day") & news["earnings"],
        "unknown": news["timing"].eq("unknown"),
    }).groupby([news["cik"], news["i"]]).any()
    panel = panel.merge(flags, left_on=["cik", "i"], right_index=True, how="left")
    for col in flags.columns:
        panel[col] = panel[col].astype("boolean").fillna(False).astype(bool)

    events = f[["cik", "i"]].drop_duplicates()
    near = pd.concat([events.assign(i=events["i"] + k) for k in range(-quiet_window, quiet_window + 1)])
    near = near.drop_duplicates().assign(near_filing=True)
    panel = panel.merge(near, on=["cik", "i"], how="left")
    panel["quiet"] = panel["near_filing"].isna()
    panel["timed"] = panel["date"].dt.year >= TIMED_START_YEAR
    return panel.drop(columns=["near_filing", "first_filing"]).reset_index(drop=True)


GROUPS = {
    "all": lambda p: pd.Series(True, index=p.index),
    "news": lambda p: p["timed"] & (p["news_night"] | p["news_day"]),
    "quiet": lambda p: p["quiet"],
}


def news_portfolios(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily equal-weighted overnight and intraday returns of covered stocks, by news status.

    all: every covered row. news: rows with an 8-K in the night or the day of that session
    (from TIMED_START_YEAR). quiet: rows with no filing within the quiet window. Night and day returns of a portfolio
    come from the same stock-days, so the night - day gap is not driven by composition.

    Returns:
        Group name -> table with date, r_night, r_day and n (number of stocks).
    """
    out = {}
    for name, rule in GROUPS.items():
        sub = panel[rule(panel)].dropna(subset=["r_night", "r_day"])
        g = sub.groupby("date")
        out[name] = pd.DataFrame({
            "r_night": g["r_night"].mean(), "r_day": g["r_day"].mean(), "n": g.size(),
        }).reset_index()
    return out


def news_paths(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Value of $1 invested in the overnight or the intraday half of the all and quiet portfolios."""
    parts = []
    for name in ["all", "quiet"]:
        p = portfolios[name]
        parts.append(pd.DataFrame({
            "date": p["date"], "group": name,
            "night": cumulative_wealth(p["r_night"]).to_numpy(), "day": cumulative_wealth(p["r_day"]).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def news_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    """How much of the overnight and intraday returns of stock-days comes from news days.

    Pooled over all covered stock-days from TIMED_START_YEAR: each row type's share of observations
    and share of the summed returns. If news explained the crime, nights with an 8-K would carry
    most of the summed overnight return although they are a small share of nights.
    The share of the summed return is reported for nights only: summed intraday returns are close
    to zero, so shares of them are meaningless.

    Returns:
        One row per type (night with 8-K, other night, day with 8-K, other day): observations,
        share of observations (%), mean (bp), share of the summed overnight return (%).
    """
    p = panel[panel["timed"]].dropna(subset=["r_night", "r_day"])
    r_night, r_day = p["r_night"].astype("float64"), p["r_day"].astype("float64")
    parts = {
        "night with 8-K": r_night[p["news_night"]], "other night": r_night[~p["news_night"]],
        "day with 8-K": r_day[p["news_day"]], "other day": r_day[~p["news_day"]],
    }
    rows = {}
    for name, r in parts.items():
        rows[name] = {
            "observations": len(r),
            "share of observations (%)": 100 * len(r) / len(p),
            "mean (bp)": 1e4 * r.mean(),
            "share of summed return (%)": 100 * r.sum() / r_night.sum() if "night" in name else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def quiet_gap_tests(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Night-indicator regression and block bootstrap of the gap, for all and quiet stock-days.

    Same tests as Part II (date-clustered SE; 21-day blocks), on the same covered stocks.

    Returns:
        One row per group: annualised gap (%), clustered 95 % CI and p-value, bootstrap CI.
    """
    rows = []
    for name in ["all", "quiet", "news"]:
        p = portfolios[name]
        reg = night_indicator_regression(p).iloc[0]
        boot = block_bootstrap_ci(p["r_night"] - p["r_day"]).iloc[0]
        rows.append({
            "stock-days": int(p["n"].sum()), "dates": len(p),
            "gap annualised (%)": reg["beta annualised (%)"],
            "CI low (clustered)": reg["CI 95% low"], "CI high (clustered)": reg["CI 95% high"],
            "p-value": reg["p-value"],
            "CI low (bootstrap)": boot["CI 95% low"], "CI high (bootstrap)": boot["CI 95% high"],
        })
    return pd.DataFrame(rows, index=["all covered", "quiet (no 8-K within ±2 sessions)", "news (8-K that session)"]).round(3)


def quiet_gap_by_period(portfolios: dict[str, pd.DataFrame], splits: tuple[int, ...] = (2003, 2008)) -> pd.DataFrame:
    """Annualised night - day gap of the all and quiet portfolios in each sub-period.

    Args:
        portfolios: Output of news_portfolios.
        splits: First years of the second, third... periods.

    Returns:
        One row per (period, group): gap, clustered 95 % CI, p-value and number of dates.
    """
    rows = {}
    for name in ["all", "quiet"]:
        p = portfolios[name]
        years = p["date"].dt.year
        bounds = [years.min(), *splits, years.max() + 1]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            reg = night_indicator_regression(p[(years >= lo) & (years < hi)]).iloc[0]
            rows[(f"{lo}–{hi - 1}", name)] = {"gap annualised (%)": reg["beta annualised (%)"], "CI low": reg["CI 95% low"],
                                              "CI high": reg["CI 95% high"], "p-value": reg["p-value"], "dates": int(reg["n days"])}
    table = pd.DataFrame.from_dict(rows, orient="index").round(3)
    return table.rename_axis(["period", "group"]).sort_index(level=0, sort_remaining=False)


def timing_validation(panel: pd.DataFrame) -> pd.DataFrame:
    """Placebo-style check that 8-Ks are placed in the right half of the session.

    If the timestamps are right, a night 8-K should make the overnight move larger but not the
    intraday one, and a day 8-K the reverse. Values are mean absolute returns (bp).

    Returns:
        One row per row type (no 8-K, night 8-K only, day 8-K only), columns |r_night| and |r_day| (bp), n.
    """
    p = panel[panel["timed"]].dropna(subset=["r_night", "r_day"])
    kind = np.select(
        [~p["news_night"] & ~p["news_day"] & ~p["unknown"], p["news_night"] & ~p["news_day"], p["news_day"] & ~p["news_night"]],
        ["no 8-K", "night 8-K only", "day 8-K only"], default="other")
    g = p.assign(kind=kind, abs_night=1e4 * p["r_night"].abs(), abs_day=1e4 * p["r_day"].abs()).groupby("kind")
    table = pd.DataFrame({"|r_night| (bp)": g["abs_night"].mean(), "|r_day| (bp)": g["abs_day"].mean(), "n": g.size()})
    return table.reindex(["no 8-K", "night 8-K only", "day 8-K only"]).round(1)


def news_regression(panel: pd.DataFrame, half: str = "night") -> pd.DataFrame:
    """Firm-level regression of returns on 8-K news, with date fixed effects.

    r_{i,t} = a_t + b1 * earnings_{i,t} + b2 * other8K_{i,t} + e_{i,t}, for the overnight (half="night")
    or intraday (half="day") return and for its absolute value. Date effects remove every
    market-wide move (they are absorbed by demeaning within each date). Standard errors are
    clustered by date and by stock (two-way). Coefficients are in bp per stock-day.

    Returns:
        One row per (outcome, regressor): coefficient (bp), 95 % CI, p-value, observations.
    """
    p = panel[panel["timed"]].dropna(subset=[f"r_{half}"])
    earnings = p[f"earnings_{half}"].astype(float)
    other = (p[f"news_{half}"] & ~p[f"earnings_{half}"]).astype(float)
    rows = []
    for outcome, y in {f"r_{half}": p[f"r_{half}"], f"|r_{half}|": p[f"r_{half}"].abs()}.items():
        data = pd.DataFrame({"y": 1e4 * y.astype(float), "earnings 8-K": earnings, "other 8-K": other})
        demeaned = data - data.groupby(p["date"]).transform("mean")
        groups = np.column_stack([pd.factorize(p["date"])[0], pd.factorize(p["ticker"])[0]])
        fit = sm.OLS(demeaned["y"], demeaned[["earnings 8-K", "other 8-K"]]).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        ci = fit.conf_int()
        for name in ["earnings 8-K", "other 8-K"]:
            rows.append({"outcome": outcome, "regressor": name, "coef (bp)": fit.params[name],
                         "CI low": ci.loc[name, 0], "CI high": ci.loc[name, 1], "p-value": fit.pvalues[name],
                         "observations": int(fit.nobs)})
    return pd.DataFrame(rows).set_index(["outcome", "regressor"]).round(3)


def annualised_means(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Annualised mean overnight and intraday returns (%) of each news portfolio."""
    return pd.DataFrame({
        name: {"night (%)": 100 * TRADING_DAYS_PER_YEAR * p["r_night"].mean(),
               "day (%)": 100 * TRADING_DAYS_PER_YEAR * p["r_day"].mean(),
               "mean stocks per date": p["n"].mean()}
        for name, p in portfolios.items()
    }).T.round(2)


ISSUER_GROUPS = {
    "US filer, same ticker today": lambda t: t["status"].eq("verified") & t["match"].eq("ticker"),
    "US filer, delisted or renamed since 2020": lambda t: t["status"].eq("verified") & t["match"].eq("name"),
    "foreign filer (6-K / 20-F)": lambda t: t["status"].eq("foreign_filer"),
    "not linked to EDGAR": lambda t: ~t["status"].isin(["verified", "foreign_filer"]),
}


def gap_by_issuer_group(stocks: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Night - day gap of equal-weighted stock portfolios, by kind of issuer.

    Shows which stocks carry the crime, and therefore what the news test (verified issuers only)
    can and cannot say about the whole sample.

    Args:
        stocks: Study sample restricted to stocks.
        identity: Table returned by verify_identity.

    Returns:
        One row per issuer group: tickers, stock-days, annualised gap (%), clustered 95 % CI, p-value.
    """
    rows = {}
    for name, rule in ISSUER_GROUPS.items():
        tickers = identity.loc[rule(identity), "ticker"]
        sub = stocks[stocks["ticker"].isin(tickers)].dropna(subset=["r_night", "r_day"])
        daily = sub.groupby("date")[["r_night", "r_day"]].mean().reset_index()
        reg = night_indicator_regression(daily).iloc[0]
        rows[name] = {"tickers": sub["ticker"].nunique(), "stock-days": len(sub),
                      "gap annualised (%)": reg["beta annualised (%)"], "CI low": reg["CI 95% low"],
                      "CI high": reg["CI 95% high"], "p-value": reg["p-value"]}
    return pd.DataFrame.from_dict(rows, orient="index").round(3)
