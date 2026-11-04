import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.crime import block_bootstrap_ci, night_indicator_regression
from src.config import STUDY_START_YEAR
from src.features.calendar import build_night_calendar
from src.features.returns import TRADING_DAYS_PER_YEAR, cumulative_wealth

# EDGAR records the time of every 8-K from 2003 (see filing_time_share_by_year).
# Before that we only know the date, which is enough for quiet days but not for the night / day split.
TIMED_START_YEAR = 2003
QUIET_WINDOW = 2
STATUS_ORDER = ["verified", "unmatched", "name_mismatch", "foreign_filer", "no_8k", "download_failed"]


def session_windows(start_date: str, end_date: str) -> pd.DataFrame:
    """NYSE sessions with the close of the previous session, the open and the close (UTC).

    Args:
        start_date: First session (YYYY-MM-DD).
        end_date: Last session (YYYY-MM-DD).

    Returns:
        One row per session: date, previous_close, market_open, market_close and i (session number).
    """
    end = pd.Timestamp(end_date) + pd.Timedelta(days=15)  # to get the close of the last session
    cal = build_night_calendar(start_date, end.strftime("%Y-%m-%d"))
    cal = cal[["date", "previous_close", "market_open"]].copy()
    cal["market_close"] = cal["previous_close"].shift(-1)
    cal = cal[cal["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    cal["i"] = np.arange(len(cal))
    return cal


def assign_sessions(filings: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Put each filing in the night or in the trading day of a session.

    A filing accepted between the close of session t-1 and the open of session t is in the night
    of t. A filing accepted between the open and the close of t is in the day of t.
    A filing without a time is put on the first session on or after its filing date, with timing
    "unknown".

    Args:
        filings: Table returned by load_filings.
        windows: Table returned by session_windows.

    Returns:
        The filings inside the calendar, with the session date, i and timing ("night", "day" or "unknown").
    """
    filings = filings.copy()
    known = filings["time_known"].to_numpy()

    # Session of each filing: the first session that closes after it (or that starts on or after its date).
    closes = windows["market_close"].to_numpy()
    by_time = np.searchsorted(closes, filings["accepted"].to_numpy(), side="right")
    by_date = np.searchsorted(windows["date"].to_numpy(), filings["filing_date"].to_numpy(), side="left")
    i = np.where(known, by_time, by_date)

    # Drop filings outside the calendar.
    first_close = windows["previous_close"].iloc[0]
    first_date = first_close.tz_convert("America/New_York").tz_localize(None).normalize()
    after_start = np.where(known, filings["accepted"] > first_close, filings["filing_date"] > first_date)
    inside = (i < len(windows)) & after_start
    filings, i, known = filings[inside].copy(), i[inside], known[inside]

    filings["i"] = i
    filings["date"] = windows["date"].to_numpy()[i]
    during_session = filings["accepted"].to_numpy() >= windows["market_open"].to_numpy()[i]
    filings["timing"] = "night"
    filings.loc[during_session, "timing"] = "day"
    filings.loc[~known, "timing"] = "unknown"
    return filings.reset_index(drop=True)


def filing_time_share_by_year(filings: pd.DataFrame) -> pd.DataFrame:
    """Number of 8-Ks per year and share of them with a recorded time.

    Args:
        filings: Table returned by load_filings.

    Returns:
        One row per year: filings and share_time_known (%).
    """
    eight_k = filings[filings["form"] == "8-K"]
    year = eight_k["filing_date"].dt.year
    table = pd.DataFrame({
        "filings": eight_k.groupby(year).size(),
        "share_time_known (%)": 100 * eight_k.groupby(year)["time_known"].mean(),
    })
    return table.rename_axis("year").round(1)


def filing_time_profile(assigned: pd.DataFrame, start_year: int = TIMED_START_YEAR) -> pd.DataFrame:
    """Share of 8-Ks accepted in each 30-minute slot of the New York clock.

    Args:
        assigned: Table returned by assign_sessions.
        start_year: First year used.

    Returns:
        One row per slot (16.5 means 16:30 to 17:00), one column for earnings 8-Ks and one for the others (%).
    """
    keep = (assigned["form"] == "8-K") & assigned["time_known"] & (assigned["date"].dt.year >= start_year)
    eight_k = assigned[keep]
    time = eight_k["accepted_ny"]
    slot = time.dt.hour + (time.dt.minute // 30) / 2
    kind = eight_k["earnings"].map({True: "earnings", False: "other 8-K"})
    counts = pd.crosstab(slot, kind)
    shares = 100 * counts / counts.sum()
    return shares.reindex(np.arange(0, 24, 0.5), fill_value=0).rename_axis("hour")


def timing_summary(assigned: pd.DataFrame, start_year: int = TIMED_START_YEAR) -> pd.DataFrame:
    """Share of 8-Ks filed in the night, in the trading day, or at an unknown time.

    Args:
        assigned: Table returned by assign_sessions.
        start_year: First year used.

    Returns:
        One row for all 8-Ks, earnings 8-Ks and other 8-Ks: night, day, unknown (%) and n.
    """
    eight_k = assigned[(assigned["form"] == "8-K") & (assigned["date"].dt.year >= start_year)]
    groups = {
        "all 8-Ks": eight_k,
        "earnings (item 2.02)": eight_k[eight_k["earnings"]],
        "other 8-Ks": eight_k[~eight_k["earnings"]],
    }
    rows = {}
    for name, group in groups.items():
        share = group["timing"].value_counts(normalize=True)
        rows[name] = {
            "night": 100 * share.get("night", 0),
            "day": 100 * share.get("day", 0),
            "unknown": 100 * share.get("unknown", 0),
            "n": len(group),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(1)


def coverage_table(identity: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """Number of stocks and share of the study sample in each identity status.

    Args:
        identity: Table returned by verify_identity.
        stocks: Study sample restricted to stocks.

    Returns:
        One row per status: tickers, share of tickers (%) and share of rows (%).
    """
    rows_per_ticker = stocks.groupby("ticker", observed=True).size().rename("rows")
    table = identity.merge(rows_per_ticker, left_on="ticker", right_index=True)
    table = table.groupby("status").agg(tickers=("ticker", "size"), rows=("rows", "sum"))
    table["share of tickers (%)"] = 100 * table["tickers"] / table["tickers"].sum()
    table["share of rows (%)"] = 100 * table["rows"] / table["rows"].sum()
    order = [s for s in STATUS_ORDER if s in table.index]
    return table.loc[order].drop(columns="rows").round(1)


def name_check_examples(identity: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Rejected matches with the highest name scores, to check the threshold by eye."""
    rejected = identity[identity["status"] == "name_mismatch"]
    rejected = rejected.sort_values("name_score", ascending=False).head(n)
    return rejected[["ticker", "security_name", "sec_name", "former_names", "name_score"]].round(2)


def build_news_panel(
    stocks: pd.DataFrame, identity: pd.DataFrame, assigned: pd.DataFrame, windows: pd.DataFrame,
    start_year: int = STUDY_START_YEAR, quiet_window: int = QUIET_WINDOW,
) -> pd.DataFrame:
    """Add 8-K news flags to each stock-day of the verified companies.

    We only keep the stock-days of verified tickers, after the company's first EDGAR filing
    ("covered" stock-days). For the other stock-days, no 8-K does not mean no news.
    A stock-day is quiet when the company filed nothing at all (8-K, 8-K/A or 8-K without a time)
    in the sessions t - quiet_window to t + quiet_window. The night / day flags need the filing
    time, so they are only reliable from TIMED_START_YEAR (column timed).

    Args:
        stocks: Study sample restricted to stocks, with ticker, date, r_night and r_day.
        identity: Table returned by verify_identity.
        assigned: Table returned by assign_sessions.
        windows: Table returned by session_windows.
        start_year: First year.
        quiet_window: Number of sessions around the day without any filing for a quiet day.

    Returns:
        The covered stock-days with cik, news_night, news_day, earnings_night, earnings_day,
        unknown (8-K without a time), quiet and timed.
    """
    verified = identity.loc[identity["status"] == "verified", ["ticker", "cik", "first_filing"]]
    panel = stocks.merge(verified, on="ticker")
    after_first_filing = panel["date"] >= pd.to_datetime(panel["first_filing"])
    panel = panel[(panel["date"].dt.year >= start_year) & after_first_filing]
    panel = panel.merge(windows[["date", "i"]], on="date", how="left")

    filings = assigned[assigned["cik"].isin(verified["cik"])]
    eight_k = filings[filings["form"] == "8-K"]
    is_night = eight_k["timing"] == "night"
    is_day = eight_k["timing"] == "day"
    flags = pd.DataFrame({
        "cik": eight_k["cik"],
        "i": eight_k["i"],
        "news_night": is_night,
        "news_day": is_day,
        "earnings_night": is_night & eight_k["earnings"],
        "earnings_day": is_day & eight_k["earnings"],
        "unknown": eight_k["timing"] == "unknown",
    })
    flags = flags.groupby(["cik", "i"]).any().reset_index()
    panel = panel.merge(flags, on=["cik", "i"], how="left")
    for col in ["news_night", "news_day", "earnings_night", "earnings_day", "unknown"]:
        panel[col] = panel[col].eq(True)

    # Sessions close to a filing: each filing session shifted by -quiet_window ... +quiet_window.
    filing_sessions = filings[["cik", "i"]].drop_duplicates()
    shifts = range(-quiet_window, quiet_window + 1)
    near = pd.concat([filing_sessions.assign(i=filing_sessions["i"] + k) for k in shifts]).drop_duplicates()
    near["near_filing"] = True
    panel = panel.merge(near, on=["cik", "i"], how="left")
    panel["quiet"] = panel["near_filing"].isna()
    panel["timed"] = panel["date"].dt.year >= TIMED_START_YEAR
    return panel.drop(columns=["near_filing", "first_filing"])


def daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted overnight and intraday returns per date, with the number of stocks n."""
    by_date = df.groupby("date")
    daily = pd.DataFrame({
        "r_night": by_date["r_night"].mean(),
        "r_day": by_date["r_day"].mean(),
        "n": by_date.size(),
    })
    return daily.reset_index()


def news_portfolios(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily equal-weighted portfolios of covered stock-days: all, with news, and quiet.

    news: stock-days with an 8-K in the night or the day (from TIMED_START_YEAR).
    quiet: stock-days without any filing in the quiet window.
    In each portfolio the night and the day returns come from the same stock-days.

    Returns:
        Portfolio name -> table with date, r_night, r_day and n.
    """
    covered = panel.dropna(subset=["r_night", "r_day"])
    with_news = covered["timed"] & (covered["news_night"] | covered["news_day"])
    return {
        "all": daily_mean(covered),
        "news": daily_mean(covered[with_news]),
        "quiet": daily_mean(covered[covered["quiet"]]),
    }


def news_paths(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Value of $1 invested overnight only or intraday only, in the all and quiet portfolios."""
    paths = []
    for name in ["all", "quiet"]:
        p = portfolios[name]
        paths.append(pd.DataFrame({
            "date": p["date"],
            "group": name,
            "night": cumulative_wealth(p["r_night"]).to_numpy(),
            "day": cumulative_wealth(p["r_day"]).to_numpy(),
        }))
    return pd.concat(paths, ignore_index=True)


def news_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    """Share of the stock-nights with an 8-K, and their share of the summed overnight return.

    If news explained the crime, the few nights with an 8-K would carry most of the overnight return.
    We do not compute the share for the day: the summed intraday return is close to zero.
    Uses covered stock-days from TIMED_START_YEAR.

    Returns:
        One row for nights with an 8-K, other nights, days with an 8-K and other days:
        observations, share of observations (%), mean (bp), share of the summed overnight return (%).
    """
    p = panel[panel["timed"]].dropna(subset=["r_night", "r_day"])
    r_night = p["r_night"].astype("float64")
    r_day = p["r_day"].astype("float64")
    parts = {
        "night with 8-K": r_night[p["news_night"]],
        "other night": r_night[~p["news_night"]],
        "day with 8-K": r_day[p["news_day"]],
        "other day": r_day[~p["news_day"]],
    }
    rows = {}
    for name, r in parts.items():
        rows[name] = {
            "observations": len(r),
            "share of observations (%)": 100 * len(r) / len(p),
            "mean (bp)": 10000 * r.mean(),
            "share of summed return (%)": 100 * r.sum() / r_night.sum() if "night" in name else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def gap_test(daily: pd.DataFrame) -> dict:
    """Annualised night - day gap with its clustered CI and p-value (Part II regression)."""
    reg = night_indicator_regression(daily).iloc[0]
    return {
        "gap annualised (%)": reg["beta annualised (%)"],
        "CI low": reg["CI 95% low"],
        "CI high": reg["CI 95% high"],
        "p-value": reg["p-value"],
        "dates": int(reg["n days"]),
    }


def quiet_gap_tests(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Night - day gap of the all, quiet and news portfolios, with the two tests of Part II.

    Returns:
        One row per portfolio: stock-days, dates, gap (%), CI and p-value (clustered by date),
        and the block bootstrap CI.
    """
    labels = {"all": "all covered", "quiet": "quiet (no filing within 2 sessions)", "news": "with an 8-K"}
    rows = {}
    for name, label in labels.items():
        p = portfolios[name]
        test = gap_test(p)
        boot = block_bootstrap_ci(p["r_night"] - p["r_day"]).iloc[0]
        rows[label] = {
            "stock-days": int(p["n"].sum()),
            "dates": test["dates"],
            "gap annualised (%)": test["gap annualised (%)"],
            "CI low (clustered)": test["CI low"],
            "CI high (clustered)": test["CI high"],
            "p-value": test["p-value"],
            "CI low (bootstrap)": boot["CI 95% low"],
            "CI high (bootstrap)": boot["CI 95% high"],
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(3)


def quiet_gap_by_period(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Night - day gap of the all and quiet portfolios in 1998-2002, 2003-2007 and 2008-2020.

    Returns:
        One row per (period, portfolio): gap (%), CI, p-value and number of dates.
    """
    rows = {}
    for first, last in [(1998, 2002), (2003, 2007), (2008, 2020)]:
        for name in ["all", "quiet"]:
            p = portfolios[name]
            in_period = p["date"].dt.year.between(first, last)
            rows[(f"{first}–{last}", name)] = gap_test(p[in_period])
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["period", "group"]).round(3)


def timing_validation(panel: pd.DataFrame) -> pd.DataFrame:
    """Check that the 8-Ks are put in the right half of the session.

    If the timing is right, an 8-K in the night should make the overnight move bigger, and an 8-K
    in the day should make the intraday move bigger. Uses covered stock-days from TIMED_START_YEAR.

    Returns:
        One row for stock-days without 8-K, with a night 8-K only and with a day 8-K only:
        mean absolute overnight and intraday returns (bp) and n.
    """
    p = panel[panel["timed"]].dropna(subset=["r_night", "r_day"])
    groups = {
        "no 8-K": ~p["news_night"] & ~p["news_day"] & ~p["unknown"],
        "night 8-K only": p["news_night"] & ~p["news_day"],
        "day 8-K only": p["news_day"] & ~p["news_night"],
    }
    rows = {}
    for name, mask in groups.items():
        rows[name] = {
            "|r_night| (bp)": 10000 * p.loc[mask, "r_night"].abs().mean(),
            "|r_day| (bp)": 10000 * p.loc[mask, "r_day"].abs().mean(),
            "n": int(mask.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(1)


def news_regression(panel: pd.DataFrame, half: str = "night") -> pd.DataFrame:
    """Effect of an 8-K on the return of a stock, with date fixed effects.

    r_it = a_t + b1 * earnings_it + b2 * other_8K_it + e_it, for the overnight (half="night") or
    intraday (half="day") return, and for its absolute value. The date effects a_t remove the
    moves of the whole market; we estimate them by subtracting the mean of each date.
    Standard errors are clustered by date and by stock. Uses covered stock-days from TIMED_START_YEAR.

    Returns:
        One row per (outcome, regressor): coefficient (bp), 95 % CI, p-value and observations.
    """
    p = panel[panel["timed"]].dropna(subset=[f"r_{half}"])
    r = 10000 * p[f"r_{half}"].astype(float)
    earnings = p[f"earnings_{half}"]
    other = p[f"news_{half}"] & ~earnings
    clusters = np.column_stack([pd.factorize(p["date"])[0], pd.factorize(p["ticker"])[0]])

    X = pd.DataFrame({"earnings 8-K": earnings.astype(float), "other 8-K": other.astype(float)})
    X = X - X.groupby(p["date"]).transform("mean")
    rows = []
    for outcome, y in {f"r_{half}": r, f"|r_{half}|": r.abs()}.items():
        y = y - y.groupby(p["date"]).transform("mean")
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        ci = fit.conf_int()
        for name in ["earnings 8-K", "other 8-K"]:
            rows.append({
                "outcome": outcome,
                "regressor": name,
                "coef (bp)": fit.params[name],
                "CI low": ci.loc[name, 0],
                "CI high": ci.loc[name, 1],
                "p-value": fit.pvalues[name],
                "observations": int(fit.nobs),
            })
    return pd.DataFrame(rows).set_index(["outcome", "regressor"]).round(3)


def annualised_means(portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Annualised mean overnight and intraday returns (%) of each news portfolio."""
    rows = {}
    for name, p in portfolios.items():
        rows[name] = {
            "night (%)": 100 * TRADING_DAYS_PER_YEAR * p["r_night"].mean(),
            "day (%)": 100 * TRADING_DAYS_PER_YEAR * p["r_day"].mean(),
            "mean stocks per date": p["n"].mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def gap_by_issuer_group(stocks: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Night - day gap of equal-weighted stock portfolios for each kind of company.

    This shows which stocks carry the crime, and so what the news test (verified companies only) can see.

    Args:
        stocks: Study sample restricted to stocks.
        identity: Table returned by verify_identity.

    Returns:
        One row per group: tickers, stock-days, gap (%), CI and p-value.
    """
    verified = identity["status"] == "verified"
    groups = {
        "US filer, same ticker today": verified & (identity["match"] == "ticker"),
        "US filer, delisted or renamed since 2020": verified & (identity["match"] == "name"),
        "foreign filer (6-K / 20-F)": identity["status"] == "foreign_filer",
        "not linked to EDGAR": ~identity["status"].isin(["verified", "foreign_filer"]),
    }
    rows = {}
    for name, mask in groups.items():
        tickers = identity.loc[mask, "ticker"]
        group = stocks[stocks["ticker"].isin(tickers)].dropna(subset=["r_night", "r_day"])
        test = gap_test(daily_mean(group))
        rows[name] = {"tickers": group["ticker"].nunique(), "stock-days": len(group)}
        rows[name].update({k: v for k, v in test.items() if k != "dates"})
    return pd.DataFrame.from_dict(rows, orient="index").round(3)
