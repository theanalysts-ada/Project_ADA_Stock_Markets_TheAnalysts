import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.crime import block_bootstrap_ci, night_indicator_regression
from src.config import STUDY_START_YEAR
from src.features.calendar import build_night_calendar
from src.features.returns import cumulative_wealth

# Return units: x100 for %, x10000 for bp; rounding only controls displayed precision.

# Filing times exist from 2003.
TIMED_START_YEAR = 2003
# Exclude filings within two sessions on either side: allow for delayed 8-K reporting.
QUIET_WINDOW = 2
# Keep coverage tables in the same order, from verified to unavailable identities.
STATUS_ORDER = ["verified", "unmatched", "name_mismatch", "foreign_filer", "no_8k", "download_failed"]


def session_windows(start_date: str, end_date: str) -> pd.DataFrame:
    """NYSE sessions with previous_close, market_open, market_close (UTC) and i (session number)."""
    # 15 extra calendar days supply a following session to recover the last close.
    end = pd.Timestamp(end_date) + pd.Timedelta(days=15)
    cal = build_night_calendar(start_date, end.strftime("%Y-%m-%d"))
    cal = cal[["date", "previous_close", "market_open"]].copy()
    cal["market_close"] = cal["previous_close"].shift(-1)
    cal = cal[cal["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    cal["i"] = np.arange(len(cal))
    return cal


def assign_sessions(filings: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Put each filing in the night (previous close -> open) or the day (open -> close) of a session.

    Filings without a time go to the first session on or after their date, with timing "unknown".
    """
    filings = filings.copy()
    known = filings["time_known"].to_numpy()

    closes = windows["market_close"].to_numpy()
    by_time = np.searchsorted(closes, filings["accepted"].to_numpy(), side="right")
    by_date = np.searchsorted(windows["date"].to_numpy(), filings["filing_date"].to_numpy(), side="left")
    i = np.where(known, by_time, by_date)

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
    """Number of 8-Ks per year and share of them with a recorded time."""
    eight_k = filings[filings["form"] == "8-K"]
    by_year = eight_k.groupby(eight_k["filing_date"].dt.year)
    table = pd.DataFrame({"filings": by_year.size(), "share_time_known (%)": 100 * by_year["time_known"].mean()})
    return table.rename_axis("year").round(1)


def filing_time_profile(assigned: pd.DataFrame) -> pd.DataFrame:
    """Share of 8-Ks accepted in each 30-minute slot of the New York clock."""
    keep = (assigned["form"] == "8-K") & assigned["time_known"] & (assigned["date"].dt.year >= TIMED_START_YEAR)
    eight_k = assigned[keep]
    time = eight_k["accepted_ny"]
    # 30-minute bins: two slots per hour over the 24-hour clock.
    slot = time.dt.hour + (time.dt.minute // 30) / 2
    kind = eight_k["earnings"].map({True: "earnings", False: "other 8-K"})
    counts = pd.crosstab(slot, kind)
    shares = 100 * counts / counts.sum()
    return shares.reindex(np.arange(0, 24, 0.5), fill_value=0).rename_axis("hour")


def timing_summary(assigned: pd.DataFrame) -> pd.DataFrame:
    """Share of 8-Ks filed in the night, in the trading day, or at an unknown time."""
    eight_k = assigned[(assigned["form"] == "8-K") & (assigned["date"].dt.year >= TIMED_START_YEAR)]
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
    """Number of stocks and share of the study sample in each identity status."""
    rows_per_ticker = stocks.groupby("ticker", observed=True).size().rename("rows")
    table = identity.merge(rows_per_ticker, left_on="ticker", right_index=True)
    table = table.groupby("status").agg(tickers=("ticker", "size"), rows=("rows", "sum"))
    table["share of tickers (%)"] = 100 * table["tickers"] / table["tickers"].sum()
    table["share of rows (%)"] = 100 * table["rows"] / table["rows"].sum()
    order = [s for s in STATUS_ORDER if s in table.index]
    return table.loc[order].drop(columns="rows").round(1)


def build_news_panel(
    stocks: pd.DataFrame, identity: pd.DataFrame, assigned: pd.DataFrame, windows: pd.DataFrame,
    start_year: int = STUDY_START_YEAR, quiet_window: int = QUIET_WINDOW,
) -> pd.DataFrame:
    """Stock-days of verified companies, after their first EDGAR filing, with 8-K flags.

    A stock-day is quiet when the company filed nothing within quiet_window sessions.
    Night / day flags need the filing time, so they are reliable only when timed is True.
    """
    verified = identity.loc[identity["status"] == "verified", ["ticker", "cik", "first_filing"]]
    panel = stocks.merge(verified, on="ticker")
    after_first_filing = panel["date"] >= pd.to_datetime(panel["first_filing"])
    panel = panel[(panel["date"].dt.year >= start_year) & after_first_filing]
    panel = panel.merge(windows[["date", "i"]], on="date", how="left")

    eight_k = assigned[assigned["form"] == "8-K"]
    is_night = eight_k["timing"] == "night"
    is_day = eight_k["timing"] == "day"
    flags = pd.DataFrame({
        "cik": eight_k["cik"],
        "i": eight_k["i"],
        "news_night": is_night,
        "news_day": is_day,
        "earnings_night": is_night & eight_k["earnings"],
        "unknown": eight_k["timing"] == "unknown",
    })
    flags = flags.groupby(["cik", "i"]).any().reset_index()
    panel = panel.merge(flags, on=["cik", "i"], how="left")
    flag_cols = ["news_night", "news_day", "earnings_night", "unknown"]
    panel[flag_cols] = panel[flag_cols].eq(True)

    filing_sessions = assigned[["cik", "i"]].drop_duplicates()
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
    daily = pd.DataFrame({"r_night": by_date["r_night"].mean(), "r_day": by_date["r_day"].mean(), "n": by_date.size()})
    return daily.reset_index()


def news_portfolios(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily equal-weighted portfolios of all, news (8-K that day, from 2003) and quiet stock-days."""
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
        daily = portfolios[name]
        paths.append(pd.DataFrame({
            "date": daily["date"],
            "group": name,
            "night": cumulative_wealth(daily["r_night"]).to_numpy(),
            "day": cumulative_wealth(daily["r_day"]).to_numpy(),
        }))
    return pd.concat(paths, ignore_index=True)


def news_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    """Share of the nights with an 8-K and their share of the summed overnight return (from 2003).

    No share for the days: the summed intraday return is close to zero.
    """
    days = panel[panel["timed"]].dropna(subset=["r_night", "r_day"])
    r_night = days["r_night"].astype("float64")
    r_day = days["r_day"].astype("float64")
    night_8k, day_8k = days["news_night"], days["news_day"]
    parts = {
        "night with 8-K": r_night[night_8k],
        "other night": r_night[~night_8k],
        "day with 8-K": r_day[day_8k],
        "other day": r_day[~day_8k],
    }
    rows = {}
    for name, r in parts.items():
        rows[name] = {
            "observations": len(r),
            "share of observations (%)": 100 * len(r) / len(days),
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
    """Night - day gap of the all, quiet and news portfolios, with the two tests of Part II."""
    labels = {"all": "all covered", "quiet": "quiet (no filing within 2 sessions)", "news": "with an 8-K"}
    rows = {}
    for name, label in labels.items():
        daily = portfolios[name]
        test = gap_test(daily)
        boot = block_bootstrap_ci(daily["r_night"] - daily["r_day"]).iloc[0]
        rows[label] = {
            "stock-days": int(daily["n"].sum()),
            "dates": test["dates"],
            "gap annualised (%)": test["gap annualised (%)"],
            "CI low (clustered)": test["CI low"],
            "CI high (clustered)": test["CI high"],
            "p-value": test["p-value"],
            "CI low (bootstrap)": boot["CI 95% low"],
            "CI high (bootstrap)": boot["CI 95% high"],
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(3)


def timing_validation(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute night and day returns with no 8-K, a night 8-K or a day 8-K (from 2003)."""
    days = panel[panel["timed"]].dropna(subset=["r_night", "r_day"])
    night_8k, day_8k = days["news_night"], days["news_day"]
    groups = {
        "no 8-K": ~(night_8k | day_8k | days["unknown"]),
        "night 8-K only": night_8k & ~day_8k,
        "day 8-K only": day_8k & ~night_8k,
    }
    rows = {}
    for name, mask in groups.items():
        rows[name] = {
            "|r_night| (bp)": 10000 * days.loc[mask, "r_night"].abs().mean(),
            "|r_day| (bp)": 10000 * days.loc[mask, "r_day"].abs().mean(),
            "n": int(mask.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(1)


def news_regression(panel: pd.DataFrame) -> pd.DataFrame:
    """Effect of an 8-K on the overnight return and its absolute value, in bp, with date fixed effects (from 2003).

    The fixed effects are removed by subtracting the mean of each date; errors are clustered by date and stock.
    """
    days = panel[panel["timed"]].dropna(subset=["r_night"])
    r = 10000 * days["r_night"].astype(float)
    earnings = days["earnings_night"]
    other = days["news_night"] & ~earnings
    clusters = np.column_stack([pd.factorize(days["date"])[0], pd.factorize(days["ticker"])[0]])

    X = pd.DataFrame({"earnings 8-K": earnings.astype(float), "other 8-K": other.astype(float)})
    X = X - X.groupby(days["date"]).transform("mean")
    rows = []
    for outcome, y in {"r_night": r, "|r_night|": r.abs()}.items():
        y = y - y.groupby(days["date"]).transform("mean")
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


def gap_by_issuer_group(stocks: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Night - day gap of equal-weighted portfolios for each kind of company."""
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
