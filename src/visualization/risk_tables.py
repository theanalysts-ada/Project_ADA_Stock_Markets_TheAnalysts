import pandas as pd

from src.features.calendar import night_calendar_summary


def calendar_table(calendar: pd.DataFrame) -> pd.DataFrame:
    """Show each closure type once, with its count and mean elapsed hours."""
    table = night_calendar_summary(calendar)[["nights", "mean_hours"]]
    return table.sort_values("mean_hours").rename(columns={"mean_hours": "mean hours"})


def _interval_table(table: pd.DataFrame, estimate: str, digits: int = 2) -> pd.DataFrame:
    """Format estimates and combine interval bounds without changing the input."""
    shown = table.copy()
    shown[estimate] = shown[estimate].map(lambda value: f"{value:.{digits}f}")
    shown["95% CI"] = [
        f"[{lo:.{digits}f}, {hi:.{digits}f}]"
        for lo, hi in zip(shown["CI 95% low"], shown["CI 95% high"])
    ]
    shown = shown.drop(columns=["CI 95% low", "CI 95% high"])
    for column in ("n nights", "HAC lags"):
        if column in shown:
            shown[column] = shown[column].astype(int)
    return shown.round({"mean hours": 2, "p-value": 3, "R-squared": 4})


def contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Show routine closures against weekday nights, with differences in bp."""
    order = ["Midweek holiday - Weekday", "Weekend - Weekday", "Long weekend - Weekday"]
    available = contrasts.index.get_level_values("contrast")
    keep = contrasts.loc[available.isin(order)]
    table = _interval_table(keep, "gap (bp)")
    table = table.reindex([name for name in order if name in available], level="contrast")
    return table[["gap (bp)", "95% CI"]]


def _combined_checks(tests: pd.DataFrame, checks: pd.DataFrame, filters: pd.DataFrame) -> pd.DataFrame:
    """Collect each full-period specification once, with one common baseline."""
    logs = tests.xs("log", level="return").groupby(level="portfolio", sort=False).head(1).droplevel("period")
    logs = pd.concat({"Log returns": logs}, names=["specification"]).reorder_levels(["portfolio", "specification"])
    alternatives = filters.drop(index="liquidity", level="filter").rename_axis(index={"filter": "specification"})
    alternatives = alternatives.rename(index={"stale": "Stale-open filter", "light": "Light filter"}, level="specification")
    return pd.concat([checks, logs, alternatives])


def robustness_table(tests: pd.DataFrame, checks: pd.DataFrame, filters: pd.DataFrame) -> pd.DataFrame:
    """Compare full-period slopes in bp/hour, with 95 % intervals.

    Args:
        tests: Regressions by period and return type.
        checks: HAC and exceptional-closure checks.
        filters: Regressions under the shared quality filters.

    Returns:
        One row per specification and one column per portfolio. Log-return effects
        are in log basis points; the remaining specifications use simple returns.
    """
    table = _interval_table(_combined_checks(tests, checks, filters), "beta (bp/hour)", digits=3)
    table["slope [95% CI]"] = table["beta (bp/hour)"] + " " + table["95% CI"]
    panels = {}
    for name in table.index.get_level_values("portfolio").unique():
        panels[name] = table.loc[name, "slope [95% CI]"]
    return pd.concat(panels, axis=1)


def proportionality_table(tests: pd.DataFrame) -> pd.DataFrame:
    """Show alpha and beta with 95 % intervals and the test of alpha = 0.

    Args:
        tests: Table returned by proportionality_tests.

    Returns:
        One row per portfolio and sample; alpha in bp per night, slopes in bp per hour.
    """
    def interval(estimate: str, low: str, high: str, digits: int) -> list[str]:
        return [
            f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
            for value, lo, hi in zip(tests[estimate], tests[low], tests[high])
        ]

    def p_value(p: float) -> str:
        return "< 0.001" if p < 0.001 else f"{p:.3f}"

    return pd.DataFrame({
        "alpha, bp/night [95% CI]": interval("alpha (bp/night)", "alpha CI low", "alpha CI high", 2),
        "p (alpha = 0)": tests["p-value alpha = 0"].map(p_value),
        "beta, bp/hour [95% CI]": interval("beta (bp/hour)", "CI 95% low", "CI 95% high", 3),
    }, index=tests.index)


def closure_counts_table(summaries: pd.DataFrame) -> pd.DataFrame:
    """Count nights by weighting when stocks and ETFs have identical coverage.

    Args:
        summaries: Closure-type summaries returned by risk_type_analysis.
    """
    counts = summaries["n nights"].unstack("portfolio")
    table = {}
    for label, suffix in (("Equal-weighted", "equal-weighted"), ("Dollar-volume-weighted", "dollar-volume")):
        stocks, etfs = counts[f"Stocks, {suffix}"], counts[f"ETFs, {suffix}"]
        if not stocks.equals(etfs):
            raise ValueError("Stocks and ETFs have different night counts; display them separately")
        table[label] = stocks
    return pd.DataFrame(table).reindex(summaries.index.get_level_values("night_type").unique()).astype("Int64")


def exceptional_closure_table(summaries: pd.DataFrame) -> pd.DataFrame:
    """Show the mean return of exceptional closures, with bootstrap intervals in bp.

    Args:
        summaries: Closure-type summaries returned by risk_type_analysis.

    Returns:
        One row per portfolio: number of nights, mean return and 95 % interval.
    """
    exceptional = summaries.xs("Exceptional closure", level="night_type")
    return _interval_table(exceptional, "mean (bp)")[["n nights", "mean (bp)", "95% CI"]]
