import pandas as pd

from src.features.calendar import night_calendar_summary


def calendar_table(calendar: pd.DataFrame) -> pd.DataFrame:
    """Show counts and mean elapsed hours by closure type."""
    table = night_calendar_summary(calendar)[["nights", "mean_hours"]]
    return table.sort_values("mean_hours").rename(columns={"mean_hours": "mean hours"})


def _interval_text(low: pd.Series, high: pd.Series, digits: int = 2) -> pd.Series:
    """Format interval bounds for display."""
    return "[" + low.round(digits).astype(str) + ", " + high.round(digits).astype(str) + "]"


def contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Show routine closures against weekday nights, in basis points."""
    routine = contrasts["contrast"] != "Exceptional closure - Weekday"
    shown = contrasts.loc[routine].copy()
    shown["gap (bp)"] = shown["gap (bp)"].round(2)
    shown["95% CI"] = _interval_text(shown["CI 95% low"], shown["CI 95% high"])
    return shown[["portfolio", "contrast", "gap (bp)", "95% CI"]]


def robustness_table(results: pd.DataFrame, filters: pd.DataFrame) -> pd.DataFrame:
    """Display full-period slopes and 95% intervals, one column per portfolio."""
    checks = results.copy()
    alternatives = filters.copy()
    alternatives["specification"] = alternatives["filter"].replace({
        "stale": "Stale-open filter", "light": "Light filter",
    })
    shown = checks
    if not alternatives.empty:
        shown = pd.concat([checks, alternatives], ignore_index=True)
    shown["slope [95% CI]"] = (
        shown["beta (bp/hour)"].round(3).astype(str) + " "
        + _interval_text(shown["CI 95% low"], shown["CI 95% high"], digits=3)
    )
    order = ["Main, 21-session blocks", "5-session blocks", "63-session blocks", "Include exceptional full-day closures",
             "Stale-open filter", "Light filter"]
    table = shown.pivot(index="specification", columns="portfolio", values="slope [95% CI]")
    return table.reindex(index=order, columns=results["portfolio"].unique())
