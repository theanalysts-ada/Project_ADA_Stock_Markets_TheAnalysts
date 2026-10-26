import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features.calendar import NIGHT_TYPES, attach_night_calendar
from src.analysis.crime import study_portfolios


def risk_samples(df: pd.DataFrame, calendar: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare the four Part II portfolios, with one observation per market night.

    Args:
        df: Part II study sample after quality and special-ETF exclusions, including
            ticker, asset_type, date, r_night, r_day and dollar_volume.
        calendar: Table returned by build_night_calendar for the study period.

    Returns:
        Four daily tables with calendar columns. All market dates are retained for
        the bootstrap; a date without an observed return has a missing r_night.
    """
    samples = {}
    for name, pair in study_portfolios(df).items():
        if pair.empty:
            raise ValueError(f"No observations for {name}")
        attach_night_calendar(pair, calendar)  # validate dates only: raises if a date is outside the calendar
        daily = pair.set_index("date").reindex(calendar["date"]).rename_axis("date").reset_index()
        samples[name] = attach_night_calendar(daily, calendar)
    return samples


def _daily_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a daily sample and reject duplicate dates or infinite returns."""
    if df["date"].duplicated().any():
        raise ValueError("Risk analysis requires one observation per date; aggregate tickers first")
    if np.isinf(df["r_night"].astype("float64")).any():
        raise ValueError("Overnight returns must be finite or missing")
    return df.sort_values("date").reset_index(drop=True)


def _night_type_bootstrap(
    df: pd.DataFrame, block: int, n: int, seed: int,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Resample contiguous blocks of sessions, keeping returns and night types together."""
    data = _daily_sample(df)
    if not 1 <= block <= len(data) or n < 2:
        raise ValueError("block must be between 1 and the sample size; n must be at least 2")
    valid = data["r_night"].notna() & data["night_type"].notna()
    names = [name for name in NIGHT_TYPES if (valid & data["night_type"].eq(name)).any()]
    if not names:
        raise ValueError("No observed overnight returns with a night type")
    codes = pd.Categorical(data["night_type"], categories=names).codes
    values = data["r_night"].to_numpy(dtype="float64")

    def means(indices: np.ndarray) -> np.ndarray:
        selected_codes, selected_values = codes[indices], values[indices]
        keep = (selected_codes >= 0) & np.isfinite(selected_values)
        counts = np.bincount(selected_codes[keep], minlength=len(names))
        sums = np.bincount(selected_codes[keep], weights=selected_values[keep], minlength=len(names))
        return np.divide(sums, counts, out=np.full(len(names), np.nan), where=counts > 0)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(data) / block))
    draws = np.empty((n, len(names)))
    for i in range(n):
        starts = rng.integers(0, len(data) - block + 1, size=n_blocks)
        indices = (starts[:, None] + np.arange(block)).ravel()[:len(data)]
        draws[i] = means(indices)
    return data, names, means(np.arange(len(data))), draws


def _type_summary(bootstrap: tuple) -> pd.DataFrame:
    """Summarise an existing bootstrap, without rounding the estimates."""
    data, names, means, draws = bootstrap
    rows = []
    for i, name in enumerate(names):
        group = data[data["night_type"].eq(name) & data["r_night"].notna()]
        lo, hi = np.nanpercentile(draws[:, i], [2.5, 97.5])
        rows.append({
            "night_type": name, "n nights": len(group), "mean hours": group["night_hours"].mean(),
            "mean (bp)": 10000 * means[i], "CI 95% low": 10000 * lo, "CI 95% high": 10000 * hi,
        })
    return pd.DataFrame(rows).set_index("night_type")


def _type_contrasts(bootstrap: tuple, reference: str) -> pd.DataFrame:
    """Calculate differences within each draw, retaining their paired uncertainty."""
    _, names, means, draws = bootstrap
    if reference not in names:
        raise ValueError(f"No observed nights for reference type {reference}")
    j = names.index(reference)
    rows = []
    for i, name in enumerate(names):
        if name == reference:
            continue
        lo, hi = np.nanpercentile(draws[:, i] - draws[:, j], [2.5, 97.5])
        rows.append({
            "contrast": f"{name} - {reference}", "gap (bp)": 10000 * (means[i] - means[j]),
            "CI 95% low": 10000 * lo, "CI 95% high": 10000 * hi,
        })
    return pd.DataFrame(rows, columns=["contrast", "gap (bp)", "CI 95% low", "CI 95% high"]).set_index("contrast")


def night_type_summary(df: pd.DataFrame, block: int = 21, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Compare closure-type means with pointwise block-bootstrap intervals.

    Args:
        df: One row per session, with date, r_night, night_hours and night_type.
        block: Number of consecutive sessions per bootstrap block.
        n: Number of bootstrap resamples.
        seed: Random seed for reproducible intervals.

    Returns:
        Counts, mean hours, mean return and 95 % interval in basis points per night.
        Draws without an observed return of a type are omitted for its interval.
    """
    return _type_summary(_night_type_bootstrap(df, block, n, seed))


def night_type_contrasts(
    df: pd.DataFrame, reference: str = "Weekday", block: int = 21, n: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Estimate differences from the reference type within bootstrap draws.

    Args:
        df: Daily sample accepted by night_type_summary.
        reference: Night type subtracted from each other type.
        block: Number of consecutive sessions per bootstrap block.
        n: Number of bootstrap resamples.
        seed: Random seed for reproducible intervals.

    Returns:
        Differences and pointwise 95 % intervals in basis points per night.
        Draws missing either category are omitted; no multiplicity correction is applied.
    """
    return _type_contrasts(_night_type_bootstrap(df, block, n, seed), reference)


def risk_type_analysis(
    samples: dict[str, pd.DataFrame], reference: str = "Weekday", block: int = 21, n: int = 1000, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute means and contrasts from one bootstrap per portfolio.

    Args:
        samples: Daily tables returned by risk_samples.
        reference: Closure type subtracted in each contrast.
        block: Number of consecutive sessions per bootstrap block.
        n: Number of bootstrap resamples.
        seed: Random seed for reproducible intervals.

    Returns:
        Summary and contrast tables indexed by portfolio, in basis points per night.
        Full precision is retained for plots and later transformations.
    """
    summaries, contrasts = {}, {}
    for name, data in samples.items():
        bootstrap = _night_type_bootstrap(data, block, n, seed)
        summaries[name] = _type_summary(bootstrap)
        contrasts[name] = _type_contrasts(bootstrap, reference)
    return pd.concat(summaries, names=["portfolio"]), pd.concat(contrasts, names=["portfolio"])


def _duration_fit(df: pd.DataFrame, log: bool, maxlags: int):
    """Fit the common duration model for coefficient and mean-return inference."""
    data = _daily_sample(df).dropna(subset=["r_night", "night_hours"])
    if maxlags < 0 or len(data) <= maxlags + 2 or data["night_hours"].nunique() < 2:
        raise ValueError("Need varying durations and more than maxlags + 2 observed nights")
    values = data["r_night"].astype("float64")
    if log:
        if (values <= -1).any():
            raise ValueError("Log returns require r_night > -1")
        values = np.log1p(values)
    design = sm.add_constant(data["night_hours"].astype("float64"), has_constant="add")
    return sm.OLS(10000 * values, design).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def duration_regression(df: pd.DataFrame, log: bool = False, maxlags: int = 5) -> pd.DataFrame:
    """Regress overnight returns on closure duration, with HAC standard errors.

    r_night(t) = alpha + beta * night_hours(t) + error(t). The coefficient describes
    an association: duration also varies with weekdays, holidays and exceptional news.

    Args:
        df: One row per date, with date, r_night and night_hours.
        log: Use log(1 + r_night) instead of the simple return.
        maxlags: Number of ordered observations used for the HAC covariance (default: 5).

    Returns:
        One row: beta and its 95 % interval (basis points per extra hour), p-value,
        number of observed nights, HAC lag count and R-squared. Nothing is annualised.
    """
    fit = _duration_fit(df, log, maxlags)
    lo, hi = fit.conf_int().loc["night_hours"]
    return pd.DataFrame([{
        "beta (bp/hour)": fit.params["night_hours"], "CI 95% low": lo, "CI 95% high": hi,
        "p-value": fit.pvalues["night_hours"], "n nights": int(fit.nobs),
        "HAC lags": maxlags, "R-squared": fit.rsquared,
    }])


def duration_predictions(samples: dict[str, pd.DataFrame], maxlags: int = 5) -> pd.DataFrame:
    """Predict mean simple returns with pointwise 95 % HAC confidence intervals.

    Fits use all observed nights, including exceptional closures. Display grids
    cover observed routine closure durations only. Bands describe uncertainty
    in the fitted mean, not the dispersion of individual nightly returns.
    The slope and its 95 % HAC interval are repeated on every row.
    """
    predictions = {}
    for name, data in samples.items():
        fit = _duration_fit(data, log=False, maxlags=maxlags)
        ordinary = data.loc[data["night_type"].ne("Exceptional closure") & data["r_night"].notna(), "night_hours"]
        hours = np.linspace(ordinary.min(), ordinary.max(), 200)
        design = sm.add_constant(pd.DataFrame({"night_hours": hours}), has_constant="add")
        frame = fit.get_prediction(design).summary_frame(alpha=0.05)
        slope_low, slope_high = fit.conf_int().loc["night_hours"]
        predictions[name] = pd.DataFrame({
            "night_hours": hours, "mean (bp)": frame["mean"],
            "beta (bp/hour)": fit.params["night_hours"],
            "beta CI low": slope_low, "beta CI high": slope_high,
            "CI 95% low": frame["mean_ci_lower"], "CI 95% high": frame["mean_ci_upper"],
        })
    return pd.concat(predictions, names=["portfolio", "point"])


def duration_tests_by_period(samples: dict[str, pd.DataFrame], maxlags: int = 5) -> pd.DataFrame:
    """Compare the duration slope before and after 2008, using simple and log returns.

    Args:
        samples: Daily tables returned by risk_samples.
        maxlags: Number of ordered observations used for the HAC covariance.

    Returns:
        One regression row per (portfolio, period, return type); beta and intervals
        are in basis points per extra hour. The full period comes first; labels
        follow the input dates. The 2008 split is included when both sides are present.
    """
    rows = []
    for name, data in samples.items():
        first_year, last_year = data["date"].dt.year.min(), data["date"].dt.year.max()
        periods = {f"{first_year}–{last_year}": data}
        if first_year < 2008 <= last_year:
            periods[f"{first_year}–2007"] = data[data["date"].dt.year <= 2007]
            periods[f"2008–{last_year}"] = data[data["date"].dt.year >= 2008]
        for period, subset in periods.items():
            for log, kind in ((False, "simple"), (True, "log")):
                row = duration_regression(subset, log=log, maxlags=maxlags).iloc[0].to_dict()
                rows.append({"portfolio": name, "period": period, "return": kind, **row})
    return pd.DataFrame(rows).set_index(["portfolio", "period", "return"]).astype({"n nights": int, "HAC lags": int})


def duration_robustness(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check the slope with a longer HAC lag window and without exceptional closures.

    Args:
        samples: Daily tables returned by risk_samples.

    Returns:
        One regression row per portfolio and specification, in basis points per hour.
    """
    rows = []
    for name, data in samples.items():
        variants = {
            "Main, HAC 5": (data, 5),
            "HAC 21": (data, 21),
            "Exclude exceptional full-day closures": (data[data["night_type"] != "Exceptional closure"], 5),
        }
        for specification, (subset, lags) in variants.items():
            row = duration_regression(subset, maxlags=lags).iloc[0].to_dict()
            rows.append({"portfolio": name, "specification": specification, **row})
    return pd.DataFrame(rows).set_index(["portfolio", "specification"]).astype({"n nights": int, "HAC lags": int})


def proportionality_tests(samples: dict[str, pd.DataFrame], maxlags: int = 5) -> pd.DataFrame:
    """Test strict proportionality to closure hours in the linear duration model.

    In r_night(t) = alpha + beta * night_hours(t) + error(t), a return proportional to
    elapsed hours (calendar time) implies alpha = 0; a fixed premium per night implies
    beta = 0. An interval containing zero does not establish that beta is zero.

    Args:
        samples: Daily tables returned by risk_samples.
        maxlags: Number of ordered observations used for the HAC covariance.

    Returns:
        One row per portfolio and sample (all nights, routine closures): alpha in bp
        per night and beta in bp per hour with 95 % HAC intervals, and a p-value
        for alpha = 0. Alpha is the fitted return at zero closure hours.
    """
    rows = []
    for name, data in samples.items():
        subsets = {
            "All nights": data,
            "Routine closures": data[data["night_type"] != "Exceptional closure"],
        }
        for sample, subset in subsets.items():
            fit = _duration_fit(subset, log=False, maxlags=maxlags)
            ci = fit.conf_int()
            rows.append({
                "portfolio": name, "sample": sample,
                "alpha (bp/night)": fit.params["const"],
                "alpha CI low": ci.loc["const", 0], "alpha CI high": ci.loc["const", 1],
                "p-value alpha = 0": fit.pvalues["const"],
                "beta (bp/hour)": fit.params["night_hours"],
                "CI 95% low": ci.loc["night_hours", 0], "CI 95% high": ci.loc["night_hours", 1],
                "n nights": int(fit.nobs),
            })
    return pd.DataFrame(rows).set_index(["portfolio", "sample"])


def risk_filter_comparison(prices: pd.DataFrame, calendar: pd.DataFrame, start_year: int, meta: pd.DataFrame) -> pd.DataFrame:
    """Repeat the duration regression under Part I's three quality filters.

    Args:
        prices: Full flagged price table, with returns calculated before filtering.
        calendar: Table returned by build_night_calendar for the study period.
        start_year: First study year.
        meta: Tagged ETF metadata from tag_special_etfs, as used in Part II.

    Returns:
        One regression row per portfolio and filter, in basis points per extra hour.
    """
    from src.data.quality import FILTERS, apply_quality_filter
    from src.data.metadata import exclude_special_etfs
    flags = sorted({flag for group in FILTERS.values() for flag in group})
    slim = prices[["date", "ticker", "asset_type", "r_night", "r_day", "dollar_volume", *flags]]
    rows = []
    for name, flags in FILTERS.items():
        sample = apply_quality_filter(slim, start_year=start_year, drop_flags=flags)
        sample = exclude_special_etfs(sample, meta)
        for portfolio, data in risk_samples(sample, calendar).items():
            row = duration_regression(data).iloc[0].to_dict()
            rows.append({"portfolio": portfolio, "filter": name, **row})
    return pd.DataFrame(rows).set_index(["portfolio", "filter"]).astype({"n nights": int, "HAC lags": int})
