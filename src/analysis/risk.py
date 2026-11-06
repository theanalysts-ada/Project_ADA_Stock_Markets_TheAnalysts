import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.crime import study_portfolios
from src.data.metadata import exclude_special_etfs
from src.data.quality import FILTERS, apply_quality_filter
from src.features.calendar import NIGHT_TYPES

# Return units: x100 for %, x10000 for bp; rounding only controls displayed precision.

# Regular 16:00-to-09:30 closure: 17 hours 30 minutes.
REFERENCE_NIGHT_HOURS = 17.5


def risk_samples(df: pd.DataFrame, calendar: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Join each daily portfolio to the calendar, retaining sessions without returns."""
    if calendar["night_hours"].isna().any():
        raise ValueError("Night calendar contains missing closure durations")
    samples = {}
    for name, portfolio in study_portfolios(df).items():
        if portfolio.empty:
            raise ValueError(f"No observations for {name}")
        outside = ~portfolio["date"].isin(calendar["date"])
        if outside.any():
            # Limit the error message to five example dates.
            dates = portfolio.loc[outside, "date"].dt.strftime("%Y-%m-%d").head(5).tolist()
            raise ValueError(f"Dates absent from the night calendar: {dates}")
        samples[name] = calendar.merge(portfolio, on="date", how="left", validate="one_to_one")
    return samples


def _daily_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Order market nights and reject duplicate dates or infinite returns."""
    if df["date"].duplicated().any():
        raise ValueError("Risk analysis requires one observation per date; aggregate tickers first")
    data = df.sort_values("date").reset_index(drop=True)
    data["r_night"] = data["r_night"].astype("float64")
    if np.isinf(data["r_night"]).any():
        raise ValueError("Overnight returns must be finite or missing")
    return data


def _routine_nights(data: pd.DataFrame) -> pd.DataFrame:
    """Exclude exceptional returns while retaining the session grid for bootstrap blocks."""
    routine = data.copy()
    routine.loc[routine["night_type"] == "Exceptional closure", "r_night"] = np.nan
    return routine


def _block_samples(data: pd.DataFrame, block: int, n: int, seed: int):
    """Resample complete session rows in contiguous blocks, retaining missing returns."""
    # Blocks must fit the sample; an interval needs more than one bootstrap draw.
    if not 1 <= block <= len(data) or n < 2:
        raise ValueError("block must be between 1 and the sample size; n must be at least 2")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(data) / block))
    for _ in range(n):
        starts = rng.integers(0, len(data) - block + 1, size=n_blocks)
        positions = np.concatenate([np.arange(start, start + block) for start in starts])
        yield data.iloc[positions[:len(data)]]


def _bootstrap_nights(data: pd.DataFrame, block: int, n: int, seed: int) -> pd.DataFrame:
    """Average returns by night type in each block-bootstrap sample."""
    observed = data.loc[data["r_night"].notna(), "night_type"].dropna().unique()
    names = [name for name in NIGHT_TYPES if name in observed]
    if not names:
        raise ValueError("No observed overnight returns with a night type")

    returns = data[["night_type", "r_night"]]
    draws = []
    for sample in _block_samples(returns, block, n, seed):
        means = sample.groupby("night_type", observed=True)["r_night"].mean()
        draws.append(means.reindex(names))
    return pd.DataFrame(draws).reset_index(drop=True)


# 21-session blocks ~ one trading month; 1000 draws limit runtime; seed=0 makes results reproducible.
def risk_type_analysis(samples, reference="Weekday", block=21, n=1000, seed=0):
    """Return routine-closure means and contrasts with 95% block-bootstrap intervals.

    Results are in basis points per night. Excluded and missing returns remain in the session
    sequence when sampling blocks. Intervals omit draws missing a category and
    are pointwise, without adjustment for multiple comparisons.
    """
    # Collect the summary and contrast tables for each portfolio.
    summaries, contrasts = [], []
    for name, data in samples.items():
        # Order sessions and exclude exceptional returns while keeping their positions.
        data = _routine_nights(_daily_sample(data))

        # Bootstrap mean returns by closure type and check that the reference is present.
        draws = _bootstrap_nights(data, block, n, seed)
        if reference not in draws.columns:
            raise ValueError(f"No observed nights for reference type {reference}")

        # Count observed nights and compute mean duration and return for each closure type.
        observed = data.loc[data["r_night"].notna()]
        summary = observed.groupby("night_type", observed=True).agg(
            n_nights=("r_night", "size"), mean_hours=("night_hours", "mean"),
            mean_return=("r_night", "mean"),
        ).reindex(draws.columns)

        # Convert returns to basis points and add the central 95% bootstrap intervals.
        summary["mean (bp)"] = 10000 * summary.pop("mean_return")
        # Quantiles 0.025 and 0.975 bound the central 95% bootstrap interval.
        summary["CI 95% low"] = 10000 * draws.quantile(0.025)
        summary["CI 95% high"] = 10000 * draws.quantile(0.975)

        # Label the summary columns and save this portfolio's table.
        summary = summary.rename(columns={"n_nights": "n nights", "mean_hours": "mean hours"})
        summary["portfolio"] = name
        summaries.append(summary.rename_axis("night_type").reset_index())

        # Compare each closure type with the reference within every bootstrap draw.
        differences = draws.subtract(draws[reference], axis="index").drop(columns=reference)

        # Report observed return gaps and their central 95% bootstrap intervals.
        means = summary["mean (bp)"]
        contrast = pd.DataFrame({
            "gap (bp)": means.drop(reference) - means.loc[reference],
            "CI 95% low": 10000 * differences.quantile(0.025),
            "CI 95% high": 10000 * differences.quantile(0.975),
        })

        # Label each contrast with its reference and save this portfolio's table.
        contrast.index = contrast.index.astype(str) + " - " + reference
        contrast["portfolio"] = name
        contrasts.append(contrast.rename_axis("contrast").reset_index())

    # Combine the portfolio tables into one summary table and one contrast table.
    return pd.concat(summaries, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def _duration_fit(data: pd.DataFrame):
    """Fit return = mu + beta * (hours - 17.5), in basis points."""
    observed = data.dropna(subset=["r_night", "night_hours"])
    # Two fitted coefficients require varying hours and at least one residual degree of freedom.
    if len(observed) < 3 or observed["night_hours"].nunique() < 2:
        raise ValueError("Each regression sample needs at least three nights and varying durations")
    hours = observed["night_hours"].astype("float64") - REFERENCE_NIGHT_HOURS
    design = sm.add_constant(hours, has_constant="add")
    return sm.OLS(10000 * observed["r_night"], design).fit()


def _bootstrap_duration(df: pd.DataFrame, block: int, n: int, seed: int):
    """Fit the observed sample and resample duration-return pairs in session blocks."""
    data = _daily_sample(df)[["night_hours", "r_night"]]
    fit = _duration_fit(data)
    draws = [_duration_fit(sample).params for sample in _block_samples(data, block, n, seed)]
    return fit, pd.DataFrame(draws).reset_index(drop=True)


def _regression_result(fit, draws: pd.DataFrame, block: int) -> dict:
    """Report coefficients and bootstrap intervals; mu - 17.5 * beta tests proportionality."""
    # Keep the central 95% of bootstrap coefficient estimates.
    low, high = draws.quantile(0.025), draws.quantile(0.975)
    gap = draws["const"] - REFERENCE_NIGHT_HOURS * draws["night_hours"]
    return {
        "mu (bp/night)": fit.params["const"],
        "mu CI low": low["const"], "mu CI high": high["const"],
        "beta (bp/hour)": fit.params["night_hours"],
        "CI 95% low": low["night_hours"], "CI 95% high": high["night_hours"],
        "proportionality gap (bp)": fit.params["const"] - REFERENCE_NIGHT_HOURS * fit.params["night_hours"],
        "proportionality CI low": gap.quantile(0.025),
        "proportionality CI high": gap.quantile(0.975),
        "n nights": int(fit.nobs), "block sessions": block,
        "bootstrap resamples": len(draws), "R-squared": fit.rsquared,
    }


# Same monthly blocks, draw count and seed as the closure-type analysis.
def duration_regression(df: pd.DataFrame, block=21, n=1000, seed=0) -> dict:
    """Return duration estimates and percentile block-bootstrap intervals."""
    fit, draws = _bootstrap_duration(df, block, n, seed)
    return _regression_result(fit, draws, block)


def _prediction_curve(fit, draws: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Predict routine-closure means with pointwise bootstrap intervals."""
    routine = data.loc[(data["night_type"] != "Exceptional closure") & data["r_night"].notna()]
    if routine.empty:
        raise ValueError("Need observed routine closures to plot duration predictions")
    # 200 grid points give a smooth fitted line without a large prediction table.
    hours = np.linspace(routine["night_hours"].min(), routine["night_hours"].max(), 200)
    rows = []
    for hour in hours:
        extra_hours = hour - REFERENCE_NIGHT_HOURS
        bootstrap_returns = draws["const"] + draws["night_hours"] * extra_hours
        # Pointwise central 95% intervals at each plotted duration.
        low, high = bootstrap_returns.quantile([0.025, 0.975])
        rows.append({
            "night_hours": hour,
            "mean (bp)": fit.params["const"] + fit.params["night_hours"] * extra_hours,
            "CI 95% low": low,
            "CI 95% high": high,
        })
    return pd.DataFrame(rows)


# Reuse 1000 draws and seed=0 so the model comparisons are reproducible.
def duration_analysis(samples: dict[str, pd.DataFrame], n=1000, seed=0):
    """Fit the duration model and robustness checks over the full study period.

    The main model uses routine closures, simple returns and 21-session blocks.
    Its bootstrap draws also supply the prediction bands.
    """
    results, predictions = [], []
    for name, data in samples.items():
        all_nights = data
        data = _routine_nights(data)
        # Baseline blocks cover about one trading month (21 sessions).
        fit, draws = _bootstrap_duration(data, 21, n, seed)
        row = _regression_result(fit, draws, block=21)
        row.update(portfolio=name, specification="Main, 21-session blocks")
        results.append(row)
        curve = _prediction_curve(fit, draws, data)
        curve["portfolio"] = name
        predictions.append(curve)

        # 5 and 63 sessions approximate a week and a quarter: shorter/longer dependence checks.
        checks = [("5-session blocks", data, 5), ("63-session blocks", data, 63),
                  ("Include exceptional full-day closures", all_nights, 21)]
        for specification, subset, block in checks:
            row = duration_regression(subset, block=block, n=n, seed=seed)
            row.update(portfolio=name, specification=specification)
            results.append(row)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


# Keep the same draw count and seed when changing the quality filter.
def risk_filter_comparison(prices, calendar, start_year, meta, n=1000, seed=0) -> pd.DataFrame:
    """Fit routine closures under alternative quality filters."""
    columns = ["date", "ticker", "asset_type", "r_night", "r_day", "dollar_volume"]
    for drop_flags in FILTERS.values():
        for flag in drop_flags:
            if flag not in columns:
                columns.append(flag)
    prices = prices[columns]

    results = []
    for filter_name, drop_flags in FILTERS.items():
        if filter_name == "liquidity":
            continue
        sample = apply_quality_filter(prices, start_year=start_year, drop_flags=drop_flags)
        sample = exclude_special_etfs(sample, meta)
        for name, data in risk_samples(sample, calendar).items():
            row = duration_regression(_routine_nights(data), n=n, seed=seed)
            row.update(portfolio=name, filter=filter_name)
            results.append(row)
    return pd.DataFrame(results)
