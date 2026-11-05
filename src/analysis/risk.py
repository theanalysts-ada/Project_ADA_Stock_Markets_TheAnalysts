import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.crime import study_portfolios
from src.data.metadata import exclude_special_etfs
from src.data.quality import FILTERS, apply_quality_filter
from src.features.calendar import NIGHT_TYPES

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


def risk_type_analysis(samples, reference="Weekday", block=21, n=1000, seed=0):
    """Return routine-closure means and contrasts with 95% block-bootstrap intervals.

    Results are in basis points per night. Excluded and missing returns remain in the session
    sequence when sampling blocks. Intervals omit draws missing a category and
    are pointwise, without adjustment for multiple comparisons.
    """
    summaries, contrasts = [], []
    for name, data in samples.items():
        data = _routine_nights(_daily_sample(data))
        draws = _bootstrap_nights(data, block, n, seed)
        if reference not in draws.columns:
            raise ValueError(f"No observed nights for reference type {reference}")

        observed = data.loc[data["r_night"].notna()]
        summary = observed.groupby("night_type", observed=True).agg(
            n_nights=("r_night", "size"), mean_hours=("night_hours", "mean"),
            mean_return=("r_night", "mean"),
        ).reindex(draws.columns)
        summary["mean (bp)"] = 10000 * summary.pop("mean_return")
        summary["CI 95% low"] = 10000 * draws.quantile(0.025)
        summary["CI 95% high"] = 10000 * draws.quantile(0.975)
        summary = summary.rename(columns={"n_nights": "n nights", "mean_hours": "mean hours"})
        summary["portfolio"] = name
        summaries.append(summary.rename_axis("night_type").reset_index())

        differences = draws.subtract(draws[reference], axis="index").drop(columns=reference)
        means = summary["mean (bp)"]
        contrast = pd.DataFrame({
            "gap (bp)": means.drop(reference) - means.loc[reference],
            "CI 95% low": 10000 * differences.quantile(0.025),
            "CI 95% high": 10000 * differences.quantile(0.975),
        })
        contrast.index = contrast.index.astype(str) + " - " + reference
        contrast["portfolio"] = name
        contrasts.append(contrast.rename_axis("contrast").reset_index())
    return pd.concat(summaries, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def _duration_fit(data: pd.DataFrame):
    """Fit return = mu + beta * (hours - 17.5), in basis points."""
    observed = data.dropna(subset=["r_night", "night_hours"])
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


def duration_regression(df: pd.DataFrame, block=21, n=1000, seed=0) -> dict:
    """Return duration estimates and percentile block-bootstrap intervals."""
    fit, draws = _bootstrap_duration(df, block, n, seed)
    return _regression_result(fit, draws, block)


def _prediction_curve(fit, draws: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Predict routine-closure means with pointwise bootstrap intervals."""
    routine = data.loc[(data["night_type"] != "Exceptional closure") & data["r_night"].notna()]
    if routine.empty:
        raise ValueError("Need observed routine closures to plot duration predictions")
    hours = np.linspace(routine["night_hours"].min(), routine["night_hours"].max(), 200)
    design = sm.add_constant(pd.DataFrame({"night_hours": hours - REFERENCE_NIGHT_HOURS}), has_constant="add")
    curves = pd.DataFrame({
        hour: draws["const"] + draws["night_hours"] * (hour - REFERENCE_NIGHT_HOURS)
        for hour in hours
    })
    return pd.DataFrame({
        "night_hours": hours, "mean (bp)": fit.predict(design),
        "CI 95% low": curves.quantile(0.025).to_numpy(),
        "CI 95% high": curves.quantile(0.975).to_numpy(),
    })


def duration_analysis(samples: dict[str, pd.DataFrame], n=1000, seed=0):
    """Fit the duration model and robustness checks over the full study period.

    The main model uses routine closures, simple returns and 21-session blocks.
    Its bootstrap draws also supply the prediction bands.
    """
    results, predictions = [], []
    for name, data in samples.items():
        all_nights = data
        data = _routine_nights(data)
        fit, draws = _bootstrap_duration(data, 21, n, seed)
        row = _regression_result(fit, draws, block=21)
        row.update(portfolio=name, specification="Main, 21-session blocks")
        results.append(row)
        curve = _prediction_curve(fit, draws, data)
        curve["portfolio"] = name
        predictions.append(curve)

        checks = [("5-session blocks", data, 5), ("63-session blocks", data, 63),
                  ("Include exceptional full-day closures", all_nights, 21)]
        for specification, subset, block in checks:
            row = duration_regression(subset, block=block, n=n, seed=seed)
            row.update(portfolio=name, specification=specification)
            results.append(row)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)


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
