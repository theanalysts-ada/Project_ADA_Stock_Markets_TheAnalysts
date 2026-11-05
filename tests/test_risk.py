import numpy as np
import pandas as pd
import pytest

from src.analysis.risk import (
    duration_analysis, duration_regression, risk_filter_comparison, risk_samples, risk_type_analysis,
)
from src.data.quality import FILTERS
from src.features.calendar import build_night_calendar
from src.visualization.risk_tables import robustness_table


def test_portfolio_has_one_observation_per_night_and_keeps_missing_dates():
    calendar = build_night_calendar("2020-01-06", "2020-02-07")
    dates = calendar["date"].drop(index=21)
    prices = pd.DataFrame({
        "date": dates.repeat(3).to_numpy(),
        "ticker": ["SPY", "A", "B"] * len(dates),
        "asset_type": ["etf", "stock", "stock"] * len(dates),
        "r_night": [0.01, 0.02, 0.04] * len(dates),
        "r_day": [0.0] * (3 * len(dates)),
        "dollar_volume": [100.0, 100.0, 300.0] * len(dates),
    }).sort_values(["ticker", "date"]).reset_index(drop=True)
    samples = risk_samples(prices, calendar)
    assert list(samples) == [
        "Stocks, equal-weighted", "Stocks, dollar-volume",
        "ETFs, equal-weighted", "ETFs, dollar-volume",
    ]
    for sample in samples.values():
        assert len(sample) == len(calendar)
        assert not sample["date"].duplicated().any()
        assert sample.loc[21, ["r_night", "r_day"]].isna().all()
        assert sample.loc[21, "night_type"] == calendar.loc[21, "night_type"]
    assert samples["Stocks, equal-weighted"].loc[20, "r_night"] == pytest.approx(0.03)
    assert samples["Stocks, dollar-volume"].loc[20, "r_night"] == pytest.approx(0.035)
    assert samples["Stocks, dollar-volume"].loc[:19, "r_night"].isna().all()
    assert samples["ETFs, dollar-volume"].loc[20, "r_night"] == pytest.approx(0.01)

    with pytest.raises(ValueError, match="Dates absent"):
        risk_samples(prices.assign(date=prices["date"] + pd.Timedelta(days=365)), calendar)
    with pytest.raises(ValueError, match="missing closure durations"):
        risk_samples(prices, calendar.assign(night_hours=np.nan))


def test_duration_coefficient_is_reported_in_basis_points_per_hour():
    hours = np.tile([17.5, 65.5, 89.5], 30)
    beta = 2.123456789
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=len(hours)),
        "night_hours": hours, "r_night": 0.001 + beta / 10000 * hours,
    })
    result = duration_regression(data, n=200)
    assert result["beta (bp/hour)"] == pytest.approx(beta, abs=1e-12, rel=0)
    assert result["mu (bp/night)"] == pytest.approx(10 + beta * 17.5)
    assert result["n nights"] == len(hours)


def test_duplicate_dates_cannot_be_treated_as_independent_nights():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-06"] * 10),
        "night_hours": [17.5, 65.5] * 5, "r_night": [0.01] * 10,
    })
    with pytest.raises(ValueError, match="one observation per date"):
        duration_regression(data, n=200)


def test_category_means_and_paired_contrasts():
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=100),
        "night_type": (["Weekday"] * 4 + ["Weekend"]) * 20,
        "night_hours": ([17.5] * 4 + [65.5]) * 20,
        "r_night": ([0.001] * 4 + [0.003]) * 20,
    })
    summaries, contrasts = risk_type_analysis({"SPY": data}, block=10, n=200)
    summary = summaries.set_index("night_type")
    contrast = contrasts.set_index("contrast").loc["Weekend - Weekday"]
    assert summary.loc["Weekend", "n nights"] == 20
    assert summary.loc["Weekday", "mean (bp)"] == pytest.approx(10)
    assert contrast["gap (bp)"] == pytest.approx(20)
    assert contrast["CI 95% low"] == pytest.approx(20)
    assert contrast["CI 95% high"] == pytest.approx(20)


def test_block_intervals_reflect_serial_dependence():
    rng = np.random.default_rng(42)
    returns = np.repeat(rng.normal(0, 0.01, 50), 20)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=len(returns)),
        "night_type": "Weekday", "night_hours": 17.5, "r_night": returns,
    })
    independent, contrasts = risk_type_analysis({"SPY": data}, block=1, n=400)
    blocks, _ = risk_type_analysis({"SPY": data}, block=20, n=400)
    independent_width = independent.loc[0, "CI 95% high"] - independent.loc[0, "CI 95% low"]
    block_width = blocks.loc[0, "CI 95% high"] - blocks.loc[0, "CI 95% low"]
    assert block_width > 2 * independent_width
    assert contrasts.empty


def test_bootstrap_retains_missing_sessions_and_pairs_category_differences():
    # Missing returns still occupy sessions inside each sampled block.
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=6),
        "night_type": ["Weekday", "Weekend"] * 3,
        "night_hours": [17.5, 65.5] * 3,
        "r_night": [0.01, 0.03, np.nan, 0.04, 0.02, np.nan],
    })
    summaries, contrasts = risk_type_analysis({"SPY": data}, block=3, n=100, seed=7)
    rng = np.random.default_rng(7)
    means = []
    for _ in range(100):
        first, second = rng.integers(0, 4, size=2)
        sample = pd.concat([data.iloc[first:first + 3], data.iloc[second:second + 3]])
        means.append(sample.groupby("night_type")["r_night"].mean())
    draws = pd.DataFrame(means)
    difference = 10000 * (draws["Weekend"] - draws["Weekday"])
    contrast = contrasts.iloc[0]
    assert contrast["CI 95% low"] == pytest.approx(difference.quantile(0.025))
    assert contrast["CI 95% high"] == pytest.approx(difference.quantile(0.975))
    assert summaries["n nights"].tolist() == [2, 2]


def test_bootstrap_rejects_invalid_parameters_and_missing_categories():
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=4),
        "night_type": ["Weekday"] * 4, "night_hours": [17.5] * 4, "r_night": [0.01] * 4,
    })
    with pytest.raises(ValueError, match="block must"):
        risk_type_analysis({"SPY": data}, block=5, n=10)
    with pytest.raises(ValueError, match="reference type"):
        risk_type_analysis({"SPY": data}, reference="Weekend", block=2, n=10)
    with pytest.raises(ValueError, match="No observed"):
        risk_type_analysis({"SPY": data.assign(r_night=np.nan)}, block=2, n=10)


@pytest.mark.parametrize("start_year", [2005, 2010])
def test_full_sample_models_and_robustness_do_not_depend_on_row_order(start_year):
    dates = pd.bdate_range(f"{start_year}-01-03", periods=90).append(pd.bdate_range("2012-01-03", periods=90))
    rng = np.random.default_rng(3)
    data = pd.DataFrame({
        "date": dates, "night_hours": [17.5, 65.5] * 90, "night_type": "Weekday",
        "r_night": rng.normal(0.001, 0.01, 180),
    })
    results, _ = duration_analysis({"SPY": data}, n=200)
    assert results["specification"].tolist() == [
        "Main, 21-session blocks", "5-session blocks", "63-session blocks",
        "Include exceptional full-day closures",
    ]
    main = results.loc[(results["specification"] == "Main, 21-session blocks")].iloc[0]
    assert main["n nights"] == len(data)
    filters = pd.DataFrame(columns=["portfolio", "filter", "beta (bp/hour)", "CI 95% low", "CI 95% high"])
    shown = robustness_table(results.iloc[::-1], filters)
    row = results.loc[(results["specification"] == "63-session blocks")].iloc[0]
    expected = f"{round(row['beta (bp/hour)'], 3)} [{round(row['CI 95% low'], 3)}, {round(row['CI 95% high'], 3)}]"
    assert shown.loc["63-session blocks", "SPY"] == expected


@pytest.mark.parametrize("proportional", [True, False])
def test_proportionality_separates_per_hour_and_per_night_returns(proportional):
    rng = np.random.default_rng(5)
    kinds = np.tile(["Weekday"] * 4 + ["Weekend"], 400)
    hours = np.where(kinds == "Weekday", 17.5, 65.5)
    mean = 0.0004 * hours / 17.5 if proportional else np.full(len(hours), 0.0004)
    data = pd.DataFrame({
        "date": pd.bdate_range("2000-01-03", periods=len(hours)),
        "night_type": kinds, "night_hours": hours,
        "r_night": mean + rng.normal(0, 0.001, len(hours)),
    })
    results, _ = duration_analysis({"Stocks": data}, n=200)
    row = results.loc[(results["specification"] == "Main, 21-session blocks")].iloc[0]
    assert row["n nights"] == len(data)
    expected_beta = 4 / 17.5 if proportional else 0
    assert row["CI 95% low"] < expected_beta < row["CI 95% high"]
    if proportional:
        assert row["proportionality CI low"] < 0 < row["proportionality CI high"]
    else:
        assert row["mu (bp/night)"] == pytest.approx(4, abs=0.5)
        assert row["proportionality CI low"] > 0


def test_duration_mean_bands_preserve_slope_and_exclude_return_noise():
    rng = np.random.default_rng(23)
    hours = np.tile([17.5, 65.5, 89.5], 500)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=len(hours)),
        "night_hours": hours, "night_type": np.tile(["Weekday", "Weekend", "Long weekend"], 500),
        "r_night": 0.0005 + 0.00001 * hours + rng.normal(0, 0.01, len(hours)),
    })
    results, predictions = duration_analysis({"Stocks": data}, n=200)
    regression = results.loc[(results["specification"] == "Main, 21-session blocks")].iloc[0]
    observed_slope = predictions["mean (bp)"].diff().iloc[1] / predictions["night_hours"].diff().iloc[1]
    assert observed_slope == pytest.approx(regression["beta (bp/hour)"])
    half_width = (predictions["CI 95% high"] - predictions["CI 95% low"]) / 2
    assert (half_width > 0).all()
    assert half_width.max() < 0.2 * (10000 * data["r_night"].std())
    assert (predictions["CI 95% low"] < predictions["mean (bp)"]).all()
    assert (predictions["CI 95% high"] > predictions["mean (bp)"]).all()
    assert "beta (bp/hour)" not in predictions


def test_exceptional_closures_enter_only_the_all_night_regression():
    hours = np.tile([17.5, 65.5, 89.5], 30)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=len(hours)),
        "night_hours": hours,
        "night_type": np.tile(["Weekday", "Weekend", "Long weekend"], 30),
        "r_night": 0.0005 + 0.00001 * hours,
    })
    data.loc[40, ["night_type", "night_hours", "r_night"]] = ["Exceptional closure", 160, -0.4]
    original = data.copy(deep=True)
    routine = data.copy()
    routine.loc[routine["night_type"] == "Exceptional closure", "r_night"] = np.nan
    results, predictions = duration_analysis({"Stocks": data}, n=200)
    for _, row in results.iterrows():
        sample = data if row["specification"] == "Include exceptional full-day closures" else routine
        expected = duration_regression(sample, block=row["block sessions"], n=200)
        for metric, value in expected.items():
            assert row[metric] == pytest.approx(value)
    main = results.loc[results["specification"] == "Main, 21-session blocks"].iloc[0]
    assert main["beta (bp/hour)"] == pytest.approx(0.1)
    assert results.iloc[-1]["beta (bp/hour)"] < 0
    np.testing.assert_allclose(predictions["mean (bp)"], 5 + 0.1 * predictions["night_hours"])
    pd.testing.assert_frame_equal(data, original)


def test_excluded_closures_keep_their_place_in_bootstrap_blocks():
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=6),
        "night_type": ["Weekday", "Weekend", "Exceptional closure", "Weekend", "Weekday", "Weekend"],
        "night_hours": [17.5, 65.5, 160, 65.5, 17.5, 65.5],
        "r_night": [0.01, 0.03, -0.4, 0.04, 0.02, 0.05],
    })
    summaries, contrasts = risk_type_analysis({"Stocks": data}, block=3, n=100, seed=7)
    masked = data.copy()
    masked.loc[2, ["night_type", "r_night"]] = ["Weekday", np.nan]
    expected_summary, expected_contrasts = risk_type_analysis({"Stocks": masked}, block=3, n=100, seed=7)
    pd.testing.assert_frame_equal(summaries, expected_summary)
    pd.testing.assert_frame_equal(contrasts, expected_contrasts)
    assert "Exceptional closure" not in summaries["night_type"].tolist()
    assert summaries["n nights"].sum() == 5
    assert data.loc[2, "r_night"] == -0.4


def test_quality_filter_checks_exclude_exceptional_closures_after_weighting():
    calendar = build_night_calendar("2018-10-01", "2018-12-31")
    prices = pd.DataFrame({
        "date": calendar["date"].repeat(2).to_numpy(),
        "ticker": ["A", "SPY"] * len(calendar),
        "asset_type": ["stock", "etf"] * len(calendar),
        "night_hours": calendar["night_hours"].repeat(2).to_numpy(),
        "r_day": 0.0, "dollar_volume": 100.0,
    })
    prices["r_night"] = 0.0005 + 0.00001 * prices["night_hours"]
    prices.loc[prices["date"] == "2018-12-06", "r_night"] = -0.4
    prices = prices.sort_values(["ticker", "date"])
    for flags in FILTERS.values():
        for flag in flags:
            prices[flag] = False
    meta = pd.DataFrame({"ticker": ["SPY"], "foreign_etf": [False], "leveraged_etf": [False]})
    results = risk_filter_comparison(prices, calendar, 2018, meta, n=200)
    assert len(results) == 8
    np.testing.assert_allclose(results["beta (bp/hour)"], 0.1)
    expected_nights = np.where(results["portfolio"].str.contains("dollar-volume"),
                              len(calendar) - 21, len(calendar) - 1)
    np.testing.assert_array_equal(results["n nights"], expected_nights)


def test_regression_intervals_resample_pairs_and_preserve_missing_sessions():
    rng = np.random.default_rng(12)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=120),
        "night_hours": np.tile([17.5, 65.5, 89.5], 40),
        "r_night": rng.normal(0.001, 0.01, 120),
    })
    data.loc[:19, "r_night"] = np.nan
    data.loc[57, "r_night"] = np.nan
    result = duration_regression(data, block=7, n=100, seed=4)
    rng = np.random.default_rng(4)
    coefficients = []
    for _ in range(100):
        starts = rng.integers(0, len(data) - 7 + 1, size=18)
        sample = pd.concat([data.iloc[start:start + 7] for start in starts]).iloc[:len(data)]
        sample = sample.dropna(subset=["r_night"])
        design = np.column_stack([np.ones(len(sample)), sample["night_hours"] - 17.5])
        coefficients.append(np.linalg.lstsq(design, 10000 * sample["r_night"], rcond=None)[0])
    coefficients = np.array(coefficients)
    gap = coefficients[:, 0] - 17.5 * coefficients[:, 1]
    np.testing.assert_allclose(
        [result["proportionality CI low"], result["proportionality CI high"]],
        np.quantile(gap, [0.025, 0.975]),
    )
    low, high = np.quantile(coefficients, [0.025, 0.975], axis=0)
    np.testing.assert_allclose([result["mu CI low"], result["CI 95% low"]], low)
    np.testing.assert_allclose([result["mu CI high"], result["CI 95% high"]], high)
    assert result["n nights"] == 99
    assert result["bootstrap resamples"] == 100
    with pytest.raises(ValueError, match="block must"):
        duration_regression(data, block=121)


def test_regression_block_intervals_reflect_serial_dependence():
    rng = np.random.default_rng(42)
    hours = np.tile([17.5, 65.5], 500)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=len(hours)),
        "night_hours": hours,
        "r_night": 0.00001 * hours + np.repeat(rng.normal(0, 0.01, 50), 20),
    })
    independent = duration_regression(data, block=1, n=400)
    blocks = duration_regression(data, block=20, n=400)
    assert blocks["mu (bp/night)"] == independent["mu (bp/night)"]
    assert blocks["beta (bp/hour)"] == independent["beta (bp/hour)"]
    assert (blocks["mu CI high"] - blocks["mu CI low"]
            > independent["mu CI high"] - independent["mu CI low"])
