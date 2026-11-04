import numpy as np
import pandas as pd
import pytest

from src.analysis.risk import duration_regression, night_type_contrasts, night_type_summary, risk_samples, risk_type_analysis
from src.features.calendar import build_night_calendar


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


def test_duration_coefficient_is_reported_in_basis_points_per_hour():
    hours = np.tile([17.5, 65.5, 89.5], 30)
    beta = 2.123456789
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=len(hours)),
        "night_hours": hours, "r_night": 0.001 + beta / 10000 * hours,
    })
    result = duration_regression(data).iloc[0]
    assert result["beta (bp/hour)"] == pytest.approx(beta, abs=1e-12, rel=0)
    assert result["n nights"] == len(hours)


def test_duplicate_dates_cannot_be_treated_as_independent_nights():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-06"] * 10),
        "night_hours": [17.5, 65.5] * 5, "r_night": [0.01] * 10,
    })
    with pytest.raises(ValueError, match="one observation per date"):
        duration_regression(data)


def test_log_regression_rejects_returns_at_or_below_minus_one():
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=30),
        "night_hours": [17.5, 65.5] * 15, "r_night": [-1.0, 0.01] * 15,
    })
    with pytest.raises(ValueError, match="r_night > -1"):
        duration_regression(data, log=True)


def test_contrasts_measure_the_difference_between_night_types():
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=100),
        "night_type": (["Weekday"] * 4 + ["Weekend"]) * 20,
        "night_hours": ([17.5] * 4 + [65.5]) * 20,
        "r_night": ([0.001] * 4 + [0.003]) * 20,
    })
    summary = night_type_summary(data, block=10, n=200)
    assert summary.loc["Weekend", "n nights"] == 20
    assert summary.loc["Weekday", "mean (bp)"] == pytest.approx(10)
    contrast = night_type_contrasts(data, block=10, n=200).loc["Weekend - Weekday"]
    assert contrast["gap (bp)"] == pytest.approx(20)
    assert contrast["CI 95% low"] == pytest.approx(20)
    assert contrast["CI 95% high"] == pytest.approx(20)


def test_shared_bootstrap_preserves_means_and_paired_contrasts(monkeypatch):
    import src.analysis.risk as risk_module

    rng = np.random.default_rng(17)
    data = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=100),
        "night_type": (["Weekday"] * 4 + ["Weekend"]) * 20,
        "night_hours": ([17.5] * 4 + [65.5]) * 20,
        "r_night": rng.normal(0.001, 0.01, 100),
    })
    expected_summary = night_type_summary(data, block=10, n=200)
    expected_contrasts = night_type_contrasts(data, block=10, n=200)
    bootstrap = risk_module._bootstrap_nights
    calls = []

    def counted_bootstrap(*args, **kwargs):
        calls.append(1)
        return bootstrap(*args, **kwargs)

    monkeypatch.setattr(risk_module, "_bootstrap_nights", counted_bootstrap)
    summaries, contrasts = risk_type_analysis({"SPY": data}, block=10, n=200)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(summaries.loc["SPY"], expected_summary)
    pd.testing.assert_frame_equal(contrasts.loc["SPY"], expected_contrasts)


def test_block_intervals_reflect_serial_dependence():
    rng = np.random.default_rng(42)
    returns = np.repeat(rng.normal(0, 0.01, 50), 20)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=len(returns)),
        "night_type": "Weekday", "night_hours": 17.5, "r_night": returns,
    })
    independent = night_type_summary(data, block=1, n=400).iloc[0]
    blocks = night_type_summary(data, block=20, n=400).iloc[0]
    independent_width = independent["CI 95% high"] - independent["CI 95% low"]
    block_width = blocks["CI 95% high"] - blocks["CI 95% low"]
    assert block_width > 2 * independent_width


@pytest.mark.parametrize("start_year", [2005, 2010])
def test_period_labels_and_checks_follow_the_input_dates(start_year):
    from src.analysis.crime import gap_tests_by_period
    from src.analysis.risk import duration_tests_by_period, duration_robustness
    from src.visualization.risk_tables import robustness_table

    dates = pd.bdate_range(f"{start_year}-01-03", periods=30).append(pd.bdate_range("2012-01-03", periods=30))
    rng = np.random.default_rng(3)
    data = pd.DataFrame({
        "date": dates, "night_hours": [17.5, 65.5] * 30, "night_type": "Weekday",
        "r_night": rng.normal(0.001, 0.01, 60), "r_day": rng.normal(0.0, 0.01, 60),
    })
    periods = [f"{start_year}–2012"]
    if start_year < 2008:
        periods += [f"{start_year}–2007", "2008–2012"]
    tests = duration_tests_by_period({"SPY": data})
    assert tests.index.get_level_values("period").unique().tolist() == periods
    assert gap_tests_by_period(data).index.get_level_values("sample").unique().tolist() == periods
    assert tests.loc[("SPY", periods[0], "simple"), "n nights"] == len(data)
    checks = duration_robustness({"SPY": data})
    filters = checks.xs("Main, HAC 5", level="specification")
    filters = pd.concat({"liquidity": filters}, names=["filter"]).reorder_levels(["portfolio", "filter"])
    shown = robustness_table(tests, checks, filters)
    assert "Log returns" in shown.index
    row = tests.loc[("SPY", periods[0], "log")]
    expected = f"{row['beta (bp/hour)']:.3f} [{row['CI 95% low']:.3f}, {row['CI 95% high']:.3f}]"
    assert shown.loc["Log returns", "SPY"] == expected


@pytest.mark.parametrize("proportional", [True, False])
def test_proportionality_separates_per_hour_and_per_night_returns(proportional):
    from src.analysis.risk import proportionality_tests
    from src.visualization.risk_tables import proportionality_table

    rng = np.random.default_rng(5)
    kinds = np.tile(["Weekday"] * 4 + ["Weekend"], 400)
    hours = np.where(kinds == "Weekday", 17.5, 65.5)
    mean = 0.0004 * hours / 17.5 if proportional else np.full(len(hours), 0.0004)
    data = pd.DataFrame({
        "date": pd.bdate_range("2000-01-03", periods=len(hours)),
        "night_type": kinds, "night_hours": hours,
        "r_night": mean + rng.normal(0, 0.001, len(hours)),
    })
    tests = proportionality_tests({"Stocks": data})
    row = tests.loc[("Stocks", "All nights")]
    assert row["n nights"] == len(data)
    expected_beta = 4 / 17.5 if proportional else 0
    assert row["CI 95% low"] < expected_beta < row["CI 95% high"]
    if proportional:
        assert row["p-value alpha = 0"] > 0.01
    else:
        assert row["alpha (bp/night)"] == pytest.approx(4, abs=0.5)
        assert row["p-value alpha = 0"] < 0.001
    assert list(proportionality_table(tests).index) == list(tests.index)


def test_duration_mean_bands_preserve_slope_and_exclude_return_noise():
    from src.analysis.risk import duration_predictions

    rng = np.random.default_rng(23)
    hours = np.tile([17.5, 65.5, 89.5], 500)
    data = pd.DataFrame({
        "date": pd.bdate_range("2010-01-01", periods=len(hours)),
        "night_hours": hours, "night_type": np.tile(["Weekday", "Weekend", "Long weekend"], 500),
        "r_night": 0.0005 + 0.00001 * hours + rng.normal(0, 0.01, len(hours)),
    })
    predictions = duration_predictions({"Stocks": data}).loc["Stocks"]
    regression = duration_regression(data).iloc[0]
    observed_slope = predictions["mean (bp)"].diff().iloc[1] / predictions["night_hours"].diff().iloc[1]
    assert observed_slope == pytest.approx(regression["beta (bp/hour)"])
    half_width = (predictions["CI 95% high"] - predictions["CI 95% low"]) / 2
    assert (half_width > 0).all()
    # Mean uncertainty should be much smaller than individual-return dispersion.
    assert half_width.max() < 0.2 * (10000 * data["r_night"].std())
    assert (predictions["CI 95% low"] < predictions["mean (bp)"]).all()
    assert (predictions["CI 95% high"] > predictions["mean (bp)"]).all()


def test_closure_counts_collapse_only_identical_stock_and_etf_coverage():
    from src.visualization.risk_tables import closure_counts_table

    index = pd.Index(["Weekday", "Weekend"], name="night_type")
    equal = pd.DataFrame({"n nights": [100, 25]}, index=index)
    weighted = pd.DataFrame({"n nights": [84, 21]}, index=index)
    summaries = pd.concat({
        "Stocks, equal-weighted": equal,
        "Stocks, dollar-volume": weighted,
        "ETFs, equal-weighted": equal,
        "ETFs, dollar-volume": weighted,
    }, names=["portfolio"])
    shown = closure_counts_table(summaries)
    assert shown.columns.tolist() == ["Equal-weighted", "Dollar-volume-weighted"]
    assert shown.loc["Weekend"].tolist() == [25, 21]
    summaries.loc[("ETFs, equal-weighted", "Weekend"), "n nights"] = 24
    with pytest.raises(ValueError, match="different night counts"):
        closure_counts_table(summaries)
