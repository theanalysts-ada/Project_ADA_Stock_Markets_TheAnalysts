import numpy as np
import pandas as pd
import pytest

from src.features.returns import add_adjusted_prices, add_returns


def make_prices(rows: list[tuple]) -> pd.DataFrame:
    """Build a small price table from (ticker, date, open, close, adj_close) rows, with adjusted prices."""
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "close", "adj_close"])
    df["date"] = pd.to_datetime(df["date"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return add_adjusted_prices(df)


def test_split_does_not_create_false_overnight_return():
    # 2-for-1 split on day 2: the raw price halves, Adj Close of day 1 is already divided by 2
    df = add_returns(make_prices([
        ("A", "2020-01-06", 98.0, 100.0, 50.0),
        ("A", "2020-01-07", 50.0, 51.0, 51.0),
    ]))
    assert df["r_night"].iloc[1] == pytest.approx(0.0)
    assert df["open"].iloc[1] / df["close"].iloc[0] - 1 == pytest.approx(-0.5)


def test_dividend_is_not_an_overnight_loss():
    # $1 dividend on day 2: the raw open drops by 1, Adj Close of day 1 is lowered by the same amount
    df = add_returns(make_prices([
        ("A", "2020-01-06", 100.0, 100.0, 99.0),
        ("A", "2020-01-07", 99.0, 99.0, 99.0),
    ]))
    assert df["r_night"].iloc[1] == pytest.approx(0.0)


def test_night_plus_day_equals_total_in_log_returns():
    df = add_returns(make_prices([
        ("A", "2020-01-06", 10.0, 10.5, 10.5),
        ("A", "2020-01-07", 11.0, 10.8, 10.8),
        ("A", "2020-01-08", 10.2, 10.9, 10.9),
    ]))
    total = np.log(df["close_adj"] / df["close_adj"].shift(1))
    decomposed = np.log1p(df["r_night"]) + np.log1p(df["r_day"])
    assert np.allclose(total.iloc[1:], decomposed.iloc[1:], atol=1e-6)


def test_first_observation_of_each_ticker_has_no_overnight_return():
    df = add_returns(make_prices([
        ("B", "2020-01-07", 20.0, 21.0, 21.0),
        ("A", "2020-01-06", 10.0, 11.0, 11.0),
        ("A", "2020-01-07", 12.0, 12.0, 12.0),
        ("B", "2020-01-08", 22.0, 22.0, 22.0),
    ]))
    assert df.groupby("ticker").head(1)["r_night"].isna().all()
    assert df.groupby("ticker").nth(1)["r_night"].notna().all()


def test_overnight_return_spans_weekends_but_not_missing_days():
    # A trades every day; B has no row on Tuesday 7 January
    df = add_returns(make_prices([
        ("A", "2020-01-03", 10.0, 10.0, 10.0),
        ("A", "2020-01-06", 11.0, 11.0, 11.0),
        ("A", "2020-01-07", 12.0, 12.0, 12.0),
        ("A", "2020-01-08", 13.0, 13.0, 13.0),
        ("B", "2020-01-06", 20.0, 20.0, 20.0),
        ("B", "2020-01-08", 22.0, 22.0, 22.0),
    ]))
    night = df.set_index(["ticker", "date"])["r_night"]
    assert night[("A", pd.Timestamp("2020-01-06"))] == pytest.approx(0.1)  # Friday close -> Monday open
    assert np.isnan(night[("B", pd.Timestamp("2020-01-08"))])


def test_buy_and_hold_wealth_is_night_times_day():
    from src.analysis.crime import ticker_wealth
    df = add_returns(make_prices([
        ("SPY", "2020-01-06", 10.0, 10.0, 10.0),
        ("SPY", "2020-01-07", 11.0, 12.1, 12.1),
        ("SPY", "2020-01-08", 11.0, 10.0, 10.0),
    ]))
    wealth = ticker_wealth(df, "SPY")
    assert wealth["buy_hold"].iloc[-1] == pytest.approx(wealth["night"].iloc[-1] * wealth["day"].iloc[-1])


def test_dollar_volume_weights_use_the_previous_twenty_days():
    from src.features.returns import add_dollar_volume, portfolio_returns
    dates = pd.bdate_range("2020-01-02", periods=21)
    rows = []
    for date in dates:
        rows += [("A", date, 10.0, 10.0, 10.0), ("B", date, 10.0, 10.0, 10.0)]
    rows[-2] = ("A", dates[-1], 10.0, 11.0, 11.0)
    rows[-1] = ("B", dates[-1], 10.0, 12.0, 12.0)
    df = add_returns(make_prices(rows))
    df["volume"] = 100
    df.loc[df["ticker"] == "B", "volume"] = 300
    df.loc[(df["ticker"] == "A") & (df["date"] == dates[-1]), "volume"] = 9999
    df.loc[(df["ticker"] == "B") & (df["date"] == dates[-1]), "volume"] = 1
    port = portfolio_returns(add_dollar_volume(df), "dollar_volume")
    assert port.loc[dates[-1], "r_day"] == pytest.approx(0.175)
