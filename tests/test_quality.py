import pandas as pd

from src.data.quality import flag_suspicious
from src.features.returns import add_adjusted_prices, add_dollar_volume, add_returns


def make_prices(rows: list[tuple]) -> pd.DataFrame:
    """Build a small flagged table from (ticker, date, open, close, adj_close, volume) rows."""
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["asset_type"] = "stock"
    return flag_suspicious(add_dollar_volume(add_returns(add_adjusted_prices(df))))


def test_spliced_history_flags_rows_before_a_tenfold_jump():
    df = make_prices([
        ("A", "2020-01-06", 100.0, 100.0, 100.0, 1000),
        ("A", "2020-01-07", 100.0, 100.0, 100.0, 1000),
        ("A", "2020-01-08", 10.0, 10.0, 10.0, 1000),
        ("B", "2020-01-06", 20.0, 20.0, 20.0, 1000),
        ("B", "2020-01-07", 21.0, 21.0, 21.0, 1000),
    ])
    flagged = df.set_index(["ticker", "date"])["spliced_history"]
    assert flagged[("A", pd.Timestamp("2020-01-06"))]
    assert flagged[("A", pd.Timestamp("2020-01-07"))]
    assert not flagged[("A", pd.Timestamp("2020-01-08"))]
    assert not flagged.loc["B"].any()
