import numpy as np
import pandas as pd

from src.analysis import confusion
from src.analysis.confusion import (
    benjamini_hochberg,
    case_summary,
    check_tickers,
    closest_in_size,
    describe_pairs,
    is_secondary_security,
    load_pair,
    name_words,
    night_score,
    one_letter_pairs,
    pair_summary,
    pairs_by_min_length,
    placebo_pvalue,
    pooled_placebo_test,
    ticker_coverage,
    ticker_name_matches,
    twin_events,
)
from src.analysis.crowd import event_study


def make_prices() -> pd.DataFrame:
    """Build a small price table: AA and AAL overlap for a few days, ZZ is listed later."""
    rows = [
        ("AA", "2020-01-01"), ("AA", "2020-01-02"), ("AA", "2020-01-03"),
        ("AAL", "2020-01-02"), ("AAL", "2020-01-03"), ("AAL", "2020-01-06"),
        ("ZZ", "2020-02-01"),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype("category")
    return df


def make_meta() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AA", "AAL", "ZZ"],
        "security_name": ["Alpha", "Alpha Air", "Zed"],
        "is_etf": [False, False, True],
    })


def test_pairs_cover_substitution_insertion_and_deletion():
    pairs = one_letter_pairs(["AAL", "AAN", "AA", "AAMC"])
    found = {(a, b): edit for a, b, edit in pairs.itertuples(index=False)}
    assert found == {
        ("AA", "AAL"): "insertion/deletion",
        ("AA", "AAN"): "insertion/deletion",
        ("AAL", "AAN"): "substitution",
    }


def test_distance_two_is_not_a_pair():
    # AAMC -> AAL needs one replacement and one deletion
    assert one_letter_pairs(["AAMC", "AAL"]).empty


def test_each_pair_appears_once_in_alphabetical_order():
    pairs = one_letter_pairs(["B", "A", "A", "C"])
    assert pairs[["ticker_a", "ticker_b"]].values.tolist() == [["A", "B"], ["A", "C"], ["B", "C"]]


def test_coverage_gives_first_last_and_count():
    coverage = ticker_coverage(make_prices())
    assert coverage.loc["AA", "n_days"] == 3
    assert coverage.loc["AAL", "first_date"] == pd.Timestamp("2020-01-02")
    assert coverage.loc["AAL", "last_date"] == pd.Timestamp("2020-01-06")


def test_overlap_of_a_pair():
    pairs = one_letter_pairs(["AA", "AAL"])
    table = describe_pairs(pairs, make_meta(), ticker_coverage(make_prices()))
    row = table.iloc[0]
    assert row["overlap_start"] == pd.Timestamp("2020-01-02")
    assert row["overlap_end"] == pd.Timestamp("2020-01-03")
    assert row["overlap_days"] == 1
    assert row["security_name_b"] == "Alpha Air"


def test_no_overlap_gives_zero_days():
    pairs = pd.DataFrame({"ticker_a": ["AAL"], "ticker_b": ["ZZ"], "edit": ["substitution"]})
    table = describe_pairs(pairs, make_meta(), ticker_coverage(make_prices()))
    assert table["overlap_days"].iloc[0] == 0


def test_summary_counts_pairs_and_tickers():
    pairs = one_letter_pairs(["AA", "AAL", "AAN", "XYZ"])
    summary = pair_summary(pairs, n_tickers=4)
    assert summary["pairs"] == 3
    assert summary["tickers involved"] == 3
    assert summary["share of tickers (%)"] == 75.0


def make_coverage(tickers: list[str], first: str = "2019-01-01", last: str = "2020-04-01") -> pd.DataFrame:
    """Give every ticker the same listing period."""
    coverage = pd.DataFrame({"first_date": pd.Timestamp(first), "last_date": pd.Timestamp(last)}, index=tickers)
    coverage["n_days"] = 100
    return coverage


def make_names() -> pd.DataFrame:
    """Three companies: ABC has the word "rapid" in its name, RAPID and RAPIDS are tickers that look like it."""
    return pd.DataFrame({
        "ticker": ["ABC", "RAPID", "RAPIDS"],
        "security_name": ["Rapid Video Communications, Inc. - Common Stock", "Rapid Holdings Ltd", "Other Corp"],
        "is_etf": [False, False, False],
    })


def make_name_coverage() -> pd.DataFrame:
    """RAPID is listed with ABC for 10 months, RAPIDS was listed years before."""
    return pd.DataFrame(
        {
            "first_date": pd.to_datetime(["2019-01-01", "2019-06-01", "2010-01-01"]),
            "last_date": pd.to_datetime(["2020-04-01", "2020-04-01", "2011-01-01"]),
            "n_days": 100,
        },
        index=["ABC", "RAPID", "RAPIDS"],
    )


def test_name_words_drops_legal_terms_and_short_words():
    assert name_words("Rapid Video Communications, Inc. - Common Stock") == {"rapid", "video", "communications"}


def test_secondary_securities_are_recognised():
    assert is_secondary_security("Alpha Corp - Depositary Shares, 5% Preferred Stock")
    assert is_secondary_security("Alpha Corp 5.6% Senior Notes due 2066")
    assert not is_secondary_security("Alpha Corp - Common Stock")


def test_ticker_equal_to_a_word_of_another_name_is_found():
    hits = ticker_name_matches(make_names(), make_name_coverage())
    assert hits[["ticker", "owner", "word", "distance"]].values.tolist() == [["RAPID", "ABC", "rapid", 0]]


def test_a_company_is_never_matched_to_its_own_name():
    # RAPID also has "rapid" in its own name, but only ABC must be reported as the owner
    hits = ticker_name_matches(make_names(), make_name_coverage())
    assert "RAPID" not in hits["owner"].tolist()


def test_one_letter_difference_and_overlap_condition():
    loose = ticker_name_matches(make_names(), make_name_coverage(), max_distance=1, min_overlap_days=0)
    assert set(loose["ticker"]) == {"RAPID", "RAPIDS"}
    strict = ticker_name_matches(make_names(), make_name_coverage(), max_distance=1)
    assert set(strict["ticker"]) == {"RAPID"}


def test_common_words_are_ignored():
    meta = pd.DataFrame({
        "ticker": ["BOND", "BOA", "BOB", "BOC", "BOD"],
        "security_name": ["Plain Corp", "Bond Fund One", "Bond Fund Two", "Bond Fund Three", "Bond Fund Four"],
        "is_etf": [False] * 5,
    })
    coverage = make_coverage(meta["ticker"].tolist())
    assert ticker_name_matches(meta, coverage, max_owners=3).empty
    assert set(ticker_name_matches(meta, coverage, max_owners=10)["owner"]) == {"BOA", "BOB", "BOC", "BOD"}


def test_secondary_securities_are_not_used_as_a_source_of_names():
    meta = pd.DataFrame({
        "ticker": ["ALPH", "ALPHP"],
        "security_name": ["Alph Inc - Common Stock", "Alph Inc - Depositary Shares, 5% Preferred Stock"],
        "is_etf": [False, False],
    })
    coverage = make_coverage(meta["ticker"].tolist())
    assert ticker_name_matches(meta, coverage).empty
    assert not ticker_name_matches(meta, coverage, ordinary_only=False).empty


def test_funds_of_the_same_family_are_dropped_but_two_companies_are_not():
    funds = pd.DataFrame({
        "ticker": ["MOAT", "GOAT"],
        "security_name": ["Delta Wide Moat ETF", "Delta Global Wide Moat ETF"],
        "is_etf": [True, True],
    })
    companies = pd.DataFrame({
        "ticker": ["SUNNY", "SNX"],
        "security_name": ["Sunny Pharma Inc", "Sunny Foods Inc"],
        "is_etf": [False, False],
    })
    assert ticker_name_matches(funds, make_coverage(["MOAT", "GOAT"])).empty
    assert len(ticker_name_matches(funds, make_coverage(["MOAT", "GOAT"]), drop_fund_families=False)) == 1
    kept = ticker_name_matches(companies, make_coverage(["SUNNY", "SNX"]))
    assert kept[["ticker", "owner"]].values.tolist() == [["SUNNY", "SNX"]]


def make_case() -> pd.DataFrame:
    """Two tickers over 30 days: BUZZ has an attention spike on day 25, RECV a night return of 2 % on day 26."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    night = np.zeros(30)
    night[26] = 0.02
    buzz_spike = np.zeros(30, dtype=bool)
    buzz_spike[25] = True
    buzz = pd.DataFrame({"ticker": "BUZZ", "date": dates, "volume": 100.0, "open": 10.0, "close": 10.0,
                         "dollar_volume": 1000.0, "r_night": 0.0, "r_day": 0.0, "spike": buzz_spike})
    recv = pd.DataFrame({"ticker": "RECV", "date": dates, "volume": 10.0, "open": 10.0, "close": 10.0,
                         "dollar_volume": 100.0, "r_night": night, "r_day": 0.0, "spike": False})
    return pd.concat([buzz, recv], ignore_index=True)


def test_twin_events_flag_the_spike_days_of_the_well_known_company():
    events = twin_events(make_case(), "BUZZ", "RECV")
    assert set(events["ticker"]) == {"RECV"}
    assert events.loc[events["spike"], "date"].tolist() == [pd.bdate_range("2020-01-01", periods=30)[25]]


def test_a_spike_day_without_trading_of_the_receiver_is_lost():
    case = make_case()
    case = case.drop(case[(case["ticker"] == "RECV") & (case["date"] == pd.bdate_range("2020-01-01", periods=30)[25])].index)
    assert not twin_events(case, "BUZZ", "RECV")["spike"].any()


def test_twin_events_feed_the_event_study_of_part_v():
    result = event_study(twin_events(make_case(), "BUZZ", "RECV"), window=2, start_year=2000)
    night_after = result[(result["day"] == 1) & (result["return"] == "r_night")].iloc[0]
    assert np.isclose(night_after["mean"], 0.02)
    assert night_after["n"] == 1


def test_case_summary_has_one_column_per_ticker_receiver_first():
    summary = case_summary(make_case(), "BUZZ", "RECV")
    assert summary.columns.tolist() == ["RECV", "BUZZ"]
    assert summary.loc["days", "BUZZ"] == 30
    assert summary.loc["median volume (shares)", "RECV"] == 10
    assert summary.loc["days without volume (%)", "RECV"] == 0.0


def make_prices_for_pair(tickers: list[str], extreme_open_day: int | None = None) -> pd.DataFrame:
    """Flat prices for the given tickers over 60 business days, with an optional wrong opening price."""
    dates = pd.bdate_range("2019-12-02", periods=60)
    prices = pd.concat([
        pd.DataFrame({"ticker": ticker, "date": dates, "open": 10.0, "close": 10.0, "open_adj": 10.0,
                      "high_adj": 10.0, "low_adj": 10.0, "close_adj": 10.0, "volume": 100.0})
        for ticker in tickers
    ], ignore_index=True)
    if extreme_open_day is not None:
        prices.loc[extreme_open_day, ["open", "open_adj"]] = 30.0  # an opening price three times too high
    return prices


def test_load_pair_adds_the_measures_of_parts_i_and_v(monkeypatch):
    prices = make_prices_for_pair(["BUZZ", "RECV"])
    monkeypatch.setattr(confusion, "load_prices", lambda tickers: prices.copy())
    case = load_pair("BUZZ", "RECV")
    assert {"r_night", "r_day", "dollar_volume", "extreme_return", "abn_volume", "spike"} <= set(case.columns)
    assert set(case["ticker"]) == {"BUZZ", "RECV"}
    assert len(case) == len(prices)


def test_load_pair_flags_extreme_returns_without_removing_them(monkeypatch):
    prices = make_prices_for_pair(["RECV"], extreme_open_day=30)
    monkeypatch.setattr(confusion, "load_prices", lambda tickers: prices.copy())
    case = load_pair("RECV", "RECV")
    assert len(case) == len(prices)
    assert case["extreme_return"].any()


def test_pairs_by_min_length_counts_the_shorter_ticker():
    pairs = pd.DataFrame({"ticker_a": ["AA", "ABC", "ABCD"], "ticker_b": ["AAL", "ABD", "ABCE"], "edit": ["x"] * 3})
    counts = pairs_by_min_length(pairs)
    assert counts.to_dict() == {2: 1, 3: 1, 4: 1}


def test_check_tickers_flags_missing_ones():
    meta = pd.DataFrame({"ticker": ["AA", "NOPRICE"], "security_name": ["Alpha", "No prices"], "is_etf": [False, False]})
    coverage = ticker_coverage(make_prices())
    table = check_tickers(["AA", "MISSING", "NOPRICE"], meta, coverage)
    assert table.loc["AA", "in_prices"]
    assert not table.loc["MISSING", "in_prices"] and not table.loc["MISSING", "in_metadata"]
    assert table.loc["NOPRICE", "in_metadata"] and not table.loc["NOPRICE", "in_prices"]
    assert pd.isna(table.loc["MISSING", "first_date"])


def make_returns(night_after: float = 0.04, day_after: float = -0.02) -> pd.DataFrame:
    """Ten sessions of one ticker; the session after day 5 has a night return and a day return as given."""
    night = np.zeros(10)
    day = np.zeros(10)
    night[6], day[6] = night_after, day_after
    return pd.DataFrame({"r_night": night, "r_day": day, "bad": False})


def test_night_score_is_the_excess_night_after_an_event():
    rows = make_returns()
    spike = pd.Series(np.arange(10) == 5)
    score, n_events = night_score(rows, spike)
    # night after the event: 0.04, mean over the ten sessions: 0.004
    assert np.isclose(score, 0.036)
    assert n_events == 1


def test_a_flagged_row_cannot_be_an_event():
    rows = make_returns()
    rows.loc[5, "bad"] = True
    score, n_events = night_score(rows, pd.Series(np.arange(10) == 5), flags=["bad"])
    assert n_events == 0 and np.isnan(score)


def test_placebo_pvalue_counts_the_placebo_scores_at_least_as_large():
    assert placebo_pvalue(0.25, np.array([0.1, 0.2, 0.3, np.nan])) == 0.5
    assert placebo_pvalue(1.0, np.zeros(9)) == 0.1


def test_benjamini_hochberg_on_a_textbook_example():
    result = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.005]), q=0.03)
    assert np.allclose(result["q_value"], [0.02, 0.04, 0.04, 0.02])
    assert result["rejected"].tolist() == [True, False, False, True]


def test_closest_in_size_uses_a_log_scale():
    pool = pd.Series({"A": 1e6, "B": 1e7, "C": 1e8})
    assert closest_in_size(2e7, pool, 2) == ["B", "C"]


def make_synthetic_prices(seed: int = 0) -> pd.DataFrame:
    """SPY, a receiver RECV with a planted overnight gain after the spikes of BUZZ, and six placebo stocks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=400)
    spike_days = np.arange(30, 390, 30)
    frames = []
    for ticker in ["SPY", "RECV", "BUZZ"] + [f"P{i}" for i in range(6)]:
        night = rng.normal(0, 0.002, len(dates))
        day = rng.normal(0, 0.002, len(dates))
        volume = rng.lognormal(0, 1.0, len(dates)) * 1e5
        if ticker == "RECV":
            night[spike_days + 1], day[spike_days + 1] = 0.05, -0.05
            volume = np.full(len(dates), 5e4)
        if ticker == "BUZZ":
            volume = np.full(len(dates), 1e5)
            volume[spike_days] = 1e6
        close = np.empty(len(dates))
        open_ = np.empty(len(dates))
        previous = 10.0
        for i in range(len(dates)):
            open_[i] = previous * (1 + night[i])
            close[i] = open_[i] * (1 + day[i])
            previous = close[i]
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "open": open_, "close": close, "open_adj": open_, "close_adj": close,
            "high_adj": np.maximum(open_, close), "low_adj": np.minimum(open_, close), "volume": volume,
            "asset_type": "stock",
        }))
    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].astype("category")
    return prices


def test_pooled_placebo_test_finds_a_planted_overnight_gain(monkeypatch):
    prices = make_synthetic_prices()

    def fake_load_prices(tickers=None, columns=None):
        rows = prices if tickers is None else prices[prices["ticker"].isin(tickers)]
        return (rows[columns] if columns else rows).copy()

    monkeypatch.setattr(confusion, "load_prices", fake_load_prices)
    matches = pd.DataFrame({"ticker": ["RECV"], "owner": ["BUZZ"]})
    table, summary = pooled_placebo_test(
        matches, flags=[], n_pool=6, n_candidates=4, n_draws=30, min_events=3, min_pool_dollar=1e3,
        min_pool_days=100,
    )
    assert table.columns.tolist() == ["receiver", "buzz", "n_events", "score", "placebo_mean", "p", "q_value", "rejected"]
    assert len(table) == 1 and table.loc[0, "n_events"] >= 10
    assert table.loc[0, "score"] > 4  # in percentage points
    assert table.loc[0, "p"] <= 0.1
    assert summary["pairs tested"] == 1

