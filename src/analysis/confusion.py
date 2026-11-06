import re

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from src.config import STUDY_START_YEAR
from src.data.load import load_prices
from src.data.quality import FILTERS, flag_suspicious
from src.features.returns import add_dollar_volume, add_returns
from src.features.volume import abnormal_volume

# Words of a security name that say nothing about the brand (legal terms, type of security...)
IGNORED_WORDS = {
    "common", "stock", "shares", "share", "class", "ordinary", "inc", "incorporated", "corp", "corporation",
    "company", "limited", "ltd", "holdings", "holding", "group", "trust", "fund", "etf", "units", "series",
    "preferred", "depositary", "receipts", "warrants", "rights", "notes", "issued", "distribution", "when",
    "with", "new", "the", "and", "for", "from",
}

# Words showing that a security is not the main stock of a company (preferred shares, notes...)
SECONDARY_WORDS = {
    "preferred", "pfd", "depositary", "notes", "unit", "units", "warrant", "warrants", "rights", "debentures",
    "subordinated", "senior", "due",
}

# Flags used for the pairs. The liquidity filter would remove the small stocks we study, so we use
# the light filter plus the stale-open flag (open = previous close makes the overnight return wrong).
PAIR_FLAGS = FILTERS["light"] + ["open_equals_prev_close"]

# A ticker that traded every market day
CALENDAR_TICKER = "SPY"


def ticker_coverage(prices):
    """First day, last day and number of days of each ticker.

    Args:
        prices: Prices with ticker and date.

    Returns:
        One row per ticker: first_date, last_date and n_days.
    """
    coverage = prices.groupby("ticker", observed=True)["date"].agg(first_date="min", last_date="max", n_days="size")
    coverage.index = coverage.index.astype(str)
    return coverage


def one_letter_pairs(tickers):
    """All pairs of tickers at edit distance 1 (one letter replaced, added or removed).

    Args:
        tickers: List of tickers.

    Returns:
        One row per pair: ticker_a, ticker_b (in alphabetical order) and the type of edit.
    """
    names = sorted(set(tickers))

    # Edit distance between every pair of tickers (anything above 1 is stored as 2)
    distance = process.cdist(names, names, scorer=Levenshtein.distance, dtype=np.uint8, score_cutoff=1, workers=-1)

    # Positions where the distance is 1. np.triu keeps the upper half of the matrix,
    # so that each pair appears once (A-B but not B-A) and a ticker is not paired with itself.
    rows, cols = np.nonzero(np.triu(distance == 1, k=1))

    pairs = pd.DataFrame({
        "ticker_a": [names[i] for i in rows],
        "ticker_b": [names[j] for j in cols],
    })
    # Same length: one letter was replaced. Different length: one letter was added or removed.
    same_length = pairs["ticker_a"].str.len() == pairs["ticker_b"].str.len()
    pairs["edit"] = np.where(same_length, "substitution", "insertion/deletion")
    return pairs


def pair_summary(pairs, n_tickers):
    """Number of pairs, and of tickers that have at least one neighbour at distance 1.

    Args:
        pairs: Output of one_letter_pairs.
        n_tickers: Number of tickers compared.

    Returns:
        The counts, and the share of tickers involved (%).
    """
    involved = len(set(pairs["ticker_a"]) | set(pairs["ticker_b"]))
    return pd.Series({
        "pairs": len(pairs),
        "substitutions": int((pairs["edit"] == "substitution").sum()),
        "insertions / deletions": int((pairs["edit"] == "insertion/deletion").sum()),
        "tickers involved": involved,
        "share of tickers (%)": round(100 * involved / n_tickers, 1),
    }, dtype="object")


def pairs_by_min_length(pairs):
    """Number of pairs by the length of the shorter ticker.

    Args:
        pairs: Output of one_letter_pairs.

    Returns:
        Number of pairs for each length.
    """
    shorter = np.minimum(pairs["ticker_a"].str.len(), pairs["ticker_b"].str.len())
    counts = shorter.value_counts().sort_index()
    counts.name = "pairs"
    counts.index.name = "letters in the shorter ticker"
    return counts


def _tokens(security_name):
    """Words of a name in lower case: everything that is not a letter or a digit becomes a space."""
    return re.sub(r"[^a-z0-9]+", " ", security_name.lower()).split()


def name_words(security_name, min_length=4):
    """Distinctive words of a security name: no legal terms and no word shorter than min_length."""
    words = set()
    for word in _tokens(security_name):
        if len(word) >= min_length and word not in IGNORED_WORDS:
            words.add(word)
    return words


def is_secondary_security(security_name):
    """True if the security is not the main stock of a company (preferred shares, notes, units...)."""
    for word in _tokens(security_name):
        if word in SECONDARY_WORDS:
            return True
    return False


def first_word(security_name):
    """First word of a name, in lower case (for ETFs, the name of the fund family)."""
    tokens = _tokens(security_name)
    if len(tokens) == 0:
        return ""
    return tokens[0]


def ticker_name_matches(meta, coverage, max_distance=0, min_length=4, min_overlap_days=90, max_owners=3,
                        ordinary_only=True, drop_fund_families=True):
    """Tickers that look like a word of another company's name (ticker ALPHA, company "Alpha Video").

    Args:
        meta: Metadata with ticker, security_name and is_etf.
        coverage: Output of ticker_coverage.
        max_distance: Max edit distance between ticker and word (0 = identical).
        min_length: Min length of tickers and words.
        min_overlap_days: Min days the two were listed at the same time.
        max_owners: A word in more company names than this is too common.
        ordinary_only: Only use the names of main stocks (no preferred shares, notes...).
        drop_fund_families: Drop pairs of ETFs from the same fund family.

    Returns:
        One row per match: ticker, ticker_name, word, distance, owner, owner_name, overlap_days.
    """
    # 1. Name and ETF flag of every ticker that has prices
    listed = meta[meta["ticker"].isin(coverage.index)].drop_duplicates("ticker").set_index("ticker")
    names = listed["security_name"].fillna("")
    is_etf = listed["is_etf"]

    # 2. Words of each company name, with the company that "owns" them
    owner_rows = []
    for ticker, name in names.items():
        if ordinary_only and is_secondary_security(name):
            continue
        for word in name_words(name, min_length):
            owner_rows.append({"word": word, "owner": ticker})
    owners = pd.DataFrame(owner_rows, columns=["word", "owner"])

    # 3. Drop common words, found in the names of too many companies
    n_owners = owners.groupby("word")["owner"].nunique()
    common_words = n_owners[n_owners > max_owners].index
    owners = owners[~owners["word"].isin(common_words)]

    # 4. Edit distance between each ticker (in lower case) and each word
    tickers = [ticker for ticker in names.index if len(ticker) >= min_length]
    words = sorted(owners["word"].unique())
    distance = process.cdist(
        [ticker.lower() for ticker in tickers], words,
        scorer=Levenshtein.distance, dtype=np.uint8, score_cutoff=max_distance, workers=-1,
    )
    rows, cols = np.nonzero(distance <= max_distance)
    hits = pd.DataFrame({
        "ticker": [tickers[i] for i in rows],
        "word": [words[j] for j in cols],
        "distance": distance[rows, cols].astype(int),
    })

    # 5. Add the company that owns the word, and never match a company with its own name
    hits = hits.merge(owners, on="word")
    hits = hits[hits["ticker"] != hits["owner"]].copy()

    # 6. Drop two ETFs of the same fund family (two funds whose names start with the same word)
    if drop_fund_families:
        both_etf = (hits["ticker"].map(is_etf) == True) & (hits["owner"].map(is_etf) == True)
        same_family = hits["ticker"].map(names).map(first_word) == hits["owner"].map(names).map(first_word)
        hits = hits[~(both_etf & same_family)].copy()

    # 7. Period when the two securities were listed at the same time
    start = pd.concat([hits["ticker"].map(coverage["first_date"]), hits["owner"].map(coverage["first_date"])], axis=1)
    end = pd.concat([hits["ticker"].map(coverage["last_date"]), hits["owner"].map(coverage["last_date"])], axis=1)
    hits["overlap_days"] = (end.min(axis=1) - start.max(axis=1)).dt.days.clip(lower=0)
    hits["ticker_name"] = hits["ticker"].map(names)
    hits["owner_name"] = hits["owner"].map(names)

    # 8. Keep the pairs listed together long enough
    hits = hits[hits["overlap_days"] >= min_overlap_days]
    columns = ["ticker", "ticker_name", "word", "distance", "owner", "owner_name", "overlap_days"]
    return hits[columns].sort_values(["ticker", "owner"]).reset_index(drop=True)


def load_pair(buzz, receiver, threshold=3.0):
    """Prices of two tickers with returns, quality flags and attention spikes (as in Parts I and V).

    Args:
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.
        threshold: Abnormal volume above which a day is a spike.

    Returns:
        One row per ticker and date. No row is removed: event_study blanks the flagged rows later.
    """
    case = load_prices(tickers=[buzz, receiver])
    case["ticker"] = case["ticker"].astype(str)
    case = add_returns(case)
    case = add_dollar_volume(case)
    case = flag_suspicious(case)
    case = abnormal_volume(case, threshold=threshold)
    return case


def twin_events(case, buzz, receiver):
    """Rows of the receiver, where the events are the spike days of the well-known company.

    The result can be passed to event_study (Part V): day 0 is a spike of the well-known company,
    and day k is the k-th trading day of the receiver after it.

    Args:
        case: Output of load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        The rows of the receiver, whose spike column now holds the spikes of the well-known company.
    """
    buzz_spikes = case[case["ticker"] == buzz].set_index("date")["spike"]
    rows = case[case["ticker"] == receiver].copy()
    # Spike of the well-known company on the same date (False if it did not trade that day)
    rows["spike"] = rows["date"].map(buzz_spikes) == True
    return rows.reset_index(drop=True)


def case_summary(case, buzz, receiver):
    """Size and data quality of the two tickers of a case.

    Args:
        case: Output of load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        One column per ticker: period, size, and share of days with no volume or a stale open.
    """
    columns = {}
    for ticker in [receiver, buzz]:
        rows = case[case["ticker"] == ticker].sort_values("date")
        columns[ticker] = pd.Series({
            "first day": rows["date"].min().date(),
            "last day": rows["date"].max().date(),
            "days": len(rows),
            "median close ($)": round(float(rows["close"].median()), 2),
            "median volume (shares)": int(rows["volume"].median()),
            "median dollar volume ($)": int(rows["dollar_volume"].median()),
            "days without volume (%)": round(100 * (rows["volume"] == 0).mean(), 1),
            "open = previous close (%)": round(100 * (rows["open"] == rows["close"].shift(1)).mean(), 1),
            "open = close (%)": round(100 * (rows["open"] == rows["close"]).mean(), 1),
        })
    return pd.DataFrame(columns)


def _spikes_on(spikes, ticker, dates):
    """Spikes of a ticker on the given dates (False on days it did not trade)."""
    return spikes[ticker].reindex(dates) == True


def night_score(rows, spike, flags=None):
    """Overnight return after the events, minus the usual overnight return. Returns (score, number of events)."""
    if flags:
        keep = ~rows[flags].any(axis=1)
    else:
        keep = pd.Series(True, index=rows.index)

    night = rows["r_night"].where(keep).astype(float)
    events = spike.to_numpy(dtype=bool) & keep.to_numpy()

    # shift(-1): on each event day, look at the overnight return of the next session
    night_after = night.shift(-1)[events].dropna()
    score = night_after.mean() - night.mean()
    return float(score), len(night_after)


def placebo_pvalue(score, placebo_scores):
    """Share of placebo scores at least as large as the observed score (one-sided p-value)."""
    scores = np.array(placebo_scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n_larger = (scores >= score).sum()
    return float((1 + n_larger) / (1 + len(scores)))


def closest_in_size(size, pool_sizes, n):
    """The n tickers of the pool whose size is closest to `size`, compared on a log scale."""
    size_gap = (np.log(pool_sizes) - np.log(size)).abs()
    return size_gap.nsmallest(n).index.tolist()


def usual_overnight_return(case, ticker, flags=PAIR_FLAGS, start_year=STUDY_START_YEAR):
    """Mean overnight return of a ticker over all its sessions, cleaned as in event_study.

    Args:
        case: Output of load_pair.
        ticker: Ticker to average.
        flags: Quality flags; flagged rows are ignored.
        start_year: Sessions before this year are ignored.

    Returns:
        The mean overnight return (0.01 = one percentage point).
    """
    rows = case[(case["ticker"] == ticker) & (case["date"].dt.year >= start_year)]
    night = rows["r_night"].astype(float)
    if flags:
        night = night.where(~rows[flags].any(axis=1))
    return float(night.mean())


def pair_placebo_test(receiver, buzz, flags=PAIR_FLAGS, n_placebos=100, min_pool_dollar=1e7, min_pool_days=2000,
                      start_year=STUDY_START_YEAR):
    """Does the receiver gain more overnight after the spikes of its well-known company than after the spikes
    of companies of similar size (placebos)?

    Args:
        receiver: Ticker that may be bought by mistake.
        buzz: Ticker of the well-known company.
        flags: Quality flags removed from the receiver's returns.
        n_placebos: Number of placebo companies.
        min_pool_dollar, min_pool_days: Min median dollar volume and min days of a placebo company.
        start_year: Data before this year is ignored.

    Returns:
        A dict with score, n_events, placebo_scores (closest in size first), p and placebos (their sizes).
    """
    # 1. Rows of the receiver, with returns and quality flags
    prices = load_prices(tickers=[receiver, CALENDAR_TICKER])
    prices["ticker"] = prices["ticker"].astype(str)
    prices = flag_suspicious(add_dollar_volume(add_returns(prices)))
    rows = prices[(prices["ticker"] == receiver) & (prices["date"].dt.year >= start_year)]
    rows = rows.sort_values("date").reset_index(drop=True)

    # 2. Size (median dollar volume), number of days and type of every ticker
    volume = load_prices(columns=["ticker", "date", "close", "volume", "asset_type"])
    volume = volume[volume["date"].dt.year >= start_year].copy()
    volume["dollar"] = volume["close"].astype(float) * volume["volume"].astype(float)
    by_ticker = volume.groupby("ticker", observed=True)
    sizes = by_ticker["dollar"].median()
    sizes.index = sizes.index.astype(str)
    days = by_ticker["dollar"].size().set_axis(sizes.index)
    is_stock = (by_ticker["asset_type"].first() == "stock").set_axis(sizes.index)
    if buzz not in sizes.index:
        raise ValueError(buzz + " has no volume data.")

    # 3. Placebos: the liquid stocks closest in size to the well-known company
    eligible = sizes[(sizes >= min_pool_dollar) & (days >= min_pool_days) & is_stock]
    placebos = closest_in_size(sizes[buzz], eligible.drop([receiver, buzz], errors="ignore"), n_placebos)

    # 4. Attention spikes of the well-known company and of the placebos: one column per ticker
    subset = volume[volume["ticker"].isin(set(placebos) | {buzz})].copy()
    subset["ticker"] = subset["ticker"].astype(str)
    spikes = abnormal_volume(subset).pivot(index="date", columns="ticker", values="spike")

    # 5. Score with the real spikes, then with the spikes of each placebo
    score, n_events = night_score(rows, _spikes_on(spikes, buzz, rows["date"]), flags)
    placebo_scores = []
    for placebo in placebos:
        placebo_score, _ = night_score(rows, _spikes_on(spikes, placebo, rows["date"]), flags)
        placebo_scores.append(placebo_score)
    placebo_scores = np.array(placebo_scores)

    return {"score": score, "n_events": n_events, "placebo_scores": placebo_scores,
            "p": placebo_pvalue(score, placebo_scores), "placebos": sizes.reindex(placebos)}