import re

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from src.data.load import load_prices
from src.data.quality import FILTERS, flag_suspicious
from src.features.returns import add_dollar_volume, add_returns
from src.features.volume import abnormal_volume

# Legal and security-type words of a Nasdaq security name: they carry no brand meaning.
IGNORED_WORDS = {
    "common", "stock", "shares", "share", "class", "ordinary", "inc", "incorporated", "corp", "corporation",
    "company", "limited", "ltd", "holdings", "holding", "group", "trust", "fund", "etf", "units", "series",
    "preferred", "depositary", "receipts", "warrants", "rights", "notes", "issued", "distribution", "when",
    "with", "new", "the", "and", "for", "from",
}
# Flags removed from the returns of a pair. The liquidity filter of Part V would remove the small stocks studied
# here, so we use the light filter of Part I plus the stale-open flags, as Part V does.
PAIR_FLAGS = FILTERS["light"] + ["open_equals_prev_close", "open_equals_close"]
# A ticker that traded every market day, used to build the calendar of the returns of a few tickers.
CALENDAR_TICKER = "SPY"
# Words that reveal a secondary security of a company (its preferred shares, notes, units...), not its main stock.
SECONDARY_WORDS = {
    "preferred", "pfd", "depositary", "notes", "unit", "units", "warrant", "warrants", "rights", "debentures",
    "subordinated", "senior", "due",
}


def ticker_coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """Find the first and last observation of each ticker.

    Args:
        prices: Table from load_prices, with at least ticker and date.

    Returns:
        One row per ticker: first_date, last_date and n_days (number of observed days).
    """
    grouped = prices.groupby("ticker", observed=True)["date"]
    coverage = grouped.agg(first_date="min", last_date="max", n_days="size")
    coverage.index = coverage.index.astype(str)
    return coverage


def one_letter_pairs(tickers: list[str]) -> pd.DataFrame:
    """List every pair of tickers whose edit distance is exactly 1.

    The edit distance (Levenshtein) is the minimum number of single-letter replacements,
    insertions or deletions needed to turn one ticker into the other. Each pair appears once,
    with ticker_a before ticker_b in alphabetical order.

    Args:
        tickers: Ticker symbols; duplicates are ignored.

    Returns:
        One row per pair: ticker_a, ticker_b and edit ("substitution" when both tickers have the
        same length, "insertion/deletion" otherwise).
    """
    names = sorted(set(tickers))
    distance = process.cdist(names, names, scorer=Levenshtein.distance, dtype=np.uint8, score_cutoff=1, workers=-1)
    rows, cols = np.nonzero(np.triu(distance == 1, k=1))
    lengths = np.array([len(name) for name in names])
    return pd.DataFrame({
        "ticker_a": np.array(names, dtype=object)[rows],
        "ticker_b": np.array(names, dtype=object)[cols],
        "edit": np.where(lengths[rows] == lengths[cols], "substitution", "insertion/deletion"),
    })


def describe_pairs(pairs: pd.DataFrame, meta: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Add names, asset types and the common trading period to each pair.

    Args:
        pairs: Table returned by one_letter_pairs.
        meta: Table returned by load_metadata, with ticker, security_name and is_etf.
        coverage: Table returned by ticker_coverage.

    Returns:
        The pairs with, for each side (suffix _a and _b), the security name, the is_etf flag, the first and
        last date and the number of days, then overlap_start, overlap_end and overlap_days
        (calendar days; 0 when the two tickers were never listed at the same time).
    """
    info = meta.set_index("ticker")[["security_name", "is_etf"]].join(coverage, how="inner")
    table = pairs
    for side in ("a", "b"):
        side_info = info.add_suffix(f"_{side}").rename_axis(f"ticker_{side}").reset_index()
        table = table.merge(side_info, on=f"ticker_{side}", how="left")
    table["overlap_start"] = table[["first_date_a", "first_date_b"]].max(axis=1)
    table["overlap_end"] = table[["last_date_a", "last_date_b"]].min(axis=1)
    table["overlap_days"] = (table["overlap_end"] - table["overlap_start"]).dt.days.clip(lower=0)
    return table


def pair_summary(pairs: pd.DataFrame, n_tickers: int) -> pd.Series:
    """Count the pairs and the tickers that have at least one neighbour at distance 1.

    Args:
        pairs: Table returned by one_letter_pairs.
        n_tickers: Number of tickers that were compared.

    Returns:
        Number of pairs (total and by type of edit), tickers involved, and their share of all tickers (%).
    """
    involved = pd.unique(pairs[["ticker_a", "ticker_b"]].to_numpy().ravel()).size
    return pd.Series({
        "pairs": len(pairs),
        "substitutions": int((pairs["edit"] == "substitution").sum()),
        "insertions / deletions": int((pairs["edit"] == "insertion/deletion").sum()),
        "tickers involved": involved,
        "share of tickers (%)": round(100 * involved / n_tickers, 1),
    }, dtype="object")


def _tokens(security_name: str) -> list[str]:
    """Split a security name into lower-case words, without punctuation."""
    return re.sub(r"[^a-z0-9]+", " ", security_name.lower()).split()


def name_words(security_name: str, min_length: int = 4) -> set[str]:
    """Extract the distinctive words of a security name, in lower case.

    Args:
        security_name: Name as written in the metadata, e.g. "Alpha Video Systems, Inc. - Common Stock".
        min_length: Words shorter than this are dropped, because short strings match by chance.

    Returns:
        The set of words, without punctuation, legal terms (Inc, Corp, Common Stock...) or short words.
    """
    return {word for word in _tokens(security_name) if len(word) >= min_length and word not in IGNORED_WORDS}


def is_secondary_security(security_name: str) -> bool:
    """Tell whether a security is a secondary product of a company (preferred shares, notes, units...)."""
    return any(word in SECONDARY_WORDS for word in _tokens(security_name))


def first_word(security_name: str) -> str:
    """Return the first word of a security name, in lower case (the brand of a fund family, e.g. "alpha")."""
    tokens = _tokens(security_name)
    return tokens[0] if tokens else ""


def ticker_name_matches(
    meta: pd.DataFrame,
    coverage: pd.DataFrame,
    max_distance: int = 0,
    min_length: int = 4,
    min_overlap_days: int = 90,
    max_owners: int = 3,
    ordinary_only: bool = True,
    drop_fund_families: bool = True,
) -> pd.DataFrame:
    """Find tickers that look like a word of another company's name.

    This is the mechanism of a mix-up: an investor who wants the company called "Alpha" searches for
    "Alpha" and buys the unrelated security whose ticker is ALPHA. A company is never matched to its own name.

    Args:
        meta: Table returned by load_metadata, with ticker, security_name and is_etf.
        coverage: Table returned by ticker_coverage.
        max_distance: Largest edit distance between the ticker and the word (0 means identical).
        min_length: Minimum length of the ticker and of the words, because short strings match by chance.
        min_overlap_days: Minimum common period of the two securities, in calendar days. The two must be
            traded at the same time for one to be mistaken for the other.
        max_owners: A word that appears in the names of more companies than this is a common word, not a brand.
        ordinary_only: If True, a name is searched only in the main security of a company, not in its
            preferred shares, notes, units or warrants (which belong to a company that is already listed).
        drop_fund_families: If True, drop the matches between two ETFs whose names start with the same word
            (the same fund family).

    Returns:
        One row per match: ticker, ticker_name, word, distance, owner (the company whose name contains the
        word), owner_name and overlap_days.
    """
    listed = meta[meta["ticker"].isin(coverage.index)].drop_duplicates("ticker")
    names = listed.set_index("ticker")["security_name"].fillna("")
    is_etf = listed.set_index("ticker")["is_etf"]
    searched = [ticker for ticker, name in names.items() if not (ordinary_only and is_secondary_security(name))]
    owners = pd.DataFrame(
        [(word, ticker) for ticker in searched for word in name_words(names[ticker], min_length)],
        columns=["word", "owner"],
    )
    owners = owners[owners["word"].map(owners.groupby("word")["owner"].nunique()) <= max_owners]
    tickers = [ticker for ticker in names.index if len(ticker) >= min_length]
    words = sorted(owners["word"].unique())
    distance = process.cdist(
        [ticker.lower() for ticker in tickers], words,
        scorer=Levenshtein.distance, dtype=np.uint8, score_cutoff=max_distance, workers=-1,
    )
    rows, cols = np.nonzero(distance <= max_distance)
    hits = pd.DataFrame({
        "ticker": np.array(tickers, dtype=object)[rows],
        "word": np.array(words, dtype=object)[cols],
        "distance": distance[rows, cols].astype(int),
    })
    hits = hits.merge(owners, on="word")
    hits = hits[hits["ticker"] != hits["owner"]].copy()
    if drop_fund_families:
        both_etf = hits["ticker"].map(is_etf).eq(True) & hits["owner"].map(is_etf).eq(True)
        same_brand = hits["ticker"].map(names).map(first_word) == hits["owner"].map(names).map(first_word)
        hits = hits[~(both_etf & same_brand)].copy()
    start = pd.concat([hits[side].map(coverage["first_date"]) for side in ("ticker", "owner")], axis=1).max(axis=1)
    end = pd.concat([hits[side].map(coverage["last_date"]) for side in ("ticker", "owner")], axis=1).min(axis=1)
    hits["overlap_days"] = (end - start).dt.days.clip(lower=0)
    hits["ticker_name"] = hits["ticker"].map(names)
    hits["owner_name"] = hits["owner"].map(names)
    columns = ["ticker", "ticker_name", "word", "distance", "owner", "owner_name", "overlap_days"]
    keep = hits["overlap_days"] >= min_overlap_days
    return hits.loc[keep, columns].sort_values(["ticker", "owner"]).reset_index(drop=True)


def load_pair(buzz: str, receiver: str, threshold: float = 3.0) -> pd.DataFrame:
    """Load the prices of two tickers with the measures of Parts I and V.

    The steps are those of the common sample: returns (add_returns), dollar volume, quality flags
    (flag_suspicious) and the abnormal volume with its attention spikes (abnormal_volume). No row is removed:
    the filters are applied afterwards by event_study, which blanks the flagged rows.

    Args:
        buzz: Ticker of the well-known company, whose news may attract attention.
        receiver: Ticker that may be bought by mistake.
        threshold: Abnormal volume above which a day is an attention spike.

    Returns:
        One row per ticker and date, with the columns of load_prices plus r_night, r_day, dollar_volume,
        one column per quality flag, abn_volume and spike.
    """
    case = load_prices(tickers=[buzz, receiver])
    case["ticker"] = case["ticker"].astype(str)
    return abnormal_volume(flag_suspicious(add_dollar_volume(add_returns(case))), threshold=threshold)


def twin_events(case: pd.DataFrame, buzz: str, receiver: str) -> pd.DataFrame:
    """Rows of the receiver, flagged as events on the days when the well-known company has an attention spike.

    The result has the format expected by event_study (Part V): day 0 is a spike day of the well-known
    company, and day k is the k-th trading day of the receiver before or after it. A spike day on which the
    receiver did not trade is lost.

    Args:
        case: Table returned by load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        The rows of the receiver, whose spike column now marks the spikes of the well-known company.
    """
    spikes = case.loc[case["ticker"] == buzz].set_index("date")["spike"]
    rows = case[case["ticker"] == receiver].copy()
    rows["spike"] = rows["date"].map(spikes).astype("boolean").fillna(False).astype(bool)
    return rows.reset_index(drop=True)


def case_summary(case: pd.DataFrame, buzz: str, receiver: str) -> pd.DataFrame:
    """Describe the size and the data quality of the two tickers of a case.

    Args:
        case: Table returned by load_pair.
        buzz: Ticker of the well-known company.
        receiver: Ticker that may be bought by mistake.

    Returns:
        One column per ticker (receiver first); rows: first and last day, number of days, median close,
        median volume, median dollar volume, share of days without volume, and share of days whose open
        equals the previous close or the close of the same day.
    """
    columns = {}
    for ticker in (receiver, buzz):
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


def pairs_by_min_length(pairs: pd.DataFrame) -> pd.Series:
    """Count the pairs by the length of the shorter ticker of the pair.

    Args:
        pairs: Table returned by one_letter_pairs.

    Returns:
        Number of pairs for each length (in letters), in increasing order of length.
    """
    lengths = pairs[["ticker_a", "ticker_b"]].apply(lambda column: column.str.len()).min(axis=1)
    return lengths.value_counts().sort_index().rename("pairs").rename_axis("letters in the shorter ticker")


def check_tickers(tickers: list[str], meta: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Check whether some tickers are in the dataset, and over which period.

    Args:
        tickers: Tickers to look for.
        meta: Table returned by load_metadata.
        coverage: Table returned by ticker_coverage.

    Returns:
        One row per requested ticker: in_prices, in_metadata, security_name, first_date, last_date and n_days.
        Dates and counts are missing for a ticker with no price data.
    """
    table = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    names = meta.drop_duplicates("ticker").set_index("ticker")["security_name"]
    table["in_prices"] = table.index.isin(coverage.index)
    table["in_metadata"] = table.index.isin(names.index)
    table["security_name"] = names.reindex(table.index)
    return table.join(coverage)


def _spikes_on(spikes: pd.DataFrame, ticker: str, dates: pd.Series) -> pd.Series:
    """Spike flags of a ticker on the given dates (False where the ticker has no row)."""
    return spikes[ticker].reindex(dates, fill_value=False).astype(bool)


def night_score(rows: pd.DataFrame, spike: pd.Series, flags: list[str] | None = None) -> tuple[float, int]:
    """Excess overnight return of a ticker on the session that follows its events.

    This is the overnight return of day +1 of event_study (Part V), minus its mean over all the sessions of the
    ticker. A positive score means that the ticker gains more than usual overnight after an event.

    Args:
        rows: Rows of one ticker sorted by date, with r_night and the flag columns.
        spike: Boolean Series aligned with rows, True on event days.
        flags: Quality flags; the returns of a flagged row are blanked and a flagged row cannot be an event.

    Returns:
        The score (a return: 0.01 is one percentage point) and the number of events with an observed overnight
        return the next session. The score is missing when there is no such event.
    """
    keep = ~rows[flags].any(axis=1) if flags else pd.Series(True, index=rows.index)
    night = rows["r_night"].where(keep).astype("float64")
    events = spike.to_numpy(dtype=bool) & keep.to_numpy()
    night_after = night.shift(-1)[events].dropna()
    return float(night_after.mean() - night.mean()), len(night_after)


def placebo_pvalue(score: float, placebo_scores: np.ndarray) -> float:
    """One-sided empirical p-value of a score against the scores of placebo draws.

    Args:
        score: Observed score.
        placebo_scores: Scores obtained with random placebos (missing values are ignored).

    Returns:
        (1 + number of placebo scores at least as large as the observed one) / (1 + number of placebo scores).
    """
    scores = np.asarray(placebo_scores, dtype="float64")
    scores = scores[~np.isnan(scores)]
    return float((1 + (scores >= score).sum()) / (1 + len(scores)))


def benjamini_hochberg(p_values: pd.Series, q: float = 0.10) -> pd.DataFrame:
    """Benjamini-Hochberg correction of a family of p-values.

    The p-values are sorted, p(1) <= ... <= p(m); the adjusted value of p(i) is the smallest p(j) * m / j with
    j >= i (capped at 1). Rejecting every hypothesis whose adjusted value is at most q controls the expected
    share of false discoveries among the rejections at q.

    Args:
        p_values: One p-value per test, with a unique index.
        q: Level at which the false discovery rate is controlled.

    Returns:
        A table in the order of p_values: p, q_value (adjusted p-value) and rejected.
    """
    p = p_values.astype("float64")
    ordered = p.sort_values()
    m = len(ordered)
    adjusted = ordered.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    table = pd.DataFrame({"p": ordered.to_numpy(), "q_value": adjusted}, index=ordered.index)
    table["rejected"] = table["q_value"] <= q
    return table.reindex(p.index)


def closest_in_size(size: float, pool_sizes: pd.Series, n: int) -> list[str]:
    """Tickers of a pool whose size is the closest to a given size, on a log scale.

    Args:
        size: Reference size (here a median dollar volume).
        pool_sizes: Size of each ticker of the pool.
        n: Number of tickers kept.

    Returns:
        The n closest tickers.
    """
    distance = (np.log(pool_sizes) - np.log(size)).abs()
    return distance.nsmallest(n).index.tolist()


def pooled_placebo_test(
    matches: pd.DataFrame,
    flags: list[str] = PAIR_FLAGS,
    n_pool: int = 500,
    n_candidates: int = 100,
    n_draws: int = 300,
    min_events: int = 15,
    min_pool_dollar: float = 1e7,
    min_pool_days: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Test the overnight gain of every receiver after the spikes of its well-known company, against placebos.

    For each pair, the score of night_score is computed with the spikes of the well-known company, then with
    the spikes of placebo companies: stocks of the pool that do not look like the receiver, drawn among the
    n_candidates whose size (median dollar volume) is the closest to that of the well-known company. A pair is
    kept when its receiver is traded enough to be measured and has at least min_events events. The p-values are
    corrected with benjamini_hochberg.

    Args:
        matches: Table returned by ticker_name_matches (ticker is the receiver, owner the well-known company).
        flags: Quality flags blanked in the returns of the receivers.
        n_pool: Number of stocks sampled to form the pool of placebo companies.
        n_candidates: Number of pool stocks, closest in size, from which the placebos of a pair are drawn.
        n_draws: Number of placebo draws per pair.
        min_events: Minimum number of events of a pair.
        min_pool_dollar: Minimum median dollar volume of a pool stock.
        min_pool_days: Minimum number of trading days of a pool stock.
        seed: Random seed.

    Returns:
        One row per tested pair (receiver, buzz, n_events, score, placebo_mean, p, q_value, rejected, in %), and a
        summary of the test over all pairs: mean score, mean and standard deviation of the placebo mean score, and the
        p-value of the mean score.
    """
    rng = np.random.default_rng(seed)
    receivers = sorted(set(matches["ticker"]))
    owners = sorted(set(matches["owner"]))

    prices = load_prices(tickers=receivers + [CALENDAR_TICKER])
    prices["ticker"] = prices["ticker"].astype(str)
    prices = flag_suspicious(add_dollar_volume(add_returns(prices)))
    rows = {t: g.sort_values("date").reset_index(drop=True) for t, g in prices.groupby("ticker") if t in receivers}

    volume = load_prices(columns=["ticker", "date", "close", "volume", "asset_type"])
    volume["dollar"] = volume["close"].astype("float64") * volume["volume"].astype("float64")
    by_ticker = volume.groupby("ticker", observed=True)
    sizes = by_ticker["dollar"].median()
    sizes.index = sizes.index.astype(str)
    days = by_ticker["dollar"].size().set_axis(sizes.index)
    is_stock = by_ticker["asset_type"].first().set_axis(sizes.index).eq("stock")
    eligible = sizes[(sizes >= min_pool_dollar) & (days >= min_pool_days) & is_stock]
    pool_sizes = eligible.sample(min(n_pool, len(eligible)), random_state=seed)
    wanted = set(pool_sizes.index) | set(owners)
    spikes = abnormal_volume(volume[volume["ticker"].isin(wanted)].assign(ticker=lambda d: d["ticker"].astype(str)))
    spikes = spikes.pivot(index="date", columns="ticker", values="spike")

    results, placebo = [], {}
    for pair in matches.itertuples():
        r = rows.get(pair.ticker)
        if r is None or pair.owner not in spikes.columns:
            continue
        measured = (r["volume"] == 0).mean() <= 0.10 and (r["close"].astype("float64") * r["volume"]).median() >= 2e4
        score, n_events = night_score(r, _spikes_on(spikes, pair.owner, r["date"]), flags)
        if not measured or n_events < min_events:
            continue
        excluded = set(matches.loc[matches["ticker"] == pair.ticker, "owner"]) | {pair.ticker}
        candidates = closest_in_size(sizes[pair.owner], pool_sizes.drop(excluded, errors="ignore"), n_candidates)
        cache = {}
        scores = []
        for candidate in rng.choice(candidates, size=n_draws):
            if candidate not in cache:
                cache[candidate] = night_score(r, _spikes_on(spikes, candidate, r["date"]), flags)[0]
            scores.append(cache[candidate])
        placebo[(pair.ticker, pair.owner)] = np.array(scores)
        results.append({"receiver": pair.ticker, "buzz": pair.owner, "n_events": n_events, "score": score,
                        "placebo_mean": np.nanmean(scores), "p": placebo_pvalue(score, np.array(scores))})
    table = pd.DataFrame(results)
    table = table.join(benjamini_hochberg(table["p"])[["q_value", "rejected"]])
    null = np.nanmean(np.array(list(placebo.values())), axis=0)
    summary = pd.Series({
        "pairs tested": len(table), "mean score": table["score"].mean(),
        "placebo mean": null.mean(), "placebo sd": null.std(),
        "p-value": placebo_pvalue(table["score"].mean(), null),
    })
    table[["score", "placebo_mean"]] *= 100
    return table, summary
