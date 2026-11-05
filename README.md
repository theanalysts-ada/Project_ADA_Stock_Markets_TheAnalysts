# Who Makes Money While Wall Street Sleeps?

## Abstract

Buy an S&P 500 fund at every close and sell it at the next open: from 1998 to 2020, one dollar becomes \$4.74. Hold it only during the trading day instead, and it shrinks to 80 cents. Yet the session is when almost all trading and news reactions happen. So who makes money while Wall Street sleeps? We treat this anomaly as an investigation. After establishing the crime, we question three suspects: **risk**, whether holding stocks through longer nights is rewarded; **news**, whether firms publish after the close; and **the crowd**, whether attention brings buyers at the open who sell back during the day. Finally, tickers that investors mistake for famous ones offer a natural experiment: attention without news.

## Research Questions

1. **Crime:** do overnight returns exceed intraday returns across US stocks and ETFs, and does the gap survive data cleaning and statistical tests?
2. **Risk:** do longer closures (weekends, holidays) earn higher overnight returns?
3. **News:** are 8-K announcements concentrated after the close, and do nights without news still earn more?
4. **Crowd:** after a spike of attention, do we see an overnight gain followed by an intraday loss?
5. **Proof:** after a famous company's attention spikes, does a stock mistaken for it gain more than usual overnight, compared with placebo stocks?

## Proposed additional datasets

### SEC EDGAR 8-K filings (Part IV)

Listed US firms must file an **8-K** within four business days of any important event; **Item 2.02** marks earnings announcements. `scripts/collect_news.py` links stocks to their SEC identifier (CIK), also by name for firms renamed or delisted since 2020: **4,197 stocks** (75.9 % of rows) are matched. Filing times exist from 2003, and the **557,104** 8-Ks since then are placed in a night or a session. Foreign firms file 6-Ks and are excluded.

### Macroeconomic calendar (Part IV)

On SPY, firm news averages out, so we add the **jobs report** and the **CPI** (8:30, during the night; FRED API) and **FOMC** statements (14:00, during the session; Federal Reserve website): **267, 267 and 177** releases from 1998 to 2020. PPI, retail sales and GDP follow in P3.

### NYSE trading calendar (Part III)

`pandas_market_calendars` gives the historical schedule (holidays, early closes, daylight saving), hence the length of each night: **17.5 hours** on weekdays, about **66** over a weekend.

### Google Trends (Part V, for P3)

Tested with the unofficial `pytrends` on ten large companies: "*X* stock" tracks interest in the stock better than the company name, but daily data come only in short windows. We go ahead in a limited way, for about 30 large companies, with abnormal volume as fallback.

## Methods

### Data handling (Part I)

Kaggle daily prices of **8,049** Nasdaq-traded stocks and ETFs (1962 to 2020, **28 million rows**), stored as typed Parquet (0.70 GB instead of 2.75 GB, loaded in about 3 seconds). Prices are adjusted for dividends, and each day is split into an **overnight** (close to open) and an **intraday** (open to close) return. From **1998**, the first year without missing opens, a **liquidity filter** uses previous-day values only (close of at least \$5, dollar volume above the bottom 20 % of its year). Foreign, leveraged and inverse ETFs are excluded.

### The crime (Part II)

We follow \$1 invested only overnight or only intraday and split the log return between the two halves, for SPY, portfolios, tickers, years and quality filters. The gap is tested with a night-indicator regression clustered by date and a 21-day block bootstrap.

### Risk (Part III)

We test two predictions: a return **per hour** (a weekend earning 3.7 times a weekday night) or **per closure**. We regress each overnight return on the closure length: per hour predicts a slope proportional to the mean, per closure a zero slope.

### News (Part IV)

Press releases often come before the 8-K, so we test whether the gap survives on **quiet stock-days**, with no filing within two sessions. A stock-level regression with **date fixed effects** measures an 8-K's effect on the overnight return and its size; macro release days use **HAC standard errors**.

### Crowd (Part V)

A spike is a day with volume above three times its previous 20-day mean. An **event study** averages overnight and intraday returns over the five days around each spike, looking for the **hangover** (an overnight gain followed by an intraday loss), repeated at two and five times to see if it grows with attention.

### Proof (Part VI)

A stock mistaken for a famous company is assumed to have no news of its own, so an overnight gain after the famous company's spikes points to the crowd. Matching tickers to words of company names gives 94 candidates; with `ZOOM` absent, we study `FORD` (Forward Industries) after the spikes of `F` (Ford Motor), against a **placebo** of the 100 stocks closest in size to `F`.

### Limitations

- *Survivorship bias:* only securities listed in 2020 are included.
- *Daily data:* opens are noisy for rarely traded names, and effects lasting hours are invisible.
- *Fragile significance:* the SPY gap is method-dependent (clustered p = 0.073; the bootstrap excludes zero).
- *Associations, not causes:* firms choose when to file and spikes often come with news; the proof rests on one case (34 events).

## Proposed timeline

- 9–20 Nov: P3 analyses of each part start.
- 20 Nov: P2 feedback, used to adjust the plan.
- 21 Nov–4 Dec: P3 analyses completed; results frozen on 4 Dec.
- 5–10 Dec: data story drafted (Jekyll), built as the investigation: crime, suspects, proof, verdict.
- 11 Dec: office hour with the TA.
- 12–17 Dec: final figures, notebook run end to end, code clean-up, README with contributions.
- 18 Dec: P3 submission.

## Organization within the team

Each member keeps their part and carries out its P3 next steps, detailed at the end of each notebook part:

- Sara: data pipeline and the crime (Parts I–II).
- Omar: risk (Part III).
- Ismail: news (Part IV).
- Rania: crowd and Google Trends (Part V).
- Majd: proof (Part VI) and code review.
- Everyone: the data story section of their part.

## Questions for TAs

1. Foreign firms earn a gap of 20.9 % a year, against 3.5 % for US firms under the same ticker (Part IV), and their US night overlaps their home session. Should we drop them from every part?
2. For some suspects our data cannot give a clear verdict: risk is not measured directly, press articles are missing, and the proof has one case. Should we look for other evidence, or deepen the current analyses as planned?


## Quickstart

1. Python 3.10 or newer, then `pip install -r requirements.txt`.
2. Run `results.ipynb` from the top. The first cells download the Kaggle files (about 520 MB) and build the price table.
3. The SEC filings (`data/edgar/`) and the macro release dates (`data/macro/`) used in Part IV are in the repository. To download the filings again (about 20 minutes), the SEC asks for a name and an email:

```bash
SEC_USER_AGENT="Your Name your.email@epfl.ch" python -m scripts.collect_news
```