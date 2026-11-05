# Who Makes Money While Wall Street Sleeps?

## Abstract

Buy an S&P 500 fund at every close and sell it at the next open: from 1998 to 2020, one dollar becomes $4.74. Buy at every open and sell at the close instead, and you have 80 cents. So the profit is made at night, while Wall Street sleeps. Who is behind it? Using US stock prices, we test three suspects. Risk: no sign that longer nights pay more. News: without news, the night-day gap only falls from 7.6 % to 6.2 % a year. The crowd, people who rush into stocks that get attention, still looks plausible, but attention often comes with news, so it is hard to see if the crowd alone raises night returns. We use an experiment: people who buy the wrong stock because its ticker looks like a famous one give attention without news. A first test on one case shows no clear effect.

## Research Questions

1. **Crime:** do overnight returns exceed intraday returns across US stocks and ETFs, and does the gap survive data cleaning and statistical tests?
2. **Risk:** do longer closures (weekends, holidays) earn higher overnight returns?
3. **News:** are 8-K announcements concentrated after the close, and do nights without news still earn more?
4. **Crowd:** after a spike of attention, do we see an overnight gain followed by an intraday loss?
5. **Proof:** when a famous company has a spike of attention, does a stock that investors mistake for it gain more than usual overnight, compared with placebo stocks?

## Proposed additional datasets

### SEC EDGAR 8-K filings (Part IV)

Listed US firms must file an **8-K** within four business days of any important event; **Item 2.02** marks earnings announcements. `scripts/collect_news.py` links stocks to their SEC identifier (CIK): **4,197 stocks** (75.9 % of rows) are matched. Filing times exist from 2003, and the **557,104** 8-Ks since then are placed in a night or a session. Foreign firms file 6-Ks and are excluded.

### Macroeconomic calendar (Part IV)

On SPY, firm news averages out, so we add the **jobs report** and the **CPI** (FRED API) and **FOMC** statements (Federal Reserve website): **267, 267 and 177** releases from 1998 to 2020. PPI, retail sales and GDP follow in P3.

### NYSE trading calendar (Part III)

`pandas_market_calendars` gives the historical schedule (holidays, early closes, daylight saving), hence the length of each night: **17.5 hours** on weekdays, about **66** over a weekend.

### Google Trends (Part V, for P3)

Tested with the unofficial `pytrends` on ten large companies (January–March 2020): "*X* stock" tracks interest in the stock better than the company name, but daily data come only in short windows. We go ahead in a limited way, for about 30 large companies, with abnormal volume as fallback.

## Methods

### Data handling (Part I)

Kaggle daily prices of **8,049** Nasdaq-traded stocks and ETFs (1962 to 2020, **28 million rows**), stored as typed Parquet. Prices are adjusted for dividends, and each day is split into an **overnight** (close to open) and an **intraday** (open to close) return. From **1998**, the first year without missing opens, a **liquidity filter** uses previous-day values only (close of at least \$5, dollar volume above the bottom 20 % of its year). Foreign, leveraged and inverse ETFs are excluded.

### The crime (Part II)

We follow \$1 invested only overnight or only intraday and split the log return between the two halves, for SPY, portfolios, tickers, years and quality filters. The gap is tested with a night-indicator regression clustered by date and a 21-day block bootstrap.

### Risk (Part III)

We test two predictions: a return **per hour** (a weekend earning 3.7 times a weekday night) or **per closure**. We regress each overnight return on the closure length, with block-bootstrap intervals: per hour predicts a slope proportional to the mean, per closure a zero slope.

### News (Part IV)

Press releases often come before the 8-K, so we test whether the gap survives on **quiet stock-days**, with no filing within two sessions. A stock-level regression with **date fixed effects** measures an 8-K's effect on the overnight return and its size; macro release days use **HAC standard errors**.

### Crowd (Part V)

A spike is a day with volume above three times its previous 20-day mean. An **event study** averages overnight and intraday returns over the five days around each spike, looking for the **hangover** (an overnight gain followed by an intraday loss), repeated at two and five times to check that the effect grows with attention.

### Proof (Part VI)

A stock mistaken for a famous company has no news of its own, so an overnight gain after the famous company's spikes points to the crowd. Matching tickers to words of company names gives 94 candidates; with `ZOOM` absent, we study `FORD` (Forward Industries) after the spikes of `F` (Ford Motor), against a **placebo** of the 100 stocks closest in size to `F`.

### Limitations

- *Survivorship bias:* only securities listed in 2020 are included.
- *Daily data:* opens are noisy for rarely traded names, and effects lasting hours are invisible.
- *No market capitalisation:* weights use dollar volume.
- *Fragile significance:* the SPY gap is method-dependent (clustered p = 0.073; the bootstrap excludes zero).
- *Associations, not causes:* firms choose when to file and volume spikes often come with news, so Parts IV and V show links, not causes; the proof, meant to isolate a cause, rests on a single case (34 events).


## Proposed timeline

- **Until the P2 feedback:** we wait for the review of Milestone P2.
- **After the feedback:** each member adapts their part to the TA's feedback, then works on it (see the next section) and fills in the final notebook.
- **11 December:** office hour with the TA, to check our progress and ask our questions.
- **After 11 December:** last organisation and code clean-up.
- **By Friday 18 December:** final submission of Milestone P3.


## Questions for TAs
 
1. Foreign firms earn a night–day gap of 20.9 % a year, against 3.5 % for US firms under the same ticker (Part IV), and their US night overlaps their home session. Should we drop them from every chapter, or report results with and without them?
2. The SPY gap is significant with the block bootstrap but not with date-clustered errors (p = 0.073). Which inference should be the main one?
3. Our only confused-ticker case rests on 34 events. Is combining several cases a sound plan for P3?


## Quickstart

1. Python 3.10 or newer, then `pip install -r requirements.txt`.
2. Part IV needs two environment variables, not a `.env` file: `FRED_API_KEY` (free FRED key) and `SEC_USER_AGENT` (your name and email).
3. Run `results.ipynb` from the top. The first cells download the Kaggle files (about 520 MB) and build the price table.
4. Before the news section, from the repository root:

```bash
SEC_USER_AGENT="Your Name your.email@epfl.ch" python -m scripts.collect_news
```

About 20 minutes. The files land in `data/edgar/`. Set `FRED_API_KEY` before the macro cells.