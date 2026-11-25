/* Charts and interactions of the crime scene (notebook Part II) and of "Has the crime stopped?". */
(function (NH) {
  const C = NH.colors;

  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const line = (name, x, y, color, extra = {}) => ({
    type: 'scatter', mode: 'lines', name, x, y,
    line: { color, width: 2.5, ...(extra.line || {}) },
    hovertemplate: extra.hovertemplate || '%{y:$.2f}<extra>' + name + '</extra>',
  });

  /* Exhibit A: value of $1 in SPY, night only, day only, buy and hold. */
  function drawSpy(el) {
    NH.getJSON('crime/spy_value.json').then((d) => {
      const crashes = [['2000-03-24', 'Dot-com crash'], ['2008-09-15', '2008 crisis'], ['2020-02-21', 'COVID crash']];
      const annotations = crashes.map(([date, text]) => {
        const i = d.dates.findIndex((x) => x >= date);
        return {
          x: d.dates[i], y: Math.log10(d.buy_hold[i]), text, showarrow: true, ax: 0, ay: -40,
          arrowcolor: C.muted, font: { size: 12, color: C.muted },
        };
      });
      Plotly.newPlot(el, [
        line('Night only', d.dates, d.night, C.night),
        line('Day only', d.dates, d.day, C.day),
        line('Buy and hold', d.dates, d.buy_hold, C.neutral, { line: { dash: 'dot', width: 2 } }),
      ], NH.layout({
        hovermode: 'x unified',
        annotations,
        xaxis: { type: 'date' },
        yaxis: { type: 'log', tickprefix: '$', title: { text: 'Value of $1 (log scale)' } },
      }), NH.config);
    }).catch((e) => NH.fail(el, e));
  }

  /* Exhibit B: histogram of the gap per ticker, stocks or ETFs. */
  function drawGaps(el) {
    NH.getJSON('crime/ticker_gaps.json').then((d) => {
      const width = d.edges[1] - d.edges[0];
      const centers = d.edges.slice(0, -1).map((e) => e + width / 2);
      const buttons = Array.from(document.querySelectorAll('[data-asset]'));
      const sentence = document.getElementById('gaps-text');
      const draw = (asset) => {
        const { counts, total, outside } = d[asset];
        Plotly.react(el, [{
          type: 'bar', x: centers, y: counts.map((c) => (100 * c) / total), width,
          marker: { color: centers.map((c) => (c > 0 ? C.night : C.day)) },
          hovertemplate: 'Gap %{x:.1f}% ± 2.5: %{y:.1f}% of tickers<extra></extra>',
        }], NH.layout({
          showlegend: false,
          bargap: 0.08,
          xaxis: { range: [-100, 200], ticksuffix: '%', title: { text: 'Annualised night − day gap (' + outside + ' tickers beyond −100% or 200% not shown)' } },
          yaxis: { title: { text: 'Share of tickers (%)' } },
          shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: C.ink, width: 1.5, dash: 'dash' } }],
          annotations: [{ x: 0, yref: 'paper', y: 1, xanchor: 'left', showarrow: false, text: ' night = day', font: { size: 12, color: C.muted } }],
        }), NH.config);
        sentence.textContent = sentence.dataset[asset];
      };
      buttons.forEach((b) => b.addEventListener('click', () => { NH.press(buttons, b); draw(b.dataset.asset); }));
      draw('stocks');
    }).catch((e) => NH.fail(el, e));
  }

  /* Exhibit B: night and day record of one ticker. The data file is loaded on first use. */
  function setupLookup() {
    const input = document.getElementById('ticker-input');
    const out = document.getElementById('ticker-result');
    if (!input || !out) return;
    const intro = out.innerHTML;
    const pct = (v) => (v > 0 ? '+' : '') + v.toFixed(1) + '% a year';
    const show = (tickers) => {
      const key = input.value.trim().toUpperCase();
      if (!key) { out.innerHTML = intro; return; }
      const t = tickers[key];
      if (!t) {
        out.innerHTML = '<p><strong>' + escapeHtml(key) + '</strong>: not in our sample (we keep liquid US stocks and ETFs with at least one year of data, 1998–2020).</p>';
        return;
      }
      const verdict = t.gap > 0
        ? 'At night, ' + escapeHtml(key) + ' earned more than during the day.'
        : 'During the day, ' + escapeHtml(key) + ' earned more than at night: one of the exceptions.';
      out.innerHTML =
        '<p><strong>' + escapeHtml(key) + '</strong>' + (t.name ? ' · ' + escapeHtml(t.name) : '') + ' (' + (t.type === 'etf' ? 'ETF' : 'stock') + ')</p>' +
        '<dl><dt>Night</dt><dd class="night">' + pct(t.night) + '</dd>' +
        '<dt>Day</dt><dd class="day">' + pct(t.day) + '</dd>' +
        '<dt>Gap</dt><dd>' + pct(t.gap) + '</dd>' +
        '<dt>Days</dt><dd>' + t.n_days.toLocaleString('en') + '</dd></dl>' +
        '<p>' + verdict + '</p>';
    };
    input.addEventListener('input', () => {
      NH.getJSON('crime/tickers.json').then(show).catch(() => {
        out.innerHTML = '<p>The ticker list could not be loaded.</p>';
      });
    });
  }

  /* Exhibit C: the COVID crash, drawn day by day when it comes into view. */
  function drawCovid(el) {
    NH.getJSON('crime/covid_crash.json').then((d) => {
      const n = d.dates.length;
      const all = d.night.concat(d.day);
      const layout = NH.layout({
        hovermode: 'x unified',
        xaxis: { type: 'date', range: [d.dates[0], d.dates[n - 1]] },
        yaxis: { tickprefix: '$', range: [Math.min(...all) * 0.97, Math.max(...all) * 1.03], title: { text: 'Value of $1' } },
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.muted, width: 1, dash: 'dot' } }],
      });
      const traces = (k) => [
        line('Night only', d.dates.slice(0, k), d.night.slice(0, k), C.night, { line: { width: 3 } }),
        line('Day only', d.dates.slice(0, k), d.day.slice(0, k), C.day, { line: { width: 3 } }),
      ];
      let timer = null;
      const play = () => {
        clearInterval(timer);
        if (NH.reducedMotion) { Plotly.react(el, traces(n), layout, NH.config); return; }
        let i = 1;
        Plotly.react(el, traces(i), layout, NH.config);
        timer = setInterval(() => {
          if (i >= n) { clearInterval(timer); return; }
          Plotly.extendTraces(el, { x: [[d.dates[i]], [d.dates[i]]], y: [[d.night[i]], [d.day[i]]] }, [0, 1]);
          i += 1;
        }, 80);
      };
      document.getElementById('covid-replay').addEventListener('click', play);
      play();
    }).catch((e) => NH.fail(el, e));
  }

  /* Exhibit D: forest plot of the gap and its interval, with or without dividends. */
  function drawIntervals(el) {
    NH.getJSON('crime/gap_intervals.json').then((rows) => {
      const button = document.getElementById('dividends-toggle');
      const text = { with: document.getElementById('intervals-with'), without: document.getElementById('intervals-without') };
      const order = rows.map((r) => r.name).reverse();
      const draw = (key) => {
        const traces = rows.map((r) => {
          const v = r[key];
          const clear = v.lo > 0;
          const color = clear ? C.night : C.stamp;
          return {
            type: 'scatter', mode: 'markers+text', x: [v.est], y: [r.name], showlegend: false,
            error_x: { type: 'data', symmetric: false, array: [v.hi - v.est], arrayminus: [v.est - v.lo], color, thickness: 2.5, width: 8 },
            marker: { color, size: 12, symbol: clear ? 'circle' : 'diamond' },
            text: [v.est + '%  [' + v.lo + '; ' + v.hi + ']' + (clear ? '' : '  includes 0')],
            textposition: 'top center',
            textfont: { family: 'IBM Plex Mono, monospace', size: 12, color },
            hovertemplate: r.name + ': ' + v.est + '% a year [' + v.lo + '; ' + v.hi + ']<extra></extra>',
          };
        });
        Plotly.react(el, traces, NH.layout({
          transition: { duration: NH.reducedMotion ? 0 : 500, easing: 'cubic-in-out' },
          xaxis: { range: [-5, 32], ticksuffix: '%', title: { text: 'Night − day gap (% a year)' } },
          yaxis: { type: 'category', categoryorder: 'array', categoryarray: order },
          shapes: [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: C.ink, width: 1.5, dash: 'dash' } }],
        }), NH.config);
        text.with.hidden = key !== 'with';
        text.without.hidden = key !== 'without';
      };
      button.addEventListener('click', () => {
        const removed = button.getAttribute('aria-pressed') !== 'true';
        button.setAttribute('aria-pressed', String(removed));
        button.textContent = removed ? 'Put the dividends back' : 'Remove the dividends';
        draw(removed ? 'without' : 'with');
      });
      draw('with');
    }).catch((e) => NH.fail(el, e));
  }

  /* Section 7: three-year rolling means of night and day returns. */
  function drawRolling(el) {
    NH.getJSON('crime/rolling_3y.json').then((d) => {
      const buttons = Array.from(document.querySelectorAll('[data-book]'));
      const draw = (key) => {
        const s = d[key];
        const hover = '%{y:.1f}% a year<extra>';
        Plotly.react(el, [
          line('Night', s.dates, s.night, C.night, { hovertemplate: hover + 'Night</extra>' }),
          line('Day', s.dates, s.day, C.day, { hovertemplate: hover + 'Day</extra>' }),
        ], NH.layout({
          hovermode: 'x unified',
          xaxis: { type: 'date' },
          yaxis: { ticksuffix: '%', zeroline: true, title: { text: 'Mean return, previous 3 years (% a year)' } },
        }), NH.config);
      };
      buttons.forEach((b) => b.addEventListener('click', () => { NH.press(buttons, b); draw(b.dataset.book); }));
      draw('stocks_dv');
    }).catch((e) => NH.fail(el, e));
  }

  NH.whenVisible(document.getElementById('chart-spy'), drawSpy);
  NH.whenVisible(document.getElementById('chart-gaps'), drawGaps);
  NH.whenVisible(document.getElementById('chart-covid'), drawCovid);
  NH.whenVisible(document.getElementById('chart-intervals'), drawIntervals);
  NH.whenVisible(document.getElementById('chart-rolling'), drawRolling);
  setupLookup();
})(window.NightHeist);
