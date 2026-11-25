# Writing your chapter of the data story

The site is one long page, built by GitHub Pages from this `docs/` folder with Jekyll.
Each chapter is **one file** in `_includes/sections/`, so nobody edits someone else's file:

| Chapter | File | Owner |
|---|---|---|
| The bet, the crime scene, has it stopped? | `bet.html`, `crime.html`, `stopped.html` | Sara |
| Interrogation n°1, risk | `risk.html` | Omar |
| Interrogation n°2, news | `news.html` | Ismail |
| Interrogation n°3, the crowd (and the proof) | `crowd.html` | Rania, Majd |
| Suspects board, verdict | `suspects.html`, `verdict.html` | together |

`index.html` only lists the sections in order. The look is in `assets/css/story.css`; do not
add styles inside your file.

## The rhythm of an interrogation

Keep the order already in your file, in 300 to 500 words with 2 or 3 charts:

1. **Motive**: the suspect's hypothesis in one sentence.
2. **Exhibit**: the key chart, with a title that states the finding.
3. **Alibi check**: what the data says, in plain words, with the numbers of the notebook.
4. **Stamp**: the verdict on the suspect, in a few words.
5. **Forensics lab**: the method in a few sentences and a link to your notebook part.

Text is in English, short and simple. Every number must be the one printed in the notebook.

## The building blocks

Use the components of `_includes/`. Examples:

```liquid
{% include exhibit.html id="chart-risk-length" label="Exhibit E · 01:00 · Longer nights"
   title="Weekend nights do not earn three times more"
   aria="Bar chart: mean overnight return by length of the closure."
   caption="Mean overnight return of SPY by closure type, 1998–2020." %}

{% include stamp.html state="verdict" text="Alibi holds" %}

{% include margin-note.html text="A weekend night lasts 65.5 hours, a weekday night 17.5." %}

{% include forensics-box.html text="We regress each overnight return on the length of the closure." section="Part III" %}

{% include placeholder.html text="[Second chart, coming soon]" %}
```

- `exhibit.html` creates an empty chart box with the id you give; your script draws in it.
- `stamp.html` takes `state` = `pending`, `verdict` or `open`.
- Write normal paragraphs as `<p class="prose">…</p>`.

## Charts

Charts are drawn in the browser with Plotly (already loaded once for the whole page, version
"basic": lines, bars, scatter and error bars) from small JSON files.

1. **Data**: export what the chart needs from the notebook, with a script in `scripts/` that
   calls the functions of `src/` (see `scripts/export_story_data.py`). Write it to
   `docs/assets/data/<your-part>/`. Keep files small: weekly instead of daily when the picture
   does not change, rounded numbers, no columns the chart does not use.
2. **Script**: create `docs/assets/js/<your-part>.js` and add it at the end of
   `_layouts/story.html`, after `style.js`. Use the shared helpers:

```js
(function (NH) {
  function drawLength(el) {
    NH.getJSON('risk/closure_length.json').then((d) => {
      Plotly.newPlot(el, [{ type: 'bar', x: d.labels, y: d.mean, marker: { color: NH.colors.night } }],
        NH.layout({ yaxis: { title: { text: 'Mean overnight return (bp)' } } }), NH.config);
    }).catch((e) => NH.fail(el, e));
  }
  NH.whenVisible(document.getElementById('chart-risk-length'), drawLength);
})(window.NightHeist);
```

- Colours: `NH.colors.night` for every night series, `NH.colors.day` for every day series,
  `NH.colors.neutral` for a third one. Never carry information by colour alone (add a label,
  a line style or a marker shape).
- `NH.layout()` gives the shared style; `NH.whenVisible()` draws the chart only when the reader
  gets close to it, so the page loads fast.

## Numbers in the text

Numbers quoted in the crime scene are not typed by hand: `scripts/export_story_data.py` writes
them to `docs/_data/crime.json`, and the page reads them as `{{ site.data.crime.spy.night }}`.
Do the same for your chapter if you can (`docs/_data/<your-part>.json`).

## Workflow

1. Create a branch from `main`: `git switch -c site/<your-part>`.
2. Edit only your section file, your JSON and your script.
3. Push the branch and open a pull request. Say which section you changed.
4. Once merged, GitHub Pages rebuilds the site in one or two minutes.

To preview locally (optional): install a recent Ruby, then in `docs/` run
`bundle install` once and `bundle exec jekyll serve`, and open the address it prints.
