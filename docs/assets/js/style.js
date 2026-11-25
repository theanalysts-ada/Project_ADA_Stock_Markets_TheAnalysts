/* Shared chart style and helpers for every chapter of the story. */
window.NightHeist = window.NightHeist || {};

(function (NH) {
  NH.colors = {
    night: '#8FB3FF',
    day: '#F2A93B',
    neutral: '#C9C4BA',
    stamp: '#E5675F',
    ink: '#E8E6E1',
    muted: '#B8B3A8',
    grid: 'rgba(232, 230, 225, 0.10)',
  };

  NH.config = { responsive: true, displayModeBar: false };

  NH.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const axis = {
    gridcolor: NH.colors.grid,
    zerolinecolor: 'rgba(232, 230, 225, 0.35)',
    linecolor: 'rgba(232, 230, 225, 0.25)',
    tickfont: { color: NH.colors.muted },
    automargin: true,
  };

  /* Plotly layout shared by all charts; `extra` overrides it (axes are merged, not replaced). */
  NH.layout = function (extra = {}) {
    return {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', size: 14, color: NH.colors.ink },
      margin: { l: 16, r: 16, t: 16, b: 16 },
      hoverlabel: { bgcolor: '#0E1628', bordercolor: NH.colors.neutral, font: { family: 'IBM Plex Mono, monospace', size: 13, color: NH.colors.ink } },
      legend: { orientation: 'h', x: 0, y: 1.12 },
      ...extra,
      xaxis: { ...axis, ...(extra.xaxis || {}) },
      yaxis: { ...axis, ...(extra.yaxis || {}) },
    };
  };

  /* URL of a data file, relative to the site (works on any GitHub Pages address). */
  NH.dataUrl = function (path) {
    return (document.body.dataset.base || '') + '/assets/data/' + path;
  };

  const cache = {};
  NH.getJSON = function (path) {
    cache[path] = cache[path] || fetch(NH.dataUrl(path)).then((r) => {
      if (!r.ok) throw new Error(path + ': ' + r.status);
      return r.json();
    });
    return cache[path];
  };

  /* Run `draw` once, when the element comes close to the screen (keeps the first load fast). */
  NH.whenVisible = function (el, draw) {
    if (!el) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        observer.disconnect();
        draw(el);
      }
    }, { rootMargin: '300px' });
    observer.observe(el);
  };

  /* Mark a group of toggle buttons: the clicked one is pressed, the others are not. */
  NH.press = function (buttons, active) {
    buttons.forEach((b) => b.setAttribute('aria-pressed', String(b === active)));
  };

  NH.fail = function (el, error) {
    el.innerHTML = '<p class="chart-fallback">This chart could not be loaded.</p>';
    console.error(error);
  };
})(window.NightHeist);
