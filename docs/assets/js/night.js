/* Scroll effects: the dusk-to-dawn sky, the investigator's clock, the mini suspects board
   and the red threads of the verdict board. Without JavaScript the page keeps its static gradient. */
(function (NH) {
  const sections = Array.from(document.querySelectorAll('[data-clock]'));
  const clock = document.querySelector('.clock-face .time');
  const mini = document.getElementById('mini-board');
  const opening = document.getElementById('bet');
  if (!sections.length) return;

  /* Clock times in minutes; the night goes on past midnight (00:30 is 24:30). */
  let dayOffset = 0;
  let previous = -1;
  const times = sections.map((s) => {
    const [h, m] = s.dataset.clock.split(':').map(Number);
    let t = h * 60 + m + dayOffset;
    if (t < previous) { dayOffset += 1440; t += 1440; }
    previous = t;
    return t;
  });
  const colors = sections.map((s) => s.dataset.sky.match(/\w\w/g).map((h) => parseInt(h, 16)));

  const mix = (a, b, t) => 'rgb(' + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',') + ')';
  const pad = (n) => String(n).padStart(2, '0');

  /* Where the reader is: index of the current section and progress (0-1) towards the next one. */
  function position() {
    const line = window.innerHeight * 0.4;
    const tops = sections.map((s) => s.getBoundingClientRect().top);
    let i = 0;
    while (i + 1 < tops.length && tops[i + 1] <= line) i += 1;
    if (i + 1 >= tops.length || tops[i] > line) return { i, t: 0 };
    return { i, t: Math.min(1, Math.max(0, (line - tops[i]) / (tops[i + 1] - tops[i]))) };
  }

  function update() {
    const { i, t } = position();
    const j = Math.min(i + 1, sections.length - 1);
    document.body.style.setProperty('--sky', mix(colors[i], colors[j], t));
    document.body.style.setProperty('--sky-2', mix(colors[i], colors[j], Math.min(1, t + 0.5)));
    if (clock) {
      const minutes = Math.round((times[i] + (times[j] - times[i]) * t) / 5) * 5 % 1440;
      clock.textContent = pad(Math.floor(minutes / 60)) + ':' + pad(minutes % 60);
    }
    if (mini && opening) {
      mini.classList.toggle('is-visible', opening.getBoundingClientRect().bottom < 0);
      const current = sections[i].id;
      mini.querySelectorAll('[data-suspect]').forEach((a) => {
        if (a.dataset.suspect === current) a.setAttribute('aria-current', 'location');
        else a.removeAttribute('aria-current');
      });
    }
  }

  let queued = false;
  const onScroll = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; update(); });
  };
  document.body.classList.add('js-sky');
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  update();

  /* Red threads between pinned cards, drawn when the board comes into view. */
  const board = document.getElementById('verdict-board');
  if (!board) return;
  const svg = board.querySelector('svg.threads');
  const pairs = JSON.parse(board.dataset.threads || '[]');
  const pin = (el, box) => {
    const r = el.getBoundingClientRect();
    return [r.left - box.left + r.width / 2, r.top - box.top + 6];
  };
  function drawThreads() {
    const box = board.getBoundingClientRect();
    svg.setAttribute('viewBox', '0 0 ' + box.width + ' ' + box.height);
    svg.innerHTML = '';
    pairs.forEach(([from, to]) => {
      const a = board.querySelector('[data-card="' + from + '"]');
      const b = board.querySelector('[data-card="' + to + '"]');
      if (!a || !b) return;
      const [x1, y1] = pin(a, box);
      const [x2, y2] = pin(b, box);
      const sag = Math.max(y1, y2) + 30 + Math.abs(x2 - x1) * 0.08;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', 'M' + x1 + ',' + y1 + ' Q' + (x1 + x2) / 2 + ',' + sag + ' ' + x2 + ',' + y2);
      path.setAttribute('pathLength', '1');
      path.setAttribute('class', 'thread');
      svg.appendChild(path);
    });
  }
  drawThreads();
  new ResizeObserver(drawThreads).observe(board);
  NH.whenVisible(board, () => board.classList.add('is-drawn'));
})(window.NightHeist);
