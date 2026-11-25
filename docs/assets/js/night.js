/* Night effects: dawn placed at the verdict, the investigator's clock, the mini suspects board
   and the red threads of the verdict board. Without JavaScript the page keeps its default gradient. */
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
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  update();

  /* Same colours as the default gradient, but the dawn starts where the verdict starts,
     whatever the length of the chapters above it. */
  const page = document.querySelector('.page');
  const stopped = document.getElementById('stopped');
  const verdict = document.getElementById('verdict');
  function placeDawn() {
    if (!page || !stopped || !verdict) return;
    const top = (el) => el.getBoundingClientRect().top - page.getBoundingClientRect().top;
    const s = top(stopped);
    const v = top(verdict);
    const stops = { '--dusk-end': s - 300, '--stopped': s, '--twilight': (s + v) / 2, '--sunrise': v - 120, '--dawn': v + 60 };
    Object.entries(stops).forEach(([name, px]) => page.style.setProperty(name, Math.max(0, px) + 'px'));
  }
  placeDawn();
  new ResizeObserver(placeDawn).observe(document.body);

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
