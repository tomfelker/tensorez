// SER player view: parse header, stream frames from disk via File.slice(),
// render through WebGL2 (see gl.js). Works identically in browser and Electron.

import { parseSerHeader, readSerFrame } from './ser-parse.js';
import { createSerRenderer } from './gl.js';

export function initSer(root) {
  root.innerHTML = `
    <h1>SER Player</h1>
    <p class="view-sub">Inspect capture videos. Frames stream from disk one at a
      time; display is linear → sRGB, no stretch.</p>

    <div class="toolbar">
      <input id="ser-file" type="file" accept=".ser" aria-label="ser file">
      <div class="grow"></div>
      <label class="opt-toggle" title="SER's endian flag is unreliable; toggle if colors/levels look wrong">
        <input id="ser-be" type="checkbox"> big-endian samples
      </label>
    </div>

    <div class="ser-layout">
      <div>
        <div class="ser-stage" id="ser-stage">
          <canvas id="ser-canvas"></canvas>
          <div class="ser-empty" id="ser-empty">Choose a .ser file to play</div>
        </div>
        <div class="ser-controls">
          <button id="ser-play" class="primary" disabled>▶ Play</button>
          <input id="ser-scrub" type="range" min="0" max="0" value="0" disabled
                 aria-label="frame scrubber">
          <span class="ser-frame-label" id="ser-frame-label">– / –</span>
          <span class="ser-fps" id="ser-fps"></span>
          <div class="sep"></div>
          <button id="ser-zoom-out" class="small" disabled>−</button>
          <span class="ser-zoom-label" id="ser-zoom-label">×1</span>
          <button id="ser-zoom-in" class="small" disabled>+</button>
          <button id="ser-reset" class="small" disabled>reset view</button>
        </div>
      </div>
      <div class="ser-info" id="ser-info">
        <h3>Header</h3>
        <div class="dim">No file loaded.</div>
      </div>
    </div>`;

  const el = (id) => root.querySelector('#' + id);
  const canvas = el('ser-canvas');
  const stage = el('ser-stage');
  const emptyMsg = el('ser-empty');
  const playBtn = el('ser-play');
  const scrub = el('ser-scrub');
  const frameLabel = el('ser-frame-label');
  const fpsEl = el('ser-fps');
  const zoomLabel = el('ser-zoom-label');
  const infoEl = el('ser-info');
  const beToggle = el('ser-be');

  const STAGE_HEIGHT = 560;

  let renderer = null;
  let file = null;
  let header = null;
  let frameIndex = 0;
  let playing = false;
  let loading = false;
  const view = { zoomExp: 0, centerX: 0, centerY: 0 };

  // fps measurement
  let fpsCount = 0;
  let fpsStart = 0;

  function resizeCanvas() {
    const w = stage.clientWidth || 800;
    canvas.width = w;
    canvas.height = STAGE_HEIGHT;
    canvas.style.height = STAGE_HEIGHT + 'px';
  }

  function ensureRenderer() {
    if (renderer) return true;
    resizeCanvas();
    try {
      renderer = createSerRenderer(canvas);
    } catch (e) {
      emptyMsg.textContent = 'WebGL2 error: ' + e.message;
      return false;
    }
    if (!renderer) {
      emptyMsg.textContent = 'WebGL2 is not available in this browser.';
      return false;
    }
    return true;
  }

  function zoom() { return Math.pow(2, view.zoomExp); }

  function updateZoomLabel() {
    const z = zoom();
    zoomLabel.textContent = z >= 1 ? `×${z}` : `×1/${1 / z}`;
  }

  function draw() {
    if (!renderer || !header) return;
    renderer.render({ zoom: zoom(), centerX: view.centerX, centerY: view.centerY });
  }

  function resetView() {
    if (!header) return;
    // largest power-of-2 zoom that fits the whole frame
    const fit = Math.min(canvas.width / header.width, canvas.height / header.height);
    view.zoomExp = Math.max(-4, Math.floor(Math.log2(fit)));
    view.centerX = header.width / 2;
    view.centerY = header.height / 2;
    updateZoomLabel();
    draw();
  }

  async function loadFrame(i, { count = false } = {}) {
    if (!file || !header || loading) return;
    loading = true;
    try {
      const data = await readSerFrame(file, header, i, { bigEndian: beToggle.checked });
      renderer.upload(data);
      frameIndex = i;
      scrub.value = String(i);
      frameLabel.textContent = `${i + 1} / ${header.frameCount}`;
      draw();
      if (count) {
        fpsCount++;
        const now = performance.now();
        if (now - fpsStart >= 500) {
          fpsEl.textContent = (fpsCount * 1000 / (now - fpsStart)).toFixed(1) + ' fps';
          fpsCount = 0;
          fpsStart = now;
        }
      }
      window.__tensorez = window.__tensorez || {};
      window.__tensorez.serFrameDrawn = (window.__tensorez.serFrameDrawn || 0) + 1;
    } finally {
      loading = false;
    }
  }

  async function playLoop() {
    fpsCount = 0;
    fpsStart = performance.now();
    while (playing && file && header) {
      const next = (frameIndex + 1) % header.frameCount;
      await loadFrame(next, { count: true });
      // yield to the event loop / vsync
      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  function setPlaying(p) {
    if (!header) return;
    playing = p;
    playBtn.textContent = p ? '❚❚ Pause' : '▶ Play';
    if (p) playLoop();
    else fpsEl.textContent = '';
  }

  function renderInfo(h) {
    const fmt = [
      ['file id', h.fileId],
      ['color', `${h.colorName} (${h.colorId})`],
      ['size', `${h.width} × ${h.height}`],
      ['bit depth', `${h.pixelDepthPerPlane}-bit, ${h.channels} ch`],
      ['frames', h.frameCount + (h.truncated ? ' (truncated file)' : '')],
      ['frame size', (h.frameBytes / 1024 / 1024).toFixed(2) + ' MiB'],
      ['endian flag', String(h.littleEndianFlag)],
      ['observer', h.observer || '—'],
      ['instrument', h.instrument || '—'],
      ['telescope', h.telescope || '—'],
      ['recorded (UTC)', h.recordedUtc ? h.recordedUtc.toISOString() : '—'],
      ['timestamps', h.hasTimestamps ? 'per-frame trailer present' : 'none'],
    ];
    infoEl.innerHTML = '<h3>Header</h3>';
    const table = document.createElement('table');
    for (const [k, v] of fmt) {
      const tr = document.createElement('tr');
      const td1 = document.createElement('td');
      const td2 = document.createElement('td');
      td1.textContent = k;
      td2.textContent = String(v);
      tr.append(td1, td2);
      table.appendChild(tr);
    }
    infoEl.appendChild(table);
  }

  async function openFile(f) {
    setPlaying(false);
    if (!ensureRenderer()) return;
    try {
      header = await parseSerHeader(f);
    } catch (e) {
      emptyMsg.hidden = false;
      emptyMsg.textContent = 'Not a readable SER file: ' + (e.message || e);
      header = null;
      return;
    }
    file = f;
    emptyMsg.hidden = true;
    renderer.configure(header);
    renderInfo(header);
    scrub.max = String(header.frameCount - 1);
    for (const b of [playBtn, scrub, el('ser-zoom-in'), el('ser-zoom-out'), el('ser-reset')]) {
      b.disabled = false;
    }
    resetView();
    await loadFrame(0);
    window.__tensorez = window.__tensorez || {};
    window.__tensorez.serHeader = header;
  }

  // ---------- events ----------

  el('ser-file').addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) openFile(f);
  });

  playBtn.addEventListener('click', () => setPlaying(!playing));

  scrub.addEventListener('input', () => {
    setPlaying(false);
    loadFrame(Number(scrub.value));
  });

  beToggle.addEventListener('change', () => { if (header) loadFrame(frameIndex); });

  function zoomStep(dir, cx, cy) {
    if (!header) return;
    const rect = canvas.getBoundingClientRect();
    const px = cx ?? rect.width / 2;
    const py = cy ?? rect.height / 2;
    // canvas CSS px == device px here (no dpr scaling on the backing store)
    const oldZ = zoom();
    view.zoomExp = Math.max(-4, Math.min(6, view.zoomExp + dir));
    const z = zoom();
    // keep the image point under the cursor fixed
    const ix = (px - canvas.width / 2) / oldZ + view.centerX;
    const iy = (py - canvas.height / 2) / oldZ + view.centerY;
    view.centerX = ix - (px - canvas.width / 2) / z;
    view.centerY = iy - (py - canvas.height / 2) / z;
    updateZoomLabel();
    draw();
  }

  el('ser-zoom-in').addEventListener('click', () => zoomStep(1));
  el('ser-zoom-out').addEventListener('click', () => zoomStep(-1));
  el('ser-reset').addEventListener('click', resetView);

  canvas.addEventListener('wheel', (e) => {
    if (!header) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomStep(e.deltaY < 0 ? 1 : -1, e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    if (!header) return;
    drag = { x: e.clientX, y: e.clientY, cx: view.centerX, cy: view.centerY };
    canvas.classList.add('panning');
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    view.centerX = drag.cx - (e.clientX - drag.x) / zoom();
    view.centerY = drag.cy - (e.clientY - drag.y) / zoom();
    draw();
  });
  canvas.addEventListener('pointerup', () => {
    drag = null;
    canvas.classList.remove('panning');
  });

  window.addEventListener('resize', () => {
    if (!header) return;
    resizeCanvas();
    draw();
  });

  return {
    onShow() {
      if (header) { resizeCanvas(); draw(); }
    },
  };
}
