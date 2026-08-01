// Run console: spawn the CLI via the bridge, consume the JSONL event stream,
// render per-stage progress, artifacts as they land, and the raw log.
//
// The CLI resolves the recipe's relative paths against its working directory,
// which defaults to the recipe's own folder — so a recipe saved beside its
// .ser files just works, and its results land there too. Run archives and the
// stage cache are bulk data and go wherever the settings say (default: the
// working directory), which is how you keep them off the capture drive.

import { openViewer } from './viewer.js';
import { loadSettings, updateSettings } from './settings.js';

const IMAGE_KINDS = new Set(['preview', 'sequence_frame']);

export function initRun(root) {
  root.innerHTML = `
    <h1>Run</h1>
    <p class="view-sub">Runs <span class="mono">python -m tensorez run &lt;recipe.toml&gt;</span>
      and follows its event stream.</p>

    <div class="toolbar">
      <input id="run-recipe" type="text" class="mono" style="width:340px"
             placeholder="path to recipe.toml" aria-label="recipe path">
      <button id="run-browse" class="small">Browse…</button>
      <span class="dim" style="font-size:12px">cwd</span>
      <input id="run-cwd" type="text" class="mono" style="width:200px"
             placeholder="(recipe directory)" aria-label="working directory">
      <div class="sep"></div>
      <button id="run-start" class="primary">▶ Run</button>
      <button id="run-cancel" class="danger" disabled>■ Cancel</button>
      <div class="grow"></div>
      <span id="run-demo-btns" hidden>
        <span class="dim" style="font-size:12px">mock:</span>
        <button id="run-demo-ok" class="small">demo run</button>
        <button id="run-demo-err" class="small">demo failure</button>
      </span>
    </div>

    <details class="rawlog" id="run-dirs">
      <summary>Where runs and cache go</summary>
      <div class="toolbar" style="margin-top:8px">
        <span class="dim" style="font-size:12px">run archives</span>
        <input id="run-runs-dir" type="text" class="mono" style="width:260px"
               placeholder="(working directory)/tensorez_runs" aria-label="runs directory">
        <button id="run-runs-browse" class="small">Browse…</button>
        <div class="sep"></div>
        <span class="dim" style="font-size:12px">stage cache</span>
        <input id="run-cache-dir" type="text" class="mono" style="width:260px"
               placeholder="(working directory)/tensorez_cache" aria-label="cache directory">
        <button id="run-cache-browse" class="small">Browse…</button>
      </div>
      <div class="field-help" style="padding:0 8px 8px">
        Kept out of the recipe on purpose: both hold bulk regenerable data, so
        point them at a fast scratch disk and every recipe uses it, leaving your
        capture drive holding only recipes and results.
      </div>
    </details>

    <div class="run-status-card">
      <div id="run-dot" class="status-dot"></div>
      <div>
        <div id="run-status" class="run-status-text">Idle</div>
        <div id="run-meta" class="run-meta">no run yet</div>
      </div>
      <div class="grow" style="flex:1"></div>
      <div id="run-elapsed" class="run-meta mono"></div>
    </div>

    <div id="run-error"></div>
    <div id="run-done"></div>
    <div id="run-stages" class="stage-list"></div>

    <details class="rawlog" id="run-rawlog-details">
      <summary>Raw event log (<span id="run-rawlog-count">0</span> lines)</summary>
      <pre id="run-rawlog"></pre>
    </details>`;

  const el = (id) => root.querySelector('#' + id);
  const recipeInput = el('run-recipe');
  const cwdInput = el('run-cwd');
  const runsDirInput = el('run-runs-dir');
  const cacheDirInput = el('run-cache-dir');
  const startBtn = el('run-start');
  const cancelBtn = el('run-cancel');
  const dot = el('run-dot');
  const statusEl = el('run-status');
  const metaEl = el('run-meta');
  const elapsedEl = el('run-elapsed');
  const stagesEl = el('run-stages');
  const errorEl = el('run-error');
  const doneEl = el('run-done');
  const rawlogEl = el('run-rawlog');
  const rawCountEl = el('run-rawlog-count');

  if (window.bridge.platform === 'mock') {
    el('run-demo-btns').hidden = false;
    recipeInput.value = 'examples/jupiter_demo.toml';
    el('run-demo-ok').addEventListener('click', () => {
      recipeInput.value = 'examples/jupiter_demo.toml';
      start();
    });
    el('run-demo-err').addEventListener('click', () => {
      recipeInput.value = 'examples/jupiter_demo_error.toml';
      start();
    });
  }

  let activeRunId = null;
  let runDir = null;
  let rawCount = 0;
  let startTime = 0;
  let elapsedTimer = null;
  const stageRows = new Map(); // stage -> {row, bar, detail, time, artifacts}

  function setStatus(kind, text, meta) {
    dot.className = 'status-dot' + (kind ? ' ' + kind : '');
    statusEl.textContent = text;
    if (meta != null) metaEl.textContent = meta;
  }

  function resetUI() {
    stagesEl.textContent = '';
    errorEl.textContent = '';
    doneEl.textContent = '';
    rawlogEl.textContent = '';
    rawCount = 0;
    rawCountEl.textContent = '0';
    stageRows.clear();
    runDir = null;
    elapsedEl.textContent = '';
  }

  function appendRaw(line, cls) {
    rawCount++;
    rawCountEl.textContent = String(rawCount);
    const span = document.createElement('span');
    if (cls) span.className = cls;
    span.textContent = line + '\n';
    rawlogEl.appendChild(span);
    if (rawlogEl.childNodes.length > 2000) rawlogEl.removeChild(rawlogEl.firstChild);
    rawlogEl.scrollTop = rawlogEl.scrollHeight;
  }

  function getStageRow(stage) {
    if (stageRows.has(stage)) return stageRows.get(stage);
    const row = document.createElement('div');
    row.className = 'stage-row';
    row.dataset.stage = stage;
    row.innerHTML = `
      <div class="stage-name">${stage}</div>
      <div>
        <div class="pbar"><div></div></div>
        <div class="artifact-strip"></div>
      </div>
      <div class="stage-detail"></div>
      <div class="stage-time"></div>`;
    stagesEl.appendChild(row);
    const rec = {
      row,
      bar: row.querySelector('.pbar > div'),
      detail: row.querySelector('.stage-detail'),
      time: row.querySelector('.stage-time'),
      strip: row.querySelector('.artifact-strip'),
    };
    stageRows.set(stage, rec);
    return rec;
  }

  function fmtSecs(s) {
    return s >= 100 ? s.toFixed(0) + ' s' : s.toFixed(1) + ' s';
  }

  function handleEvent(ev) {
    switch (ev.event) {
      case 'run_start': {
        runDir = ev.run_dir;
        setStatus('running', 'Running', `${ev.recipe_path} → ${ev.run_dir}`);
        metaEl.title = JSON.stringify(ev.recipe, null, 2);
        elapsedEl.textContent = `${ev.frame_count} frames`;
        break;
      }
      case 'stage_start': {
        const r = getStageRow(ev.stage);
        if (ev.cached) {
          r.detail.innerHTML = '<span class="badge cached">cached</span>';
          r.bar.style.width = '100%';
        } else {
          r.detail.textContent = 'running…';
        }
        break;
      }
      case 'progress': {
        const r = getStageRow(ev.stage);
        const pct = ev.total ? (100 * ev.current / ev.total) : 0;
        r.bar.style.width = pct.toFixed(1) + '%';
        r.detail.textContent =
          `${ev.current}/${ev.total}` + (ev.message ? ` — ${ev.message}` : '');
        break;
      }
      case 'artifact': {
        const r = getStageRow(ev.stage);
        if (IMAGE_KINDS.has(ev.kind)) {
          const url = window.bridge.fileUrl(runDir || '', ev.path);
          const card = document.createElement('div');
          card.className = 'artifact-thumb';
          card.title = `${ev.name} (${ev.kind})` +
            (ev.width ? ` ${ev.width}×${ev.height}` : '') +
            (ev.frame != null ? ` frame ${ev.frame}` : '');
          card.innerHTML = `<img src="${url}" alt="${ev.name}">
            <div class="a-name">${ev.name}${ev.frame != null ? ' #' + ev.frame : ''}</div>`;
          card.addEventListener('click', () => openViewer(ev.path, url));
          r.strip.appendChild(card);
        } else {
          const chip = document.createElement('div');
          chip.className = 'artifact-chip';
          chip.title = ev.path;
          chip.textContent = `${ev.name} (.${ev.path.split('.').pop()})`;
          r.strip.appendChild(chip);
        }
        break;
      }
      case 'stage_end': {
        const r = getStageRow(ev.stage);
        r.row.classList.add('stage-done');
        r.bar.style.width = '100%';
        r.time.textContent = fmtSecs(ev.seconds);
        if (!r.detail.querySelector('.badge')) {
          r.detail.innerHTML = '<span class="badge done">done</span>';
        }
        break;
      }
      case 'log': {
        appendRaw(`[${ev.level}] ${ev.message}`, 'log-' + ev.level);
        break;
      }
      case 'error': {
        setStatus('error', 'Failed', ev.message);
        if (ev.stage && stageRows.has(ev.stage)) {
          const r = stageRows.get(ev.stage);
          r.row.classList.add('stage-error');
          r.detail.innerHTML = '<span class="badge error">error</span>';
        }
        errorEl.innerHTML = '';
        const banner = document.createElement('div');
        banner.className = 'error-banner';
        banner.innerHTML = `<div class="e-title">Run failed${ev.stage ? ` in stage “${ev.stage}”` : ''}</div>`;
        const msg = document.createElement('pre');
        msg.textContent = ev.message + (ev.traceback ? '\n\n' + ev.traceback : '');
        banner.appendChild(msg);
        errorEl.appendChild(banner);
        break;
      }
      case 'done': {
        setStatus('done', 'Done', `finished in ${fmtSecs(ev.seconds)}`);
        doneEl.innerHTML = '';
        const products = ev.products || [];
        const banner = document.createElement('div');
        banner.className = 'done-banner';
        banner.innerHTML = `
          <div>
            <div class="d-title">Run complete — ${fmtSecs(ev.seconds)}</div>
            <div class="run-meta mono">${products.join(', ') || 'no products'}</div>
            <div class="run-meta mono dim">${ev.output_dir || ''}</div>
          </div>
          <div style="flex:1"></div>`;
        const btn = document.createElement('button');
        btn.textContent = 'Open in Results';
        btn.addEventListener('click', () => {
          window.dispatchEvent(new CustomEvent('tensorez:navigate', {
            detail: { view: 'results', runDir },
          }));
        });
        // one thumbnail per product, each named for its stage's output
        for (const name of products) {
          const url = window.bridge.fileUrl(runDir || '', name + '.png');
          const thumb = document.createElement('img');
          thumb.src = url;
          thumb.alt = name;
          thumb.title = name;
          thumb.style.cssText = 'height:64px;border-radius:6px;background:#000;cursor:pointer';
          thumb.addEventListener('click', () => openViewer(name + '.png', url));
          banner.appendChild(thumb);
        }
        banner.appendChild(btn);
        doneEl.appendChild(banner);
        break;
      }
      default:
        // Unknown event types must be ignored (contract §2).
        break;
    }
  }

  // ---------- bridge stream ----------

  window.bridge.onRunLine(({ runId, line }) => {
    if (runId !== activeRunId) return;
    appendRaw(line);
    let ev;
    try { ev = JSON.parse(line); } catch { return; } // tolerate junk on stdout
    handleEvent(ev);
  });

  window.bridge.onRunStderr(({ runId, line }) => {
    if (runId !== activeRunId) return;
    appendRaw('[stderr] ' + line, 'log-stderr');
  });

  window.bridge.onRunExit(({ runId, code, signal }) => {
    if (runId !== activeRunId) return;
    activeRunId = null;
    startBtn.disabled = false;
    cancelBtn.disabled = true;
    clearInterval(elapsedTimer);
    if (signal || code === null) {
      setStatus('cancelled', 'Cancelled', `killed (${signal || 'signal'})`);
    } else if (code !== 0 && dot.className.indexOf('error') < 0) {
      setStatus('error', 'Failed', `exit code ${code}`);
    }
    appendRaw(`[exit] code=${code} signal=${signal || 'none'}`, code === 0 ? undefined : 'log-error');
  });

  async function start() {
    const recipePath = recipeInput.value.trim();
    if (!recipePath) { recipeInput.focus(); return; }
    resetUI();
    setStatus('running', 'Starting…', recipePath);
    startBtn.disabled = true;
    cancelBtn.disabled = false;
    startTime = performance.now();
    elapsedTimer = setInterval(() => {
      elapsedEl.textContent = ((performance.now() - startTime) / 1000).toFixed(1) + ' s';
    }, 100);
    activeRunId = await window.bridge.spawnRun({
      recipePath,
      cwd: cwdInput.value.trim() || undefined,
      runsDir: runsDirInput.value.trim() || undefined,
      cacheDir: cacheDirInput.value.trim() || undefined,
    });
  }

  // ---------- runs/cache preferences ----------

  for (const [input, key, title] of [
    [runsDirInput, 'runsDir', 'Choose where run archives go'],
    [cacheDirInput, 'cacheDir', 'Choose where the stage cache goes'],
  ]) {
    input.addEventListener('change', () => updateSettings({ [key]: input.value.trim() }));
    const browse = el(input.id.replace('-dir', '-browse'));
    browse.addEventListener('click', async () => {
      const p = await window.bridge.chooseDirectory({ title, defaultPath: input.value.trim() });
      if (p) { input.value = p; await updateSettings({ [key]: p }); }
    });
  }

  loadSettings().then(({ values }) => {
    runsDirInput.value = values.runsDir || '';
    cacheDirInput.value = values.cacheDir || '';
  });

  startBtn.addEventListener('click', start);
  cancelBtn.addEventListener('click', () => {
    if (activeRunId != null) window.bridge.killRun(activeRunId);
  });
  el('run-browse').addEventListener('click', async () => {
    const p = await window.bridge.openFileDialog({
      title: 'Choose recipe',
      filters: [{ name: 'TOML recipe', extensions: ['toml'] }],
    });
    if (p) recipeInput.value = p;
  });

  return {
    onShow() {
      // convenience: pick up the recipe editor's saved path if we have none
      if (!recipeInput.value && window.__tensorez?.recipePath?.()) {
        recipeInput.value = window.__tensorez.recipePath();
      }
    },
    // the recipe view's Run button: it has already saved the file
    receive(detail) {
      if (detail.recipePath) recipeInput.value = detail.recipePath;
      if (detail.autostart && activeRunId == null) start();
    },
  };
}
