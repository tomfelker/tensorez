// Results viewer: load a completed run directory's manifest.json and show its
// artifacts grouped by stage. Click any preview for full-size zoom/pan.

import { openViewer } from './viewer.js';

const IMAGE_KINDS = new Set(['preview', 'sequence_frame']);

export function initResults(root) {
  root.innerHTML = `
    <h1>Results</h1>
    <p class="view-sub">Browse a completed run directory —
      reads <span class="mono">manifest.json</span> and shows every declared artifact.</p>

    <div class="toolbar">
      <input id="res-dir" type="text" class="mono" style="width:420px"
             placeholder="path to run directory" aria-label="run directory">
      <button id="res-browse" class="small">Browse…</button>
      <button id="res-load" class="primary">Load</button>
      <span id="res-mock-hint" class="dim" style="font-size:12px" hidden>
        (mock run dir prefilled)</span>
    </div>

    <div id="res-error"></div>
    <div id="res-body">
      <div class="placeholder-note">No run loaded. Pick a run directory and press Load.</div>
    </div>`;

  const dirInput = root.querySelector('#res-dir');
  const body = root.querySelector('#res-body');
  const errBox = root.querySelector('#res-error');

  if (window.bridge.platform === 'mock') {
    dirInput.value = '/mockrun/run1';
    root.querySelector('#res-mock-hint').hidden = false;
  }

  root.querySelector('#res-browse').addEventListener('click', async () => {
    const p = await window.bridge.chooseDirectory({ title: 'Choose run directory' });
    if (p) { dirInput.value = p; load(); }
  });
  root.querySelector('#res-load').addEventListener('click', load);
  dirInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') load(); });

  async function load() {
    const dir = dirInput.value.trim().replace(/\/+$/, '');
    if (!dir) return;
    errBox.textContent = '';
    let manifest;
    try {
      const manifestPath = window.bridge.platform === 'mock'
        ? dir + '/manifest.json'
        : dir + '/manifest.json'; // same join; electron readTextFile handles OS paths
      manifest = JSON.parse(await window.bridge.readTextFile(manifestPath));
    } catch (e) {
      errBox.innerHTML = '';
      const banner = document.createElement('div');
      banner.className = 'error-banner';
      banner.innerHTML = '<div class="e-title">Could not load manifest.json</div>';
      const pre = document.createElement('pre');
      pre.textContent = `${dir}/manifest.json\n${e.message || e}\n\nA missing manifest means the run is incomplete or still going (the manifest is written last, atomically).`;
      banner.appendChild(pre);
      errBox.appendChild(banner);
      return;
    }
    render(dir, manifest);
  }

  function render(dir, m) {
    body.textContent = '';

    // ---- summary card ----
    const sum = document.createElement('div');
    sum.className = 'results-summary';
    const items = [
      ['recipe', m.run?.name ?? '—'],
      ['started (UTC)', m.run?.started_utc ?? '—'],
      ['total time', m.run?.seconds != null ? m.run.seconds.toFixed(1) + ' s' : '—'],
      ['frames', m.run?.frame_count ?? '—'],
      ['products', (m.run?.products || []).join(', ') || '—'],
      ['artifacts', (m.artifacts || []).length],
    ];
    for (const [label, value] of items) {
      const d = document.createElement('div');
      d.className = 'sum-item';
      d.innerHTML = `<div class="sum-label">${label}</div><div class="sum-value"></div>`;
      d.querySelector('.sum-value').textContent = String(value);
      sum.appendChild(d);
    }
    body.appendChild(sum);

    // ---- stage timing/cached info ----
    const stageInfo = new Map((m.stages || []).map((s) => [s.name, s]));

    // ---- artifacts grouped by stage ----
    const byStage = new Map();
    for (const a of m.artifacts || []) {
      if (!byStage.has(a.stage)) byStage.set(a.stage, []);
      byStage.get(a.stage).push(a);
    }
    // keep contract stage order, then anything unknown at the end
    const order = ['lights', 'darks', 'align', 'lucky_scoring', 'local_lucky',
                   'lucky_stack', 'mfbd', 'output'];
    const rank = (s) => { const i = order.indexOf(s); return i < 0 ? order.length : i; };
    const stages = [...byStage.keys()].sort((a, b) => rank(a) - rank(b));

    for (const stage of stages) {
      const grp = document.createElement('div');
      grp.className = 'stage-group';
      const info = stageInfo.get(stage);
      const h = document.createElement('h3');
      h.textContent = stage;
      if (info?.cached) h.innerHTML += ' <span class="badge cached">cached</span>';
      if (info?.seconds != null) {
        const t = document.createElement('span');
        t.className = 'dim mono';
        t.style.cssText = 'font-size:11px;margin-left:8px;text-transform:none;letter-spacing:0';
        t.textContent = info.seconds.toFixed(1) + ' s';
        h.appendChild(t);
      }
      grp.appendChild(h);

      const gal = document.createElement('div');
      gal.className = 'gallery';
      for (const a of byStage.get(stage)) {
        if (IMAGE_KINDS.has(a.kind)) {
          const url = window.bridge.fileUrl(dir, a.path);
          const card = document.createElement('div');
          card.className = 'artifact-thumb';
          card.title = `${a.path}` + (a.width ? ` — ${a.width}×${a.height}` : '');
          card.innerHTML = `<img src="${url}" alt="${a.name}">
            <div class="a-name">${a.name}${a.frame != null ? ' #' + a.frame : ''}</div>`;
          card.addEventListener('click', () =>
            openViewer(`${stage}/${a.name}` + (a.width ? ` — ${a.width}×${a.height}` : ''), url));
          gal.appendChild(card);
        } else {
          const chip = document.createElement('div');
          chip.className = 'artifact-chip';
          chip.title = a.path;
          chip.textContent = `${a.name} (${a.kind}, .${a.path.split('.').pop()})`;
          gal.appendChild(chip);
        }
      }
      grp.appendChild(gal);
      body.appendChild(grp);
    }

    if (!stages.length) {
      const note = document.createElement('div');
      note.className = 'placeholder-note';
      note.textContent = 'Manifest loaded but declares no artifacts.';
      body.appendChild(note);
    }
  }

  return {
    onShow() {},
    receive(detail) {
      if (detail.runDir) {
        dirInput.value = window.bridge.platform === 'mock'
          ? window.bridge.fileUrl(detail.runDir) // map mock run_dir to served dir
          : detail.runDir;
        load();
      }
    },
  };
}
