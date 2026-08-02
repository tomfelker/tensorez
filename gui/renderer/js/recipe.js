// Recipe editor: schema-generated form <-> raw TOML pane, two-way sync.
//
// Document model, like any editor: the recipe you are editing is always
// backed by a real file, because the CLI can only run a file. A recipe you
// haven't saved anywhere lives in a scratch file in the user's profile
// directory (bridge.appPaths().scratchRecipe) and is written on every edit;
// Save As moves it somewhere you care about — beside the .ser files, ideally,
// since that is where its results will land — after which Save writes there.
// The last-opened path is remembered in the settings file next to the scratch
// recipe, so reopening the app returns you to what you were working on.

import { parse as parseToml, stringify as stringifyToml } from '../vendor/index.js';
import {
  SCHEMA, defaultRecipe, validateRecipe, hydrate,
  isFieldAtDefault, resetFieldToDefault, modifiedFields, describeDefault,
} from './schema.js';
import { loadSettings, updateSettings } from './settings.js';

const SQRT2 = Math.SQRT2;

function roundSig(x, sig = 3) {
  if (x === 0) return 0;
  const mag = Math.ceil(Math.log10(Math.abs(x)));
  const f = Math.pow(10, sig - mag);
  return Math.round(x * f) / f;
}

// Rebuild an object with sections/keys in schema order (stable TOML output).
function canonicalize(obj) {
  const out = {};
  for (const sec of SCHEMA) {
    const sv = obj[sec.section];
    if (sv == null) continue;
    const t = {};
    for (const f of sec.fields) if (f.key in sv) t[f.key] = sv[f.key];
    out[sec.section] = t;
  }
  return out;
}

export async function initRecipe(root) {
  let state = defaultRecipe();
  let currentPath = null;   // always a real file once boot() has run
  let scratchPath = null;   // …which is this one until the user saves elsewhere
  let unsaved = false;      // editor content differs from the file on disk
  let loading = false;      // suppress "unsaved" while adopting a file's content
  let tomlDirty = false;    // textarea is the source of truth while it has errors

  root.innerHTML = `
    <h1>Recipe</h1>
    <p class="view-sub">Describe what to process and how. Save it beside your capture
      files — the recipe names the run, and its results land next to it.</p>
    <div class="toolbar">
      <button id="rc-new">New</button>
      <button id="rc-open">Open…</button>
      <button id="rc-save">Save</button>
      <button id="rc-saveas">Save As…</button>
      <div class="sep"></div>
      <button id="rc-run" class="primary" title="saves, then runs this recipe">▶ Run</button>
      <div class="sep"></div>
      <div id="rc-path" class="file-label">unsaved recipe</div>
    </div>
    <!-- Problems stay out here: the TOML pane below is collapsed by default,
         and a validation error you can't see is worse than useless. -->
    <div id="rc-toml-error" class="toml-error"></div>
    <div class="recipe-form" id="rc-form"></div>
    <details class="toml-details" id="rc-toml-details">
      <summary>Raw TOML<span class="dim"> — this is the whole recipe; the form above
        just edits it</span></summary>
      <div class="toml-pane">
        <textarea id="rc-toml" spellcheck="false" aria-label="raw toml"></textarea>
      </div>
    </details>`;

  const formEl = root.querySelector('#rc-form');
  const tomlEl = root.querySelector('#rc-toml');
  const errEl = root.querySelector('#rc-toml-error');
  const pathEl = root.querySelector('#rc-path');

  function setError(msg) {
    if (msg) {
      errEl.textContent = msg;
      errEl.classList.add('show');
      tomlEl.classList.add('toml-bad');
    } else {
      errEl.classList.remove('show');
      tomlEl.classList.remove('toml-bad');
    }
  }

  function syncTomlFromState() {
    tomlEl.value = stringifyToml(state) + '\n';
    tomlDirty = false;
    // form-driven state always passes per-field checks, but cross-field
    // constraints (e.g. align.per_channel combos) can still be violated —
    // surface those live so the user learns before the CLI hard-errors
    const problems = validateRecipe(state);
    setError(problems.length ? 'Recipe validation:\n• ' + problems.join('\n• ') : null);
    if (!loading) markUnsaved();
  }

  function stateChanged({ rerender = false } = {}) {
    if (rerender) {
      // Canonicalize ONLY when re-rendering: field controls close over the
      // current section objects, so replacing them without a re-render would
      // silently orphan later edits.
      state = canonicalize(state);
      renderForm();
    } else {
      refreshDefaultMarkers();
    }
    syncTomlFromState();
  }

  // ---------- form rendering ----------

  function renderForm() {
    formEl.textContent = '';
    for (const sec of SCHEMA) formEl.appendChild(renderSection(sec));
    refreshDefaultMarkers();
  }

  // ---------- off-default marking ----------
  // Applied after rendering AND after every in-place edit, because most commits
  // deliberately skip the re-render (rebuilding controls mid-keystroke would
  // steal focus). So these marks are maintained on the live DOM instead of
  // being baked in when a row is built.

  function makeRevertButton(sec, f) {
    const revert = document.createElement('button');
    revert.className = 'small revert-btn';
    revert.type = 'button';
    revert.textContent = '↺';
    revert.title = `Revert to default (${describeDefault(f)})`;
    revert.setAttribute('aria-label', `Revert ${f.key} to its default`);
    revert.addEventListener('click', () => {
      const secState = state[sec.section];  // looked up now: a re-render replaces it
      if (!secState) return;
      resetFieldToDefault(f, secState);
      stateChanged({ rerender: true });
    });
    return revert;
  }

  function makeSectionReset(sec) {
    const reset = document.createElement('button');
    reset.className = 'small reset-section';
    reset.type = 'button';
    reset.addEventListener('click', () => {
      const secState = state[sec.section];
      if (!secState) return;
      for (const f of modifiedFields(sec, secState)) resetFieldToDefault(f, secState);
      stateChanged({ rerender: true });
    });
    return reset;
  }

  function refreshDefaultMarkers() {
    for (const sec of SCHEMA) {
      const card = formEl.querySelector(`.section-card[data-section="${sec.section}"]`);
      if (!card) continue;
      // A disabled optional section renders detached defaults, so it is scored
      // against the real state and correctly reports nothing.
      const secState = state[sec.section];

      for (const f of sec.fields) {
        if (f.hidden) continue; // a composite control reports for these
        const row = card.querySelector(`[data-field="${sec.section}.${f.key}"]`);
        if (!row) continue;
        const off = !!secState && !isFieldAtDefault(f, secState);
        row.classList.toggle('field-modified', off); // semantic hook; no styling
        const button = row.querySelector('.revert-btn');
        // third grid track, past the control — a reserved column of its own
        if (off && !button) row.appendChild(makeRevertButton(sec, f));
        else if (!off && button) button.remove();
      }

      const changed = modifiedFields(sec, secState);
      card.classList.toggle('section-modified', changed.length > 0);
      let badge = card.querySelector('.reset-section');
      if (!changed.length) {
        badge?.remove();
        continue;
      }
      if (!badge) {
        badge = makeSectionReset(sec);
        const head = card.querySelector('.section-head');
        // ahead of an optional section's enable checkbox, which sits hard right
        head.insertBefore(badge, head.querySelector('label'));
      }
      badge.textContent = `↺ ${changed.length} changed`;
      badge.title = 'Revert to defaults: ' +
        changed.map((f) => f.labelText ?? f.key).join(', ');
    }
  }

  function renderSection(sec) {
    const card = document.createElement('div');
    card.className = 'section-card';
    card.dataset.section = sec.section;
    const enabled = sec.section in state;
    if (sec.optionalSection && !enabled) card.classList.add('disabled-section');

    const head = document.createElement('div');
    head.className = 'section-head';
    head.innerHTML = `<h2>${sec.label}</h2><span class="sec-toml">[${sec.section}]</span>`;
    if (sec.optionalSection) {
      const lab = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = enabled;
      cb.dataset.sectionToggle = sec.section;
      cb.addEventListener('change', () => {
        if (cb.checked) {
          const t = {};
          for (const f of sec.fields) if (!f.optional) t[f.key] = structuredClone(f.default);
          state[sec.section] = t;
        } else {
          delete state[sec.section];
        }
        stateChanged({ rerender: true });
      });
      lab.append(cb, document.createTextNode('enabled'));
      head.appendChild(lab);
    }
    card.appendChild(head);

    if (sec.help) {
      const h = document.createElement('div');
      h.className = 'field-help';
      h.style.marginBottom = '10px';
      h.textContent = sec.help;
      card.appendChild(h);
    }

    const fields = document.createElement('div');
    fields.className = 'fields';
    // A disabled optional section renders greyed-out defaults on a detached
    // object (pointer-events are off, so nothing can commit to it).
    const secState = state[sec.section] ||
      Object.fromEntries(sec.fields.filter((f) => !f.optional)
        .map((f) => [f.key, structuredClone(f.default)]));
    for (const f of sec.fields) {
      if (f.hidden) continue; // rendered by a composite control (e.g. pixelscale)
      fields.appendChild(renderField(sec, f, secState));
    }

    card.appendChild(fields);
    return card;
  }

  function renderField(sec, f, secState) {
    const row = document.createElement('div');
    row.className = 'field-row';
    row.dataset.field = `${sec.section}.${f.key}`;
    if (f.showIf && !f.showIf(secState)) row.classList.add('hidden-field');

    const label = document.createElement('div');
    label.className = 'field-label';
    label.innerHTML = `<span class="fname mono">${f.labelText ?? f.key}</span>` +
      (f.required
        ? ' <span class="req-star" title="required physical parameter">required</span>' : '') +
      (f.help ? `<div class="field-help">${f.help}</div>` : '');
    row.appendChild(label);

    const ctl = document.createElement('div');
    ctl.className = 'field-control';
    const present = f.key in secState;

    if (f.optional) {
      const tog = document.createElement('label');
      tog.className = 'opt-toggle';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = present;
      cb.addEventListener('change', () => {
        if (cb.checked) secState[f.key] = structuredClone(f.default);
        else delete secState[f.key];
        stateChanged({ rerender: true });
      });
      tog.append(cb, document.createTextNode('set'));
      ctl.appendChild(tog);
      if (!present) {
        const note = document.createElement('span');
        note.className = 'dim';
        note.style.fontSize = '12px';
        note.textContent = f.key === 'end_frame' ? '(all frames)'
          : f.key === 'crop' ? '(full frame)' : '(unset)';
        ctl.appendChild(note);
        row.appendChild(ctl);
        return row;
      }
    }

    const commit = (v, opts) => {
      secState[f.key] = v;
      stateChanged(opts);
    };

    switch (f.type) {
      case 'string': {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.value = secState[f.key] ?? '';
        inp.readOnly = !!f.readonly;
        inp.addEventListener('input', () => commit(inp.value)); // live sync
        ctl.appendChild(inp);
        break;
      }
      case 'bool': {
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = !!secState[f.key];
        // bool toggles re-render so dependent constraint hints refresh
        cb.addEventListener('change', () => commit(cb.checked, { rerender: true }));
        ctl.appendChild(cb);
        const conflict = f.conflictIf?.(secState);
        if (conflict && !cb.checked) {
          // prevent creating a combo the CLI hard-errors on
          cb.disabled = true;
          const note = document.createElement('span');
          note.className = 'dim';
          note.style.fontSize = '12px';
          note.textContent = `(${conflict})`;
          ctl.appendChild(note);
        } else if (conflict && cb.checked) {
          // already-invalid combo (e.g. loaded from a file): warn, allow unchecking
          const warn = document.createElement('span');
          warn.className = 'field-warn';
          warn.textContent = `⚠ ${conflict} — the CLI will refuse this`;
          ctl.appendChild(warn);
        }
        break;
      }
      case 'enum':
      case 'int_enum': {
        const sel = document.createElement('select');
        for (const o of f.options) {
          const opt = document.createElement('option');
          opt.value = String(o);
          opt.textContent = String(o);
          sel.appendChild(opt);
        }
        sel.value = String(secState[f.key]);
        sel.addEventListener('change', () =>
          commit(f.type === 'int_enum' ? Number(sel.value) : sel.value, { rerender: true }));
        ctl.appendChild(sel);
        break;
      }
      case 'int':
      case 'float':
        ctl.appendChild(makeStepper(f, secState));
        break;
      case 'int2':
      case 'float2': {
        const isInt = f.type === 'int2';
        const wrap = document.createElement('span');
        wrap.className = 'pair-input field-control';
        (f.labels || ['a', 'b']).forEach((axis, i) => {
          const lab = document.createElement('span');
          lab.className = 'dim mono';
          lab.style.fontSize = '12px';
          lab.textContent = axis;
          const inp = document.createElement('input');
          inp.type = 'number';
          inp.step = isInt ? '1' : 'any';
          inp.value = secState[f.key][i];
          const commitPair = (normalize) => {
            let v = Number(inp.value);
            if (!Number.isFinite(v)) { if (!normalize) return; v = 0; }
            if (isInt) v = Math.round(v);
            if (normalize) inp.value = String(v);
            secState[f.key] = secState[f.key].slice();
            secState[f.key][i] = v;
            stateChanged();
          };
          inp.addEventListener('input', () => commitPair(false));
          inp.addEventListener('change', () => commitPair(true));
          wrap.append(lab, inp);
        });
        ctl.appendChild(wrap);
        break;
      }
      case 'floatlist': {
        // comma-separated numbers in one monospace input (e.g. "700, 530, 470")
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'mono';
        inp.style.width = '180px';
        inp.value = (secState[f.key] || []).join(', ');
        const parseList = () => {
          const parts = inp.value.split(',').map((s) => s.trim()).filter((s) => s !== '');
          if (!parts.length) return null;
          const nums = parts.map(Number);
          return nums.every(Number.isFinite) ? nums : null;
        };
        inp.addEventListener('input', () => {
          const nums = parseList();
          inp.classList.toggle('bad-input', !nums);
          if (nums) commit(nums);
        });
        inp.addEventListener('change', () => {
          const nums = parseList();
          if (!nums) { inp.value = (secState[f.key] || f.default).join(', '); }
          else inp.value = nums.join(', ');
          inp.classList.remove('bad-input');
        });
        ctl.appendChild(inp);
        break;
      }
      case 'pixelscale': {
        // Either/or: a direct arcsec/px value, OR camera parameters with the
        // resulting scale computed read-only. Exactly one representation lives
        // in the state (and thus the TOML); pixel_size_um present = camera.
        const cameraMode = 'pixel_size_um' in secState;
        const computeScale = () => {
          const fl = secState.focal_length_mm ?? f.camera.focal_length_mm;
          const bl = secState.barlow ?? f.camera.barlow;
          const px = secState.pixel_size_um ?? f.camera.pixel_size_um;
          return 206.265 * px / (fl * bl);
        };

        const mode = document.createElement('select');
        for (const [v, t] of [['direct', 'arcsec/px'], ['camera', 'from camera']]) {
          const o = document.createElement('option');
          o.value = v;
          o.textContent = t;
          mode.appendChild(o);
        }
        mode.value = cameraMode ? 'camera' : 'direct';
        mode.dataset.role = 'ps-mode';
        mode.addEventListener('change', () => {
          if (mode.value === 'camera') {
            delete secState[f.key];
            for (const [k, dv] of Object.entries(f.camera)) secState[k] = dv;
          } else {
            // carry the computed scale over as the direct value (3 decimals)
            const s = Number(computeScale().toFixed(3));
            for (const k of Object.keys(f.camera)) delete secState[k];
            secState[f.key] = s;
          }
          stateChanged({ rerender: true });
        });
        ctl.appendChild(mode);

        if (!cameraMode) {
          ctl.appendChild(makeStepper({ ...f, type: 'float', logStep: true }, secState));
        } else {
          const wrap = document.createElement('span');
          wrap.className = 'field-control';
          const readout = document.createElement('span');
          readout.className = 'ps-computed mono';
          readout.dataset.role = 'ps-computed';
          readout.title = '206.265 × pixel_size_um / (focal_length_mm × barlow)';
          const updateReadout = () => {
            readout.textContent = `= ${computeScale().toFixed(3)}″/px`;
          };
          const parts = [
            ['focal_length_mm', 'focal mm'],
            ['barlow', 'barlow ×'],
            ['pixel_size_um', 'pixel µm'],
          ];
          for (const [key, labelTxt] of parts) {
            const lab = document.createElement('span');
            lab.className = 'dim mono';
            lab.style.fontSize = '11px';
            lab.textContent = labelTxt;
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.step = 'any';
            inp.style.width = '74px';
            inp.value = secState[key];
            inp.dataset.role = 'ps-' + key;
            const commitNum = (normalize) => {
              const v = Number(inp.value);
              if (!Number.isFinite(v) || v <= 0) {
                if (normalize) inp.value = secState[key];
                return;
              }
              secState[key] = v;
              updateReadout();
              stateChanged();
            };
            inp.addEventListener('input', () => commitNum(false));
            inp.addEventListener('change', () => commitNum(true));
            wrap.append(lab, inp);
          }
          wrap.appendChild(readout);
          updateReadout();
          ctl.appendChild(wrap);
        }
        break;
      }
      case 'path': {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'mono';
        inp.style.flex = '1';
        inp.value = secState[f.key] ?? '';
        inp.addEventListener('input', () => commit(inp.value));
        const btn = document.createElement('button');
        btn.className = 'small';
        btn.textContent = 'Browse…';
        btn.addEventListener('click', async () => {
          const p = f.pathKind === 'directory'
            ? await window.bridge.chooseDirectory({ title: `Choose ${f.key}` })
            : await window.bridge.openFileDialog({ title: `Choose ${f.key}`, filters: f.filters });
          if (p) { inp.value = p; commit(p); }
        });
        ctl.append(inp, btn);
        break;
      }
      case 'paths': {
        ctl.appendChild(makePathsList(f, secState));
        break;
      }
    }
    row.appendChild(ctl);
    return row;  // off-default marks are applied by refreshDefaultMarkers()
  }

  function makeStepper(f, secState) {
    const wrap = document.createElement('span');
    wrap.className = 'stepper';
    const dec = document.createElement('button');
    const inc = document.createElement('button');
    dec.className = inc.className = 'small';
    dec.type = inc.type = 'button';
    dec.textContent = f.logStep ? '÷' : '−';
    inc.textContent = f.logStep ? '×' : '+';
    dec.title = f.logStep ? 'divide by √2' : 'decrement';
    inc.title = f.logStep ? 'multiply by √2' : 'increment';

    const inp = document.createElement('input');
    inp.type = 'number';
    inp.value = secState[f.key];
    if (f.type === 'int') inp.step = '1';
    else inp.step = String(f.step ?? 'any');
    if (f.min != null) inp.min = f.min;
    if (f.max != null) inp.max = f.max;
    inp.readOnly = !!f.readonly;
    if (f.readonly) { dec.disabled = inc.disabled = true; }

    const clamp = (v) => {
      if (f.min != null) v = Math.max(f.min, v);
      if (f.max != null) v = Math.min(f.max, v);
      return v;
    };
    const set = (v) => {
      v = clamp(v);
      if (f.type === 'int') v = Math.round(v);
      inp.value = String(v);
      secState[f.key] = v;
      stateChanged();
    };
    // live sync while typing (no clamping mid-edit); normalize on change
    inp.addEventListener('input', () => {
      const v = Number(inp.value);
      if (!Number.isFinite(v) || f.readonly) return;
      secState[f.key] = f.type === 'int' ? Math.round(v) : v;
      stateChanged();
    });
    inp.addEventListener('change', () => {
      const v = Number(inp.value);
      set(Number.isFinite(v) ? v : f.default);
    });
    const step = (dir) => {
      const cur = Number(inp.value) || f.default;
      if (f.logStep) set(roundSig(dir > 0 ? cur * SQRT2 : cur / SQRT2, 3));
      else set(roundSig(cur + dir * (f.step ?? 1), 6));
    };
    dec.addEventListener('click', () => step(-1));
    inc.addEventListener('click', () => step(+1));

    wrap.append(dec, inp, inc);
    if (f.logStep) {
      const note = document.createElement('span');
      note.className = 'step-note';
      note.textContent = '×√2';
      wrap.appendChild(note);
    }
    return wrap;
  }

  function makePathsList(f, secState) {
    const list = document.createElement('div');
    list.className = 'paths-list';
    const paths = secState[f.key];

    const rebuild = () => {
      list.textContent = '';
      paths.forEach((p, i) => {
        const row = document.createElement('div');
        row.className = 'path-row';
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.value = p;
        inp.placeholder = 'path or glob';
        inp.addEventListener('input', () => { paths[i] = inp.value; stateChanged(); });
        const browse = document.createElement('button');
        browse.className = 'small';
        browse.textContent = 'Browse…';
        browse.addEventListener('click', async () => {
          const sel = await window.bridge.openFileDialog({ title: `Choose ${f.key}`, filters: f.filters });
          if (sel) { paths[i] = sel; inp.value = sel; stateChanged(); }
        });
        const del = document.createElement('button');
        del.className = 'small danger';
        del.textContent = '✕';
        del.title = 'remove path';
        del.disabled = paths.length <= 1;
        del.addEventListener('click', () => { paths.splice(i, 1); rebuild(); stateChanged(); });
        row.append(inp, browse, del);
        list.appendChild(row);
      });
      const add = document.createElement('button');
      add.className = 'small';
      add.textContent = '+ add path';
      add.addEventListener('click', () => { paths.push(''); rebuild(); stateChanged(); });
      list.appendChild(add);
    };
    rebuild();
    return list;
  }

  // ---------- TOML pane -> form ----------

  let debounce = null;
  tomlEl.addEventListener('input', () => {
    tomlDirty = true;
    markUnsaved();
    clearTimeout(debounce);
    debounce = setTimeout(applyTomlPane, 300);
  });

  function applyTomlPane() {
    let parsed;
    try {
      parsed = parseToml(tomlEl.value);
    } catch (e) {
      setError('TOML parse error — form not updated.\n' + (e.message || String(e)).split('\n')[0]);
      return;
    }
    const problems = validateRecipe(parsed);
    if (problems.length) {
      setError('Invalid recipe — form not updated.\n• ' + problems.join('\n• '));
      return;
    }
    setError(null);
    tomlDirty = false;
    state = canonicalize(hydrate(parsed));
    renderForm();
  }

  // ---------- the document ----------

  function baseName(p) {
    return String(p).split(/[\\/]/).pop();
  }

  function setPath(p) {
    currentPath = p;
    const scratch = p != null && p === scratchPath;
    const label = p == null ? 'unsaved recipe' : scratch ? 'untitled' : baseName(p);
    pathEl.textContent = label + (unsaved && p != null ? ' •' : '');
    pathEl.title = scratch ? `unsaved — scratch copy kept in ${p}` : (p || '');
    pathEl.dataset.path = p || '';
    pathEl.dataset.scratch = scratch ? '1' : '';
  }

  function markUnsaved() {
    if (!unsaved) { unsaved = true; setPath(currentPath); }
    // The scratch file isn't a document the user chose, so keep it current
    // rather than nagging: it exists purely so an "unsaved" recipe can run.
    if (currentPath && currentPath === scratchPath) scheduleScratchSave();
  }

  let scratchTimer = null;
  function scheduleScratchSave() {
    clearTimeout(scratchTimer);
    scratchTimer = setTimeout(() => { save().catch(() => {}); }, 400);
  }

  async function save() {
    if (!currentPath) return false;
    await window.bridge.writeTextFile(currentPath, tomlEl.value);
    unsaved = false;
    setPath(currentPath);
    return true;
  }

  async function saveReporting() {
    try {
      return await save();
    } catch (e) {
      setError(`Could not save to ${currentPath}:\n${e.message || e}`);
      return false;
    }
  }

  function loadText(text, { warn = true } = {}) {
    const parsed = parseToml(text);
    const problems = validateRecipe(parsed);
    loading = true;
    try {
      state = canonicalize(hydrate(parsed));
      stateChanged({ rerender: true });
    } finally {
      loading = false;
    }
    if (warn && problems.length) setError('Loaded with warnings:\n• ' + problems.join('\n• '));
  }

  async function openPath(p) {
    const text = await window.bridge.readTextFile(p);
    loadText(text);
    unsaved = false;
    setPath(p);
    await updateSettings({ lastRecipePath: p });
  }

  // ---------- toolbar ----------

  root.querySelector('#rc-new').addEventListener('click', async () => {
    state = defaultRecipe();
    unsaved = false;
    setPath(scratchPath);
    stateChanged({ rerender: true });
    if (await saveReporting()) await updateSettings({ lastRecipePath: scratchPath });
  });

  root.querySelector('#rc-open').addEventListener('click', async () => {
    const p = await window.bridge.openFileDialog({
      title: 'Open recipe',
      filters: [{ name: 'TOML recipe', extensions: ['toml'] }],
      defaultPath: currentPath === scratchPath ? undefined : currentPath,
    });
    if (!p) return;
    try {
      await openPath(p);
    } catch (e) {
      setError(`Could not open ${p}:\n${e.message || e}`);
    }
  });

  root.querySelector('#rc-save').addEventListener('click', () => saveReporting());

  root.querySelector('#rc-saveas').addEventListener('click', async () => {
    // Default next to the lights: that's where the results will land, and it
    // makes the recipe's own paths short and relative.
    const lights = state.lights?.paths?.[0] || '';
    const dir = lights.includes('/') || lights.includes('\\')
      ? lights.slice(0, Math.max(lights.lastIndexOf('/'), lights.lastIndexOf('\\')) + 1)
      : '';
    const p = await window.bridge.saveFileDialog({
      title: 'Save recipe as',
      defaultPath: dir + 'my_recipe.toml',
      filters: [{ name: 'TOML recipe', extensions: ['toml'] }],
    });
    if (!p) return;
    setPath(p);
    if (await saveReporting()) await updateSettings({ lastRecipePath: p });
  });

  root.querySelector('#rc-run').addEventListener('click', async () => {
    if (!(await saveReporting())) return; // the CLI runs the file, not the editor
    window.dispatchEvent(new CustomEvent('tensorez:navigate', {
      detail: { view: 'run', recipePath: currentPath, autostart: true },
    }));
  });

  // ---------- boot ----------

  async function boot() {
    const { paths, values } = await loadSettings();
    if (paths) scratchPath = paths.scratchRecipe;

    // Reopen what was last worked on; fall back to the scratch recipe, and
    // create it from defaults the very first time.
    const candidates = [values.lastRecipePath, scratchPath].filter(Boolean);
    for (const p of candidates) {
      try {
        if (!(await window.bridge.exists(p))) continue;
        await openPath(p);
        return;
      } catch { /* unreadable or unparseable: try the next */ }
    }
    setPath(scratchPath);
    stateChanged({ rerender: true });
    if (!scratchPath) return;
    try {
      await save();
      await updateSettings({ lastRecipePath: scratchPath });
    } catch (e) {
      // an unwritable profile directory shouldn't take the whole app down
      setError(`Could not write the scratch recipe to ${scratchPath}:\n${e.message || e}`);
    }
  }

  // initial paint, then adopt whatever the profile directory holds
  renderForm();
  syncTomlFromState();
  try {
    await boot();
  } catch (e) {
    setError(`Could not restore the last recipe:\n${e.message || e}`);
  }

  // debug/testing hooks
  window.__tensorez = window.__tensorez || {};
  window.__tensorez.recipeState = () => structuredClone(state);
  window.__tensorez.recipePath = () => currentPath;

  return {
    onShow() {},
    getRecipePath: () => currentPath,
  };
}
