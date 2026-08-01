// Mock implementation of the window.bridge API (documented in ../../preload.js)
// for running the renderer in a plain browser via dev-server.mjs.
//
// - dialogs        -> window.prompt with sensible defaults
// - filesystem     -> fetch() for server-served paths ("/examples/...",
//                     "/mockrun/..."), plus an in-memory overlay for writes
// - spawnRun       -> replays examples/mock_events.jsonl line by line with
//                     small delays so progress animates. If the recipe path
//                     contains "error", replays the failing-run variant. If it
//                     points at a *.jsonl or log.txt (e.g. a REAL run's
//                     mirrored event stream), that file is replayed instead.
// - fileUrl        -> maps paths so artifact thumbnails resolve: mock events'
//                     relative run_dir goes to the served mockrun dir; real
//                     absolute paths go through the dev server's /fs/ route.

const MOCK_RUN_URL = '/mockrun/run1';

// Map a path the way a browser can fetch it:
//  - server-served URLs pass through
//  - absolute filesystem paths go through the dev server's /fs/ route
//  - anything else (mock events' relative run_dir) -> null (caller maps it)
function toUrl(p) {
  if (/^\/(mockrun|examples|fs)\//.test(p)) return p;
  if (p.startsWith('/')) return '/fs' + p;
  if (/^[A-Za-z]:[\\/]/.test(p)) return '/fs/' + p.replaceAll('\\', '/'); // Windows
  return null;
}

const memFs = new Map(); // path -> text, overlay for writeTextFile

const listeners = { line: new Set(), stderr: new Set(), exit: new Set(), dir: new Set() };
const emit = (kind, payload) => { for (const cb of listeners[kind]) cb(payload); };

const activeRuns = new Map(); // runId -> {cancelled}
let nextId = 1;

async function replay(runId, eventsUrl) {
  const state = activeRuns.get(runId);
  let text;
  try {
    const resp = await fetch(eventsUrl);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    text = await resp.text();
  } catch (err) {
    emit('stderr', { runId, line: `mock: cannot load ${eventsUrl}: ${err}` });
    emit('exit', { runId, code: 127, signal: null });
    activeRuns.delete(runId);
    return;
  }
  const lines = text.split('\n').filter((l) => l.trim());
  let exitCode = 0;
  for (const line of lines) {
    if (state.cancelled) {
      emit('exit', { runId, code: null, signal: 'SIGTERM' });
      activeRuns.delete(runId);
      return;
    }
    // small delay so progress bars animate; slightly longer on stage boundaries
    const isProgress = line.includes('"progress"');
    await new Promise((r) => setTimeout(r, isProgress ? 35 : 90));
    emit('line', { runId, line });
    try {
      const ev = JSON.parse(line);
      if (ev.event === 'error') exitCode = 1;
    } catch {}
  }
  emit('exit', { runId, code: exitCode, signal: null });
  activeRuns.delete(runId);
}

export function installMockBridge() {
  window.bridge = {
    platform: 'mock',

    async openFileDialog(opts = {}) {
      const v = window.prompt(
        (opts.title || 'Open file') + '\n(mock dialog — enter a path)',
        opts.defaultPath || '/examples/jupiter_demo.toml',
      );
      return v || null;
    },

    async saveFileDialog(opts = {}) {
      const v = window.prompt(
        (opts.title || 'Save file') + '\n(mock dialog — enter a path)',
        opts.defaultPath || '/recipes/my_recipe.toml',
      );
      return v || null;
    },

    async chooseDirectory(opts = {}) {
      const v = window.prompt(
        (opts.title || 'Choose directory') + '\n(mock dialog — enter a path)',
        opts.defaultPath || MOCK_RUN_URL,
      );
      return v || null;
    },

    // No real profile directory in the browser: the in-memory overlay stands
    // in for it, so the scratch recipe and settings behave the same way.
    async appPaths() {
      return {
        userData: '/appdata',
        settings: '/appdata/settings.json',
        scratchRecipe: '/appdata/untitled.toml',
      };
    },

    async readTextFile(p) {
      if (memFs.has(p)) return memFs.get(p);
      const url = toUrl(p) ?? p;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`cannot read ${p}: ${resp.status}`);
      return resp.text();
    },

    async writeTextFile(p, text) {
      memFs.set(p, text);
      console.log(`[mock bridge] wrote ${text.length} chars to ${p}`);
    },

    async exists(p) {
      if (memFs.has(p)) return true;
      try { return (await fetch(p, { method: 'HEAD' })).ok; } catch { return false; }
    },

    fileUrl(base, rel) {
      // Real absolute run dirs go through /fs/; mock events' relative run_dir
      // ("output/jupiter_demo/…") maps onto the served mock run dir.
      const b = toUrl(base) ?? MOCK_RUN_URL;
      if (!rel) return b;
      return b.replace(/\/$/, '') + '/' + rel.replace(/^\//, '');
    },

    async spawnRun({ recipePath }) {
      const runId = nextId++;
      activeRuns.set(runId, { cancelled: false });
      const p = recipePath || '';
      let url;
      if (/(\.jsonl|log\.txt)$/.test(p)) {
        url = toUrl(p) ?? p; // replay a real (or hand-picked) event stream
      } else if (/error/i.test(p)) {
        url = '/examples/mock_events_error.jsonl';
      } else {
        url = '/examples/mock_events.jsonl';
      }
      replay(runId, url); // fire and forget
      return runId;
    },

    async killRun(runId) {
      const st = activeRuns.get(runId);
      if (st) st.cancelled = true;
      return !!st;
    },

    onRunLine(cb) { listeners.line.add(cb); return () => listeners.line.delete(cb); },
    onRunStderr(cb) { listeners.stderr.add(cb); return () => listeners.stderr.delete(cb); },
    onRunExit(cb) { listeners.exit.add(cb); return () => listeners.exit.delete(cb); },

    async watchDirectory() { return null; }, // no-op in mock
    async unwatchDirectory() {},
    onDirChanged(cb) { listeners.dir.add(cb); return () => listeners.dir.delete(cb); },
  };
}
