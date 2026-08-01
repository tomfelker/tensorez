// TensoRez GUI — Electron main process.
//
// DELIBERATELY THIN. Everything interesting lives in the renderer, which also
// runs in a plain browser against a mock bridge (see renderer/js/bridge-mock.js).
// The main process only provides what a browser cannot:
//   - native open/save dialogs
//   - filesystem read/write
//   - spawning/killing the tensorez CLI and forwarding its stdout lines
//   - directory watching
// The full bridge surface is documented in preload.js.

'use strict';

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const readline = require('node:readline');

const runs = new Map(); // runId -> ChildProcess
const watchers = new Map(); // watchId -> fs.FSWatcher
let nextId = 1;

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: '#14171c',
    title: 'TensoRez',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  return win;
}

// ---------- dialogs ----------

ipcMain.handle('dialog:openFile', async (e, opts = {}) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  const r = await dialog.showOpenDialog(win, {
    title: opts.title || 'Open',
    properties: ['openFile'],
    filters: opts.filters || [],
    defaultPath: opts.defaultPath,
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('dialog:saveFile', async (e, opts = {}) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  const r = await dialog.showSaveDialog(win, {
    title: opts.title || 'Save',
    filters: opts.filters || [],
    defaultPath: opts.defaultPath,
  });
  return r.canceled ? null : r.filePath;
});

ipcMain.handle('dialog:chooseDirectory', async (e, opts = {}) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  const r = await dialog.showOpenDialog(win, {
    title: opts.title || 'Choose directory',
    properties: ['openDirectory', 'createDirectory'],
    defaultPath: opts.defaultPath,
  });
  return r.canceled ? null : r.filePaths[0];
});

// ---------- app paths ----------
// The renderer edits recipes like a normal document app: there is always a
// real file behind the editor (the CLI can only run a file), so an unsaved
// recipe lives in a scratch file in the user's profile directory until they
// Save As somewhere they care about. Settings live beside it.

ipcMain.handle('app:paths', () => {
  const userData = app.getPath('userData');
  fs.mkdirSync(userData, { recursive: true });
  return {
    userData,
    settings: path.join(userData, 'settings.json'),
    scratchRecipe: path.join(userData, 'untitled.toml'),
  };
});

// ---------- filesystem ----------

ipcMain.handle('fs:readText', (e, p) => fsp.readFile(p, 'utf8'));
ipcMain.handle('fs:writeText', (e, p, text) => fsp.writeFile(p, text, 'utf8'));
ipcMain.handle('fs:exists', async (e, p) => {
  try { await fsp.access(p); return true; } catch { return false; }
});

// ---------- CLI subprocess ----------
// `python -m tensorez run <recipe.toml>` with configurable cwd.
// stdout is the JSONL event stream; lines are forwarded verbatim to the
// renderer, which owns all parsing (so mock and real paths share code).

ipcMain.handle('run:spawn', (e, { recipePath, cwd, runsDir, cacheDir }) => {
  const runId = nextId++;
  // --events: the CLI's default output is human-readable; we want the JSONL
  // event stream (the GUI is the consumer that needs it).
  const args = ['-m', 'tensorez', 'run', recipePath, '--events'];
  if (runsDir) args.push('--runs-dir', runsDir);
  if (cacheDir) args.push('--cache-dir', cacheDir);
  // The CLI resolves the recipe's relative paths against its working
  // directory; defaulting that to the recipe's own folder is what makes
  // "recipe saved beside the .ser files" work with bare filenames.
  const child = spawn(process.env.TENSOREZ_PYTHON || 'python', args, {
    cwd: cwd || path.dirname(recipePath),
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  runs.set(runId, child);
  const wc = e.sender;
  const send = (ch, payload) => { if (!wc.isDestroyed()) wc.send(ch, payload); };

  readline.createInterface({ input: child.stdout }).on('line', (line) => {
    send('run:line', { runId, line });
  });
  readline.createInterface({ input: child.stderr }).on('line', (line) => {
    send('run:stderr', { runId, line });
  });
  child.on('error', (err) => send('run:stderr', { runId, line: String(err) }));
  child.on('close', (code, signal) => {
    runs.delete(runId);
    send('run:exit', { runId, code, signal });
  });
  return runId;
});

ipcMain.handle('run:kill', (e, runId) => {
  const child = runs.get(runId);
  if (child) child.kill('SIGTERM');
  return !!child;
});

// ---------- directory watching ----------

ipcMain.handle('watch:start', (e, dirPath) => {
  const watchId = nextId++;
  const wc = e.sender;
  try {
    const w = fs.watch(dirPath, { recursive: true }, (eventType, filename) => {
      if (!wc.isDestroyed()) wc.send('watch:changed', { watchId, eventType, filename });
    });
    watchers.set(watchId, w);
    return watchId;
  } catch (err) {
    return null;
  }
});

ipcMain.handle('watch:stop', (e, watchId) => {
  const w = watchers.get(watchId);
  if (w) { w.close(); watchers.delete(watchId); }
});

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  for (const child of runs.values()) child.kill('SIGTERM');
  for (const w of watchers.values()) w.close();
  if (process.platform !== 'darwin') app.quit();
});
