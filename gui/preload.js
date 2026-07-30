// TensoRez GUI — preload. Exposes the *entire* main-process surface as
// `window.bridge`. The renderer never touches Electron APIs directly, and the
// same renderer runs in a plain browser with the mock bridge
// (renderer/js/bridge-mock.js) implementing this identical interface.
//
// ----------------------------------------------------------------------------
// window.bridge API
// ----------------------------------------------------------------------------
// platform: 'electron' | 'mock'
//
// Dialogs (all resolve to an absolute path string, or null on cancel):
//   openFileDialog({title?, filters?, defaultPath?})   -> Promise<string|null>
//   saveFileDialog({title?, filters?, defaultPath?})   -> Promise<string|null>
//   chooseDirectory({title?, defaultPath?})            -> Promise<string|null>
//
// Filesystem:
//   readTextFile(path)         -> Promise<string>       (rejects if missing)
//   writeTextFile(path, text)  -> Promise<void>
//   exists(path)               -> Promise<boolean>
//   fileUrl(base, rel?)        -> string  usable as <img src>. Joins and
//                                 converts to a URL the renderer can load.
//
// CLI runs (JSONL event stream per CONTRACT.md §2):
//   spawnRun({recipePath, cwd?, cacheDir?}) -> Promise<runId>
//   killRun(runId)                          -> Promise<boolean>
//   onRunLine(cb)    cb({runId, line})   one raw stdout line (JSONL) at a time
//   onRunStderr(cb)  cb({runId, line})
//   onRunExit(cb)    cb({runId, code, signal})
//
// Directory watching:
//   watchDirectory(path) -> Promise<watchId|null>
//   unwatchDirectory(watchId) -> Promise<void>
//   onDirChanged(cb) cb({watchId, eventType, filename})
// ----------------------------------------------------------------------------

const { contextBridge, ipcRenderer } = require('electron');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

function on(channel, cb) {
  const handler = (_e, payload) => cb(payload);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
}

contextBridge.exposeInMainWorld('bridge', {
  platform: 'electron',

  openFileDialog: (opts) => ipcRenderer.invoke('dialog:openFile', opts),
  saveFileDialog: (opts) => ipcRenderer.invoke('dialog:saveFile', opts),
  chooseDirectory: (opts) => ipcRenderer.invoke('dialog:chooseDirectory', opts),

  readTextFile: (p) => ipcRenderer.invoke('fs:readText', p),
  writeTextFile: (p, text) => ipcRenderer.invoke('fs:writeText', p, text),
  exists: (p) => ipcRenderer.invoke('fs:exists', p),
  fileUrl: (base, rel) => pathToFileURL(rel ? path.join(base, rel) : base).href,

  spawnRun: (opts) => ipcRenderer.invoke('run:spawn', opts),
  killRun: (runId) => ipcRenderer.invoke('run:kill', runId),
  onRunLine: (cb) => on('run:line', cb),
  onRunStderr: (cb) => on('run:stderr', cb),
  onRunExit: (cb) => on('run:exit', cb),

  watchDirectory: (p) => ipcRenderer.invoke('watch:start', p),
  unwatchDirectory: (id) => ipcRenderer.invoke('watch:stop', id),
  onDirChanged: (cb) => on('watch:changed', cb),
});
