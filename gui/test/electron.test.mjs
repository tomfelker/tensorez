// The one suite that exercises the Electron main process.
//
// Every other test drives the renderer in plain Chromium against the mock
// bridge, which is deliberate — but it means main.js and preload.js could break
// completely and the whole suite would still pass. That gap is exactly what
// makes an Electron upgrade risky, so this launches the real app and
// round-trips the things only the main process can do.
//
// Runs against a throwaway --user-data-dir: the app writes a scratch recipe on
// boot, and that must not land on the real one someone has open.
//
// Note this pops a real window for a few seconds; Electron has no headless mode.

import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { promises as fsp } from 'node:fs';
import { createRequire } from 'node:module';
import { _electron as electron } from 'playwright';
import { GUI } from './helper.mjs';

const require = createRequire(import.meta.url);

export default async function run() {
  const userDataDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'tensorez-electron-'));
  const app = await electron.launch({
    executablePath: require('electron'),
    args: [GUI, `--user-data-dir=${userDataDir}`],
    // the spawn path shells out to TENSOREZ_PYTHON; node rejects the CLI's
    // argv and exits immediately, which exercises it without needing Python
    env: { ...process.env, TENSOREZ_PYTHON: process.execPath },
  });

  try {
    const page = await app.firstWindow();
    await page.waitForSelector('body[data-ready]');
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));

    // the real preload is in play, not the mock the other suites use
    assert.equal(await page.evaluate(() => window.bridge.platform), 'electron',
      'the Electron preload must be what the renderer sees');

    // app:paths — a plain ipcMain.handle round trip, and proof the throwaway
    // profile took effect rather than the developer's real one
    const paths = await page.evaluate(() => window.bridge.appPaths());
    assert.ok(paths.userData.startsWith(userDataDir.slice(0, 8)) ||
      path.resolve(paths.userData).startsWith(path.resolve(userDataDir)),
      `userData ${paths.userData} should be under the test profile ${userDataDir}`);
    assert.equal(path.basename(paths.scratchRecipe), 'untitled.toml');

    // the scratch recipe really is written on boot, by the main process
    assert.match(await fsp.readFile(paths.scratchRecipe, 'utf8'), /^\[lights\]/,
      'boot writes a runnable scratch recipe');

    // fs bridge: write -> exists -> read, all over IPC
    const scratch = path.join(userDataDir, 'probe.toml');
    const wrote = await page.evaluate(async (p) => {
      await window.bridge.writeTextFile(p, '[lights]\npaths = ["x.ser"]\n');
      return { exists: await window.bridge.exists(p), text: await window.bridge.readTextFile(p) };
    }, scratch);
    assert.equal(wrote.exists, true);
    assert.match(wrote.text, /x\.ser/);
    assert.equal(await page.evaluate((p) => window.bridge.exists(p + '.nope'), scratch), false);

    // fileUrl is computed in the preload, where node:url actually exists
    assert.match(await page.evaluate((p) => window.bridge.fileUrl(p), scratch), /^file:\/\//);

    // main -> renderer push: spawn, line forwarding, and exit notification.
    // This is the one mechanism a request/response round trip does not cover.
    const spawned = await page.evaluate(() => new Promise((resolve) => {
      const seen = { stderr: 0, exit: undefined };
      window.bridge.onRunStderr(() => { seen.stderr++; });
      window.bridge.onRunExit((e) => { seen.exit = e.code; resolve(seen); });
      window.bridge.spawnRun({ recipePath: 'nonexistent.toml' });
      setTimeout(() => resolve(seen), 15000);
    }));
    assert.notEqual(spawned.exit, undefined, 'run:exit must reach the renderer');
    assert.ok(spawned.stderr > 0, 'child stderr must be forwarded to the renderer');

    // killing an unknown run is a clean false, not a throw
    assert.equal(await page.evaluate(() => window.bridge.killRun(9999)), false);

    // directory watching starts and stops without error
    const watchId = await page.evaluate((d) => window.bridge.watchDirectory(d), userDataDir);
    assert.ok(watchId !== null, 'watchDirectory should return an id for a real directory');
    await page.evaluate((id) => window.bridge.unwatchDirectory(id), watchId);
    assert.equal(await page.evaluate(() => window.bridge.watchDirectory('/no/such/dir/here')), null,
      'watching a missing directory returns null rather than throwing');

    assert.deepEqual(errors, [], 'no renderer JS errors under the real bridge');
  } finally {
    await app.close();
    await fsp.rm(userDataDir, { recursive: true, force: true }).catch(() => {});
  }
}
