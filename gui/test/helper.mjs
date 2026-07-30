// Shared test harness: starts the dev server on a scratch port, launches
// headless Chromium (SwiftShader for WebGL2), and provides screenshot/assert
// helpers. Tests are plain node scripts using node:assert.

import { spawn } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import { promises as fsp } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

export const GUI = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
// repo root with forward slashes: safe both as a filesystem path (Windows
// accepts /) and inside URLs handed to the mock bridge's /fs/ route
export const REPO = path.resolve(GUI, '..').replaceAll('\\', '/');
export const SCREENSHOTS = path.join(GUI, 'screenshots');
export const PORT = 8199;
export const BASE = `http://localhost:${PORT}`;

export async function startServer() {
  const proc = spawn(process.execPath, [path.join(GUI, 'dev-server.mjs'), String(PORT)], {
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  await once(proc.stdout, 'data'); // "dev server: http://..."
  return proc;
}

export async function launchBrowser() {
  return chromium.launch({
    args: [
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--disable-gpu-sandbox',
    ],
  });
}

export async function newPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.consoleErrors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') page.consoleErrors.push(m.text());
  });
  page.on('pageerror', (e) => page.consoleErrors.push(String(e)));
  return page;
}

export async function gotoApp(page) {
  await page.goto(BASE);
  await page.waitForSelector('body[data-ready]');
}

export async function shoot(page, name) {
  await fsp.mkdir(SCREENSHOTS, { recursive: true });
  const file = path.join(SCREENSHOTS, name);
  await page.screenshot({ path: file, fullPage: false });
  console.log('  screenshot:', path.relative(GUI, file));
}

export function checkNoPageErrors(page) {
  // image 404s in console would surface as failed requests, not pageerrors;
  // we only fail hard on JS errors
  const errs = page.consoleErrors.filter((e) => !/Failed to load resource/.test(e));
  if (errs.length) {
    throw new Error('page had JS errors:\n' + errs.join('\n'));
  }
}
