// Test runner: node test/run-all.mjs [name-filter]
// Starts one dev server + one browser, runs every test module in sequence.

import { startServer, launchBrowser } from './helper.mjs';

const TESTS = [
  ['recipe', () => import('./recipe.test.mjs')],
  ['run-console', () => import('./run-console.test.mjs')],
  ['results', () => import('./results.test.mjs')],
  ['ser', () => import('./ser.test.mjs')],
  ['real', () => import('./real.test.mjs')],
  ['mfbd', () => import('./mfbd.test.mjs')],
];

const filter = process.argv[2];
const server = await startServer();
const browser = await launchBrowser();

let failed = 0;
for (const [name, load] of TESTS) {
  if (filter && !name.includes(filter)) continue;
  process.stdout.write(`\n=== ${name} ===\n`);
  try {
    const mod = await load();
    await mod.default(browser);
    console.log(`PASS ${name}`);
  } catch (e) {
    failed++;
    console.error(`FAIL ${name}:`, e && e.stack || e);
  }
}

await browser.close();
server.kill();
process.exit(failed ? 1 : 0);
