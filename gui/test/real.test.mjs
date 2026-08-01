// Integration against REAL CLI output (no mocks): loads the manifest of an
// actual `python -m tensorez run` and replays its mirrored event stream
// (log.txt) through the run console. The latest completed run under the CLI
// output dir is discovered at test time; expected artifact counts are derived
// from its own manifest so CLI re-runs don't break the test.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { REPO, gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';

const IMAGE_KINDS = new Set(['preview', 'sequence_frame']);

// Latest timestamped run dir (containing a manifest) under a tensorez_runs/.
export function latestRun(base) {
  if (!fs.existsSync(base)) return null;
  const runs = fs.readdirSync(base)
    .map((d) => `${base}/${d}`) // forward slashes: these paths also go into /fs/ URLs
    .filter((d) => fs.existsSync(path.join(d, 'manifest.json')))
    .sort();
  return runs.at(-1) ?? null;
}

export function expectations(runDir) {
  const manifest = JSON.parse(fs.readFileSync(path.join(runDir, 'manifest.json'), 'utf8'));
  const imgs = manifest.artifacts.filter((a) => IMAGE_KINDS.has(a.kind)).length;
  const chips = manifest.artifacts.length - imgs;
  const logLines = fs.readFileSync(path.join(runDir, 'log.txt'), 'utf8')
    .split('\n').filter((l) => l.trim()).length;
  return { manifest, imgs, chips, logLines };
}

export default async function run(browser) {
  const REAL_RUN = latestRun(`${REPO}/cli/examples/tensorez_runs/jupiter`);
  assert.ok(REAL_RUN,
    'no completed jupiter run found — from cli/examples, run: python -m tensorez run jupiter.toml');
  const exp = expectations(REAL_RUN);

  const page = await newPage(browser);
  await gotoApp(page);

  // ---------- Results view on the real run dir ----------
  await page.click('.nav-btn[data-view=results]');
  await page.fill('#res-dir', REAL_RUN);
  await page.click('#res-load');
  await page.waitForSelector('.stage-group');

  const summary = await page.locator('.results-summary').textContent();
  assert.match(summary, /jupiter/);
  assert.match(summary, /local_lucky/); // the products list
  assert.match(summary, new RegExp(String(exp.manifest.run.frame_count)));

  const groups = await page.locator('.stage-group h3').allTextContents();
  const expectedGroups = [...new Set(exp.manifest.artifacts.map((a) => a.stage))];
  assert.deepEqual(groups.map((g) => g.trim().match(/^[a-z_]+/)[0]).sort(),
    expectedGroups.sort());

  await page.waitForFunction(() => {
    const imgs = [...document.querySelectorAll('.artifact-thumb img')];
    return imgs.length && imgs.every((i) => i.complete);
  });
  const imgCount = await page.locator('.artifact-thumb img').count();
  const chipCount = await page.locator('.artifact-chip').count();
  assert.equal(imgCount, exp.imgs, `expected ${exp.imgs} image artifacts, got ${imgCount}`);
  assert.equal(chipCount, exp.chips, `expected ${exp.chips} non-image artifacts, got ${chipCount}`);
  const broken = await page.locator('.artifact-thumb img').evaluateAll(
    (els) => els.filter((i) => i.naturalWidth === 0).map((i) => i.src));
  assert.deepEqual(broken, [], 'real thumbnails failed to load');
  await shoot(page, '11-real-results.png');

  // full-size zoom on a real product preview
  await page.locator('.artifact-thumb', { hasText: 'local_lucky' }).first().click();
  await page.waitForFunction(() => {
    const img = document.querySelector('#lightbox img');
    return img && img.complete && img.naturalWidth > 0;
  });
  await page.click('[data-lb=in]');
  assert.equal(await page.locator('.lb-zoom').textContent(), '×2');
  await shoot(page, '11-real-results-zoom.png');
  await page.keyboard.press('Escape');

  // ---------- Run console replaying the real event stream ----------
  await page.click('.nav-btn[data-view=run]');
  await page.fill('#run-recipe', REAL_RUN + '/log.txt');
  await page.click('#run-start');

  // catch any stage mid-flight (which stages emit progress depends on caching)
  await page.waitForFunction(() => {
    return [...document.querySelectorAll('.stage-row .pbar > div')].some((bar) => {
      const w = parseFloat(bar.style.width) || 0;
      return w > 0 && w < 100;
    });
  }, null, { timeout: 20000 });
  assert.equal(await page.locator('#run-status').textContent(), 'Running');
  await shoot(page, '11-real-run-mid.png');

  await page.waitForSelector('.done-banner', { timeout: 60000 });
  assert.equal(await page.locator('#run-status').textContent(), 'Done');
  // real lucky stage_start carries an extra pass1_cached field, and unknown
  // events like validate_result may appear — both must be ignored, not crash
  checkNoPageErrors(page);

  const thumbs = await page.locator('#view-run .artifact-thumb').count();
  assert.equal(thumbs, exp.imgs, `expected ${exp.imgs} live thumbnails, got ${thumbs}`);
  const chips = await page.locator('#view-run .artifact-chip').count();
  assert.equal(chips, exp.chips, `expected ${exp.chips} chips from real events, got ${chips}`);
  const rawCount = Number(await page.locator('#run-rawlog-count').textContent());
  assert.ok(rawCount >= exp.logLines,
    `all ${exp.logLines} real lines should be logged, got ${rawCount}`);
  // real thumbnails from artifact events actually resolved
  const brokenLive = await page.locator('#view-run .artifact-thumb img').evaluateAll(
    (els) => els.filter((i) => !i.complete || i.naturalWidth === 0).length);
  assert.equal(brokenLive, 0, 'live artifact thumbnails must load from the real run dir');

  await page.evaluate(() => document.getElementById('views').scrollTo(0, 0));
  await page.evaluate(() => window.scrollTo(0, 0));
  await shoot(page, '11-real-run-done.png');

  checkNoPageErrors(page);
  await page.close();
}
