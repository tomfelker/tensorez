import assert from 'node:assert/strict';
import { gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';

export default async function run(browser) {
  const page = await newPage(browser);
  await gotoApp(page);
  await page.click('.nav-btn[data-view=run]');
  await page.waitForSelector('#run-demo-ok');

  // ---- successful demo run ----
  await page.click('#run-demo-ok');
  // catch the lucky stage mid-flight (progress events land ~35ms apart)
  await page.waitForFunction(() => {
    const bar = document.querySelector('.stage-row[data-stage=lucky] .pbar > div');
    if (!bar) return false;
    const w = parseFloat(bar.style.width) || 0;
    return w > 0 && w < 100;
  }, null, { timeout: 20000 });

  // mid-run assertions
  assert.equal(await page.locator('#run-status').textContent(), 'Running');
  assert.ok(await page.locator('.stage-row[data-stage=darks] .badge.cached').isVisible(),
    'darks stage should show cached badge');
  const lightsTime = await page.locator('.stage-row[data-stage=lights] .stage-time').textContent();
  assert.match(lightsTime, /3\.1 s/, 'completed stage shows its timing');
  await shoot(page, '04-run-mid.png');

  // artifacts appear live before the run is over
  await page.waitForSelector('.stage-row[data-stage=lucky] .artifact-thumb img');

  // deconv stage: latest per-iteration loss message shows inline on the row
  await page.waitForFunction(() => {
    const d = document.querySelector('.stage-row[data-stage=deconv] .stage-detail');
    return d && /torchmfbd loss \d/.test(d.textContent);
  }, null, { timeout: 30000 });
  await shoot(page, '12-deconv-mock-run.png');

  // done
  await page.waitForSelector('.done-banner', { timeout: 30000 });
  assert.equal(await page.locator('#run-status').textContent(), 'Done');
  const thumbs = await page.locator('.artifact-thumb').count();
  assert.ok(thumbs >= 8, `expected >= 8 artifact thumbnails, got ${thumbs}`);
  const chips = await page.locator('.artifact-chip').count();
  assert.ok(chips >= 2, `non-image artifacts should appear as chips, got ${chips}`);
  const rawCount = Number(await page.locator('#run-rawlog-count').textContent());
  assert.ok(rawCount > 50, `raw log should have all lines, got ${rawCount}`);
  // open the raw log, then scroll back up so the done banner is in frame
  await page.click('#run-rawlog-details summary');
  await page.evaluate(() => document.getElementById('views').scrollTo(0, 0));
  await page.evaluate(() => window.scrollTo(0, 0));
  await shoot(page, '05-run-done.png');

  // clicking an artifact opens the zoom viewer
  await page.locator('.artifact-thumb').first().click();
  await page.waitForSelector('#lightbox:not([hidden]) img');
  await page.keyboard.press('Escape');
  assert.ok(await page.locator('#lightbox').isHidden());

  // ---- cancel ----
  await page.click('#run-demo-ok');
  await page.waitForSelector('.stage-row[data-stage=lights]');
  await page.click('#run-cancel');
  await page.waitForFunction(() =>
    document.querySelector('#run-status').textContent === 'Cancelled');
  assert.ok(await page.locator('#run-start').isEnabled());

  // ---- failing run ----
  await page.click('#run-demo-err');
  await page.waitForSelector('.error-banner', { timeout: 30000 });
  assert.equal(await page.locator('#run-status').textContent(), 'Failed');
  assert.match(await page.locator('.error-banner').textContent(), /CUDA out of memory/);
  assert.match(await page.locator('.error-banner').textContent(), /Traceback/);
  assert.ok(await page.locator('.stage-row[data-stage=lucky] .badge.error').isVisible());
  await shoot(page, '06-run-error.png');

  checkNoPageErrors(page);
  await page.close();
}
