import assert from 'node:assert/strict';
import { gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';

export default async function run(browser) {
  const page = await newPage(browser);
  await gotoApp(page);
  await page.click('.nav-btn[data-view=results]');

  assert.equal(await page.locator('#res-dir').inputValue(), '/mockrun/run1',
    'mock run dir should be prefilled');
  await page.click('#res-load');
  await page.waitForSelector('.stage-group');

  // summary card
  const summary = await page.locator('.results-summary').textContent();
  assert.match(summary, /jupiter_demo/);
  assert.match(summary, /local_lucky, lucky_stack_p10, mfbd/); // the products list
  assert.match(summary, /300/);
  assert.match(summary, /89\.4 s/);

  // groups in pipeline order with artifacts
  const groups = await page.locator('.stage-group h3').allTextContents();
  // the output stage copies products out; it declares no artifacts of its own
  assert.deepEqual(groups.map((g) => g.trim().match(/^[a-z_]+/)[0]),
    ['align', 'lucky_scoring', 'local_lucky', 'lucky_stack', 'mfbd']);

  const imgs = await page.locator('.artifact-thumb img').count();
  assert.equal(imgs, 13, `expected 13 image artifacts, got ${imgs}`);
  const chips = await page.locator('.artifact-chip').count();
  assert.equal(chips, 9, `expected 9 non-image artifact chips, got ${chips}`);

  // all thumbnails actually load (no broken images)
  await page.waitForFunction(() => {
    const imgs = [...document.querySelectorAll('.artifact-thumb img')];
    return imgs.length && imgs.every((i) => i.complete);
  });
  const broken = await page.locator('.artifact-thumb img').evaluateAll(
    (els) => els.filter((i) => i.naturalWidth === 0).map((i) => i.src));
  assert.deepEqual(broken, [], 'thumbnails failed to load');

  await shoot(page, '07-results-gallery.png');

  // ---- full-size viewer with pow2 zoom + pan ----
  await page.locator('.artifact-thumb', { hasText: 'mfbd' }).last().click();
  await page.waitForSelector('#lightbox:not([hidden]) img');
  await page.waitForFunction(() => {
    const img = document.querySelector('#lightbox img');
    return img && img.complete && img.naturalWidth > 0;
  });
  await page.click('[data-lb=in]');
  await page.click('[data-lb=in]');
  assert.equal(await page.locator('.lb-zoom').textContent(), '×4');
  // pan by dragging
  const canvas = page.locator('#lightbox .lb-canvas');
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 150, box.y + box.height / 2 - 80);
  await page.mouse.up();
  await shoot(page, '08-results-zoom.png');
  await page.keyboard.press('Escape');
  assert.ok(await page.locator('#lightbox').isHidden());

  // ---- missing manifest shows a helpful error ----
  await page.fill('#res-dir', '/mockrun/nope');
  await page.click('#res-load');
  await page.waitForSelector('#res-error .error-banner');
  assert.match(await page.locator('#res-error').textContent(), /manifest\.json/);

  checkNoPageErrors(page);
  await page.close();
}
