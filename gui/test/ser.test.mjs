import assert from 'node:assert/strict';
import fs from 'node:fs';
import { REPO, gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';

const SER_FILE = `${REPO}/examples/synthetic_planet.ser`;

// Copy the WebGL canvas into a 2D canvas and count non-black pixels.
async function nonBlackPixels(page) {
  return page.evaluate(() => {
    const src = document.querySelector('#ser-canvas');
    const c = document.createElement('canvas');
    c.width = src.width; c.height = src.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(src, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let n = 0, maxV = 0;
    for (let i = 0; i < d.length; i += 4) {
      const v = Math.max(d[i], d[i + 1], d[i + 2]);
      if (v > 12) n++;
      if (v > maxV) maxV = v;
    }
    return { nonBlack: n, max: maxV, total: d.length / 4 };
  });
}

export default async function run(browser) {
  assert.ok(fs.existsSync(SER_FILE),
    `${SER_FILE} missing — run \`python examples/gen_synthetic.py\` (or the CLI tests) first`);
  const page = await newPage(browser);
  await gotoApp(page);
  await page.click('.nav-btn[data-view=ser]');

  await page.setInputFiles('#ser-file', SER_FILE);
  await page.waitForFunction(() => (window.__tensorez?.serFrameDrawn || 0) >= 1,
    null, { timeout: 20000 });

  // header parsed correctly
  const header = await page.evaluate(() => {
    const h = window.__tensorez.serHeader;
    return { colorName: h.colorName, width: h.width, height: h.height,
             frameCount: h.frameCount, depth: h.pixelDepthPerPlane, channels: h.channels };
  });
  assert.deepEqual(header, {
    colorName: 'RGB', width: 256, height: 256, frameCount: 60, depth: 16, channels: 3,
  });
  const info = await page.locator('#ser-info').textContent();
  assert.match(info, /RGB \(100\)/);
  assert.match(info, /256 × 256/);
  assert.match(info, /16-bit, 3 ch/);
  assert.match(info, /LUCAM-RECORDER/);

  // the canvas must actually show the planet (not all-black)
  const px = await nonBlackPixels(page);
  assert.ok(px.nonBlack > 5000,
    `expected non-trivial pixels, got ${px.nonBlack}/${px.total} (max ${px.max})`);
  assert.equal(await page.locator('#ser-frame-label').textContent(), '1 / 60');
  await shoot(page, '09-ser-player.png');

  // scrub to a later frame
  await page.locator('#ser-scrub').evaluate((el) => {
    el.value = '30';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForFunction(() =>
    document.querySelector('#ser-frame-label').textContent === '31 / 60');
  const px30 = await nonBlackPixels(page);
  assert.ok(px30.nonBlack > 5000, 'frame 30 should render too');

  // play: frames advance and fps readout appears
  await page.click('#ser-play');
  await page.waitForFunction(() =>
    /fps/.test(document.querySelector('#ser-fps').textContent), null, { timeout: 10000 });
  await page.waitForTimeout(600);
  await page.click('#ser-play'); // pause
  const label = await page.locator('#ser-frame-label').textContent();
  assert.notEqual(label, '31 / 60', 'playback should have advanced frames');

  // pow2 zoom + pan
  const zoomBefore = await page.locator('#ser-zoom-label').textContent();
  await page.click('#ser-zoom-in');
  await page.click('#ser-zoom-in');
  const zoomAfter = await page.locator('#ser-zoom-label').textContent();
  assert.notEqual(zoomAfter, zoomBefore);
  assert.match(zoomAfter, /^×[\d/]+$/);
  const canvas = page.locator('#ser-canvas');
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 60);
  await page.mouse.up();
  const pxZoom = await nonBlackPixels(page);
  assert.ok(pxZoom.nonBlack > 5000, 'zoomed view should still show the planet');
  await shoot(page, '10-ser-zoom.png');

  checkNoPageErrors(page);
  await page.close();
}
