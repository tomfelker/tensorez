// [mfbd] stage verification against REAL CLI data:
//   - recipe editor round-trip on the CLI's jupiter_mfbd.toml (sparse file)
//   - run console replay of the real mfbd run's log.txt
//   - results view on the real mfbd run dir

import assert from 'node:assert/strict';
import { REPO, gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';
import { latestRun, expectations } from './real.test.mjs';

const MFBD_RECIPE = `${REPO}/cli/examples/jupiter_mfbd.toml`;

export default async function run(browser) {
  const MFBD_RUN = latestRun(`${REPO}/cli/examples/tensorez_runs/jupiter_mfbd`);
  assert.ok(MFBD_RUN, 'no completed jupiter_mfbd run found — from cli/examples, run: ' +
    'python -m tensorez run jupiter_mfbd.toml');
  const exp = expectations(MFBD_RUN);

  const page = await newPage(browser);
  await gotoApp(page);

  // ---------- recipe editor: open the real (sparse) deconv recipe ----------
  page.once('dialog', (d) => d.accept(MFBD_RECIPE));
  await page.click('#rc-open');
  await page.waitForFunction((p) =>
    document.querySelector('#rc-path').dataset.path === p, MFBD_RECIPE);

  // deconv section is enabled and populated
  assert.ok(await page.locator('input[data-section-toggle=mfbd]').isChecked(),
    'deconv section toggle should be on');
  // C11 default hydrated for the omitted diameter
  assert.equal(await page
    .locator('[data-field="mfbd.diameter_cm"] .stepper input').inputValue(), '27.94');
  assert.equal(await page
    .locator('[data-field="mfbd.n_modes"] select').inputValue(), '20');
  assert.equal(await page
    .locator('[data-field="mfbd.wavelengths_nm"] input').inputValue(), '700, 530, 470');
  assert.equal(await page
    .locator('[data-field="mfbd.top_n"] .stepper input').inputValue(), '10');
  // sparse file: defaults were hydrated without complaint
  assert.ok(!(await page.locator('#rc-toml-error.show').isVisible()),
    'sparse CLI recipe must load without errors');
  const toml = page.locator('#rc-toml');
  assert.match(await toml.inputValue(), /method = "torchmfbd"/);
  assert.match(await toml.inputValue(), /frequency_cutoff = \[ 0\.2, 0\.3 \]/);

  // ---- pixel scale: file uses camera keys -> camera mode auto-selected ----
  const psField = page.locator('[data-field="mfbd.pixel_scale_arcsec"]');
  assert.equal(await psField.locator('[data-role=ps-mode]').inputValue(), 'camera');
  assert.equal(await psField.locator('[data-role="ps-focal_length_mm"]').inputValue(), '2800');
  assert.equal(await psField.locator('[data-role="ps-barlow"]').inputValue(), '2');
  assert.equal(await psField.locator('[data-role="ps-pixel_size_um"]').inputValue(), '4.3');
  // computed read-only scale: 206.265 * 4.3 / (2800 * 2) = 0.158
  assert.match(await psField.locator('[data-role=ps-computed]').textContent(), /0\.158/);
  // camera keys serialized, pixel_scale_arcsec NOT (either/or, never both)
  assert.match(await toml.inputValue(), /pixel_size_um = 4\.3/);
  assert.match(await toml.inputValue(), /focal_length_mm = 2800/);
  assert.match(await toml.inputValue(), /barlow = 2/);
  assert.doesNotMatch(await toml.inputValue(), /pixel_scale_arcsec/);

  // round-trip: TOML pane parses back to exactly the form state
  const roundTrip = () => page.evaluate(async () => {
    const { parse } = await import('./vendor/index.js');
    const a = parse(document.querySelector('#rc-toml').value);
    const b = window.__tensorez.recipeState();
    return JSON.stringify(a) === JSON.stringify(b);
  });
  assert.ok(await roundTrip(), 'camera-mode deconv recipe must round-trip form <-> TOML');

  // screenshot the as-opened state (camera mode, barlow 2 -> 0.158″/px)
  await page.locator('.section-card[data-section=mfbd]').scrollIntoViewIfNeeded();
  await shoot(page, '12-deconv-recipe.png');

  // ---- switch to direct mode: carries the computed value, swaps the keys ----
  await psField.locator('[data-role=ps-mode]').selectOption('direct');
  assert.match(await toml.inputValue(), /pixel_scale_arcsec = 0\.158/);
  assert.doesNotMatch(await toml.inputValue(), /pixel_size_um|focal_length_mm|barlow/);
  assert.ok(await roundTrip(), 'direct-mode recipe must round-trip form <-> TOML');
  // and back to camera mode (defaults restored)
  await page.locator('[data-field="mfbd.pixel_scale_arcsec"] [data-role=ps-mode]')
    .selectOption('camera');
  assert.match(await toml.inputValue(), /pixel_size_um/);
  assert.doesNotMatch(await toml.inputValue(), /pixel_scale_arcsec/);
  assert.ok(await roundTrip(), 'restored camera mode must round-trip form <-> TOML');
  // editing a camera field updates the computed readout and the TOML
  const pxInput = page
    .locator('[data-field="mfbd.pixel_scale_arcsec"] [data-role="ps-pixel_size_um"]');
  await pxInput.fill('2.9');
  assert.match(await page
    .locator('[data-field="mfbd.pixel_scale_arcsec"] [data-role=ps-computed]')
    .textContent(), /0\.214/); // 206.265*2.9/2800
  assert.match(await toml.inputValue(), /pixel_size_um = 2\.9/);
  await pxInput.fill('4.3');

  // interact: change n_modes via the constrained dropdown
  await page.locator('[data-field="mfbd.n_modes"] select').selectOption('27');
  assert.match(await toml.inputValue(), /n_modes = 27/);
  await page.locator('[data-field="mfbd.n_modes"] select').selectOption('20');

  // frames=all hides top_n
  await page.locator('[data-field="mfbd.frames"] select').selectOption('all');
  assert.ok(await page.locator('[data-field="mfbd.top_n"]')
    .evaluate((el) => el.classList.contains('hidden-field')), 'top_n hidden for frames=all');
  await page.locator('[data-field="mfbd.frames"] select').selectOption('lucky_top');

  // ---------- run console: replay the real deconv event stream ----------
  await page.click('.nav-btn[data-view=run]');
  await page.fill('#run-recipe', MFBD_RUN + '/log.txt');
  await page.click('#run-start');

  // catch deconv mid-flight with a loss message inline on the row
  await page.waitForFunction(() => {
    const row = document.querySelector('.stage-row[data-stage=mfbd]');
    if (!row) return false;
    const w = parseFloat(row.querySelector('.pbar > div').style.width) || 0;
    return w > 0 && w < 100 &&
      /torchmfbd loss \d/.test(row.querySelector('.stage-detail').textContent);
  }, null, { timeout: 30000 });
  await shoot(page, '12-deconv-run-mid.png');

  await page.waitForSelector('.done-banner', { timeout: 60000 });
  assert.equal(await page.locator('#run-status').textContent(), 'Done');
  checkNoPageErrors(page);

  // psf_examples and lucky_stack rendered as thumbnails from live events
  assert.ok(await page.locator('#view-run .artifact-thumb', { hasText: 'psf_examples' }).isVisible());
  assert.ok(await page.locator('#view-run .artifact-thumb', { hasText: 'lucky_stack' }).isVisible());
  assert.ok(await page.locator('#view-run .artifact-chip', { hasText: 'loss_history' }).isVisible());
  const brokenLive = await page.locator('#view-run .artifact-thumb img').evaluateAll(
    (els) => els.filter((i) => !i.complete || i.naturalWidth === 0).length);
  assert.equal(brokenLive, 0, 'live deconv thumbnails must load');

  await page.evaluate(() => document.getElementById('views').scrollTo(0, 0));
  await page.evaluate(() => window.scrollTo(0, 0));
  await shoot(page, '12-deconv-run-done.png');

  // ---------- results view on the real deconv run dir ----------
  await page.click('.nav-btn[data-view=results]');
  await page.fill('#res-dir', MFBD_RUN);
  await page.click('#res-load');
  await page.waitForSelector('#view-results .stage-group');

  const groups = await page.locator('#view-results .stage-group h3').allTextContents();
  // the output stage only copies products out; it declares no artifacts
  assert.deepEqual(groups.map((g) => g.trim().match(/^[a-z_]+/)[0]),
    ['lucky_scoring', 'lucky_stack', 'mfbd']);

  await page.waitForFunction(() => {
    const imgs = [...document.querySelectorAll('#view-results .artifact-thumb img')];
    return imgs.length && imgs.every((i) => i.complete);
  });
  const imgs = await page.locator('#view-results .artifact-thumb img').count();
  const chips = await page.locator('#view-results .artifact-chip').count();
  assert.equal(imgs, exp.imgs, `expected ${exp.imgs} preview artifacts, got ${imgs}`);
  assert.equal(chips, exp.chips, `expected ${exp.chips} non-image artifacts, got ${chips}`);
  const broken = await page.locator('#view-results .artifact-thumb img').evaluateAll(
    (els) => els.filter((i) => i.naturalWidth === 0).map((i) => i.src));
  assert.deepEqual(broken, [], 'real deconv thumbnails failed to load');
  assert.ok(await page.locator('#view-results .artifact-thumb', { hasText: 'psf_examples' }).isVisible());
  assert.ok(await page.locator('#view-results .artifact-thumb', { hasText: 'lucky_stack' }).isVisible());
  await shoot(page, '12-deconv-results.png');

  checkNoPageErrors(page);
  await page.close();
}
