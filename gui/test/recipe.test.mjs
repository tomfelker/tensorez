import assert from 'node:assert/strict';
import { gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';

export default async function run(browser) {
  const page = await newPage(browser);
  await gotoApp(page);
  await page.click('.nav-btn[data-view=recipe]');
  await page.waitForSelector('#rc-form .section-card');

  const toml = page.locator('#rc-toml');
  const pathEl = page.locator('#rc-path');

  // ---- a fresh app starts on the scratch recipe, already backed by a file ----
  assert.equal(await pathEl.textContent(), 'untitled');
  assert.equal(await pathEl.getAttribute('data-path'), '/appdata/untitled.toml');
  assert.match(await page.evaluate(() => window.bridge.readTextFile('/appdata/untitled.toml')),
    /^\[lights\]/, 'scratch recipe written on first boot');

  // ---- the raw TOML is an escape hatch, not the interface: collapsed ----
  assert.equal(await page.locator('#rc-toml-details').getAttribute('open'), null,
    'raw TOML starts collapsed');
  assert.ok(await toml.isHidden(), 'raw TOML hidden until asked for');
  await page.click('#rc-toml-details summary');
  assert.ok(await toml.isVisible());

  // ---- form -> toml ----
  // log stepper: 35 * sqrt(2) -> 49.5 (3 significant digits)
  const cross = page.locator('[data-field="local_lucky.crossover_wavelength_pixels"]');
  await cross.locator('button[title="multiply by √2"]').click();
  assert.equal(await cross.locator('input').inputValue(), '49.5');
  assert.match(await toml.inputValue(), /crossover_wavelength_pixels = 49\.5/);
  await cross.locator('button[title="divide by √2"]').click();
  assert.equal(await cross.locator('input').inputValue(), '35');

  // int stepper +
  const dbg = page.locator('[data-field="output.debug_frames"]');
  await dbg.locator('button[title=increment]').click();
  assert.match(await toml.inputValue(), /debug_frames = 11/);

  // enable optional darks section
  assert.doesNotMatch(await toml.inputValue(), /\[darks\]/);
  await page.check('input[data-section-toggle=darks]');
  assert.match(await toml.inputValue(), /\[darks\]\npaths = \[ "data\/darks\.ser" \]/);

  // optional field: set end_frame
  const endf = page.locator('[data-field="lights.end_frame"]');
  await endf.locator('.opt-toggle input').check();
  await page.locator('[data-field="lights.end_frame"] .stepper input').fill('250');
  await page.locator('[data-field="lights.end_frame"] .stepper input').press('Tab');
  assert.match(await toml.inputValue(), /end_frame = 250/);

  // enum field: debayer dropdown
  await page.locator('[data-field="lights.debayer"] select').selectOption('superpixel_rgb');
  assert.match(await toml.inputValue(), /debayer = "superpixel_rgb"/);

  // paths list: add a second path
  await page.locator('[data-field="lights.paths"] .paths-list button:has-text("add path")').click();
  const secondPath = page.locator('[data-field="lights.paths"] .path-row').nth(1).locator('input');
  await secondPath.fill('data/jupiter_2.ser');
  await secondPath.press('Tab');
  assert.match(await toml.inputValue(), /"data\/jupiter_2\.ser"/);

  // round-trip: the TOML pane parses back to exactly the form state
  const roundTrip = await page.evaluate(async () => {
    const { parse } = await import('./vendor/index.js');
    const fromToml = parse(document.querySelector('#rc-toml').value);
    const state = window.__tensorez.recipeState();
    return { equal: JSON.stringify(fromToml) === JSON.stringify(state), fromToml, state };
  });
  assert.ok(roundTrip.equal,
    'form state and TOML pane must round-trip:\n' + JSON.stringify(roundTrip, null, 2));

  await page.evaluate(() => document.getElementById('views').scrollTo(0, 0));
  await shoot(page, '01-recipe-editor.png');

  // ---- toml -> form ----
  const newToml = (await toml.inputValue())
    .replace('steepness = 3', 'steepness = 7.5')
    .replace('center_of_mass = true', 'center_of_mass = false');
  await toml.fill(newToml);
  await page.waitForTimeout(500); // debounce
  assert.equal(
    await page.locator('[data-field="local_lucky.steepness"] .stepper input').inputValue(), '7.5');
  assert.equal(
    await page.locator('[data-field="align.center_of_mass"] input[type=checkbox]').isChecked(),
    false);
  await shoot(page, '02-toml-sync.png');

  // ---- invalid TOML: inline error, form untouched ----
  await toml.fill(newToml + '\nthis is [ not toml ===');
  await page.waitForTimeout(500);
  assert.ok(await page.locator('#rc-toml-error.show').isVisible(), 'parse error should show');
  assert.match(await page.locator('#rc-toml-error').textContent(), /form not updated/);
  assert.equal(
    await page.locator('[data-field="local_lucky.steepness"] .stepper input').inputValue(),
    '7.5', 'form must not be clobbered');
  await shoot(page, '03-toml-error.png');

  // ---- unknown key (contract hard error) also rejected ----
  await toml.fill(newToml.replace('[output]', 'typo_key = 3\n[output]'));
  await page.waitForTimeout(500);
  assert.match(await page.locator('#rc-toml-error').textContent(),
    /unknown key local_lucky\.typo_key/);
  // fix it again -> error clears
  await toml.fill(newToml);
  await page.waitForTimeout(500);
  assert.ok(!(await page.locator('#rc-toml-error.show').isVisible()), 'error should clear');

  // ---- Open via bridge dialog (mock uses window.prompt) ----
  page.once('dialog', (d) => d.accept('/examples/jupiter_demo.toml'));
  await page.click('#rc-open');
  await page.waitForFunction(() =>
    document.querySelector('#rc-path').dataset.path === '/examples/jupiter_demo.toml');
  assert.equal(await pathEl.textContent(), 'jupiter_demo.toml', 'shows the file name');
  assert.equal(
    await page.locator('[data-field="local_lucky.isoplanatic_patch_pixels"] .stepper input')
      .inputValue(), '55');
  // the opened file is remembered for next launch
  const settings = JSON.parse(await page.evaluate(() =>
    window.bridge.readTextFile('/appdata/settings.json')));
  assert.equal(settings.lastRecipePath, '/examples/jupiter_demo.toml');

  // ---- editing marks unsaved; Save writes to the remembered path ----
  const dbg2 = page.locator('[data-field="output.debug_frames"]');
  await dbg2.locator('button[title=increment]').click();
  assert.equal(await pathEl.textContent(), 'jupiter_demo.toml •', 'unsaved marker');
  await page.click('#rc-save');
  await page.waitForFunction(() =>
    document.querySelector('#rc-path').textContent === 'jupiter_demo.toml');
  const saved = await page.evaluate(async () =>
    window.bridge.readTextFile('/examples/jupiter_demo.toml'));
  assert.match(saved, /debug_frames = 11/);

  // ---- Save As relocates the document ----
  page.once('dialog', (d) => d.accept('/captures/iss_pass.toml'));
  await page.click('#rc-saveas');
  await page.waitForFunction(() =>
    document.querySelector('#rc-path').dataset.path === '/captures/iss_pass.toml');
  assert.match(await page.evaluate(() => window.bridge.readTextFile('/captures/iss_pass.toml')),
    /debug_frames = 11/);

  // ---- New goes back to the scratch file, which is written immediately ----
  await page.click('#rc-new');
  assert.equal(await pathEl.textContent(), 'untitled');
  assert.equal(await pathEl.getAttribute('data-path'), '/appdata/untitled.toml');
  assert.match(await page.evaluate(() => window.bridge.readTextFile('/appdata/untitled.toml')),
    /\[local_lucky\]/);
  assert.doesNotMatch(await toml.inputValue(), /^name = /m, '[recipe] has no name key');

  // ---- Run saves and hands the path to the run console ----
  await page.click('#rc-run');
  await page.waitForFunction(() =>
    document.querySelector('#run-recipe')?.value === '/appdata/untitled.toml');
  assert.ok(await page.locator('#view-run').isVisible(), 'Run switches to the run view');
  await page.click('#run-cancel');
  await page.click('.nav-btn[data-view=recipe]');

  // ---- align.per_channel + constraint hints ----
  const pcRow = page.locator('[data-field="align.per_channel"]');
  const pcBox = pcRow.locator('input[type=checkbox]');
  const comBox = page.locator('[data-field="align.center_of_mass"] input[type=checkbox]');
  const errBox = page.locator('#rc-toml-error');

  assert.match(await toml.inputValue(), /per_channel = false/, 'default serialized');
  await pcBox.check();
  assert.match(await toml.inputValue(), /per_channel = true/);

  // round-trip with the new key
  const rt2 = await page.evaluate(async () => {
    const { parse } = await import('./vendor/index.js');
    return JSON.stringify(parse(document.querySelector('#rc-toml').value)) ===
           JSON.stringify(window.__tensorez.recipeState());
  });
  assert.ok(rt2, 'per_channel must round-trip form <-> TOML');

  // conflicting combo from the form: warning note + validation problem
  // (warned while per_channel is on, disabled once it's off)
  await comBox.uncheck();
  assert.ok(await pcRow.locator('.field-warn').isVisible(), 'conflict warning shown');
  assert.match(await errBox.textContent(), /requires center_of_mass/);
  await page.locator('[data-section="align"]').scrollIntoViewIfNeeded();
  await shoot(page, '13-recipe-per-channel-conflict.png');
  await pcBox.uncheck();
  assert.ok(!(await errBox.isVisible()), 'validation clears when combo is fixed');
  assert.ok(await pcBox.isDisabled(), 'cannot enable per_channel without center_of_mass');
  await comBox.check();
  assert.ok(await pcBox.isEnabled(), 're-enabled with center_of_mass');

  // TOML -> form: a hand-authored hard-error combo is rejected like any other
  // invalid recipe (same class as unknown keys — the CLI refuses it)
  const conflictToml = (await toml.inputValue())
    .replace('per_channel = false', 'per_channel = true')
    .replace('center_of_mass = true', 'center_of_mass = false');
  await toml.fill(conflictToml);
  await page.waitForTimeout(500);
  assert.match(await errBox.textContent(), /form not updated/);
  assert.match(await errBox.textContent(), /requires center_of_mass/);

  // reset for the align-card screenshot refresh
  await page.click('#rc-new');
  await page.evaluate(() => document.getElementById('views').scrollTo(0, 0));

  // ---- off-default values are marked, and revertible ----
  // A fresh recipe is all defaults by construction, so nothing may be marked.
  assert.equal(await page.locator('#rc-form .field-modified').count(), 0,
    'a new recipe has nothing marked as changed');
  assert.equal(await page.locator('#rc-form .reset-section').count(), 0);

  // a plain field: gutter mark, revert control, and a section badge counting it
  const dbgRow = page.locator('[data-field="output.debug_frames"]');
  await dbgRow.locator('button[title=increment]').click();
  assert.ok(await dbgRow.locator('.revert-btn').isVisible(), 'revert offered once changed');
  assert.match(await dbgRow.locator('.revert-btn').getAttribute('title'), /\(10\)/,
    'tooltip names the default it would restore');
  assert.equal(await page.locator('[data-section="output"] .reset-section').textContent(),
    '↺ 1 changed');
  await dbgRow.locator('.revert-btn').click();
  assert.equal(await dbgRow.locator('.stepper input').inputValue(), '10');
  assert.equal(await page.locator('[data-field="output.debug_frames"].field-modified').count(), 0,
    'reverting clears the mark');
  assert.equal(await page.locator('[data-section="output"] .reset-section').count(), 0);

  // an optional key's default is ABSENCE, so setting it at all is a change and
  // reverting takes it back out of the TOML rather than to some value
  const endRow = page.locator('[data-field="lights.end_frame"]');
  await endRow.locator('.opt-toggle input').check();
  assert.ok(await endRow.locator('.revert-btn').isVisible(), 'a set optional key is a change');
  assert.match(await endRow.locator('.revert-btn').getAttribute('title'), /unset/);
  assert.match(await toml.inputValue(), /end_frame = /);
  await endRow.locator('.revert-btn').click();
  assert.doesNotMatch(await toml.inputValue(), /end_frame = /, 'reverted to absent');
  assert.equal(await endRow.locator('.opt-toggle input').isChecked(), false);

  // the section badge reverts every changed field in its card at once
  await page.locator('[data-field="lights.debayer"] select').selectOption('none');
  const startInput = page.locator('[data-field="lights.start_frame"] .stepper input');
  await startInput.fill('12');
  await startInput.press('Tab');
  assert.equal(await page.locator('[data-section="lights"] .reset-section').textContent(),
    '↺ 2 changed');
  await page.locator('[data-section="lights"]').scrollIntoViewIfNeeded();
  await shoot(page, '14-recipe-modified-fields.png');
  await page.click('[data-section="lights"] .reset-section');
  assert.equal(await page.locator('[data-section="lights"] .field-modified').count(), 0);
  assert.match(await toml.inputValue(), /debayer = "bilinear"/);
  assert.match(await toml.inputValue(), /start_frame = 0/);

  // the either/or pixel scale: camera mode is a departure in itself, because it
  // serializes different keys, and reverting restores the direct representation
  await page.check('input[data-section-toggle=mfbd]');
  const scaleRow = page.locator('[data-field="mfbd.pixel_scale_arcsec"]');
  assert.equal(await scaleRow.locator('.revert-btn').count(), 0,
    'a freshly enabled section is at its defaults');
  await scaleRow.locator('select[data-role=ps-mode]').selectOption('camera');
  assert.ok(await scaleRow.locator('.revert-btn').isVisible(), 'camera mode is a change');
  assert.match(await toml.inputValue(), /pixel_size_um/);
  await scaleRow.locator('.revert-btn').click();
  assert.doesNotMatch(await toml.inputValue(), /pixel_size_um/, 'camera keys removed');
  assert.match(await toml.inputValue(), /pixel_scale_arcsec = 0\.25/);
  assert.equal(await scaleRow.locator('.revert-btn').count(), 0);

  await page.click('#rc-new');

  checkNoPageErrors(page);
  await page.close();
}
