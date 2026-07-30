import assert from 'node:assert/strict';
import { gotoApp, newPage, shoot, checkNoPageErrors } from './helper.mjs';

export default async function run(browser) {
  const page = await newPage(browser);
  await gotoApp(page);
  await page.click('.nav-btn[data-view=recipe]');
  await page.waitForSelector('#rc-form .section-card');

  const toml = page.locator('#rc-toml');

  // ---- form -> toml ----
  const nameInput = page.locator('[data-field="recipe.name"] input[type=text]');
  await nameInput.fill('jupiter_night1');
  await nameInput.press('Tab');
  assert.match(await toml.inputValue(), /name = "jupiter_night1"/);

  // log stepper: 35 * sqrt(2) -> 49.5 (3 significant digits)
  const cross = page.locator('[data-field="lucky.crossover_wavelength_pixels"]');
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

  // checkbox field
  await page.check('[data-field="align.only_even_shifts"] input[type=checkbox]');
  assert.match(await toml.inputValue(), /only_even_shifts = true/);

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
    .replace('name = "jupiter_night1"', 'name = "saturn_take2"')
    .replace('steepness = 3', 'steepness = 7.5')
    .replace('center_of_mass = true', 'center_of_mass = false');
  await toml.fill(newToml);
  await page.waitForTimeout(500); // debounce
  assert.equal(await nameInput.inputValue(), 'saturn_take2');
  assert.equal(
    await page.locator('[data-field="lucky.steepness"] .stepper input').inputValue(), '7.5');
  assert.equal(
    await page.locator('[data-field="align.center_of_mass"] input[type=checkbox]').isChecked(),
    false);
  await shoot(page, '02-toml-sync.png');

  // ---- invalid TOML: inline error, form untouched ----
  await toml.fill(newToml + '\nthis is [ not toml ===');
  await page.waitForTimeout(500);
  assert.ok(await page.locator('#rc-toml-error.show').isVisible(), 'parse error should show');
  assert.match(await page.locator('#rc-toml-error').textContent(), /form not updated/);
  assert.equal(await nameInput.inputValue(), 'saturn_take2', 'form must not be clobbered');
  await shoot(page, '03-toml-error.png');

  // ---- unknown key (contract hard error) also rejected ----
  await toml.fill(newToml.replace('[output]', 'typo_key = 3\n[output]'));
  await page.waitForTimeout(500);
  assert.match(await page.locator('#rc-toml-error').textContent(),
    /unknown key lucky\.typo_key/);
  // fix it again -> error clears
  await toml.fill(newToml);
  await page.waitForTimeout(500);
  assert.ok(!(await page.locator('#rc-toml-error.show').isVisible()), 'error should clear');

  // ---- Open via bridge dialog (mock uses window.prompt) ----
  page.once('dialog', (d) => d.accept('/examples/jupiter_demo.toml'));
  await page.click('#rc-open');
  await page.waitForFunction(() =>
    document.querySelector('#rc-path').textContent === '/examples/jupiter_demo.toml');
  assert.match(await toml.inputValue(), /name = "jupiter_demo"/);
  assert.equal(await nameInput.inputValue(), 'jupiter_demo');
  assert.equal(
    await page.locator('[data-field="lucky.isoplanatic_patch_pixels"] .stepper input')
      .inputValue(), '55');

  // ---- Save (already has a path; mock write goes to memory fs) ----
  await page.click('#rc-save');
  const saved = await page.evaluate(async () =>
    window.bridge.readTextFile('/examples/jupiter_demo.toml'));
  assert.match(saved, /name = "jupiter_demo"/);

  // ---- New resets ----
  await page.click('#rc-new');
  assert.match(await toml.inputValue(), /name = "my_run"/);
  assert.equal(await page.locator('#rc-path').textContent(), 'unsaved recipe');

  // ---- align.per_channel + constraint hints ----
  const pcRow = page.locator('[data-field="align.per_channel"]');
  const pcBox = pcRow.locator('input[type=checkbox]');
  const oesBox = page.locator('[data-field="align.only_even_shifts"] input[type=checkbox]');
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
  await oesBox.check();
  assert.ok(await pcRow.locator('.field-warn').isVisible(), 'conflict warning shown');
  assert.match(await errBox.textContent(), /incompatible with only_even_shifts/);
  await page.locator('[data-section="align"]').scrollIntoViewIfNeeded();
  await shoot(page, '13-recipe-per-channel-conflict.png');
  await oesBox.uncheck();
  assert.ok(!(await errBox.isVisible()), 'validation clears when combo is fixed');

  // center_of_mass off: warned while on, disabled once off
  await comBox.uncheck();
  assert.match(await errBox.textContent(), /requires center_of_mass/);
  await pcBox.uncheck();
  assert.ok(!(await errBox.isVisible()));
  assert.ok(await pcBox.isDisabled(), 'cannot enable per_channel without center_of_mass');
  await comBox.check();
  assert.ok(await pcBox.isEnabled(), 're-enabled with center_of_mass');

  // TOML -> form: a hand-authored hard-error combo is rejected like any other
  // invalid recipe (same class as unknown keys — the CLI refuses it)
  const conflictToml = (await toml.inputValue())
    .replace('per_channel = false', 'per_channel = true')
    .replace('only_even_shifts = false', 'only_even_shifts = true');
  await toml.fill(conflictToml);
  await page.waitForTimeout(500);
  assert.match(await errBox.textContent(), /form not updated/);
  assert.match(await errBox.textContent(), /incompatible with only_even_shifts/);

  // reset for the align-card screenshot refresh
  await page.click('#rc-new');
  await page.evaluate(() => document.getElementById('views').scrollTo(0, 0));

  checkNoPageErrors(page);
  await page.close();
}
