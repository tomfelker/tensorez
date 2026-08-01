// GUI preferences, kept in a JSON file in the user's profile directory
// (bridge.appPaths().settings) alongside the scratch recipe.
//
// These are *machine* preferences, deliberately not recipe keys: which recipe
// you had open, and where bulky run archives and caches should go — the CLI
// takes the latter two as --runs-dir / --cache-dir, so a recipe stays portable
// while one machine can keep its scratch data on a fast disk.
//
// Every view reads and writes through this module so they can't fight over
// the file. Loading is best-effort: a missing or corrupt settings file just
// means defaults.

let paths = null;      // {userData, settings, scratchRecipe}
let values = {};
let loaded = null;     // in-flight or completed load, so we only do it once

export async function loadSettings() {
  if (!loaded) loaded = load();
  return loaded;
}

async function load() {
  try {
    paths = await window.bridge.appPaths();
  } catch {
    return { paths: null, values };  // no profile directory: memory only
  }
  try {
    values = JSON.parse(await window.bridge.readTextFile(paths.settings)) || {};
  } catch {
    values = {};
  }
  return { paths, values };
}

export function getSettings() {
  return values;
}

export function appPaths() {
  return paths;
}

export async function updateSettings(patch) {
  values = { ...values, ...patch };
  if (!paths) return values;
  try {
    await window.bridge.writeTextFile(paths.settings, JSON.stringify(values, null, 2) + '\n');
  } catch { /* preferences are best-effort; never block the app on them */ }
  return values;
}
