// TensoRez renderer entry point: install mock bridge if needed, then wire up
// navigation and the four views.

async function boot() {
  if (!window.bridge) {
    const { installMockBridge } = await import('./bridge-mock.js');
    installMockBridge();
  }

  document.getElementById('bridge-badge').textContent =
    window.bridge.platform === 'electron' ? 'bridge: electron' : 'bridge: mock';

  const [{ initRecipe }, { initRun }, { initResults }, { initSer }] = await Promise.all([
    import('./recipe.js'),
    import('./run.js'),
    import('./results.js'),
    import('./ser.js'),
  ]);

  // initRecipe is async: it adopts the last-opened (or scratch) recipe from
  // the profile directory before we declare the app ready.
  const views = {
    recipe: await initRecipe(document.getElementById('view-recipe')),
    run: initRun(document.getElementById('view-run')),
    results: initResults(document.getElementById('view-results')),
    ser: initSer(document.getElementById('view-ser')),
  };

  const navBtns = [...document.querySelectorAll('.nav-btn')];
  function show(name) {
    for (const btn of navBtns) btn.classList.toggle('active', btn.dataset.view === name);
    for (const sec of document.querySelectorAll('.view')) {
      sec.hidden = sec.id !== 'view-' + name;
    }
    views[name]?.onShow?.();
    location.hash = name;
  }
  for (const btn of navBtns) btn.addEventListener('click', () => show(btn.dataset.view));

  const initial = location.hash.replace('#', '');
  show(views[initial] ? initial : 'recipe');
  document.body.dataset.ready = '1'; // signals tests that listeners are wired

  // cross-view hop: run console "view results" button etc.
  window.addEventListener('tensorez:navigate', (e) => {
    show(e.detail.view);
    views[e.detail.view]?.receive?.(e.detail);
  });
}

boot();
