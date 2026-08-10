# TensoRez — notes for Claude Code

Planetary lucky-imaging stacker. PyTorch CLI does all processing; Electron GUI
just edits recipe TOMLs, spawns the CLI, and renders its JSONL event stream.
**DESIGN_CONTRACT.md is the source of truth** for the recipe schema, event
stream, and output/manifest layout — keep CLI and GUI changes in sync with it.

**Don't preserve backwards compatibility.** The design is still moving: change
both sides together, let removed keys become unknown-key/unknown-section errors
rather than deprecations, and note the break in the contract's Amendments. There
is deliberately no recipe schema version to bump.

## Layout

- `cli/` — the `tensorez` Python package (`tensorez run|validate <recipe.toml>`)
- `gui/` — Electron app; renderer also runs in a plain browser against a mock
  bridge (`gui/renderer/js/bridge-mock.js`), which is how most GUI tests work
- `legacy/` — the old TensorFlow codebase, reference only; don't build on it
- `examples/` — synthetic-data generator shared by CLI tests and GUI SER tests
- `data/jupiter_mvi_6906/` — committed real Jupiter PNGs used by example recipes

## Setup (Windows dev box)

Venv lives at repo root: `.venv` (Python 3.14, torch+cu130). Install:
`cd cli && pip install -r requirements.txt` (must run from `cli/` — the `-e .`
inside resolves against the shell cwd). The GUI finds Python via
`TENSOREZ_PYTHON` env var, falling back to `python` on PATH.

MP4/AVI input needs FFmpeg's **shared** libraries as well, installed here with
`winget install BtbN.FFmpeg.LGPL.Shared.7.1` (which puts the real bin directory
on the user PATH, so a fresh shell picks it up). `TENSOREZ_FFMPEG_DIR`
overrides the search when a shell's PATH is stale.

## Tests

- CLI: `cd cli && pytest` (~80 s; needs `pip install -e .[dev]`). conftest
  auto-generates `examples/synthetic_planet.ser` and `examples/speckle_planet.ser`
  (~24 MB each, gitignored) on first run.
- GUI: `cd gui && npm test`. Playwright/headless Chromium
  (`npx playwright install chromium` once). `pretest` auto-runs
  `scripts/gen-mock.mjs` which writes `gui/mockrun/` + mock event streams.
- The `real` and `mfbd` GUI tests replay actual CLI output: they need
  completed runs of `cli/examples/jupiter.toml` and `jupiter_mfbd.toml`,
  **run from `cli/examples/`** (recipe paths resolve against the cwd). Their
  output lands in `cli/examples/{<recipe stem>,tensorez_runs,tensorez_cache}/`,
  all gitignored. They fail with a "run the CLI first" message otherwise.
- GUI tests rewrite `gui/screenshots/` on every run — untracked on purpose.
- **Only the `electron` suite runs Electron.** Every other GUI test drives the
  renderer in plain Chromium against the mock bridge, so `main.js`/`preload.js`
  can break without any of them noticing — that suite is the only thing
  covering them, which is what makes an Electron upgrade verifiable. It
  launches the real app (a window appears for a few seconds; Electron has no
  headless mode) under a throwaway `--user-data-dir`, so it can't clobber the
  scratch recipe of a GUI you have open.
- `test_video.py` generates its fixtures by piping raw frames through the
  `ffmpeg` binary, so it skips (with the reason) when FFmpeg or torchcodec is
  missing. Three of its tests cover the degraded path and always run.

## Gotchas (each of these has bitten before)

- **Windows paths in TOML**: tests that f-string a path into recipe TOML must
  use `.as_posix()` — raw backslashes are TOML escape sequences and fail to
  parse. Same rule for any path the GUI hands to `/fs/` URLs (mock bridge
  normalizes `\` → `/`).
- **No absolute paths in committed files**: example recipes and tests must use
  repo-relative paths (recipe paths resolve against the recipe file's dir).
  The vibe-coding VM once left `/root/tensorez/...` everywhere.
- Recipe parsing is strict: unknown keys are hard errors, and exactly one
  pixel-scale representation is allowed in `[mfbd]` (direct or camera keys).
- The `local_lucky` and `lucky_fourier` stages are never fully cached (pass 2
  always runs); `mfbd` is never cached at all. Don't "fix" that.
- `lucky_fourier` only helps when seeing distorts Fourier *phases*. The
  zero-phase gaussian blur in `synthetic_planet.ser` gives it nothing to
  recover, so it (correctly) loses to the plain average there — its tests run
  on `speckle_planet.ser` instead. Don't move them over.
- Event-stream consumers must ignore unknown event types and fields.
- **Most recipe-form commits deliberately skip the re-render** (rebuilding a
  control mid-keystroke steals focus), so anything derived from field values —
  the off-default marks and revert buttons — has to be maintained on the live
  DOM by `refreshDefaultMarkers()`, not baked in when a row is built. Computing
  it in `renderField` looks right and silently never updates.
- **Video needs a *shared* FFmpeg, and PATH is not enough.** torchcodec loads
  the system FFmpeg (majors 4–8) instead of bundling it. Static builds — which
  is most Windows packages, `Gyan.FFmpeg` included — ship `ffmpeg.exe` and no
  DLLs, so they look installed and aren't. And since Python 3.8 the loader
  doesn't search PATH for an extension module's dependencies, so having ffmpeg
  on PATH still fails: `video.py` must call `os.add_dll_directory` on the
  library directory *before* importing torchcodec. Symptom of getting this
  wrong is a bare `Could not load this library: libtorchcodec_core4.dll`.
- **torchcodec's 8-bit float conversion is lossy; its deeper ones are fine.**
  Asking for float32 on an 8-bit source gives `v*256/65535`, not `v/255` — 0.39%
  low, white never reaching 1.0. At 10 bits it's `v/1023` to within 4e-6. So
  `video.py` asks for `output_dtype="auto"`: uint8 for 8-bit, which it scales by
  255 itself, float32 for anything deeper, passed straight through. Don't
  "simplify" that to a single dtype. Covered by
  `test_lossless_video_decodes_exactly` and `test_ten_bit_video_keeps_its_depth`.
- **Recipe paths resolve against the cwd**, not the recipe's directory (the
  GUI spawns the CLI with cwd = the recipe's folder). The recipe file names
  everything: there is no `[recipe]` section at all, and products default to
  `<cwd>/<recipe stem>/`. No `final.*` — each producer writes
  `<its name>.{npy,tif,png}`.
- Run archives and the cache are `--runs-dir` / `--cache-dir` (default
  `./tensorez_runs/<recipe stem>/<timestamp>/` and `./tensorez_cache`), never
  recipe keys: they're machine preferences, so they can live on a scratch disk.
  The GUI stores both in its settings file (`renderer/js/settings.js`).
- The CLI prints for humans by default; `--events` is what emits JSONL. Tests
  and the GUI must pass it (conftest's `run_cli` adds it automatically).

## Style

- Torch tensors are NCHW float32 linear-light RGB; the `.npy` products on disk
  are HWC.
- GUI is dependency-light vanilla JS (only `smol-toml` at runtime); keep it that
  way — the renderer must keep working in both Electron and the mock-bridge
  browser mode.
