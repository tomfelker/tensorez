# TensoRez — notes for Claude Code

Planetary lucky-imaging stacker. PyTorch CLI does all processing; Electron GUI
just edits recipe TOMLs, spawns the CLI, and renders its JSONL event stream.
**DESIGN_CONTRACT.md is the source of truth** for the recipe schema, event
stream, and output/manifest layout — keep CLI and GUI changes in sync with it
and bump its version fields.

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

## Tests

- CLI: `cd cli && pytest` (~80 s; needs `pip install -e .[dev]`). conftest
  auto-generates `examples/synthetic_planet.ser` (~24 MB, gitignored) on first run.
- GUI: `cd gui && npm test`. Playwright/headless Chromium
  (`npx playwright install chromium` once). `pretest` auto-runs
  `scripts/gen-mock.mjs` which writes `gui/mockrun/` + mock event streams.
- The `real` and `deconv` GUI tests replay actual CLI output: they need
  completed runs of `cli/examples/jupiter.toml` and `jupiter_deconv.toml`
  (run from `cli/`; outputs land in `cli/output/`, gitignored). They fail with
  a "run the CLI first" message otherwise.
- GUI tests rewrite `gui/screenshots/` on every run — untracked on purpose.

## Gotchas (each of these has bitten before)

- **Windows paths in TOML**: tests that f-string a path into recipe TOML must
  use `.as_posix()` — raw backslashes are TOML escape sequences and fail to
  parse. Same rule for any path the GUI hands to `/fs/` URLs (mock bridge
  normalizes `\` → `/`).
- **No absolute paths in committed files**: example recipes and tests must use
  repo-relative paths (recipe paths resolve against the recipe file's dir).
  The vibe-coding VM once left `/root/tensorez/...` everywhere.
- Recipe parsing is strict: unknown keys are hard errors, and exactly one
  pixel-scale representation is allowed in `[deconv]` (direct or camera keys).
- The lucky stage is never fully cached (pass 2 always runs); `deconv` is never
  cached at all. Don't "fix" that.
- Event-stream consumers must ignore unknown event types and fields.

## Style

- Torch tensors are NCHW float32 linear-light RGB; `final.npy` on disk is HWC.
- GUI is dependency-light vanilla JS (only `smol-toml` at runtime); keep it that
  way — the renderer must keep working in both Electron and the mock-bridge
  browser mode.
