Tensorez
===
Using modern tensor libraries to enhance the resolution of your telescope.

The idea here is to make an easy to use, principled, opinionated, fast, and modern application to help with alignment, stacking, lucky imaging, deconvolution, and various other techniques for processing videos from a telescope, especially of planets and satellites.

Installing
==

You'll need:
* **Python 3.11+** (3.14 works) for the processing CLI
* **Node.js** (v20+) only if you want the GUI
* An **NVIDIA GPU** is optional but strongly recommended — the pipeline is PyTorch and runs fine on CPU, just slower.

CLI
--
Create a virtualenv at the repo root and install the CLI into it. The requirements file pulls PyTorch built for CUDA 13.0, plus the optional `[deconv]` extra (torchmfbd multi-frame blind deconvolution).

Windows (PowerShell):
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
cd cli
pip install -r requirements.txt
```

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
cd cli
pip install -r requirements.txt
```

Notes:
* No NVIDIA GPU? Skip the big CUDA download: `pip install -e .[deconv] --extra-index-url https://download.pytorch.org/whl/cpu` (from `cli/`).
* Don't need deconvolution? A minimal install is just `pip install -e .` (from `cli/`).
* Check that the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`

GUI
--
The GUI is an Electron app that edits recipe files and runs the CLI for you.

```bash
cd gui
npm install
npm start
```

The GUI spawns `python -m tensorez ...`, so launch it from a shell where the venv is **activated** — or set `TENSOREZ_PYTHON` to the venv's interpreter (e.g. `C:\projects\tensorez\.venv\Scripts\python.exe`) and it will use that regardless.

There's also a browser-hosted dev mode with a mocked CLI bridge (no Electron, no Python needed): `npm run dev`, then open http://localhost:8123/.

Tests
--
CLI tests (the first run generates `examples/synthetic_planet.ser`, ~24 MB, and exercises the whole pipeline — a few minutes on CPU, quick on a GPU):
```bash
cd cli
pip install -e .[dev]
pytest
```

GUI tests drive the renderer in headless Chromium via Playwright (one-time browser download):
```bash
cd gui
npx playwright install chromium
npm test
```

Using
==
The CLI takes a recipe `.toml` describing your input files and processing options (see `cli/examples/` for real ones, and DESIGN_CONTRACT.md for every key):

Keep the recipe with the capture files it processes — that's the intended workflow, and it makes everything else fall out:

```bash
cd data/2026-01-02_jupiter    # where jupiter.ser and jupiter.toml live
tensorez validate jupiter.toml   # parse + resolve, report cache state, do no work
tensorez run jupiter.toml        # run the pipeline
```

(`python -m tensorez ...` works too. Add `--events` for the JSONL event stream the GUI consumes.)

Paths in a recipe resolve against the current directory, the usual shell rule — so run recipes from where their inputs are. The recipe file names the run, and unless `[output] dir` says otherwise you get:

- `jupiter/` — the products, one set per enabled stage: `local_lucky.tif`, `mfbd.tif`, `lucky_stack_p10.tif`, … each also as a `.png` preview and an exact float32 `.npy`. Each run overwrites them.
- `tensorez_runs/jupiter/<timestamp>/` — every past run, keeping its own copy of the products, the recipe it used, the log, and `examples/` full of per-stage debug imagery.
- `tensorez_cache/` — completed stages, so reruns after tweaking one stage only redo what changed.

Those last two hold bulk regenerable data, so `--runs-dir` and `--cache-dir` can send them somewhere else entirely — a big fast scratch disk, say — leaving your capture drive holding only recipes and results.

In the GUI: open or build a recipe in the left panel, hit Run, and watch progress and artifacts appear live. It runs the CLI from the recipe's own folder, so the same rules apply. An unsaved recipe is kept in a scratch file so it can run immediately; Save As puts it wherever you like (beside your captures, ideally). The runs and cache locations are settings, under "Where runs and cache go" in the Run view.

Architecture
==
There's a command-line python script, written with PyTorch, that handles all the actual image processing.  It uses 'recipe' files, which specify all the input filenames, and options for all the processing.  Then, there's a GUI, which basically provides a nice way to edit these recipes, and to run the CLI.

See DESIGN_CONTRACT.md for more on the details.

History
==
This project started some years ago, and at the time was written in Tensorflow.  The original idea was to build a forward model of a true image, being croupted by a series of PSFs, generating observed images - and then to fit this model using gradient descent to the actual observations, and then to extract the true image, which would be among the weights.  It didn't work very well, and while it could have been improved by factoring the problem to avoid directly solving for the true image, the logical endpoint has already been reached by others, and implemented in torchmfbd, so we can just use that as one step in our pipeline.

Then I implemented a few other random ideas:
* Online blind deconvolution, basically a tensorflow implementation of that paper's matlab code
* Some traditional lucky imaging stuff
* Some "local" lucky imaging, where we make a luckiness extimate per pixel (not just per frame), so it can work well on extended (e.g., lunar) images.
* Alignment code, using backprop to precisely align images
* "Local align", which aligns images using a flow field - also great for lunar images.

All of these ideas were just standalone python scripts, using some of the libary code developed here.  And in particular, a pipeline for loading the files and doing some preprocessing, like dark frame subtraction and computing per-pixel variance.

But now it's 2026, and vibe coding is a thing, and so now it's easy to rewrite this in PyTorch, with a cleaner 'recipe' architecture rather than hardcoding filenames at the top of the scripts, and even with a fancy UI, so that eventually it can be accessible to hobbyists who aren't comforable coding.