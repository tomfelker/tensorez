# TensoRez Next — CLI ↔ GUI Contract (v0)

The CLI and the GUI are independent programs. The CLI never knows a GUI exists.
The GUI authors recipe files, spawns the CLI, and reads what the CLI leaves behind.
Everything the GUI can do, a person with a text editor and a terminal can do identically.

The three interfaces below are the entire boundary. Changes to any of them bump
`version` fields and get noted here.

---

## 1. Recipe file (`*.toml`) — the only input

The pipeline is **fixed and opinionated**: stages run in a hard-coded order; the
recipe only parameterizes them and toggles optional ones. It is not a DAG language.

Stage order: `lights` → `darks` (calibration) → `align` → `lucky` → `deconv` (optional) → `output`.

```toml
[recipe]
version = 0
name = "jupiter_demo"        # used in output paths; [A-Za-z0-9_-]+

[lights]
# One or more paths/globs, concatenated in order. .ser (MONO / RGB / BAYER_RGGB /
# BAYER_GRBG), or globs of stills (.png, .tif, .jpg — decoded to linear light).
paths = ["data/jupiter.ser"]
start_frame = 0              # inclusive, default 0
frame_step = 1               # default 1
end_frame = 300              # exclusive; omit for "all frames"
debayer = "bilinear"         # how Bayer sources become channels (ignored otherwise):
                             #   "bilinear"        full-size 3-ch, missing colors interpolated
                             #   "superpixel_rgb"  half-size 3-ch, real photosites, greens averaged
                             #   "superpixel_rggb" half-size 4-ch (R, G1, G2, B), real photosites
                             #   "none"            keep the mosaic as 1-ch mono (IR-filtered captures)

[darks]                      # optional section; omit to skip dark calibration
paths = ["data/darks.ser"]
# Produces master dark (mean) AND per-pixel variance (fed to luckiness as noise term).

[align]
center_of_mass = true        # integer-shift CoM centering
per_channel = false          # align each color channel independently — corrects atmospheric
                             # dispersion; requires center_of_mass
crop = [512, 512]            # [w, h], centered after alignment; omit for full frame
crop_align = 2               # crop size/offset rounded to multiple of this
crop_offsets = [0, 0]        # [x, y] from image center, default [0, 0]
# (global content-align and local/flow align are planned stages, not in v0;
#  when they land they appear here as `content = true`, `local = true` + kwargs)

[lucky]
algorithm = "frequency_bands"          # only algorithm in v0
noise_wavelength_pixels = 2.0          # below this wavelength: treated as noise
crossover_wavelength_pixels = 35.0     # known/interesting frequency split
isoplanatic_patch_pixels = 55.0        # spatial smoothing scale of luckiness
channel_crosstalk = 0.0                # 0 = per-channel luck, 1 = min across channels

selection = "sigmoid"                  # "sigmoid" (v0) | "top_k" (planned)
stdevs_above_mean = 2.5                # sigmoid: gate center, in σ of per-pixel luck
steepness = 3.0                        # sigmoid: gate sharpness
# top_fraction = 0.05                  # top_k variant, when implemented

[deconv]                     # optional; multi-frame blind deconvolution (torchmfbd)
method = "torchmfbd"         # only value in v0
frames = "lucky_top"         # "lucky_top" (top_n frames by luckiness score) | "all"
top_n = 12
diameter_cm = 27.94          # default: Celestron C11 aperture
central_obscuration_cm = 9.5 # default: Celestron C11
# Pixel scale — EXACTLY ONE representation per recipe file (hard error if both/neither):
pixel_scale_arcsec = 0.158   # (a) direct; must oversample λ/D (hard error; warning under 2×)
# focal_length_mm = 2800.0   # (b) camera mode: scale = 206.265·pixel_size_um/(focal_length_mm·barlow)
# barlow = 1.0
# pixel_size_um = 4.3        #     required in camera mode
wavelengths_nm = [700, 530, 470]  # REQUIRED — one per output channel (1 entry for mono)
psf_model = "kl"             # "kl" | "zernike"
n_modes = 20                 # must complete a radial degree: 2, 5, 9, 14, 20, 27, 35, …
iterations = 100
optimizer = "adam"           # "adam" | "lbfgs"
lr_obj = 0.02
lr_modes = 0.08
apodization_border = 0       # keep 0 for planets on dark sky
frequency_cutoff = [0.2, 0.3]  # reconstruction filter, fractions of the diffraction limit
# Requires a square [align] crop. With [deconv] present, final.* is the deconvolution
# and the lucky stack is published as stages/lucky/lucky_stack.{tif,png}.

[output]
dir = "output"               # runs land in <dir>/<name>/<UTC timestamp>/
debug_frames = 10            # per-frame debug artifacts kept for first N frames
```

Unknown keys are a **hard error** (catches typos; the GUI round-trips files it
didn't write). Omitting a key that has a default is legal — sparse recipes are
encouraged; only the values shown as REQUIRED have no default. Relative paths in the recipe resolve against the recipe file's
directory. The cache directory is `cache/` under the current working directory
(override with `--cache-dir`).

### CLI invocations

```
tensorez run <recipe.toml> [--cache-dir DIR] [--pretty]
tensorez validate <recipe.toml>     # parse + resolve; report per-stage cache hit/miss; no work
```

`--pretty` renders human-readable progress instead of JSONL. Exit code 0 on
success, nonzero on error (after emitting an `error` event).

---

## 2. Event stream — JSONL on stdout

One JSON object per line. Every event has `"event"` and `"t"` (monotonic seconds
since run start, float). Unknown event types must be ignored by consumers.

| event | fields | notes |
|---|---|---|
| `run_start` | `recipe_path`, `recipe` (resolved dict), `run_dir`, `frame_count` | first event |
| `stage_start` | `stage`, `cached` (bool) | `cached: true` ⇒ no progress events follow, `stage_end` is immediate |
| `progress` | `stage`, `current`, `total`, `message?` | `current` counts from 1; throttled to ≤ ~10/s |
| `artifact` | `stage`, `name`, `kind`, `path`, `width?`, `height?`, `frame?` | emitted as soon as the file is complete; path relative to `run_dir` |
| `stage_end` | `stage`, `seconds` | |
| `log` | `level` (`info`\|`warning`), `message` | freeform |
| `error` | `message`, `stage?`, `traceback?` | terminal; process exits nonzero |
| `done` | `seconds`, `final` (path of final artifact) | terminal on success |

Artifact `kind`: `preview` (8-bit PNG, for display), `image` (16-bit TIFF,
linear light), `array` (`.npy` float32), `sequence_frame` (per-frame debug
preview, has `frame`).

Cancellation = kill the process (SIGTERM/SIGKILL both fine). Stages are
restartable; completed stages are served from cache on rerun.

---

## 3. Output layout + manifest

```
<output.dir>/<recipe.name>/<YYYY-MM-DDTHH-MM-SSZ>/
  manifest.json
  recipe.toml              # verbatim copy of the input recipe
  log.txt                  # the event stream, mirrored
  final_preview.png        # 8-bit sRGB preview
  final.tif                # 16-bit linear-light result (the deliverable)
  final.npy                # float32, exact
  stages/<stage>/...       # per-stage debug artifacts (luckiness maps, weights,
                           # unweighted average, per-frame previews …)
```

`manifest.json`:

```json
{
  "manifest_version": 0,
  "recipe": { ... resolved recipe ... },
  "run": {"started_utc": "...", "seconds": 123.4, "frame_count": 300},
  "stages": [{"name": "align", "cached": false, "seconds": 5.2}, ...],
  "artifacts": [
    {"stage": "lucky", "name": "luckiness_mean", "kind": "preview",
     "path": "stages/lucky/luckiness_mean.png", "width": 512, "height": 512},
    ...
  ]
}
```

The manifest is written **last**, atomically (tmp + rename); its presence marks a
complete run. The GUI displays whatever `artifacts` declares — adding a debug
output to a stage must never require GUI changes.

### Caching (CLI-internal, but layout is stable for inspection)

`cache/<stage>/<sha256[:16]>/` with a sibling human-readable `key.txt` holding
the hash-info string (inputs described by path + size + mtime, plus all
parameters that affect the stage, plus upstream stage keys). Same scheme as the
tensorez dev branch, with file identity added to the key.

---

## Amendments (v0, post-integration)

- `stage_start` for `lucky` carries `pass1_cached: bool` — the lucky stage is never
  fully `cached` (pass 2 always runs), but pass-1 statistics may be served from cache.
  Consumers must ignore unknown fields on any event (confirmed both sides).
- `tensorez validate` emits one `validate_result` event:
  `{recipe, frame_count, stages: [{stage, cached}]}` over stages `darks`, `align`, `lucky_stats`.
- `done.final` is a path **relative to `run_dir`** (e.g. `"final.tif"`); its preview
  is `final_preview.png` by convention.
- `final.npy` on disk is **HWC** float32 (NCHW applies to in-memory torch tensors only).
- Artifact `name` is not a unique key (the same name may appear with multiple kinds);
  `(name, kind, frame)` is unique.
- TOML integer literals are accepted anywhere a float is expected (JS serializers
  write `2.0` as `2`).
- The lucky stage always emits `stages/lucky/frame_scores.npy` (per-frame
  spatial-mean luckiness, cached with pass-1 stats) and a `log` line naming the
  best frames. The `deconv` stage is never `cached: true` (its output is the
  product); its per-iteration `progress` carries the current loss in `message`,
  and it emits `loss_history` (array) and `psf_examples` (preview) artifacts.
- Recipe *files* carry exactly one pixel-scale representation; the *resolved*
  recipe in `run_start`/`validate_result`/manifest carries the camera keys (when
  used) plus the computed effective `pixel_scale_arcsec`.
- `[lights] debayer` replaced the Bayer-phase machinery: `[align]
  only_even_shifts` is GONE (unknown-key error if present) and Bayer lights no
  longer require it — debayering happens on read, so alignment and cropping are
  mosaic-agnostic. With `debayer = "bilinear"`, the lucky stage still weights
  each pixel by the per-channel Bayer sample mask, now shifted per frame along
  with the image. With `"superpixel_rggb"`, `[deconv] wavelengths_nm` needs 4
  entries (R, G1, G2, B).

## 4. Pixel conventions

Float32, linear light, shape `(N, C, H, W)` in torch code. Channel order RGB
(or R, G1, G2, B for `debayer = "superpixel_rggb"`; mono is 1 channel).
Display conversion is linear → sRGB (IEC 61966-2-1) at the very end only —
no auto-stretch, no gamma knobs; WYSIWYG like AstroLock Seeker.
SER 16-bit values scale to [0, 1] by /(2^bit_depth − 1). Bayer sources are
debayered on read per `[lights] debayer`; the superpixel modes halve width
and height (crop sizes and all *_pixels tunings are in output pixels).
4-channel results collapse to RGB (greens averaged) in `final.tif` and every
preview PNG; `final.npy` keeps the exact channels.
