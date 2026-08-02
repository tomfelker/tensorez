# TensoRez Next — CLI ↔ GUI Contract

The CLI and the GUI are independent programs. The CLI never knows a GUI exists.
The GUI authors recipe files, spawns the CLI, and reads what the CLI leaves behind.
Everything the GUI can do, a person with a text editor and a terminal can do identically.

The three interfaces below are the entire boundary: this document exists so both
sides agree on them *today*, not to promise anything about tomorrow. **Nothing is
kept backwards compatible for now** — recipes have no schema version, old keys
become unknown-key errors rather than deprecations, and both sides change
together. Breaking changes get an entry in Amendments so a stale recipe's error
message can be looked up, and that's the whole compatibility story until the
design settles.

---

## 1. Recipe file (`*.toml`) — the only input

The pipeline is **fixed and opinionated**: stages run in a hard-coded order; the
recipe only parameterizes them and toggles optional ones. It is not a DAG language.

Stage order: `lights` → `darks` (calibration) → `align` → `lucky_scoring` →
`local_lucky` → `lucky_stack` → `mfbd` → `output`.

After alignment there are three independent, individually optional **producers**,
each yielding one or more single-image products: `local_lucky` (per-pixel lucky
stacking), `lucky_stack` (classic whole-frame lucky stacks), and `mfbd`
(multi-frame blind deconvolution). At least one must be enabled. `lucky_scoring`
computes one cached scalar score per frame and is required by `lucky_stack` and
by `mfbd` with `frames = "lucky_top"`.

A recipe is nothing but its stages — there is no identity section and no schema
version to declare. The recipe file *is* the run's name: `iss_pass.toml` produces
`iss_pass/`.

**Paths in a recipe — inputs and `[output] dir` alike — resolve against the
working directory**, not the recipe's location: the ordinary shell rule. The
GUI runs the CLI with the working directory set to the recipe's own folder, so
the intended workflow (recipe saved beside the `.SER` files it processes) needs
nothing but bare filenames, and its results appear right there. A recipe copied
into a run archive therefore needs no path rewriting — it just has to be run
from the same place.

```toml
[lights]
# One or more paths/globs, concatenated in order. .ser (MONO / RGB / BAYER_RGGB /
# BAYER_GRBG), globs of stills (.png, .tif, .jpg), or videos (.mp4, .avi, .mov,
# .mkv, …) — all decoded to linear light. Video needs the optional [video] extra
# plus FFmpeg's shared libraries; without them a video input is a clean error and
# every other format still works.
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
start_frame = 0              # same selection keys as [lights] — carve the darks out
frame_step = 1               # of a capture that contains them (e.g. the empty sky
end_frame = 100              # before/after an ISS pass); end_frame exclusive
keep_level = false           # true: subtract only the dark's *pattern*, adding its
                             # scalar mean back, so calibrated pixels keep their
                             # pedestal instead of scattering around zero (.tif/.png
                             # are unsigned and clip negatives to black; .npy keeps
                             # them). Useful when the "darks" are really sky frames.
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

[lucky_scoring]              # optional; scalar luckiness per frame, cached like alignment
metric = "fourier_bandpass"  # mean |FFT| in the band below | "image_squared"
                             # (Muller & Buffington 1974 sharpness, no parameters)
min_wavelength_pixels = 5.0  # fourier_bandpass band edges (fine detail vs noise floor)
max_wavelength_pixels = 50.0

[lucky_stack]                # optional; classic lucky stacks. Requires [lucky_scoring].
top_fractions = [0.1]        # one product per fraction f: the plain average of the
                             # best ceil(f * N) frames (min 1); 1.0 = mean of everything.
                             # Products: lucky_stack_p10.{npy,tif,png} etc.
                             # (label = percent, '.'->'_' : 0.05->p5, 0.125->p12_5)

[local_lucky]                # optional; per-pixel lucky stacking (best for extended scenes)
algorithm = "frequency_bands"          # only algorithm in v0
noise_wavelength_pixels = 2.0          # below this wavelength: treated as noise
crossover_wavelength_pixels = 35.0     # known/interesting frequency split
isoplanatic_patch_pixels = 55.0        # spatial smoothing scale of luckiness
channel_crosstalk = 0.0                # 0 = per-channel luck, 1 = min across channels
stdevs_above_mean = 2.5                # sigmoid gate center, in σ of per-pixel luck
steepness = 3.0                        # sigmoid gate sharpness
# Product: local_lucky.{npy,tif,png}

[mfbd]                       # optional; multi-frame blind deconvolution (torchmfbd)
method = "torchmfbd"         # only value in v0
frames = "lucky_top"         # "lucky_top" (needs [lucky_scoring]) | "all"
top_n = 12                   # count of luckiest frames…
# top_fraction = 0.1         # …or a fraction of all frames — never both
# Product: mfbd.{npy,tif,png}
diameter_cm = 27.94          # default: Celestron C11 aperture
central_obscuration_cm = 9.5 # default: Celestron C11
# Pixel scale — EXACTLY ONE representation per recipe file (hard error if both/neither):
pixel_scale_arcsec = 0.158   # (a) direct. Describes the SENSOR photosites; superpixel
                             # debayer modes double the effective scale automatically.
                             # Undersampling (< 2 px per λ/D, or even < 1) warns, never fails.
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
# Requires a square [align] crop.

[output]                     # optional section
# dir = "results"            # default: the recipe's filename minus its extension, in
                             # the working directory — jupiter.toml -> jupiter/ (see §3).
                             # Only the *products* live here; run archives and the cache
                             # are CLI options, not recipe keys.
debug_frames = 10            # per-frame debug artifacts kept for first N frames
```

Unknown keys are a **hard error** (catches typos; the GUI round-trips files it
didn't write). Omitting a key that has a default is legal — sparse recipes are
encouraged; only the values shown as REQUIRED have no default.

### CLI invocations

```
tensorez run <recipe.toml> [--runs-dir DIR] [--cache-dir DIR] [--events]
tensorez validate <recipe.toml>     # parse + resolve; report per-stage cache hit/miss; no work
```

Output is human-readable by default; `--events` switches stdout to the JSONL
event stream of §2, which is what the GUI spawns.

`--runs-dir` (default `./tensorez_runs`) and `--cache-dir` (default
`./tensorez_cache`) say where bulk regenerable data goes, and are options
rather than recipe keys precisely so a recipe stays portable while one machine
can keep its archives and cache on a fast scratch disk. Like every other
relative path they resolve against the working directory; the GUI remembers a
setting for each.

Exit code 0 on success, nonzero on error (after emitting an `error` event).

---

## 2. Event stream — JSONL on stdout

One JSON object per line. Every event has `"event"` and `"t"` (monotonic seconds
since run start, float). Unknown event types must be ignored by consumers.

| event | fields | notes |
|---|---|---|
| `run_start` | `recipe_path`, `name`, `recipe` (resolved dict), `output_dir`, `run_dir`, `frame_count` | first event; `name` is the recipe's filename stem |
| `stage_start` | `stage`, `cached` (bool) | `cached: true` ⇒ no progress events follow, `stage_end` is immediate |
| `progress` | `stage`, `current`, `total`, `message?` | `current` counts from 1; throttled to ≤ ~10/s |
| `artifact` | `stage`, `name`, `kind`, `path`, `width?`, `height?`, `frame?` | emitted as soon as the file is complete; path relative to `run_dir` |
| `stage_end` | `stage`, `seconds` | |
| `log` | `level` (`info`\|`warning`), `message` | freeform |
| `error` | `message`, `stage?`, `traceback?` | terminal; process exits nonzero |
| `done` | `seconds`, `products` (product base names), `output_dir` | terminal on success |

Artifact `kind`: `preview` (8-bit PNG, for display), `image` (16-bit TIFF,
linear light), `array` (`.npy` float32), `sequence_frame` (per-frame debug
preview, has `frame`).

Cancellation = kill the process (SIGTERM/SIGKILL both fine). Stages are
restartable; completed stages are served from cache on rerun.

---

## 3. Output layout + manifest

Every producer publishes its result under a **stable name of its own** —
`local_lucky`, `lucky_stack_p10`, `mfbd`, … — as `.npy` (exact float32),
`.tif` (16-bit linear) and `.png` (8-bit sRGB preview). No product is
"the" result; there is no `final.*`.

Running `jupiter.toml` from the directory it lives in, with no options:

```
<cwd>/                                 # where the recipe and its .SER files live
  jupiter.toml
  jupiter.ser
  jupiter/                             # <output.dir>: default <recipe name>/
    local_lucky.npy/.tif/.png          # the products, overwritten by each run
    mfbd.npy/.tif/.png
  tensorez_runs/jupiter/<YYYY-MM-DDTHH-MM-SSZ>/   # --runs-dir; one dir per run, kept
    manifest.json
    recipe.toml                        # verbatim copy of the recipe as run
    log.txt                            # the event stream, mirrored
    local_lucky.npy/.tif/.png          # that run's own copy of each product
    mfbd.npy/.tif/.png
    examples/<stage>/...               # debug imagery (luckiness maps, weights,
                                       # unweighted average, PSFs, per-frame previews …)
  tensorez_cache/<stage>/...           # --cache-dir; stage cache (see below)
```

Artifact paths are relative to the **run** directory. The output directory
holds only the products, so "the latest result" is always one predictable
path, while the runs directory accumulates the history — under a per-recipe
level, so that pointing `--runs-dir` at one shared scratch disk still keeps
each recipe's history its own.

`manifest.json`:

```json
{
  "manifest_version": 0,
  "recipe": { ... resolved recipe ... },
  "run": {"name": "jupiter", "started_utc": "...", "seconds": 123.4,
          "frame_count": 300, "working_dir": "...", "output_dir": "...",
          "products": ["local_lucky", "mfbd"]},
  "stages": [{"name": "align", "cached": false, "seconds": 5.2}, ...],
  "artifacts": [
    {"stage": "local_lucky", "name": "luckiness_mean", "kind": "preview",
     "path": "examples/local_lucky/luckiness_mean.png", "width": 512, "height": 512},
    ...
  ]
}
```

The manifest is written **last**, atomically (tmp + rename); its presence marks a
complete run. The GUI displays whatever `artifacts` declares — adding a debug
output to a stage must never require GUI changes.

### Caching (CLI-internal, but layout is stable for inspection)

`tensorez_cache/<stage>/<sha256[:16]>/` with a sibling human-readable `key.txt` holding
the hash-info string (inputs described by path + size + mtime, plus all
parameters that affect the stage, plus upstream stage keys). Same scheme as the
tensorez dev branch, with file identity added to the key.

---

## Amendments (post-integration)

- **`[recipe]` is gone entirely (breaking)** — both its keys are: `name`
  (the filename says it) and `version` (a schema version nobody would know
  when to change, on a project explicitly not maintaining compatibility).
  A recipe now starts at `[lights]`; an old one fails with
  `unknown section [recipe]`.

- `stage_start` for `local_lucky` carries `pass1_cached: bool` — that stage is never
  fully `cached` (pass 2 always runs), but pass-1 statistics may be served from cache.
  Consumers must ignore unknown fields on any event (confirmed both sides).
- `tensorez validate` emits one `validate_result` event:
  `{recipe, frame_count, stages: [{stage, cached}]}` over the cacheable stages —
  `darks`, `align`, `lucky_scoring`, `local_lucky_stats` — each present only when
  the corresponding recipe section is enabled.
- Product `.npy` files on disk are **HWC** float32 (NCHW applies to in-memory
  torch tensors only).
- Artifact `name` is not a unique key (the same name may appear with multiple kinds);
  `(name, kind, frame)` is unique.
- TOML integer literals are accepted anywhere a float is expected (JS serializers
  write `2.0` as `2`).
- The `lucky_scoring` stage emits `examples/lucky_scoring/frame_scores.npy` and a
  `log` line naming the best frames; the scores themselves are cached. The `mfbd`
  stage is never `cached: true` (its output is a product); its per-iteration
  `progress` carries the current loss in `message`, and it emits `loss_history`
  (array) and `psf_examples` (preview) artifacts.
- Recipe *files* carry exactly one pixel-scale representation; the *resolved*
  recipe in `run_start`/`validate_result`/manifest carries the camera keys (when
  used) plus the computed effective `pixel_scale_arcsec`.
- `[lights] debayer` replaced the Bayer-phase machinery: `[align]
  only_even_shifts` is GONE (unknown-key error if present) and Bayer lights no
  longer require it — debayering happens on read, so alignment and cropping are
  mosaic-agnostic. With `debayer = "bilinear"`, the local_lucky stage still
  weights each pixel by the per-channel Bayer sample mask, now shifted per frame
  along with the image. With `"superpixel_rggb"`, `[mfbd] wavelengths_nm` needs 4
  entries (R, G1, G2, B).
- `[mfbd]` pixel-scale keys (both forms) always describe the sensor photosites;
  with a superpixel debayer mode the pipeline doubles the effective
  `pixel_scale_arcsec` itself (and logs it). Undersampling relative to λ/D is a
  warning, never a hard error — deconvolving seeing-blurred undersampled data is
  a legitimate, knowingly-degraded choice.
- The old `[lucky]` and `[deconv]` sections were split/renamed into
  `[lucky_scoring]` / `[lucky_stack]` / `[local_lucky]` / `[mfbd]` (all
  optional, at least one producer required); old names are unknown-section
  hard errors. `[lucky]`'s `selection` key is gone — classic top-K selection
  is now the `lucky_stack` stage.
- **Path resolution changed (breaking).** Relative paths in a recipe now
  resolve against the **working directory**, not the recipe file's directory.
  The GUI spawns the CLI with the working directory set to the recipe's folder,
  which preserves the intended workflow while making a recipe mean the same
  thing wherever it is copied.
- **Output layout, products, and naming (breaking).** The recipe file names the
  run, and `[output] dir` defaults to that name in the working directory. Runs land in
  `<runs-dir>/<recipe name>/<timestamp>/` where `--runs-dir` defaults to
  `./tensorez_runs`, debug artifacts moved from `stages/` to
  `examples/`, and `--cache-dir` defaults to `./tensorez_cache`. Neither is a
  recipe key — they are machine preferences (the GUI keeps a setting for each,
  in a JSON file in the user's profile directory). There is no
  `final.*`: each producer writes `<product>.{npy,tif,png}` at the top of the
  run directory, and the `output` stage copies them into `<dir>` (where each
  run overwrites the last) while declaring no artifacts of its own. `done`
  carries `products` + `output_dir` instead of `final`; `run_start` gained
  `name` + `output_dir`; the manifest's `run` gained `name`, `working_dir`,
  `output_dir` and `products`; `validate_result` gained `name`, `working_dir`,
  `output_dir` and `runs_dir`.
- `--pretty` is gone because human-readable output is now the default;
  `--events` opts into the JSONL stream (the GUI passes it).
- `[darks] keep_level` subtracts only the master dark's pattern (adding its
  scalar mean back). Dark subtraction otherwise leaves genuinely negative
  pixels: alignment (mean-relative) and the luckiness/scoring bands (DC-free)
  are unaffected, `.npy` preserves them, and `.tif`/`.png` clip them to black.
  The `align` stage logs what fraction of frame 0 went negative.
- **Video inputs (additive).** `[lights] paths` and `[darks] paths` accept
  MP4/AVI/MOV/MKV alongside `.ser` and stills, decoded by torchcodec with
  frame-accurate indexing (`seek_mode="exact"`, so frame *i* is frame *i* even
  on a long-GOP file rather than the nearest keyframe). There is no recipe key
  for this: a video is just another path. Support is optional in both halves —
  the `[video]` pip extra and FFmpeg's shared libraries, which torchcodec loads
  rather than bundles — and when either is absent a video path fails as an
  ordinary bad-input error naming both remedies, while SER and stills are
  unaffected. See §4 for how video pixels are interpreted.

## 4. Pixel conventions

Float32, linear light, shape `(N, C, H, W)` in torch code. Channel order RGB
(or R, G1, G2, B for `debayer = "superpixel_rggb"`; mono is 1 channel).
Display conversion is linear → sRGB (IEC 61966-2-1) at the very end only —
no auto-stretch, no gamma knobs; WYSIWYG like AstroLock Seeker.
SER 16-bit values scale to [0, 1] by /(2^bit_depth − 1). Bayer sources are
debayered on read per `[lights] debayer`; the superpixel modes halve width
and height (crop sizes and all *_pixels tunings are in output pixels).
4-channel results collapse to RGB (greens averaged) in every product `.tif`
and preview `.png`; the `.npy` keeps the exact channels.

Video frames arrive already demosaiced and lossily compressed, so `debayer`
never applies to them and no mosaic can be recovered. They are decoded to
8-bit RGB and divided by 255 before the same sRGB → linear curve the stills
take, which makes a video frame numerically identical to a still of the same
value: BT.709, which consumer video declares, shares sRGB's primaries and
white point, so this is the canonical inverse rather than an approximation.
Sources deeper than 8 bits are reduced to 8 on the way in, and HDR transfer
functions (PQ, HLG) are linearized as if they were sRGB — wrong, and warned
about. A grayscale video decodes as 1 channel, not 3 identical ones.
