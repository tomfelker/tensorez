// Recipe schema, derived from CONTRACT.md §1. The form in recipe.js is
// generated entirely from this table.
//
// Field types:
//   string | int | float | bool | enum | int_enum | paths (list of path strings)
//   path (single path) | int2 / float2 (pairs, e.g. crop / frequency_cutoff)
//   floatlist (comma-separated numbers, e.g. wavelengths_nm)
// Field flags:
//   optional        key may be absent from the TOML (checkbox to include)
//   required        physically required parameter — highlighted in the form
//   logStep         stepper multiplies/divides by sqrt(2) instead of adding
//   showIf(section) hide unless predicate on the section's values passes
//   conflictIf(section)  constraint hint for bool fields: returns a message
//                   when the combo is a CLI hard error (checkbox is disabled
//                   while off, warned while on); section.validate() flags the
//                   same combos as validation problems
//   pathKind        'file' | 'directory' — what Browse picks
//
// Keys with defaults may be OMITTED in recipe files (the CLI applies the
// defaults — real CLI-authored recipes are sparse); hydrate() fills them in
// for form display.

export const SCHEMA = [
  {
    section: 'recipe',
    label: 'Recipe',
    help: 'Identity of this processing run.',
    fields: [
      { key: 'version', type: 'int', default: 0, readonly: true, help: 'contract version' },
      { key: 'name', type: 'string', default: 'my_run', pattern: '^[A-Za-z0-9_-]+$',
        help: 'used in output paths; letters, digits, _ and - only' },
    ],
  },
  {
    section: 'lights',
    label: 'Lights',
    help: 'Input frames: .ser video or globs of stills, concatenated in order.',
    fields: [
      { key: 'paths', type: 'paths', default: ['data/lights.ser'], pathKind: 'file',
        filters: [{ name: 'SER / stills', extensions: ['ser', 'png', 'tif', 'jpg'] }] },
      { key: 'start_frame', type: 'int', default: 0, min: 0, help: 'inclusive' },
      { key: 'frame_step', type: 'int', default: 1, min: 1 },
      { key: 'end_frame', type: 'int', optional: true, default: 300, min: 1,
        help: 'exclusive; leave unset to use all frames' },
      { key: 'debayer', type: 'enum',
        options: ['bilinear', 'superpixel_rgb', 'superpixel_rggb', 'none'], default: 'bilinear',
        help: 'Bayer sources only (ignored otherwise). bilinear: full-size, missing colors ' +
          'interpolated. superpixel_*: half-size, real photosites only (rggb keeps both ' +
          'greens as 4 channels — [deconv] then needs 4 wavelengths). none: treat the ' +
          'mosaic as mono (IR-filtered captures)' },
    ],
  },
  {
    section: 'darks',
    label: 'Darks',
    optionalSection: true,
    help: 'Optional dark calibration; master dark (mean) + per-pixel variance.',
    fields: [
      { key: 'paths', type: 'paths', default: ['data/darks.ser'], pathKind: 'file',
        filters: [{ name: 'SER / stills', extensions: ['ser', 'png', 'tif', 'jpg'] }] },
    ],
  },
  {
    section: 'align',
    label: 'Align',
    help: 'Integer-shift center-of-mass alignment and centered crop.',
    validate: (s) => {
      // sparse recipes: center_of_mass defaults true
      const problems = [];
      if (s.per_channel === true && s.center_of_mass === false) {
        problems.push('align: per_channel requires center_of_mass = true (CLI hard error)');
      }
      return problems;
    },
    fields: [
      { key: 'center_of_mass', type: 'bool', default: true, help: 'integer-shift CoM centering' },
      { key: 'per_channel', type: 'bool', default: false,
        help: 'align each color channel independently — corrects atmospheric dispersion ' +
          '(requires center_of_mass)',
        conflictIf: (s) => s.center_of_mass === false ? 'requires center_of_mass' : null },
      { key: 'crop', type: 'int2', optional: true, default: [512, 512], labels: ['w', 'h'],
        min: 2, help: 'centered after alignment; unset = full frame. [deconv] needs a square crop' },
      { key: 'crop_align', type: 'int', default: 2, min: 1,
        help: 'crop size/offset rounded to multiple of this' },
      { key: 'crop_offsets', type: 'int2', default: [0, 0], labels: ['x', 'y'],
        help: 'offset from image center' },
    ],
  },
  {
    section: 'lucky',
    label: 'Lucky',
    help: 'Frequency-band luckiness weighting and frame selection.',
    fields: [
      { key: 'algorithm', type: 'enum', options: ['frequency_bands'], default: 'frequency_bands' },
      { key: 'noise_wavelength_pixels', type: 'float', default: 2.0, min: 0.5, max: 64,
        logStep: true, help: 'below this wavelength: treated as noise' },
      { key: 'crossover_wavelength_pixels', type: 'float', default: 35.0, min: 1, max: 1024,
        logStep: true, help: 'known/interesting frequency split' },
      { key: 'isoplanatic_patch_pixels', type: 'float', default: 55.0, min: 1, max: 1024,
        logStep: true, help: 'spatial smoothing scale of luckiness' },
      { key: 'channel_crosstalk', type: 'float', default: 0.0, min: 0, max: 1, step: 0.05,
        help: '0 = per-channel luck, 1 = min across channels' },
      { key: 'selection', type: 'enum', options: ['sigmoid', 'top_k'], default: 'sigmoid',
        help: 'top_k is planned, not yet in the CLI' },
      { key: 'stdevs_above_mean', type: 'float', default: 2.5, min: -5, max: 10, step: 0.1,
        help: 'sigmoid gate center, in σ of per-pixel luck',
        showIf: (s) => s.selection !== 'top_k' },
      { key: 'steepness', type: 'float', default: 3.0, min: 0.1, max: 50, step: 0.1,
        help: 'sigmoid gate sharpness',
        showIf: (s) => s.selection !== 'top_k' },
      { key: 'top_fraction', type: 'float', optional: true, default: 0.05, min: 0.001, max: 1,
        step: 0.01, help: 'top_k variant, when implemented',
        showIf: (s) => s.selection === 'top_k' },
    ],
  },
  {
    section: 'deconv',
    label: 'Deconv',
    optionalSection: true,
    help: 'Optional multi-frame blind deconvolution (torchmfbd). Requires a square ' +
      '[align] crop. When enabled, final.* is the deconvolution and the lucky stack ' +
      'is published as stages/lucky/lucky_stack.*',
    validate: (s) => {
      const problems = [];
      if ('pixel_scale_arcsec' in s && 'pixel_size_um' in s) {
        problems.push('deconv: specify either pixel_scale_arcsec or the camera keys ' +
          '(focal_length_mm/barlow/pixel_size_um), never both');
      }
      if (!('pixel_scale_arcsec' in s) && !('pixel_size_um' in s) &&
          ('focal_length_mm' in s || 'barlow' in s)) {
        problems.push('deconv: camera mode needs pixel_size_um (it has no default)');
      }
      return problems;
    },
    fields: [
      { key: 'method', type: 'enum', options: ['torchmfbd'], default: 'torchmfbd',
        help: 'only value in v0' },
      { key: 'frames', type: 'enum', options: ['lucky_top', 'all'], default: 'lucky_top',
        help: 'lucky_top: top_n frames by luckiness score' },
      { key: 'top_n', type: 'int', default: 12, min: 1,
        help: 'number of luckiest frames fed to the deconvolution',
        showIf: (s) => s.frames !== 'all' },
      { key: 'diameter_cm', type: 'float', default: 27.94, min: 0.1, logStep: true,
        help: 'telescope aperture (default: Celestron C11)' },
      { key: 'central_obscuration_cm', type: 'float', default: 9.5, min: 0,
        help: 'secondary obstruction diameter (default: Celestron C11)' },
      // Pixel scale is EITHER/OR: pixel_scale_arcsec directly, or camera mode
      // via focal_length_mm + barlow + pixel_size_um (scale computed as
      // 206.265 * pixel_size_um / (focal_length_mm * barlow)). Exactly one
      // representation is serialized; pixel_size_um present = camera mode.
      { key: 'pixel_scale_arcsec', type: 'pixelscale', labelText: 'pixel scale',
        default: 0.25, min: 0.001, required: true,
        help: 'must oversample λ/D (hard error otherwise; warning under 2×)',
        camera: { focal_length_mm: 2800, barlow: 1.0, pixel_size_um: 4.3 },
        skipIf: (s) => 'pixel_size_um' in s },
      { key: 'focal_length_mm', type: 'float', hidden: true, optional: true,
        default: 2800, min: 1, presentIf: (s) => 'pixel_size_um' in s },
      { key: 'barlow', type: 'float', hidden: true, optional: true,
        default: 1.0, min: 0.1, presentIf: (s) => 'pixel_size_um' in s },
      { key: 'pixel_size_um', type: 'float', hidden: true, optional: true,
        default: 4.3, min: 0.1 },
      { key: 'wavelengths_nm', type: 'floatlist', default: [700, 530, 470], required: true,
        help: 'one per output channel (1 entry for mono); no CLI default' },
      { key: 'psf_model', type: 'enum', options: ['kl', 'zernike'], default: 'kl' },
      { key: 'n_modes', type: 'int_enum', options: [2, 5, 9, 14, 20, 27, 35, 44], default: 20,
        help: 'must complete a radial degree: 2, 5, 9, 14, 20, 27, 35, 44' },
      { key: 'iterations', type: 'int', default: 100, min: 1 },
      { key: 'optimizer', type: 'enum', options: ['adam', 'lbfgs'], default: 'adam' },
      { key: 'lr_obj', type: 'float', default: 0.02, min: 1e-5, max: 10, logStep: true,
        help: 'learning rate for the object estimate' },
      { key: 'lr_modes', type: 'float', default: 0.08, min: 1e-5, max: 10, logStep: true,
        help: 'learning rate for the PSF modes' },
      { key: 'apodization_border', type: 'int', default: 0, min: 0,
        help: 'keep 0 for planets on dark sky' },
      { key: 'frequency_cutoff', type: 'float2', default: [0.2, 0.3], labels: ['low', 'high'],
        min: 0, max: 1, help: 'reconstruction filter, fractions of the diffraction limit' },
    ],
  },
  {
    section: 'output',
    label: 'Output',
    help: 'Runs land in <dir>/<name>/<UTC timestamp>/.',
    fields: [
      { key: 'dir', type: 'path', default: 'output', pathKind: 'directory' },
      { key: 'debug_frames', type: 'int', default: 10, min: 0,
        help: 'per-frame debug artifacts kept for first N frames' },
    ],
  },
];

// Fresh recipe with all default values (optional sections/fields omitted).
export function defaultRecipe() {
  const out = {};
  for (const sec of SCHEMA) {
    if (sec.optionalSection) continue;
    const t = {};
    for (const f of sec.fields) {
      if (f.optional) continue;
      t[f.key] = structuredClone(f.default);
    }
    out[sec.section] = t;
  }
  return out;
}

// Fill in schema defaults for keys the file omitted (recipes may be sparse —
// the CLI applies the same defaults). Missing non-optional sections are created;
// absent optional sections stay absent. Returns a new object.
export function hydrate(obj) {
  const out = structuredClone(obj);
  for (const sec of SCHEMA) {
    if (!(sec.section in out)) {
      if (sec.optionalSection) continue;
      out[sec.section] = {};
    }
    const t = out[sec.section];
    if (typeof t !== 'object' || t === null || Array.isArray(t)) continue;
    for (const f of sec.fields) {
      if (f.optional) {
        // optional keys that become defaulted in a mode (camera-mode companions)
        if (f.presentIf?.(t) && !(f.key in t)) t[f.key] = structuredClone(f.default);
        continue;
      }
      if (f.skipIf?.(t)) continue; // e.g. pixel_scale_arcsec in camera mode
      if (!(f.key in t)) t[f.key] = structuredClone(f.default);
    }
  }
  return out;
}

// Validate a parsed recipe object against the schema. Returns a list of
// problem strings; empty list = valid. Unknown keys are flagged because the
// contract makes them a hard error in the CLI. Missing keys are NOT flagged —
// they have defaults the CLI applies (use hydrate() to fill them in).
export function validateRecipe(obj) {
  const problems = [];
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) {
    return ['recipe must be a table of sections'];
  }
  const known = new Map(SCHEMA.map((s) => [s.section, s]));
  for (const [secName, secVal] of Object.entries(obj)) {
    const sec = known.get(secName);
    if (!sec) { problems.push(`unknown section [${secName}] (contract: hard error)`); continue; }
    if (typeof secVal !== 'object' || secVal === null || Array.isArray(secVal)) {
      problems.push(`[${secName}] must be a table`); continue;
    }
    const fields = new Map(sec.fields.map((f) => [f.key, f]));
    for (const [k, v] of Object.entries(secVal)) {
      const f = fields.get(k);
      if (!f) { problems.push(`unknown key ${secName}.${k} (contract: hard error)`); continue; }
      const err = checkValue(f, v);
      if (err) problems.push(`${secName}.${k}: ${err}`);
    }
    if (sec.validate) problems.push(...sec.validate(secVal));
  }
  return problems;
}

function isNum(x) { return typeof x === 'number' && Number.isFinite(x); }

function checkValue(f, v) {
  switch (f.type) {
    case 'string':
    case 'path':
      if (typeof v !== 'string') return 'expected a string';
      if (f.pattern && !new RegExp(f.pattern).test(v)) return `must match ${f.pattern}`;
      return null;
    case 'int':
      if (typeof v !== 'number' || !Number.isInteger(v)) return 'expected an integer';
      if (f.min != null && v < f.min) return `must be ≥ ${f.min}`;
      if (f.max != null && v > f.max) return `must be ≤ ${f.max}`;
      return null;
    case 'float':
    case 'pixelscale':
      if (!isNum(v)) return 'expected a number';
      if (f.min != null && v < f.min) return `must be ≥ ${f.min}`;
      if (f.max != null && v > f.max) return `must be ≤ ${f.max}`;
      return null;
    case 'bool':
      return typeof v === 'boolean' ? null : 'expected true/false';
    case 'enum':
    case 'int_enum':
      return f.options.includes(v) ? null : `expected one of: ${f.options.join(', ')}`;
    case 'paths':
      if (!Array.isArray(v) || v.some((x) => typeof x !== 'string')) {
        return 'expected an array of path strings';
      }
      return v.length ? null : 'needs at least one path';
    case 'int2':
      if (!Array.isArray(v) || v.length !== 2 ||
          v.some((x) => typeof x !== 'number' || !Number.isInteger(x))) {
        return 'expected a pair of integers like [512, 512]';
      }
      return null;
    case 'float2':
      if (!Array.isArray(v) || v.length !== 2 || !v.every(isNum)) {
        return 'expected a pair of numbers like [0.2, 0.3]';
      }
      return null;
    case 'floatlist':
      if (!Array.isArray(v) || !v.length || !v.every(isNum)) {
        return 'expected a non-empty array of numbers';
      }
      return null;
    default:
      return null;
  }
}
