// Generates all mock data the renderer needs in dev/test:
//   gui/mockrun/run1/            fake completed run dir (manifest + PNGs)
//   gui/examples/mock_events.jsonl        realistic full-run event stream
//   gui/examples/mock_events_error.jsonl  variant that fails during lucky
//   gui/examples/jupiter_demo.toml        example recipe
//
// Run: node scripts/gen-mock.mjs

import { promises as fsp } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const GUI = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const RUN = path.join(GUI, 'mockrun', 'run1');
const EXAMPLES = path.join(GUI, 'examples');

// ---------- minimal PNG encoder (8-bit RGB) ----------

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'latin1');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

function encodePng(width, height, rgb) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 2;  // color type: truecolor
  const raw = Buffer.alloc(height * (1 + width * 3));
  for (let y = 0; y < height; y++) {
    const row = y * (1 + width * 3);
    raw[row] = 0; // filter: none
    rgb.copy(raw, row + 1, y * width * 3, (y + 1) * width * 3);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

// ---------- procedural images ----------

function makeImage(size, fn) {
  const rgb = Buffer.alloc(size * size * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [r, g, b] = fn(x, y, size);
      const i = (y * size + x) * 3;
      rgb[i] = Math.max(0, Math.min(255, Math.round(r * 255)));
      rgb[i + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      rgb[i + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
    }
  }
  return encodePng(size, size, rgb);
}

let seed = 42;
function rand() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 0xffffffff;
}

function planet(x, y, size, { sharp = 1, tint = [1, 0.85, 0.65], bands = true } = {}) {
  const cx = size / 2, cy = size / 2, R = size * 0.36;
  const dx = x - cx, dy = y - cy;
  const r = Math.hypot(dx, dy);
  const edge = 1 / (1 + Math.exp((r - R) / (2.5 / sharp)));
  if (edge < 0.003) return [0, 0, 0];
  const mu = Math.sqrt(Math.max(0, 1 - (r / R) * (r / R)));
  const limb = 0.35 + 0.65 * mu;
  let band = 1;
  if (bands) {
    const lat = dy / R;
    band = 0.78 + 0.22 * Math.sin(lat * 9 * sharp) * Math.cos(lat * 3.7);
    band += 0.05 * Math.sin((dx / R) * 5 + lat * 14); // festoons
  }
  const v = edge * limb * band;
  return [v * tint[0], v * tint[1], v * tint[2]];
}

function noiseMap(x, y, size, { scale = 12, tint = [1, 1, 1], floor = 0.08 } = {}) {
  const u = x / size * scale, v = y / size * scale;
  let n = 0.5
    + 0.28 * Math.sin(u * 1.3 + Math.cos(v)) * Math.cos(v * 0.9)
    + 0.18 * Math.sin(u * 3.1 + v * 2.3)
    + 0.09 * Math.sin(u * 6.7 - v * 5.1);
  n = Math.max(floor, Math.min(1, n));
  return [n * tint[0], n * tint[1], n * tint[2]];
}

function psfGrid(x, y, size) {
  // 3x2 grid of speckly PSF estimates on black (torchmfbd psf_examples style)
  const cols = 3, rows = 2;
  const cw = size / cols, ch = size / rows;
  const cx = (Math.floor(x / cw) + 0.5) * cw;
  const cy = (Math.floor(y / ch) + 0.5) * ch;
  const dx = x - cx, dy = y - cy;
  const r2 = dx * dx + dy * dy;
  const core = Math.exp(-r2 / (2 * 6 * 6));
  const speckle = 0.35 * Math.exp(-r2 / (2 * 16 * 16)) *
    (0.5 + 0.5 * Math.sin(dx * 0.9 + dy * 0.4) * Math.cos(dy * 0.8 - dx * 0.3));
  const v = Math.min(1, core + Math.max(0, speckle));
  const border = (x % Math.round(cw) < 1 || y % Math.round(ch) < 1) ? 0.07 : 0;
  return [v + border, v * 0.95 + border, v * 0.85 + border];
}

function comTrack(x, y, size) {
  // plot-like: dark bg, grid, wiggly track
  let v = 0.04;
  if (x % 32 === 0 || y % 32 === 0) v = 0.08;
  const t = x / size;
  const cy = size / 2 + Math.sin(t * 12) * 20 + Math.sin(t * 31) * 8;
  const d = Math.abs(y - cy);
  if (d < 1.6) return [0.35, 0.65, 1.0];
  if (d < 4) return [0.12 + v, 0.2 + v, 0.35 + v];
  return [v, v, v + 0.01];
}

// ---------- artifact + event definitions ----------

const ARTIFACTS = [
  { stage: 'align', name: 'com_track', kind: 'preview',
    path: 'stages/align/com_track.png', width: 256, height: 256,
    gen: (s) => makeImage(s, comTrack) },
  { stage: 'align', name: 'aligned_first_frame', kind: 'preview',
    path: 'stages/align/aligned_first_frame.png', width: 256, height: 256,
    gen: (s) => makeImage(s, (x, y, n) => planet(x, y, n, { sharp: 0.5 })) },
  { stage: 'lucky', name: 'weights_pass1', kind: 'preview',
    path: 'stages/lucky/weights_pass1.png', width: 256, height: 256,
    gen: (s) => makeImage(s, (x, y, n) => noiseMap(x, y, n, { tint: [0.8, 0.75, 1.0] })) },
  { stage: 'lucky', name: 'frame_0000', kind: 'sequence_frame', frame: 0,
    path: 'stages/lucky/frames/frame_0000.png', width: 128, height: 128,
    gen: (s) => makeImage(128, (x, y, n) => planet(x + 3, y - 2, n, { sharp: 0.45 })) },
  { stage: 'lucky', name: 'frame_0001', kind: 'sequence_frame', frame: 1,
    path: 'stages/lucky/frames/frame_0001.png', width: 128, height: 128,
    gen: (s) => makeImage(128, (x, y, n) => planet(x - 2, y + 1, n, { sharp: 0.55 })) },
  { stage: 'lucky', name: 'frame_0002', kind: 'sequence_frame', frame: 2,
    path: 'stages/lucky/frames/frame_0002.png', width: 128, height: 128,
    gen: (s) => makeImage(128, (x, y, n) => planet(x, y + 3, n, { sharp: 0.4 })) },
  { stage: 'lucky', name: 'frame_0003', kind: 'sequence_frame', frame: 3,
    path: 'stages/lucky/frames/frame_0003.png', width: 128, height: 128,
    gen: (s) => makeImage(128, (x, y, n) => planet(x - 1, y, n, { sharp: 0.6 })) },
  { stage: 'lucky', name: 'luckiness_mean', kind: 'preview',
    path: 'stages/lucky/luckiness_mean.png', width: 256, height: 256,
    gen: (s) => makeImage(s, (x, y, n) => noiseMap(x, y, n, { scale: 7, tint: [1, 0.9, 0.7] })) },
  { stage: 'lucky', name: 'unweighted_average', kind: 'preview',
    path: 'stages/lucky/unweighted_average.png', width: 256, height: 256,
    gen: (s) => makeImage(s, (x, y, n) => planet(x, y, n, { sharp: 0.35 })) },
  { stage: 'lucky', name: 'luckiness', kind: 'array',
    path: 'stages/lucky/luckiness.npy',
    gen: () => Buffer.from('\x93NUMPY mock — not a real npy\n', 'latin1') },
  { stage: 'lucky', name: 'frame_scores', kind: 'array',
    path: 'stages/lucky/frame_scores.npy',
    gen: () => Buffer.from('\x93NUMPY mock — not a real npy\n', 'latin1') },
  { stage: 'lucky', name: 'lucky_stack', kind: 'image',
    path: 'stages/lucky/lucky_stack.tif',
    gen: () => Buffer.from('II*\0 mock tiff placeholder', 'latin1') },
  { stage: 'lucky', name: 'lucky_stack', kind: 'preview',
    path: 'stages/lucky/lucky_stack.png', width: 256, height: 256,
    gen: (s) => makeImage(s, (x, y, n) => planet(x, y, n, { sharp: 1.1 })) },
  { stage: 'deconv', name: 'psf_examples', kind: 'preview',
    path: 'stages/deconv/psf_examples.png', width: 256, height: 256,
    gen: (s) => makeImage(s, psfGrid) },
  { stage: 'deconv', name: 'loss_history', kind: 'array',
    path: 'stages/deconv/loss_history.npy',
    gen: () => Buffer.from('\x93NUMPY mock — not a real npy\n', 'latin1') },
  { stage: 'output', name: 'final_preview', kind: 'preview',
    path: 'final_preview.png', width: 256, height: 256,
    gen: (s) => makeImage(s, (x, y, n) => planet(x, y, n, { sharp: 1.4 })) },
  { stage: 'output', name: 'final', kind: 'image',
    path: 'final.tif',
    gen: () => Buffer.from('II*\0 mock tiff placeholder', 'latin1') },
  { stage: 'output', name: 'final_exact', kind: 'array',
    path: 'final.npy',
    gen: () => Buffer.from('\x93NUMPY mock — not a real npy\n', 'latin1') },
];

const RECIPE = {
  recipe: { version: 0, name: 'jupiter_demo' },
  lights: { paths: ['data/jupiter.ser'], start_frame: 0, frame_step: 1, end_frame: 300 },
  darks: { paths: ['data/darks.ser'] },
  align: { center_of_mass: true, only_even_shifts: false, crop: [512, 512],
           crop_align: 2, crop_offsets: [0, 0] },
  lucky: { algorithm: 'frequency_bands', noise_wavelength_pixels: 2.0,
           crossover_wavelength_pixels: 35.0, isoplanatic_patch_pixels: 55.0,
           channel_crosstalk: 0.0, selection: 'sigmoid',
           stdevs_above_mean: 2.5, steepness: 3.0 },
  deconv: { method: 'torchmfbd', frames: 'lucky_top', top_n: 12,
            diameter_cm: 20.0, central_obscuration_cm: 0.0,
            pixel_scale_arcsec: 0.25, wavelengths_nm: [700.0, 530.0, 470.0],
            psf_model: 'kl', n_modes: 20, iterations: 100, optimizer: 'adam',
            lr_obj: 0.02, lr_modes: 0.08, apodization_border: 0,
            frequency_cutoff: [0.2, 0.3] },
  output: { dir: 'output', debug_frames: 10 },
};

const RECIPE_TOML = `[recipe]
version = 0
name = "jupiter_demo"

[lights]
paths = ["data/jupiter.ser"]
start_frame = 0
frame_step = 1
end_frame = 300

[darks]
paths = ["data/darks.ser"]

[align]
center_of_mass = true
only_even_shifts = false
crop = [512, 512]
crop_align = 2
crop_offsets = [0, 0]

[lucky]
algorithm = "frequency_bands"
noise_wavelength_pixels = 2.0
crossover_wavelength_pixels = 35.0
isoplanatic_patch_pixels = 55.0
channel_crosstalk = 0.0
selection = "sigmoid"
stdevs_above_mean = 2.5
steepness = 3.0

[deconv]
method = "torchmfbd"
frames = "lucky_top"
top_n = 12
diameter_cm = 20.0
central_obscuration_cm = 0.0
pixel_scale_arcsec = 0.25
wavelengths_nm = [700.0, 530.0, 470.0]
psf_model = "kl"
n_modes = 20
iterations = 100
optimizer = "adam"
lr_obj = 0.02
lr_modes = 0.08
apodization_border = 0
frequency_cutoff = [0.2, 0.3]

[output]
dir = "output"
debug_frames = 10
`;

const RUN_DIR = 'output/jupiter_demo/2026-07-29T17-30-00Z';

function buildEvents({ fail = false } = {}) {
  const ev = [];
  let t = 0;
  const push = (o) => { t += 0.05 + rand() * 0.4; ev.push({ ...o, t: Number(t.toFixed(3)) }); };
  const art = (a) => {
    const { gen, ...rest } = a;
    push({ event: 'artifact', ...rest });
  };
  const progress = (stage, total, msg, steps = 10) => {
    for (let i = 1; i <= steps; i++) {
      push({ event: 'progress', stage, current: Math.round(total * i / steps), total,
             ...(msg ? { message: msg } : {}) });
    }
  };

  push({ event: 'run_start', recipe_path: 'examples/jupiter_demo.toml',
         recipe: RECIPE, run_dir: RUN_DIR, frame_count: 300 });
  push({ event: 'log', level: 'info',
         message: 'loaded 300 frames from data/jupiter.ser (RGB, 16-bit, 512×512)' });

  push({ event: 'stage_start', stage: 'lights', cached: false });
  progress('lights', 300, 'decoding frames', 8);
  push({ event: 'stage_end', stage: 'lights', seconds: 3.1 });

  push({ event: 'stage_start', stage: 'darks', cached: true });
  push({ event: 'stage_end', stage: 'darks', seconds: 0.0 });
  push({ event: 'log', level: 'info', message: 'darks served from cache (cache/darks/8c1f2a90d3b4e5f6)' });

  push({ event: 'stage_start', stage: 'align', cached: false });
  progress('align', 300, 'center-of-mass alignment', 10);
  ARTIFACTS.filter((a) => a.stage === 'align').forEach(art);
  push({ event: 'stage_end', stage: 'align', seconds: 5.2 });

  push({ event: 'stage_start', stage: 'lucky', cached: false });
  progress('lucky', 300, 'pass 1/2 — measuring luckiness', 10);
  art(ARTIFACTS.find((a) => a.name === 'weights_pass1'));
  if (fail) {
    push({ event: 'log', level: 'warning',
           message: 'GPU memory low: 214 MiB free before pass 2' });
    push({ event: 'error', stage: 'lucky',
           message: 'CUDA out of memory: tried to allocate 2.50 GiB (GPU 0; 3.94 GiB total)',
           traceback:
             'Traceback (most recent call last):\n' +
             '  File "tensorez/lucky.py", line 214, in weighted_accumulate\n' +
             '    acc = acc + weights[:, None] * bands\n' +
             'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB' });
    return ev;
  }
  ARTIFACTS.filter((a) => a.kind === 'sequence_frame').forEach(art);
  progress('lucky', 300, 'pass 2/2 — weighted accumulation', 10);
  art(ARTIFACTS.find((a) => a.name === 'luckiness_mean'));
  art(ARTIFACTS.find((a) => a.name === 'unweighted_average'));
  art(ARTIFACTS.find((a) => a.name === 'luckiness'));
  art(ARTIFACTS.find((a) => a.name === 'frame_scores'));
  push({ event: 'log', level: 'info',
         message: 'lucky: best frames by mean luckiness: 81 (0.3129), 85 (0.3120), 28 (0.3117), …' });
  art(ARTIFACTS.find((a) => a.path === 'stages/lucky/lucky_stack.tif'));
  art(ARTIFACTS.find((a) => a.path === 'stages/lucky/lucky_stack.png'));
  push({ event: 'stage_end', stage: 'lucky', seconds: 41.7 });

  // deconv stage: per-iteration progress with the loss in `message`
  push({ event: 'stage_start', stage: 'deconv', cached: false });
  push({ event: 'log', level: 'info',
         message: 'deconv: torchmfbd on 12 frame(s): [28, 29, 31, 81, 82, 85, 86, 87, 88, 91, 92, 95]' });
  for (let i = 1; i <= 10; i++) {
    const iter = i * 10;
    const loss = (0.97 * Math.exp(-i * 0.18) + 0.18).toFixed(6);
    push({ event: 'progress', stage: 'deconv', current: iter, total: 100,
           message: `torchmfbd loss ${loss}` });
  }
  art(ARTIFACTS.find((a) => a.name === 'psf_examples'));
  art(ARTIFACTS.find((a) => a.name === 'loss_history'));
  push({ event: 'stage_end', stage: 'deconv', seconds: 38.4 });

  push({ event: 'stage_start', stage: 'output', cached: false });
  art(ARTIFACTS.find((a) => a.name === 'final_preview'));
  art(ARTIFACTS.find((a) => a.name === 'final'));
  art(ARTIFACTS.find((a) => a.name === 'final_exact'));
  push({ event: 'stage_end', stage: 'output', seconds: 0.8 });

  push({ event: 'done', seconds: 89.4, final: 'final.tif' });
  return ev;
}

// ---------- write everything ----------

async function main() {
  // fake run dir
  const events = buildEvents();
  await fsp.mkdir(path.join(RUN, 'stages/align'), { recursive: true });
  await fsp.mkdir(path.join(RUN, 'stages/lucky/frames'), { recursive: true });
  await fsp.mkdir(path.join(RUN, 'stages/deconv'), { recursive: true });
  for (const a of ARTIFACTS) {
    const data = a.gen(a.width || 256);
    await fsp.writeFile(path.join(RUN, a.path), data);
  }
  await fsp.writeFile(path.join(RUN, 'recipe.toml'), RECIPE_TOML);
  await fsp.writeFile(path.join(RUN, 'log.txt'),
    events.map((e) => JSON.stringify(e)).join('\n') + '\n');

  const manifest = {
    manifest_version: 0,
    recipe: RECIPE,
    run: { started_utc: '2026-07-29T17:30:00Z', seconds: 89.4, frame_count: 300 },
    stages: [
      { name: 'lights', cached: false, seconds: 3.1 },
      { name: 'darks', cached: true, seconds: 0.0 },
      { name: 'align', cached: false, seconds: 5.2 },
      { name: 'lucky', cached: false, seconds: 41.7 },
      { name: 'deconv', cached: false, seconds: 38.4 },
      { name: 'output', cached: false, seconds: 0.8 },
    ],
    artifacts: ARTIFACTS.map(({ gen, ...a }) => a),
  };
  await fsp.writeFile(path.join(RUN, 'manifest.json'), JSON.stringify(manifest, null, 2));

  // event streams
  await fsp.mkdir(EXAMPLES, { recursive: true });
  await fsp.writeFile(path.join(EXAMPLES, 'mock_events.jsonl'),
    events.map((e) => JSON.stringify(e)).join('\n') + '\n');
  await fsp.writeFile(path.join(EXAMPLES, 'mock_events_error.jsonl'),
    buildEvents({ fail: true }).map((e) => JSON.stringify(e)).join('\n') + '\n');

  // example recipe for the mock Open dialog
  await fsp.writeFile(path.join(EXAMPLES, 'jupiter_demo.toml'), RECIPE_TOML);

  console.log('mock data written to', RUN, 'and', EXAMPLES);
}

main();
