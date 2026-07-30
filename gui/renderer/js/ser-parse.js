// SER file parsing. Header struct '<14slllllll40s40s40sqq' = 178 bytes.
// Frames are read lazily via File.slice() — the file is never loaded whole.

export const COLOR_IDS = {
  0: 'MONO',
  8: 'BAYER_RGGB',
  9: 'BAYER_GRBG',
  10: 'BAYER_GBRG',
  11: 'BAYER_BGGR',
  16: 'BAYER_CYYM',
  17: 'BAYER_YCMY',
  18: 'BAYER_YMCY',
  19: 'BAYER_MYYC',
  100: 'RGB',
  101: 'BGR',
};

const HEADER_BYTES = 178;

function cstr(bytes) {
  let end = bytes.indexOf(0);
  if (end < 0) end = bytes.length;
  return new TextDecoder('latin1').decode(bytes.subarray(0, end)).trim();
}

// .NET ticks (100ns since 0001-01-01) -> Date, or null if zero.
function serDate(ticks) {
  if (!ticks) return null;
  const EPOCH_TICKS = 621355968000000000n; // 1970-01-01 in .NET ticks
  return new Date(Number((ticks - EPOCH_TICKS) / 10000n));
}

export async function parseSerHeader(file) {
  if (file.size < HEADER_BYTES) throw new Error('file too small to be a SER file');
  const buf = await file.slice(0, HEADER_BYTES).arrayBuffer();
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);

  const h = {
    fileId: cstr(u8.subarray(0, 14)),
    luId: dv.getInt32(14, true),
    colorId: dv.getInt32(18, true),
    littleEndianFlag: dv.getInt32(22, true),
    width: dv.getInt32(26, true),
    height: dv.getInt32(30, true),
    pixelDepthPerPlane: dv.getInt32(34, true),
    frameCount: dv.getInt32(38, true),
    observer: cstr(u8.subarray(42, 82)),
    instrument: cstr(u8.subarray(82, 122)),
    telescope: cstr(u8.subarray(122, 162)),
    dateTime: dv.getBigInt64(162, true),
    dateTimeUtc: dv.getBigInt64(170, true),
  };

  if (h.width <= 0 || h.height <= 0 || h.frameCount <= 0 ||
      h.pixelDepthPerPlane < 1 || h.pixelDepthPerPlane > 16) {
    throw new Error(
      `implausible SER header (id="${h.fileId}", ${h.width}×${h.height}, ` +
      `depth ${h.pixelDepthPerPlane}, ${h.frameCount} frames)`);
  }

  h.colorName = COLOR_IDS[h.colorId] || `unknown(${h.colorId})`;
  h.channels = (h.colorId === 100 || h.colorId === 101) ? 3 : 1;
  h.bytesPerSample = h.pixelDepthPerPlane > 8 ? 2 : 1;
  h.frameBytes = h.width * h.height * h.channels * h.bytesPerSample;
  h.maxValue = (1 << h.pixelDepthPerPlane) - 1;
  h.dataBytes = h.frameBytes * h.frameCount;
  // Optional trailer: 8-byte UTC timestamps per frame after all frame data.
  h.hasTimestamps = file.size >= HEADER_BYTES + h.dataBytes + 8 * h.frameCount;
  h.recordedUtc = serDate(h.dateTimeUtc) || serDate(h.dateTime);

  if (file.size < HEADER_BYTES + h.dataBytes) {
    const fit = Math.floor((file.size - HEADER_BYTES) / h.frameBytes);
    if (fit < 1) throw new Error('SER file truncated: no complete frames');
    h.frameCount = fit; // tolerate truncated captures
    h.truncated = true;
  }
  return h;
}

// Read frame `index` -> Uint16Array or Uint8Array of w*h*channels samples.
// NOTE on endianness: the SER spec's LittleEndian flag is famously inverted in
// the wild — nearly all writers store little-endian data regardless of the
// flag. We therefore default to little-endian and let the caller override.
export async function readSerFrame(file, header, index, { bigEndian = false } = {}) {
  if (index < 0 || index >= header.frameCount) throw new Error('frame out of range');
  const start = HEADER_BYTES + index * header.frameBytes;
  const buf = await file.slice(start, start + header.frameBytes).arrayBuffer();
  if (header.bytesPerSample === 1) return new Uint8Array(buf);
  const arr = new Uint16Array(buf); // fresh ArrayBuffer -> aligned, host LE
  if (bigEndian) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      arr[i] = ((v & 0xff) << 8) | (v >> 8);
    }
  }
  return arr;
}
