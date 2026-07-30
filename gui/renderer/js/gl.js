// WebGL2 renderer for SER frames.
//
// Frames are uploaded as *integer* textures (R8UI / R16UI). WebGL2 has no
// 3-channel renderable/paddable RGB16UI path worth fighting with, so RGB/BGR
// frames are uploaded as a single-channel texture of width*3 and the fragment
// shader does manual channel indexing (texelFetch at x*channels + c). The same
// path handles MONO and Bayer (channels = 1), with bilinear debayer for
// RGGB / GRBG / GBRG / BGGR done in the shader.
//
// Display conversion is linear -> sRGB only (IEC 61966-2-1). No auto-stretch.

const VS = `#version 300 es
void main() {
  // fullscreen triangle from gl_VertexID
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D u_tex;    // width = imageWidth * channels
uniform ivec2 u_size;        // image size in pixels
uniform int u_channels;      // 1 or 3
uniform int u_mode;          // 0 mono, 1 rgb, 2 bgr, 3 bayer
uniform ivec2 u_bayerRed;    // position of R within the 2x2 tile
uniform float u_maxVal;      // (1 << bit_depth) - 1
uniform vec2 u_canvas;       // canvas size in device pixels
uniform float u_zoom;        // power-of-2 scale, image px -> canvas px
uniform vec2 u_center;       // image-space point at the canvas center

out vec4 outColor;

float fetchSample(ivec2 p, int c) {
  p = clamp(p, ivec2(0), u_size - 1);
  return float(texelFetch(u_tex, ivec2(p.x * u_channels + c, p.y), 0).r) / u_maxVal;
}

vec3 debayer(ivec2 p) {
  ivec2 par = ivec2(p.x & 1, p.y & 1);
  float self = fetchSample(p, 0);
  float cross = 0.25 * (fetchSample(p + ivec2(-1, 0), 0) + fetchSample(p + ivec2(1, 0), 0)
                      + fetchSample(p + ivec2(0, -1), 0) + fetchSample(p + ivec2(0, 1), 0));
  float diag  = 0.25 * (fetchSample(p + ivec2(-1, -1), 0) + fetchSample(p + ivec2(1, -1), 0)
                      + fetchSample(p + ivec2(-1, 1), 0) + fetchSample(p + ivec2(1, 1), 0));
  float horiz = 0.5 * (fetchSample(p + ivec2(-1, 0), 0) + fetchSample(p + ivec2(1, 0), 0));
  float vert  = 0.5 * (fetchSample(p + ivec2(0, -1), 0) + fetchSample(p + ivec2(0, 1), 0));

  if (par == u_bayerRed) {
    return vec3(self, cross, diag);                     // red site
  } else if (par == (ivec2(1) - u_bayerRed)) {
    return vec3(diag, cross, self);                     // blue site
  } else if (par.y == u_bayerRed.y) {
    return vec3(horiz, self, vert);                     // green on red row
  } else {
    return vec3(vert, self, horiz);                     // green on blue row
  }
}

vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
  // canvas pixel (y-down) -> image pixel
  vec2 fragPx = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y);
  vec2 imgPos = (fragPx - 0.5 * u_canvas) / u_zoom + u_center;

  if (imgPos.x < 0.0 || imgPos.y < 0.0 ||
      imgPos.x >= float(u_size.x) || imgPos.y >= float(u_size.y)) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  ivec2 p = ivec2(floor(imgPos));

  vec3 rgb;
  if (u_mode == 0) {
    rgb = vec3(fetchSample(p, 0));
  } else if (u_mode == 1) {
    rgb = vec3(fetchSample(p, 0), fetchSample(p, 1), fetchSample(p, 2));
  } else if (u_mode == 2) {
    rgb = vec3(fetchSample(p, 2), fetchSample(p, 1), fetchSample(p, 0));
  } else {
    rgb = debayer(p);
  }
  outColor = vec4(linearToSrgb(rgb), 1.0);
}`;

const BAYER_RED = {
  8: [0, 0],   // RGGB
  9: [1, 0],   // GRBG
  10: [0, 1],  // GBRG
  11: [1, 1],  // BGGR
};

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader compile failed: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

export function createSerRenderer(canvas) {
  const gl = canvas.getContext('webgl2', {
    preserveDrawingBuffer: true, // lets tests read pixels back
    antialias: false,
    depth: false,
  });
  if (!gl) return null;

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('program link failed: ' + gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);
  const U = {};
  for (const name of ['u_tex', 'u_size', 'u_channels', 'u_mode', 'u_bayerRed',
                      'u_maxVal', 'u_canvas', 'u_zoom', 'u_center']) {
    U[name] = gl.getUniformLocation(prog, name);
  }
  gl.uniform1i(U.u_tex, 0);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  let tex = null;
  let header = null;

  return {
    gl,

    configure(h) {
      header = h;
      if (tex) gl.deleteTexture(tex);
      tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const texW = h.width * h.channels;
      const internal = h.bytesPerSample === 2 ? gl.R16UI : gl.R8UI;
      gl.texStorage2D(gl.TEXTURE_2D, 1, internal, texW, h.height);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

      gl.uniform2i(U.u_size, h.width, h.height);
      gl.uniform1i(U.u_channels, h.channels);
      gl.uniform1f(U.u_maxVal, h.maxValue);
      let mode = 0;
      if (h.colorId === 100) mode = 1;
      else if (h.colorId === 101) mode = 2;
      else if (BAYER_RED[h.colorId]) {
        mode = 3;
        gl.uniform2i(U.u_bayerRed, ...BAYER_RED[h.colorId]);
      }
      gl.uniform1i(U.u_mode, mode);
    },

    upload(data) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const type = header.bytesPerSample === 2 ? gl.UNSIGNED_SHORT : gl.UNSIGNED_BYTE;
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0,
        header.width * header.channels, header.height,
        gl.RED_INTEGER, type, data);
    },

    render(view) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(U.u_canvas, canvas.width, canvas.height);
      gl.uniform1f(U.u_zoom, view.zoom);
      gl.uniform2f(U.u_center, view.centerX, view.centerY);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}
