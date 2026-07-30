// Tiny dev server for the TensoRez renderer.
//
//   node dev-server.mjs [port]     (default 8123)
//
// Serves:
//   /              -> renderer/            (the web app)
//   /examples/*    -> gui/examples/*  falling back to  ../examples/*
//                     (mock JSONL event streams live in gui/examples,
//                      synthetic_planet.ser etc. in the repo examples dir)
//   /mockrun/*     -> gui/mockrun/*        (fake completed run directory)
//   /fs/<abs path> -> real files, restricted to FS_ROOTS below. Lets the
//                     browser-hosted renderer open REAL run directories
//                     (e.g. CLI output under /root/tensorez-next/cli/output)
//                     through the mock bridge. Dev convenience only.
//
// The renderer auto-installs the mock bridge when window.bridge is absent,
// so opening http://localhost:8123/ in any browser gives a fully working
// (mocked) app.

import http from 'node:http';
import { promises as fsp } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.argv[2] || process.env.PORT || 8123);

// absolute-path roots the /fs/ route may serve from
const FS_ROOTS = [path.resolve(HERE, '..'), '/tmp'];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.jsonl': 'application/x-ndjson',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.toml': 'text/plain; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.ser': 'application/octet-stream',
  '.tif': 'image/tiff',
  '.npy': 'application/octet-stream',
};

async function tryFile(p) {
  try {
    const st = await fsp.stat(p);
    if (st.isFile()) return p;
  } catch {}
  return null;
}

async function resolvePath(urlPath) {
  // strip query, decode, and neutralize traversal
  let p = decodeURIComponent(urlPath.split('?')[0]);
  p = path.posix.normalize(p);
  if (p.includes('..')) return null;
  if (p === '/') p = '/index.html';

  if (p.startsWith('/examples/')) {
    const rel = p.slice('/examples/'.length);
    return (await tryFile(path.join(HERE, 'examples', rel)))
        ?? (await tryFile(path.join(HERE, '..', 'examples', rel)));
  }
  if (p.startsWith('/mockrun/')) {
    return tryFile(path.join(HERE, p.slice(1)));
  }
  if (p.startsWith('/fs/')) {
    const abs = path.resolve('/', p.slice('/fs/'.length));
    if (!FS_ROOTS.some((root) => abs === root || abs.startsWith(root + path.sep))) return null;
    return tryFile(abs);
  }
  return tryFile(path.join(HERE, 'renderer', p.slice(1)));
}

const server = http.createServer(async (req, res) => {
  const file = await resolvePath(req.url);
  if (!file) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('not found: ' + req.url);
    return;
  }
  const data = await fsp.readFile(file);
  res.writeHead(200, {
    'content-type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
    'cache-control': 'no-store',
  });
  res.end(data);
});

server.listen(PORT, () => {
  console.log(`TensoRez dev server: http://localhost:${PORT}/`);
});
