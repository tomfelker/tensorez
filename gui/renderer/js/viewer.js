// Full-screen image viewer with power-of-2 zoom and drag pan.
// Used by the run console and the results viewer. No auto-stretch: images are
// shown exactly as encoded (previews are already linear->sRGB per contract).

const lb = document.getElementById('lightbox');
let zoomExp = 0; // zoom = 2^zoomExp
let panX = 0, panY = 0;
let img = null;

function zoomLabel() {
  const z = Math.pow(2, zoomExp);
  return z >= 1 ? `×${z}` : `×1/${1 / z}`;
}

function apply() {
  if (!img) return;
  const z = Math.pow(2, zoomExp);
  img.style.transform = `translate(${panX}px, ${panY}px) scale(${z})`;
  lb.querySelector('.lb-zoom').textContent = zoomLabel();
}

function build(title, src) {
  lb.innerHTML = `
    <div class="lb-bar">
      <button class="small" data-lb="close">✕ close</button>
      <div class="lb-title">${title}</div>
      <div class="grow" style="flex:1"></div>
      <button class="small" data-lb="out">−</button>
      <div class="lb-zoom">×1</div>
      <button class="small" data-lb="in">+</button>
      <button class="small" data-lb="fit">reset</button>
      <span class="dim" style="font-size:11px">pow2 zoom · drag to pan · ESC to close</span>
    </div>
    <div class="lb-canvas"></div>`;
  const canvas = lb.querySelector('.lb-canvas');
  img = document.createElement('img');
  img.src = src;
  img.alt = title;
  img.draggable = false;
  canvas.appendChild(img);

  img.addEventListener('load', () => {
    // center at 1:1
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    panX = (cw - img.naturalWidth) / 2;
    panY = (ch - img.naturalHeight) / 2;
    zoomExp = 0;
    apply();
  });

  lb.querySelector('[data-lb=close]').addEventListener('click', close);
  lb.querySelector('[data-lb=in]').addEventListener('click', () => zoomAt(1));
  lb.querySelector('[data-lb=out]').addEventListener('click', () => zoomAt(-1));
  lb.querySelector('[data-lb=fit]').addEventListener('click', () => {
    zoomExp = 0;
    panX = (canvas.clientWidth - img.naturalWidth) / 2;
    panY = (canvas.clientHeight - img.naturalHeight) / 2;
    apply();
  });

  function zoomAt(dir, cx, cy) {
    const rect = canvas.getBoundingClientRect();
    const px = cx ?? rect.width / 2;
    const py = cy ?? rect.height / 2;
    const oldZ = Math.pow(2, zoomExp);
    zoomExp = Math.max(-4, Math.min(6, zoomExp + dir));
    const z = Math.pow(2, zoomExp);
    // keep the point under (px,py) fixed
    panX = px - (px - panX) * (z / oldZ);
    panY = py - (py - panY) * (z / oldZ);
    apply();
  }

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1 : -1, e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, px: panX, py: panY };
    canvas.classList.add('panning');
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    panX = drag.px + (e.clientX - drag.x);
    panY = drag.py + (e.clientY - drag.y);
    apply();
  });
  canvas.addEventListener('pointerup', () => { drag = null; canvas.classList.remove('panning'); });
}

function onKey(e) {
  if (e.key === 'Escape') close();
}

export function openViewer(title, src) {
  build(title, src);
  lb.hidden = false;
  window.addEventListener('keydown', onKey);
}

export function close() {
  lb.hidden = true;
  lb.innerHTML = '';
  img = null;
  window.removeEventListener('keydown', onKey);
}
