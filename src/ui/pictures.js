
/* ---------- activation pictures ---------- */
// Pictures are drawn into <canvas> elements right after their markup lands, in the same task, so a
// refresh never shows an empty frame (images from data URLs decode later and blink black).
const vizJobs = new Map();
let vizN = 0;
const toURL = (w, h, fill) => ({w, h, fill});
function pic(p, cls='', label=''){
  if (vizJobs.size > 4000) vizJobs.clear();
  const id = String(++vizN); vizJobs.set(id, p);
  return `<canvas${cls ? ` class="${cls}"` : ''} width="${p.w}" height="${p.h}" data-viz="${id}"${label ? ` role="img" aria-label="${esc(label)}"` : ' aria-hidden="true"'}></canvas>`;
}
// Each picture is enlarged by a whole number first (every cell an exact block of pixels), then the browser
// scales it smoothly to its box. Stretching the tiny original with nearest-neighbour scaling made some rows
// and columns one screen pixel thicker than others, which showed as faint lines that shifted on every pan.
const vizSrc = document.createElement('canvas'), vizSrcCtx = vizSrc.getContext('2d');
function paintViz(el){
  drawPlots(el);
  el.querySelectorAll('canvas[data-viz]').forEach(c => {
    const p = vizJobs.get(c.dataset.viz); c.removeAttribute('data-viz'); if (!p) return;
    vizJobs.delete(c.dataset.viz);
    const S = Math.max(1, Math.min(8, Math.ceil(256/Math.max(p.w, p.h))));
    vizSrc.width = p.w; vizSrc.height = p.h;
    const im = vizSrcCtx.createImageData(p.w, p.h); p.fill(im.data); vizSrcCtx.putImageData(im, 0, 0);
    c.width = p.w*S; c.height = p.h*S;
    const ctx = c.getContext('2d'); ctx.imageSmoothingEnabled = false; ctx.drawImage(vizSrc, 0, 0, c.width, c.height);
  });
}
const norm = (v, lo, hi) => hi > lo ? (v - lo)/(hi - lo) : hi > 0 ? 1 : 0;
const gray = (px, i, a) => { const v = Math.round(255*a); px[i] = px[i+1] = px[i+2] = v; px[i+3] = 255; };
// Diverging colormap for activations: black at zero, blue for negative, orange for positive,
// both brightening toward white at the extremes. a is the value divided by the tensor's max |value|.
const NEG = [[40,120,255],[200,235,255]], POS = [[255,110,40],[255,235,200]];
function divRGB(a){
  const t = Math.min(1, Math.abs(a)), [c1, c2] = a < 0 ? NEG : POS;
  if (t < .6) return c1.map(c => Math.round(c*t/.6));
  const k = (t - .6)/.4; return c1.map((c, i) => Math.round(c + (c2[i] - c)*k));
}
const divCSS = a => `rgb(${divRGB(a).join(',')})`;
const paint = (px, i, a) => { const [r, g, b] = divRGB(a); px[i] = r; px[i+1] = g; px[i+2] = b; px[i+3] = 255; };
const amax = v => { let m = 0; for (const x of v){ const y = Math.abs(x); if (y > m) m = y; } return m || 1; };
const CBAR = `<div class="cbar"><i></i><span>−max</span><span>0</span><span>+max</span></div>`;
// A Data component's sample: an imported image in grayscale; a collected one (any values) in activation colors.
function sampleHTML(s, big){
  if (s.pixels) return pic(sampleURL(s), big ? 'dbig' : 'dimg', `A handwritten ${s.label}`);
  const t = s.t;
  if (t.shape.length === 3){ let g = ''; for (let c = 0; c < Math.min(big ? 16 : 4, t.shape[0]); c++) g += pic(mapURL(t, c), '', `Channel ${c+1}`); return `<div class="chgrid${big ? '' : ' sm'}">${g}</div>`; }
  return pic(vecURL(t.data, big ? 128 : 64), 'act v');
}
function sampleURL(s){ return toURL(s.ds.w, s.ds.h, px => { for (let i = 0; i < s.pixels.length; i++) gray(px, i*4, s.pixels[i]/255); }); }
// Raw input pixels (0 to 1) in grayscale, for the Data component.
function grayURL(t, c){ const [, h, w] = t.shape, v = t.data.subarray(c*h*w, (c+1)*h*w); return toURL(w, h, px => { for (let i = 0; i < v.length; i++) gray(px, i*4, Math.max(0, Math.min(1, v[i]))); }); }
function mapURL(t, c){
  const [, h, w] = t.shape, v = t.data.subarray(c*h*w, (c+1)*h*w), m = amax(v);
  return toURL(w, h, px => { for (let i = 0; i < v.length; i++) paint(px, i*4, v[i]/m); });
}
function vecURL(v, cells=64){
  const n = Math.min(cells, v.length), b = v.length / n, a = new Float32Array(n);
  for (let i = 0; i < n; i++){ let s = 0, c = 0; for (let j = Math.floor(i*b); j < Math.floor((i+1)*b); j++){ s += v[j]; c++; } a[i] = c ? s/c : 0; }
  const m = amax(a);
  return toURL(n, 1, px => { for (let i = 0; i < n; i++) paint(px, i*4, a[i]/m); });
}
// Feature maps tiled into one picture, each channel on its own scale, with thin gaps between tiles.
function montageURL(t, max=16){
  const [C, h, w] = t.shape, n = Math.min(C, max), g = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n/g), W = g*w + g - 1, H = rows*h + rows - 1;
  return toURL(W, H, px => {
    for (let i = 0; i < px.length; i += 4){ px[i] = px[i+1] = px[i+2] = 38; px[i+3] = 255; }
    for (let c = 0; c < n; c++){
      const v = t.data.subarray(c*h*w, (c+1)*h*w), m = amax(v), ox = (c % g)*(w + 1), oy = Math.floor(c/g)*(h + 1);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) paint(px, ((oy + y)*W + ox + x)*4, v[y*w + x]/m);
    }
  });
}
function panel(t, label){
  let inner = '';
  if (t && isPred(t)){
    const [lo, hi] = range(t.data), am = argmax(t.data);
    inner = `<div class="pred">${[...t.data].map((x, i) => `<span class="pb${i === am ? ' hi' : ''}${i === t.label ? ' lab' : ''}" title="${i}: ${x.toFixed(3)}"><s><i style="height:${Math.round(3 + 97*norm(x, Math.min(0, lo), hi))}%"></i></s><b>${i}</b></span>`).join('')}</div>`;
  } else if (t) inner = pic(tensorURL(t), t.shape.length === 3 ? '' : 'strip');
  return `<div class="pn${t ? '' : ' empty'}"><span class="pl">${label}</span>${inner}</div>`;
}
function toggleView(ids){
  const ns = ids.map(id => byId(cur(), id)).filter(Boolean); if (!ns.length) return;
  const on = !ns.every(n => n.view);
  ns.forEach(n => { if (on) n.view = true; else delete n.view; });
  render();
}
const viewBtn = n => `<button class="tb" data-act="view" aria-pressed="${!!n.view}">View activations</button>`;
// Feature maps as tiles; anything else as one horizontal strip (long vectors averaged down to 80 cells).
const tensorURL = t => t.shape.length === 3 ? montageURL(t) : vecURL(t.data, 80);
function actHTML(t){
  if (t.shape.length === 1) return pic(vecURL(t.data), 'act v');
  if (t.shape.length !== 3) return '';
  const C = Math.min(6, t.shape[0]); let h = '';
  for (let c = 0; c < C; c++) h += pic(mapURL(t, c));
  return `<div class="act m">${h}${t.shape[0] > C ? `<span>+${t.shape[0]-C}</span>` : ''}</div>`;
}
const isPred = t => t && t.shape.length === 1 && t.data.length <= 20;
function predHTML(t, big){
  const v = t.data, [lo, hi] = range(v), am = argmax(v);
  // Only a vector with one score per class of the dataset in use is read as a prediction.
  const dn = root.nodes.find(m => m.type === 'data'), ds = dn && dsById(dn.props.dataset), isClass = ds && v.length === ds.classes, lab = isClass ? t.label : undefined;
  let bars = '';
  v.forEach((x, i) => bars += `<span class="pb${i === am ? ' hi' : ''}${i === lab ? ' lab' : ''}" title="${i}: ${x.toFixed(3)}"><s><i style="height:${Math.round(3 + 97*norm(x, Math.min(0, lo), hi))}%"></i></s><b>${i}</b></span>`);
  if (!isClass) return `<div class="pred${big ? ' big' : ''}">${bars}</div>`;
  const verdict = lab === undefined ? '' : am === lab ? '<em class="ok">✓ matches the label</em>' : `<em class="no">label is ${lab}</em>`;
  return `<div class="pred${big ? ' big' : ''}">${bars}</div><div class="pv">Predicts<b>${am}</b>${verdict}</div>`;
}
function actPanel(n){
  const t = n._act; if (!t) return '';
  const [lo, hi] = range(t.data); let s = 0; for (const v of t.data) s += v;
  let vis = '';
  if (isPred(t)) vis = predHTML(t, true);
  else if (t.shape.length === 1) vis = pic(vecURL(t.data, 128), 'act v');
  else if (t.shape.length === 3){ let g = ''; for (let c = 0; c < Math.min(16, t.shape[0]); c++) g += pic(mapURL(t, c), '', `Channel ${c+1}`); vis = `<div class="chgrid">${g}</div>`; }
  return `<div class="actp"><span class="cap">Activation</span>${vis}${isPred(t) ? '' : CBAR}${dimsRows([['Min', lo.toFixed(3)], ['Mean', (s/t.data.length).toFixed(3)], ['Max', hi.toFixed(3)]])}</div>`;
}

/* ---------- Plot 2D ---------- */
const CAT = ['#4FC3FF', '#FF8A3D', '#7CFF8A', '#FF5C8A', '#C49BFF', '#FFE15C', '#5CFFE0', '#FF9CE6', '#B8FF5C', '#9DB4FF'];   // one colour per label
const plotHTML = (n, big) => `<canvas class="plot${n.type === 'plot3d' ? ' p3' : ''}${big ? ' big' : ''}" data-plot="${n.id}" role="img" aria-label="${n.type === 'plot3d' ? '3D' : ''} scatter plot of ${esc(n.name)}"></canvas>`;
function drawPlots(el){ el.querySelectorAll('canvas[data-plot]').forEach(c => { const n = findNode(c.dataset.plot); if (n && n.type === 'plot3d') drawPlot3(c, n); else drawPlot(c, n); }); }
function drawPlot(c, n){
  const big = c.classList.contains('big'), dpr = window.devicePixelRatio || 1, w = c.clientWidth || 200, h = c.clientHeight || w;
  c.width = Math.round(w*dpr); c.height = Math.round(h*dpr);
  const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  g.font = `${big ? 10 : 9}px "JetBrains Mono", ui-monospace, monospace`; g.textBaseline = 'middle';
  if (!n) return;
  const xd = Number(n.props.x_dim), yd = Number(n.props.y_dim);
  if (n._pts && n._pts.raw && (n._pts.xd !== xd || n._pts.yd !== yd) && Math.max(xd, yd) < sizeOf(n._pts.raw.shape)) keepPlot(n, n._pts.raw, n._pts.classes, n._pts.from);   // other axes
  const P = n._pts, t = !(n._r && n._r.err) && n._actIn && n._actIn[0];
  // What arrives now: a matrix is one point per row; anything else is one point, the sample on screen.
  let live = [];
  if (t){ if (t.shape.length === 2){ const K = t.shape[1]; for (let r = 0; r < t.shape[0]; r++) live.push([t.data[r*K + xd], t.data[r*K + yd]]); } else live = [[t.data[xd], t.data[yd]]]; }
  const many = t && t.shape.length === 2;
  if (!(P && P.n) && !live.length){ g.fillStyle = 'rgba(255,255,255,.45)'; g.textAlign = 'center'; g.fillText('—', w/2, h/2); return; }
  const span = (a, extra) => { let lo = Infinity, hi = -Infinity; const see = v => { if (isFinite(v)){ if (v < lo) lo = v; if (v > hi) hi = v; } }; if (a) for (const v of a) see(v); extra.forEach(see); if (!(hi > lo)){ lo = (isFinite(lo) ? lo : 0) - 1; hi = (isFinite(hi) ? hi : 0) + 1; } const pad = (hi - lo)*.06; return [lo - pad, hi + pad]; };
  const [x0, x1] = span(P && P.x, live.map(q => q[0])), [y0, y1] = span(P && P.y, live.map(q => q[1])), m = big ? 22 : 14;
  const X = v => m + (v - x0)/(x1 - x0)*(w - 2*m), Y = v => h - m - (v - y0)/(y1 - y0)*(h - 2*m);
  g.strokeStyle = 'rgba(255,255,255,.16)'; g.lineWidth = 1; g.beginPath();
  if (x0 < 0 && x1 > 0){ g.moveTo(X(0), m/2); g.lineTo(X(0), h - m/2); }
  if (y0 < 0 && y1 > 0){ g.moveTo(m/2, Y(0)); g.lineTo(w - m/2, Y(0)); }
  g.stroke(); g.strokeStyle = 'rgba(255,255,255,.28)'; g.strokeRect(m - .5, m - .5, w - 2*m + 1, h - 2*m + 1);
  const r = big ? 2.2 : 1.5;
  g.globalAlpha = .8;
  if (P) for (let i = 0; i < P.n; i++){ g.fillStyle = CAT[((P.labels[i] % 10) + 10) % 10]; g.beginPath(); g.arc(X(P.x[i]), Y(P.y[i]), r, 0, 7); g.fill(); }
  g.globalAlpha = 1;
  if (many){ g.fillStyle = '#fff'; for (const [a, b] of live) if (isFinite(a) && isFinite(b)){ g.beginPath(); g.arc(X(a), Y(b), r + .6, 0, 7); g.fill(); } }
  else if (live.length && isFinite(live[0][0]) && isFinite(live[0][1])){   // the sample on screen
    g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.shadowColor = '#fff'; g.shadowBlur = 8;
    g.beginPath(); g.arc(X(live[0][0]), Y(live[0][1]), big ? 6 : 4.5, 0, 7); g.stroke(); g.shadowBlur = 0;
  }
  g.fillStyle = 'rgba(255,255,255,.55)'; const f = v => Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(2);
  g.textAlign = 'left'; g.fillText(f(x0), m, h - m/2); g.textAlign = 'right'; g.fillText(f(x1), w - m, h - m/2);
  g.save(); g.translate(m/2, h - m); g.rotate(-Math.PI/2); g.textAlign = 'left'; g.fillText(f(y0), 0, 0); g.textAlign = 'right'; g.fillText(f(y1), -(h - 2*m), 0); g.restore();
}
// The points an inference run left in a Plot 2D: two values of each sample that reached it, with its label.
function keepPlot(m, c, classes, from){
  const S = sizeOf(c.shape), xd = Number(m.props.x_dim), yd = Number(m.props.y_dim), zd = m.type === 'plot3d' ? Number(m.props.z_dim) : null;
  const x = new Float32Array(c.n), y = new Float32Array(c.n), z = zd == null ? null : new Float32Array(c.n);
  for (let i = 0; i < c.n; i++){ x[i] = c.x[i*S + xd]; y[i] = c.x[i*S + yd]; if (z) z[i] = c.x[i*S + zd]; }
  m._pts = {n:c.n, x, y, z, labels:c.labels, classes, from, raw:c, xd, yd, zd};
}
function plotNote(n){
  const P = n._pts;
  if (!P || !P.n) return `<small class="vnote">Shows what reaches it: the sample on screen as a ring, or a matrix's rows as points. Run inference on a Data component upstream and it keeps every sample that arrives.</small>`;
  const k = Math.min(10, P.classes || 0);
  return `${k ? `<div class="legend">${Array.from({length:k}, (_, i) => `<span><i style="background:${CAT[i]}"></i>${i}</span>`).join('')}</div>` : ''}<small class="vnote">${P.n.toLocaleString()} samples that reached it in the last inference run (${esc(P.from)}), coloured by label. The ring is the sample on screen.${n.type === 'plot3d' ? ' Drag to turn it, scroll to zoom, Shift-drag to move, double-click to reset.' : ''}</small>`;
}

/* ---------- Plot 3D ---------- */
/* Each axis is scaled to the cube −1…1; the camera orbits it: yaw turns around the vertical (third) axis, pitch tilts,
   zoom and pan move the picture. The camera lives on the component (n.cam), so it stays as it was left. */
const CAM0 = {yaw:-0.7, pitch:0.45, zoom:1, px:0, py:0};
const camOf = n => n.cam || (n.cam = {...CAM0});
function drawPlot3(c, n){
  const big = c.classList.contains('big'), dpr = window.devicePixelRatio || 1, w = c.clientWidth || 200, h = c.clientHeight || w;
  c.width = Math.round(w*dpr); c.height = Math.round(h*dpr);
  const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  g.font = `${big ? 10 : 9}px "JetBrains Mono", ui-monospace, monospace`; g.textBaseline = 'middle'; g.textAlign = 'center';
  if (!n) return;
  const dims = [n.props.x_dim, n.props.y_dim, n.props.z_dim].map(Number);
  if (n._pts && n._pts.raw && (n._pts.xd !== dims[0] || n._pts.yd !== dims[1] || n._pts.zd !== dims[2]) && Math.max(...dims) < sizeOf(n._pts.raw.shape)) keepPlot(n, n._pts.raw, n._pts.classes, n._pts.from);
  const P = n._pts && n._pts.z ? n._pts : null, t = !(n._r && n._r.err) && n._actIn && n._actIn[0];
  let live = [];
  if (t){ if (t.shape.length === 2){ const K = t.shape[1]; for (let r = 0; r < t.shape[0]; r++) live.push(dims.map(d => t.data[r*K + d])); } else live = [dims.map(d => t.data[d])]; }
  const many = t && t.shape.length === 2;
  if (!(P && P.n) && !live.length){ g.fillStyle = 'rgba(255,255,255,.45)'; g.fillText('—', w/2, h/2); return; }
  // Each axis onto −1…1.
  const rng3 = [0, 1, 2].map(a => { let lo = Infinity, hi = -Infinity; const see = v => { if (isFinite(v)){ if (v < lo) lo = v; if (v > hi) hi = v; } }; if (P) for (const v of [P.x, P.y, P.z][a]) see(v); live.forEach(q => see(q[a])); if (!(hi > lo)){ lo = (isFinite(lo) ? lo : 0) - 1; hi = (isFinite(hi) ? hi : 0) + 1; } return [lo, hi]; });
  const nrm = (v, a) => (v - rng3[a][0])/(rng3[a][1] - rng3[a][0])*2 - 1;
  const cam = camOf(n), cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch), D = 4.5, S = Math.min(w, h)*0.25*cam.zoom;
  const proj = (x, y, z) => { const x1 = x*cy - y*sy, d1 = x*sy + y*cy, up = z*cp - d1*sp, depth = z*sp + d1*cp, f = D/(D + depth); return [w/2 + cam.px + x1*f*S, h/2 + cam.py - up*f*S, depth, f]; };
  // The cube and its axes.
  g.strokeStyle = 'rgba(255,255,255,.14)'; g.lineWidth = 1; g.beginPath();
  const C = [-1, 1];
  for (const a of C) for (const b of C){ for (const [p0, p1] of [[[-1, a, b], [1, a, b]], [[a, -1, b], [a, 1, b]], [[a, b, -1], [a, b, 1]]]){ const u = proj(...p0), v = proj(...p1); g.moveTo(u[0], u[1]); g.lineTo(v[0], v[1]); } }
  g.stroke();
  const o = proj(-1, -1, -1);
  [['#FF7F7F', [1, -1, -1]], ['#7CFF8A', [-1, 1, -1]], ['#7CC4FF', [-1, -1, 1]]].forEach(([col, e], a) => {
    const q = proj(...e); g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(o[0], o[1]); g.lineTo(q[0], q[1]); g.stroke();
    const lx = q[0] + (q[0] - o[0])*.08, ly = q[1] + (q[1] - o[1])*.08; g.fillStyle = col; g.fillText(`${'xyz'[a]}${big ? ' [' + dims[a] + ']' : ''}`, lx, ly);
  });
  // Points, far ones first.
  if (P){
    const N = P.n, sx = new Float32Array(N), sy2 = new Float32Array(N), dz = new Float32Array(N), fs = new Float32Array(N), ord = new Uint32Array(N);
    for (let i = 0; i < N; i++){ const q = proj(nrm(P.x[i], 0), nrm(P.y[i], 1), nrm(P.z[i], 2)); sx[i] = q[0]; sy2[i] = q[1]; dz[i] = q[2]; fs[i] = q[3]; ord[i] = i; }
    ord.sort((a, b) => dz[b] - dz[a]);
    const dense = Math.min(1, Math.sqrt(2000/N)), r0 = (big ? 1.5 : 1)*(.55 + .45*dense)*Math.sqrt(cam.zoom), a0 = .3 + .55*dense;   // many points: smaller and fainter
    for (const i of ord){ const r = r0*fs[i]; g.globalAlpha = Math.max(.15, Math.min(.95, a0*(1.15 - .3*dz[i]))); g.fillStyle = CAT[((P.labels[i] % 10) + 10) % 10]; g.fillRect(sx[i] - r, sy2[i] - r, 2*r, 2*r); }
    g.globalAlpha = 1;
  }
  if (many){ g.fillStyle = '#fff'; for (const q3 of live){ const q = proj(nrm(q3[0], 0), nrm(q3[1], 1), nrm(q3[2], 2)); g.fillRect(q[0] - 1.6, q[1] - 1.6, 3.2, 3.2); } }
  else if (live.length && live[0].every(isFinite)){   // the sample on screen
    const q = proj(nrm(live[0][0], 0), nrm(live[0][1], 1), nrm(live[0][2], 2));
    g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.shadowColor = '#fff'; g.shadowBlur = 8; g.beginPath(); g.arc(q[0], q[1], big ? 6 : 4.5, 0, 7); g.stroke(); g.shadowBlur = 0;
  }
}
const redrawPlot3 = n => document.querySelectorAll(`canvas[data-plot="${n.id}"]`).forEach(c => drawPlot3(c, n));
// Drag turns the camera (Shift-drag moves it); the wheel zooms; a double-click puts it back.
function startOrbit(e, c){
  const n = findNode(c.dataset.plot); if (!n) return;
  e.preventDefault(); e.stopPropagation();
  const cam = camOf(n), sx = e.clientX, sy = e.clientY, c0 = {...cam}, pan = e.shiftKey; let raf = 0;
  c.style.cursor = 'grabbing';
  track(m => {
    const dx = m.clientX - sx, dy = m.clientY - sy;
    if (pan){ cam.px = c0.px + dx; cam.py = c0.py + dy; }
    else { cam.yaw = c0.yaw - dx*0.01; cam.pitch = Math.max(-1.55, Math.min(1.55, c0.pitch + dy*0.01)); }
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; redrawPlot3(n); });
  }, () => { c.style.cursor = ''; save(); });
}
function zoomOrbit(e, c){
  const n = findNode(c.dataset.plot); if (!n) return;
  e.preventDefault(); e.stopPropagation();
  const cam = camOf(n); cam.zoom = Math.max(0.2, Math.min(20, cam.zoom*Math.exp(-e.deltaY*0.0015)));
  redrawPlot3(n); save();
}
function resetOrbit(c){ const n = findNode(c.dataset.plot); if (!n) return; n.cam = {...CAM0}; redrawPlot3(n); save(); }
// Where a press can turn a 3D plot: in the info window, or on the component in view mode.
const orbitTarget = e => { const c = e.target.closest && e.target.closest('canvas.p3'); return c && (c.closest('.insp') || c.closest('.node.view')) ? c : null; };
