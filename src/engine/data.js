/* Netgraph engine: datasets. An imported dataset is {id, name, c, h, w, classes, n, images:Uint8Array, labels}: c channels
   (1 for grayscale, 3 for RGB; missing means 1), each image stored channel by channel, and labels as a Uint8Array, or a
   Uint16Array for more than 256 classes.
   A collected one (what reached a Data component's input during an inference run) is {id, name, shape, classes, n,
   x:Float32Array, labels, owner (the collecting component's id), from (where the samples came from)}. */

/* ---------- datasets ---------- */
let datasets = [];
const dsById = id => datasets.find(d => d.id === id) || null;
const imgShape = ds => [ds.c || 1, ds.h, ds.w];
function joinParts(base, parts){
  const px = sizeOf(imgShape(base)), n = parts.reduce((a, p) => a + p.n, 0), images = new Uint8Array(n*px), labels = new (base.classes > 256 ? Uint16Array : Uint8Array)(n);
  let o = 0; for (const p of parts){ images.set(p.images, o*px); labels.set(p.labels, o); o += p.n; }
  const {splits, ...rest} = base;
  return {...rest, n, images, labels};
}
const dsShape = (ds, flatten) => { const s = ds.x ? ds.shape : imgShape(ds); return flatten ? [sizeOf(s)] : s; };
function sampleAt(ds, i, flatten){
  if (ds.x){ const S = sizeOf(ds.shape); return {pixels:null, t:tensor(dsShape(ds, flatten), 1, ds.x.slice(i*S, (i + 1)*S), [ds.labels[i]])}; }
  const px = sizeOf(imgShape(ds)), pixels = ds.images.subarray(i*px, (i+1)*px), data = new Float32Array(px);
  for (let k = 0; k < px; k++) data[k] = pixels[k] / 255;
  return {pixels, t:tensor(dsShape(ds, flatten), 1, data, [ds.labels[i]])};
}
function sampleOf(n){
  const ds = dsById(n.props.dataset); if (!ds || (n._collect && !ds.x)) return null;   // a collecting component shows only what it collected
  const i = Math.max(0, Math.min(ds.n - 1, Math.floor(Number(n.props.index)) || 0)), s = sampleAt(ds, i, n.props.flatten);
  return {i, label:ds.labels[i], pixels:s.pixels, ds, t:s.t};
}
// Train/test split: shuffle every sample with the seed, then hold out the first test% as the test set.
const splitCache = new Map();
function splitOf(n){
  const ds = dsById(n.props.dataset); if (!ds) return null;
  const pct = Math.max(0, Math.min(90, Number(n.props.test ?? 20) || 0)), seed = Math.floor(Number(n.props.seed ?? 1)) || 0;
  const key = `${ds.id}:${ds.n}:${seed}:${pct}`;
  let sp = splitCache.get(key);
  if (!sp){
    if (splitCache.size > 8) splitCache.clear();
    const perm = new Uint32Array(ds.n), r = rng(hashStr(ds.id) ^ Math.imul(seed, 2654435761));
    for (let i = 0; i < ds.n; i++) perm[i] = i;
    for (let i = ds.n - 1; i > 0; i--){ const j = Math.floor(r()*(i + 1)), t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    const nt = Math.round(ds.n*pct/100), isTest = new Uint8Array(ds.n);
    for (let i = 0; i < nt; i++) isTest[perm[i]] = 1;
    sp = {test:perm.subarray(0, nt), train:perm.subarray(nt), isTest};
    splitCache.set(key, sp);
  }
  return sp;
}
// A batch of samples as one tensor, pixels scaled to 0..1, with their labels.
function batchOf(ds, idxs, flatten){
  if (ds.x){
    const S = sizeOf(ds.shape), B = idxs.length, data = new Float32Array(B*S), labels = new Int32Array(B);
    for (let b = 0; b < B; b++){ data.set(ds.x.subarray(idxs[b]*S, (idxs[b] + 1)*S), b*S); labels[b] = ds.labels[idxs[b]]; }
    return tensor(dsShape(ds, flatten), B, data, labels);
  }
  const px = sizeOf(imgShape(ds)), B = idxs.length, data = new Float32Array(B*px), labels = new Int32Array(B);
  for (let b = 0; b < B; b++){
    const o = idxs[b]*px;
    for (let k = 0; k < px; k++) data[b*px + k] = ds.images[o + k]/255;
    labels[b] = ds.labels[idxs[b]];
  }
  return tensor(dsShape(ds, flatten), B, data, labels);
}
