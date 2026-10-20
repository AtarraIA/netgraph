/* Netgraph engine: weights.
   Every trainable layer's parameters live here by layer id: {key, W, B, trained}, plus for BatchNorm the
   running statistics rm (mean), rv (variance) and nbt (batches seen). key records the shape they were made
   for, so a layer whose shape changes gets fresh weights. */
// wver counts changes to it, so the page can tell whether a result it shows is still current.
let wver = 0;
const wcache = new class extends Map {
  set(k, v){ wver++; return super.set(k, v); }
  delete(k){ wver++; return super.delete(k); }
  clear(){ wver++; return super.clear(); }
}();

// Fresh weights follow PyTorch's defaults (uniform ±1/√fan_in for weights and biases), seeded by the layer id
// so they stay put. BatchNorm starts at scale 1 and shift 0, with running mean 0 and running variance 1.
function weightsOf(n, wn, bn, fanIn, kind=''){
  const key = `${wn}:${bn}:${kind}`;
  let p = wcache.get(n.id);
  if (!p || p.key !== key){
    const r = rng(hashStr(n.id)), b = 1/Math.sqrt(fanIn), W = new Float32Array(wn), B = new Float32Array(bn);
    if (kind === 'bn') W.fill(1);
    else { for (let i = 0; i < wn; i++) W[i] = (r()*2 - 1)*b; for (let i = 0; i < bn; i++) B[i] = (r()*2 - 1)*b; }
    p = {key, W, B};
    wcache.set(n.id, p);
  }
  if (kind === 'bn' && !p.rm){ p.rm = new Float32Array(bn); p.rv = new Float32Array(bn).fill(1); p.nbt = 0; }   // also for weights saved before BatchNorm kept statistics
  return p;
}
const bnParams = (n, C) => weightsOf(n, C, C, 1, 'bn');

// Plain copies of the weights, for moving them between the page and the training worker, and for storage.
function exportWeights(onlyTrained){
  return [...wcache].filter(([, P]) => !onlyTrained || P.trained).map(([id, P]) => ({
    id, key:P.key, W:P.W.slice(), B:P.B.slice(), rm:P.rm ? P.rm.slice() : null, rv:P.rv ? P.rv.slice() : null, nbt:P.nbt || 0, trained:!!P.trained,
  }));
}
function importWeights(recs){
  for (const r of recs) wcache.set(r.id, {key:r.key, W:r.W, B:r.B, rm:r.rm || undefined, rv:r.rv || undefined, nbt:r.nbt || 0, trained:r.trained !== false});
}
