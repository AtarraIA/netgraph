/* Netgraph engine: running the network.
   A tensor holds B samples of one shape back to back: {shape (one sample's), B, data, labels}. labels[b] is
   sample b's class when the data has labels; a one-sample tensor also keeps it as .label for the UI.
   forward() runs the sample shown on screen (forwardSteps does the same a step at a time, for the page to show
   results as they come); fwdT/bwdT run a training batch forward and backward. */
const sizeOf = s => s.reduce((a, b) => a*b, 1);
function tensor(shape, B, data, labels){
  const t = {shape, B, data, labels:labels || null};
  if (B === 1 && labels) t.label = labels[0];
  return t;
}
const BN_EPS = 1e-5, BN_MOMENTUM = 0.1;   // PyTorch's BatchNorm defaults
const now = () => typeof performance !== 'undefined' ? performance.now() : Date.now();

// One layer, forward. mode is undefined at inference. In training it is {train:true, bn:Map}: BatchNorm then
// normalizes with the batch's own statistics and updates its running ones, like PyTorch's model.train().
function runNode(n, ins, mode){
  const x = ins[0], p = n._p || n.props, X = x.data, B = x.B, S = X.length/B, out = (shape, data) => tensor(shape, B, data, x.labels);
  switch (n.type){
    case 'dense': case 'conv2d': { const it = layerSteps(n, ins); let r; while (!(r = it.next()).done); return r.value; }
    case 'maxpool': case 'avgpool': {
      const [c,h,w] = x.shape, [,ho,wo] = n._out, k = +p.kernel, st = isAuto(p.stride) ? k : +p.stride, mx = n.type === 'maxpool', Y = new Float32Array(B*c*ho*wo);
      for (let bt = 0; bt < B; bt++){
        const xo = bt*c*h*w, yo = bt*c*ho*wo;
        for (let ci = 0; ci < c; ci++) for (let i = 0; i < ho; i++) for (let j = 0; j < wo; j++){
          let v = mx ? -Infinity : 0;
          for (let a = 0; a < k; a++) for (let b = 0; b < k; b++){ const q = X[xo + (ci*h+i*st+a)*w+j*st+b]; v = mx ? Math.max(v, q) : v + q; }
          Y[yo + (ci*ho+i)*wo+j] = mx ? v : v/(k*k);
        }
      }
      return out(n._out, Y);
    }
    case 'gap': {
      const [c,h,w] = x.shape, hw = h*w, Y = new Float32Array(B*c);
      for (let bt = 0; bt < B; bt++) for (let ci = 0; ci < c; ci++){ let s = 0; const o = bt*S + ci*hw; for (let i = 0; i < hw; i++) s += X[o+i]; Y[bt*c + ci] = s/hw; }
      return out(n._out, Y);
    }
    case 'flatten': case 'unflatten': return out(n._out, X);
    case 'activation': { const k = presetOf(n); return k ? runNode({...n, type:k}, ins, mode) : actCustom(n, x); }
    case 'relu': return out(x.shape, X.map(v => v > 0 ? v : 0));
    case 'gelu': return out(x.shape, X.map(v => 0.5*v*(1 + Math.tanh(0.7978845608*(v + 0.044715*v*v*v)))));   // PyTorch's GELU(approximate="tanh")
    case 'sigmoid': return out(x.shape, X.map(v => 1/(1 + Math.exp(-v))));
    case 'tanh': return out(x.shape, X.map(v => Math.tanh(v)));
    case 'softmax': {   // over features for vectors, over channels for maps (dim=1 with the batch)
      const C = x.shape.length === 1 ? S : x.shape[0], G = S/C, Y = new Float32Array(X.length);
      for (let bt = 0; bt < B; bt++) for (let g = 0; g < G; g++){
        const o = bt*S + g; let m = -Infinity, s = 0;
        for (let c = 0; c < C; c++) m = Math.max(m, X[o + c*G]);
        for (let c = 0; c < C; c++) s += (Y[o + c*G] = Math.exp(X[o + c*G] - m));
        for (let c = 0; c < C; c++) Y[o + c*G] /= s;
      }
      return out(x.shape, Y);
    }
    case 'batchnorm': {
      const C = x.shape[0], per = S/C, P = bnParams(n, C), Y = new Float32Array(X.length);
      if (mode && mode.train){
        const m = B*per, mean = new Float64Array(C), vr = new Float64Array(C);
        for (let bt = 0; bt < B; bt++) for (let c = 0; c < C; c++){ const o = bt*S + c*per; for (let j = 0; j < per; j++) mean[c] += X[o+j]; }
        for (let c = 0; c < C; c++) mean[c] /= m;
        for (let bt = 0; bt < B; bt++) for (let c = 0; c < C; c++){ const o = bt*S + c*per; for (let j = 0; j < per; j++){ const d = X[o+j] - mean[c]; vr[c] += d*d; } }
        for (let c = 0; c < C; c++) vr[c] /= m;
        const inv = Float32Array.from(vr, v => 1/Math.sqrt(v + BN_EPS)), xhat = new Float32Array(X.length);
        for (let bt = 0; bt < B; bt++) for (let c = 0; c < C; c++){
          const o = bt*S + c*per;
          for (let j = 0; j < per; j++){ const i = o + j; xhat[i] = (X[i] - mean[c])*inv[c]; Y[i] = P.W[c]*xhat[i] + P.B[c]; }
        }
        if (mode.updateStats !== false){   // running statistics use the unbiased variance, as in PyTorch
          for (let c = 0; c < C; c++){ P.rm[c] = (1 - BN_MOMENTUM)*P.rm[c] + BN_MOMENTUM*mean[c]; P.rv[c] = (1 - BN_MOMENTUM)*P.rv[c] + BN_MOMENTUM*vr[c]*(m > 1 ? m/(m - 1) : 1); }
          P.nbt = (P.nbt || 0) + 1;
        }
        mode.bn.set(n.id, {xhat, inv});
      } else {
        for (let c = 0; c < C; c++){
          const inv = 1/Math.sqrt(P.rv[c] + BN_EPS);
          for (let bt = 0; bt < B; bt++){ const o = bt*S + c*per; for (let j = 0; j < per; j++) Y[o+j] = P.W[c]*(X[o+j] - P.rm[c])*inv + P.B[c]; }
        }
      }
      return out(x.shape, Y);
    }
    case 'dropout': case 'output': case 'gout': case 'ptrin': return out(x.shape, X);
    case 'plot2d': case 'plot3d': return null;   // it shows what reaches it; nothing goes out
    case 'pca': {   // y = components · (x − mean)
      const k = lastDim(n._out), P = pcaParams(n, S, k); if (!P) return null;
      const Y = new Float32Array(B*k), c = new Float32Array(S);
      for (let b = 0; b < B; b++){ for (let i = 0; i < S; i++) c[i] = X[b*S + i] - P.B[i]; for (let j = 0; j < k; j++){ let s = 0; for (let i = 0, o = j*S; i < S; i++) s += P.W[o + i]*c[i]; Y[b*k + j] = s; } }
      return out([k], Y);
    }   // dropout is off at inference
    case 'add': { const Y2 = ins[1].data; return out(x.shape, X.map((v, i) => v + Y2[i])); }
    case 'concat': {
      const Z = ins[1].data, Sb = Z.length/B, Y = new Float32Array(X.length + Z.length);
      for (let bt = 0; bt < B; bt++){ Y.set(X.subarray(bt*S, (bt+1)*S), bt*(S + Sb)); Y.set(Z.subarray(bt*Sb, (bt+1)*Sb), bt*(S + Sb) + S); }
      return out(n._out, Y);
    }
  }
  return null;
}

// A Sliders component's values, the same for every sample of the batch. They are settings, so training leaves them alone.
function slidersTensor(n, B){
  const v = (n.props.sliders || []).map(s => Number(s.value) || 0), N = v.length, data = new Float32Array(B*N);
  for (let b = 0; b < B; b++) data.set(v, b*N);
  return tensor([N], B, data, null);
}
// An Activation with a function of its own: evaluated sample by sample from the formula (x is the sample).
function actCustom(n, x){
  const ast = bindVars(parseLoss(n.props.fn), n, true), B = x.B, S = x.data.length/B, So = sizeOf(n._out), Y = new Float32Array(B*So);
  for (let b = 0; b < B; b++){
    const r = evalLoss(ast, () => ({shape:x.shape, data:x.data.subarray(b*S, (b + 1)*S)}), () => undefined, false);
    if (typeof r.v === 'number') Y.fill(r.v, b*So, (b + 1)*So); else Y.set(r.v, b*So);
  }
  return tensor(n._out, B, Y, x.labels);
}
function actCustomBack(n, x, gy){
  const ast = bindVars(parseLoss(n.props.fn), n, true), B = x.B, S = x.data.length/B, So = gy.length/B, gx = new Float32Array(x.data.length);
  for (let b = 0; b < B; b++){
    const seed = gy.subarray(b*So, (b + 1)*So);
    const r = evalLoss(ast, () => ({shape:x.shape, data:x.data.subarray(b*S, (b + 1)*S)}), () => undefined, true, 1, seed);
    const g = r.grads.get('x'); if (g) gx.set(g, b*S);
  }
  return gx;
}
/* Expressions over a batch: each sample is evaluated on its own slice of every tensor it names. src is the
   expression's text (a Loss block's, or one input cell's); getT(component) gives a component's batch tensor. seeds(b), when given, starts the gradient for sample b: a number for a Loss
   (d total / d this loss) or d(total)/d(output values) for a Flatten. Gradients come back gathered per component
   over the whole batch, and straight into the weights for .pre_weights/.post_weights. */
function exprEval(n, src, getT, B, seeds){
  const ast = bindVars(parseLoss(src), n), Ls = new Float32Array(B), vals = [], grads = new Map();
  let correct = 0, scored = 0, labels = null;
  for (let b = 0; b < B; b++){
    let label;
    const get = key => {
      const m = lookup(key, n._g).node, t = m && getT(m);
      if (!t) throw new Error(`${key.split(SEP).join('.')} has no value for this sample. Is it fed by the Data component?`);
      const S = t.data.length/t.B, bb = t.B === 1 ? 0 : b;
      if (label === undefined && t.labels) label = t.labels[bb];
      return {shape:t.shape, data:t.data.subarray(bb*S, (bb + 1)*S)};
    };
    const seed = seeds ? seeds(b) : null;
    const r = evalLoss(ast, get, () => label, !!seeds, typeof seed === 'number' ? seed : 1, seed && typeof seed !== 'number' ? seed : null);
    Ls[b] = r.L; vals.push(r.v);
    if (r.acc != null){ correct += r.acc; scored++; }
    if (label !== undefined) (labels ||= new Int32Array(B))[b] = label;
    if (seeds) for (const [key, gr] of r.grads){
      if (gr.P){   // weights: summed over the batch
        const o = grads.get(key);
        if (o) for (let i = 0; i < o.g.length; i++) o.g[i] += gr.g[i]; else grads.set(key, {P:gr.P, pos:gr.pos, g:Float32Array.from(gr.g)});
        continue;
      }
      const t = getT(lookup(key, n._g).node), S = t.data.length/t.B, bb = t.B === 1 ? 0 : b;
      let o = grads.get(key); if (!o){ o = new Float32Array(t.data.length); grads.set(key, o); }
      for (let i = 0; i < S; i++) o[bb*S + i] += gr[i];
    }
  }
  return {Ls, vals, grads, correct, scored, labels};
}
// An input given by an expression: its values for each sample, laid out in the shape inference worked out.
function exprInput(n, i, getT, B){
  const r = exprEval(n, portExpr(n, i), getT, B), shape = n._in[i], N = sizeOf(shape), data = new Float32Array(B*N);
  r.vals.forEach((v, b) => { if (typeof v === 'number') data.fill(v, b*N, (b + 1)*N); else data.set(v, b*N); });
  return tensor(shape, B, data, r.labels);
}
// Sends an input's gradient on into the components its expression reads.
function exprInputBack(ctx, n, i, gx, getT){
  const B = ctx.B, S = gx.length/B;
  seedGrads(ctx, exprEval(n, portExpr(n, i), getT, B, b => gx.subarray(b*S, (b + 1)*S)).grads, n._g);
}

// The sample shown on screen, through every layer a Data component reaches. Sets n._act (output) and
// n._actIn (inputs) on every node, and n._lossNow on Loss blocks. BatchNorm uses its running statistics.
function forward(g, ginT){ const it = forwardSteps(g, ginT); let r; while (!(r = it.next()).done); return r.value; }
// The on-screen run, a step at a time. Yields {id, frac} as a component makes progress and {id, done:true} once its
// result (n._act) is in; a block also yields its own progress, by how many of its components are done. Each component
// keeps its previous result until its turn comes, so the page can show the old one meanwhile.
function* forwardSteps(g, ginT){
  let res = null;
  const seen = new Set();
  for (const n of topo(g)){
    seen.add(n.id); n._act = null; n._actIn = null; n._lossNow = null;
    const d = DEFS[n.type], ins = new Array(d.ins).fill(null);
    for (const e of g.edges) if (e.to === n.id && e.port < d.ins && wired(n, e.port)){ const s = byId(g, e.from); ins[e.port] = s ? s._act : null; }
    if (n._r && !n._r.err) for (let i = 0; i < d.ins; i++) if (portExpr(n, i)) try { ins[i] = exprInput(n, i, m => m._act, 1); } catch(e){ ins[i] = null; }
    if (n.type === 'loss'){
      try { if (n._r && !n._r.err){ const r = exprEval(n, n.props.expr, m => m._act, 1); n._lossNow = r.Ls[0]; n._act = tensor([1], 1, r.Ls, r.labels); } } catch(e){}
      yield {id:n.id, done:true};
      continue;
    }
    let t = null;
    if (n.type === 'data') t = n._r && !n._r.err ? (sampleOf(n) || {}).t || null : null;
    else if (n.type === 'sliders') t = n._r && !n._r.err ? slidersTensor(n, 1) : null;
    else if (n.type === 'gin') t = ginT;
    else if (n.type === 'ptrout'){ const x = n._ptr && n._r && !n._r.err && n._ptr.I._act; try { t = x ? runPointer(n, x) : null; } catch(e){ t = null; } }   // the stretch again, on its Pointer in's input
    else if (d.ins && !n._src && n._r && !n._r.err && !n._r.wait && ins.every(Boolean)){
      n._actIn = ins;
      try {
        if (n.type === 'group') t = yield* blockSteps(n, ins[0]);
        else if (n.type === 'dense' || n.type === 'conv2d'){ const it = layerSteps(n, ins); let r; while (!(r = it.next()).done) yield {id:n.id, frac:r.value}; t = r.value; }
        else t = runNode(n, ins);
      } catch(e){ t = null; }
    } else if (n.type === 'group') yield* blockSteps(n, null);
    n._act = t;
    if (n.type === 'gout') res = t;
    yield {id:n.id, done:true};
  }
  for (const n of g.nodes) if (!seen.has(n.id)){ n._act = null; n._actIn = null; n._lossNow = null; }   // not reached (a cycle)
  return res;
}
function* blockSteps(n, x){
  const it = forwardSteps(n.graph, x), N = n.graph.nodes.length || 1;
  let k = 0, r;
  while (!(r = it.next()).done){
    const e = r.value; yield e;
    if (n.graph.nodes.some(m => m.id === e.id)){ if (e.done) k++; yield {id:n.id, frac:(k + (e.done ? 0 : e.frac))/N}; }
  }
  return r.value;
}

/* ---------- training ---------- */
function pgrad(ctx, P){ let G = ctx.pg.get(P); if (!G){ G = {W:new Float32Array(P.W.length), B:new Float32Array(P.B.length)}; ctx.pg.set(P, G); } return G; }
function orderOf(ctx, g){ let o = ctx.order.get(g); if (!o){ o = topo(g); ctx.order.set(g, o); } return o; }
function addGrad(ctx, id, gx){ const o = ctx.grad.get(id); if (!o) ctx.grad.set(id, gx); else ctx.grad.set(id, o.map((v, i) => v + gx[i])); }
function seedGrads(ctx, grads, g){
  for (const [key, gr] of grads){
    if (gr.P){ const G = pgrad(ctx, gr.P); gr.pos.forEach((q, k) => G.W[q] += gr.g[k]); continue; }
    const rr = lookup(key, g); if (!rr.node) continue;
    addGrad(ctx, rr.node.id, gr); rr.groups.forEach(b => ctx.seeded.add(b.id));
  }
}
// Forward pass for a training batch: keeps every layer's inputs and output so the backward pass can use them.
function fwdT(g, ginT, xT, ctx){
  let res = null;
  const getT = m => ctx.out.get(m.id), mode = ctx.train ? {train:true, bn:ctx.bn} : undefined;
  for (const n of orderOf(ctx, g)){
    const d = DEFS[n.type], ins = new Array(d.ins).fill(null);
    for (const e of g.edges) if (e.to === n.id && e.port < d.ins && wired(n, e.port)) ins[e.port] = ctx.out.get(e.from) || null;
    if (n._r && !n._r.err) for (let i = 0; i < d.ins; i++) if (portExpr(n, i)) try { ins[i] = exprInput(n, i, getT, ctx.B); } catch(e){ ins[i] = null; }
    if (n.type === 'loss'){
      if (!n._r || n._r.err) continue;
      if (n === ctx.loss){
        const r = exprEval(n, n.props.expr, getT, ctx.B);
        ctx.out.set(n.id, tensor([1], ctx.B, r.Ls, r.labels));
        ctx.L = total(r.Ls)/ctx.B; ctx.correct = r.correct; ctx.scored = r.scored;
      } else try { const r = exprEval(n, n.props.expr, getT, ctx.B); ctx.out.set(n.id, tensor([1], ctx.B, r.Ls, r.labels)); } catch(e){}
      continue;
    }
    let t = null;
    if (ctx.cap && n.type === 'pca' && ins[0]) ctx.cap.set(n.id, ins[0]);   // what a PCA is fitted on
    if (n === ctx.src) t = xT;
    else if (n.type === 'data'){ if (ctx.cap && n._collect && ins[0]) ctx.cap.set(n.id, ins[0]); }   // an inference run: keep what reached it
    else if (PLOTS.has(n.type)){ if (ctx.cap && ins[0]) ctx.cap.set(n.id, ins[0]); }
    else if (n.type === 'sliders') t = n._r && !n._r.err ? slidersTensor(n, ctx.B) : null;
    else if (n.type === 'gin') t = ginT;
    else if (n.type === 'ptrout'){ const x = n._ptr && n._r && !n._r.err && ctx.out.get(n._ptr.I.id); if (x){ ctx.ins.set(n.id, [x]); t = runPointer(n, x, ctx); } }
    else if (d.ins && (!n._src || n.id === ctx.ptrFrom) && n._r && !n._r.err && !n._r.wait && ins.every(Boolean)){
      ctx.ins.set(n.id, ins);
      if (n.type === 'group') t = fwdT(n.graph, ins[0], xT, ctx);
      else if (n.type === 'dropout' && ctx.train){
        const q = Number((n._p || n.props).p), X = ins[0].data, m = new Float32Array(X.length);
        for (let i = 0; i < m.length; i++) m[i] = Math.random() >= q ? 1/(1 - q) : 0;
        ctx.masks.set(n.id, m); t = tensor(ins[0].shape, ins[0].B, X.map((v, i) => v*m[i]), ins[0].labels);
      } else t = runNode(n, ins, mode);
    }
    else if (n.type === 'group') t = fwdT(n.graph, null, xT, ctx);   // nothing reaches it, but it may hold its own Data
    if (t) ctx.out.set(n.id, t);
    if (n.type === 'gout') res = t;
  }
  return res;
}
// Dense and Conv2D, a few output units (or channels) at a time: yields the fraction done after each slice of about
// LAYER_SLICE multiply-adds, and returns the output.
const LAYER_SLICE = 1e6;
function* layerSteps(n, ins){
  const x = ins[0], p = n._p || n.props, X = x.data, B = x.B;
  if (n.type === 'dense'){
    const fi = lastDim(x.shape), fo = lastDim(n._out), rows = X.length/fi, P = weightsOf(n, fi*fo, fo, fi), Y = new Float32Array(rows*fo);
    const step = Math.max(1, Math.floor(LAYER_SLICE/(rows*fi)));
    for (let o0 = 0; o0 < fo; o0 += step){ denseUnits(P, !!p.bias, X, Y, rows, fi, fo, o0, Math.min(fo, o0 + step)); if (o0 + step < fo) yield (o0 + step)/fo; }
    return tensor(n._out, B, Y, x.labels);
  }
  // Conv2D as a matrix product per sample: Y[o, p] = bias[o] + Σ_q W[o, q]·patch[p, q], where patch p is the input under
  // the kernel at output position p (im2col), q runs over in-channel × kernel row × kernel column, as W does.
  const [c,h,w] = x.shape, [co,ho,wo] = n._out, k = +p.kernel, st = +p.stride, pd = +p.padding, K = c*k*k, N = ho*wo;
  const P = weightsOf(n, co*K, co, K), Y = new Float32Array(B*co*N), cols = new Float32Array(N*K);
  const step = Math.min(co, Math.max(4, 4*Math.floor(LAYER_SLICE/(4*N*K))));
  for (let bt = 0; bt < B; bt++){
    im2colT(X, bt*c*h*w, c, h, w, k, st, pd, ho, wo, cols);
    const Yb = Y.subarray(bt*co*N, (bt + 1)*co*N);
    for (let o0 = 0; o0 < co; o0 += step){
      const o1 = Math.min(co, o0 + step);
      matmulNT(P.W.subarray(o0*K, o1*K), o1 - o0, cols, N, K, Yb.subarray(o0*N), N, 1, P.B.subarray(o0, o1), false);
      if (bt < B - 1 || o1 < co) yield (bt*co + o1)/(B*co);
    }
  }
  return tensor(n._out, B, Y, x.labels);
}
// The patches under a k×k kernel, one row per output position: cols[p*K + q] (zero where the kernel overhangs the padding).
function im2colT(X, xo, c, h, w, k, st, pd, ho, wo, cols){
  const K = c*k*k;
  for (let i = 0; i < ho; i++) for (let j = 0; j < wo; j++){
    let r = (i*wo + j)*K;
    for (let ci = 0; ci < c; ci++) for (let a = 0; a < k; a++){
      const y = i*st - pd + a, xr = xo + (ci*h + y)*w, inY = y >= 0 && y < h;
      for (let b = 0; b < k; b++){ const xx = j*st - pd + b; cols[r++] = inY && xx >= 0 && xx < w ? X[xr + xx] : 0; }
    }
  }
}
// The reverse: adds each patch's values back where they came from (patches overlap, so values add up).
function col2imT(cols, gx, xo, c, h, w, k, st, pd, ho, wo){
  for (let i = 0; i < ho; i++) for (let j = 0; j < wo; j++){
    let r = (i*wo + j)*c*k*k;
    for (let ci = 0; ci < c; ci++) for (let a = 0; a < k; a++){
      const y = i*st - pd + a, xr = xo + (ci*h + y)*w, inY = y >= 0 && y < h;
      for (let b = 0; b < k; b++, r++){ const xx = j*st - pd + b; if (inY && xx >= 0 && xx < w) gx[xr + xx] += cols[r]; }
    }
  }
}
/* Y[m*ym + n*yn] = bias[m] + Σ_k A[m*K + k]·Bt[n*K + k] (or += without the bias when add): the product of A (M×K)
   with Bt (N×K) transposed, both read along k. Done in 4×4 blocks with the 16 sums in registers, which needs far
   fewer memory reads than one dot product at a time. Sums run in k order in double precision, as a plain loop would. */
function matmulNT(A, M, Bt, N, K, Y, ym, yn, bias, add){
  let m = 0;
  for (; m + 4 <= M; m += 4){
    const a0 = m*K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
    let n = 0;
    for (; n + 4 <= N; n += 4){
      const c0 = n*K, c1 = c0 + K, c2 = c1 + K, c3 = c2 + K;
      let s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0, s20 = 0, s21 = 0, s22 = 0, s23 = 0, s30 = 0, s31 = 0, s32 = 0, s33 = 0;
      for (let q = 0; q < K; q++){
        const w0 = A[a0+q], w1 = A[a1+q], w2 = A[a2+q], w3 = A[a3+q], x0 = Bt[c0+q], x1 = Bt[c1+q], x2 = Bt[c2+q], x3 = Bt[c3+q];
        s00 += w0*x0; s01 += w0*x1; s02 += w0*x2; s03 += w0*x3;
        s10 += w1*x0; s11 += w1*x1; s12 += w1*x2; s13 += w1*x3;
        s20 += w2*x0; s21 += w2*x1; s22 += w2*x2; s23 += w2*x3;
        s30 += w3*x0; s31 += w3*x1; s32 += w3*x2; s33 += w3*x3;
      }
      put(Y, m*ym + n*yn, yn, s00, s01, s02, s03, bias ? bias[m] : 0, add);
      put(Y, (m+1)*ym + n*yn, yn, s10, s11, s12, s13, bias ? bias[m+1] : 0, add);
      put(Y, (m+2)*ym + n*yn, yn, s20, s21, s22, s23, bias ? bias[m+2] : 0, add);
      put(Y, (m+3)*ym + n*yn, yn, s30, s31, s32, s33, bias ? bias[m+3] : 0, add);
    }
    for (; n < N; n++) for (let r = 0; r < 4; r++) dotInto(A, (m + r)*K, Bt, n*K, K, Y, (m + r)*ym + n*yn, bias ? bias[m + r] : 0, add);
  }
  for (; m < M; m++) for (let n = 0; n < N; n++) dotInto(A, m*K, Bt, n*K, K, Y, m*ym + n*yn, bias ? bias[m] : 0, add);
}
function put(Y, i, yn, s0, s1, s2, s3, b, add){
  if (add){ Y[i] += s0; Y[i + yn] += s1; Y[i + 2*yn] += s2; Y[i + 3*yn] += s3; }
  else { Y[i] = b + s0; Y[i + yn] = b + s1; Y[i + 2*yn] = b + s2; Y[i + 3*yn] = b + s3; }
}
function dotInto(A, ao, Bt, bo, K, Y, i, b, add){ let s = 0; for (let q = 0; q < K; q++) s += A[ao + q]*Bt[bo + q]; if (add) Y[i] += s; else Y[i] = b + s; }
const transpose = (X, R, C, out = new Float32Array(R*C)) => { for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) out[c*R + r] = X[r*C + c]; return out; };
function denseUnits(P, bias, X, Y, rows, fi, fo, o0, o1){
  const W = P.W;
  for (let r = 0; r < rows; r++) for (let o = o0; o < o1; o++){
    let s = bias ? P.B[o] : 0;
    for (let i = 0, wo = o*fi, xo = r*fi; i < fi; i++) s += W[wo+i]*X[xo+i];
    Y[r*fo+o] = s;
  }
}

// A pointer pair's stretch, run on x with its own record of inputs and outputs (the stretch also runs in the main flow) but
// the same weights and weight gradients. Without ctx it runs as at inference, for the screen.
function runPointer(n, x, ctx){
  if (!n._pg) return null;
  const sub = {B:x.B, train:!!(ctx && ctx.train), out:new Map(), ins:new Map(), grad:new Map(), masks:new Map(), bn:new Map(), seeded:new Set(),
    order:ctx ? ctx.order : new Map(), pg:ctx ? ctx.pg : new Map(), ptrFrom:n._ptr.a.id};
  const y = fwdT(n._pg, x, null, sub);
  if (ctx) (ctx.subs ||= new Map()).set(n.id, sub);
  return y;
}
// Backward pass: walks the graph in reverse, collecting weight gradients in ctx.pg. Returns the gradient for a
// block's input. The batch loss is the mean over samples, so each sample's loss gradient starts at 1/B.
function bwdT(g, gOut, ctx){
  const order = orderOf(ctx, g), getT = m => ctx.out.get(m.id), B = ctx.B; let gIn = null;
  for (let k = order.length - 1; k >= 0; k--){
    const n = order[k];
    let gy = n.type === 'gout' ? gOut : ctx.grad.get(n.id);
    if (n.type === 'loss'){   // the trained loss starts the gradient; other losses it uses pass theirs on
      const seeds = n === ctx.loss ? () => 1/B : gy ? b => gy[b] : null;
      if (seeds && n._r && !n._r.err) seedGrads(ctx, exprEval(n, n.props.expr, getT, B, seeds).grads, n._g);
      continue;
    }
    if (!gy && !(n.type === 'group' && ctx.seeded.has(n.id))) continue;   // a block also runs backward when an expression reads a layer inside it
    if (n.type === 'gin'){ gIn = gy; continue; }
    if (n.type === 'group' && !ctx.ins.get(n.id)){ if (ctx.seeded.has(n.id)) bwdT(n.graph, null, ctx); continue; }   // a block that runs on its own Data
    const ins = ctx.ins.get(n.id); if (!ins) continue;
    if (n.type === 'ptrout'){ const gx = bwdT(n._pg, gy, ctx.subs.get(n.id)); if (gx) addGrad(ctx, n._ptr.I.id, gx); continue; }   // back to its Pointer in, not to the wire it reads the end from
    const gxs = n.type === 'group' ? [bwdT(n.graph, gy, ctx)] : backNode(n, ins, ctx.out.get(n.id), gy, ctx);
    for (const e of g.edges) if (e.to === n.id && wired(n, e.port) && gxs[e.port]) addGrad(ctx, e.from, gxs[e.port]);
    for (let i = 0; i < gxs.length; i++) if (gxs[i] && portExpr(n, i)) exprInputBack(ctx, n, i, gxs[i], getT);   // into what the expression reads
  }
  return gIn;
}
function backNode(n, ins, y, gy, ctx){
  const x = ins[0], X = x.data, p = n._p || n.props, B = x.B, S = X.length/B;
  switch (n.type){
    case 'pca': {   // fixed: the gradient passes back through the projection; the components don't train
      const k = lastDim(n._out), P = pcaParams(n, S, k), gx = new Float32Array(X.length);
      for (let b = 0; b < B; b++) for (let j = 0; j < k; j++){ const g = gy[b*k + j]; if (!g) continue; for (let i = 0, o = j*S; i < S; i++) gx[b*S + i] += P.W[o + i]*g; }
      return [gx];
    }
    case 'dense': {
      const fi = lastDim(x.shape), fo = lastDim(n._out), rows = X.length/fi, P = weightsOf(n, fi*fo, fo, fi), G = pgrad(ctx, P), W = P.W, gx = new Float32Array(X.length);
      for (let r = 0; r < rows; r++) for (let o = 0; o < fo; o++){
        const g = gy[r*fo+o]; if (!g) continue;
        if (p.bias) G.B[o] += g;
        for (let i = 0, wo = o*fi, xo = r*fi; i < fi; i++){ gx[xo+i] += W[wo+i]*g; G.W[wo+i] += g*X[xo+i]; }
      }
      return [gx];
    }
    case 'conv2d': {
      const [c,h,w] = x.shape, [co,ho,wo] = n._out, k = +p.kernel, st = +p.stride, pd = +p.padding;
      // As matrix products per sample, like the forward pass: dW[o, q] += Σ_p gy[o, p]·patch[p, q], and each patch's
      // gradient Σ_o gy[o, p]·W[o, q] added back into gx where the patch came from.
      const K = c*k*k, N = ho*wo, P = weightsOf(n, co*K, co, K), G = pgrad(ctx, P), gx = new Float32Array(X.length);
      const cols = new Float32Array(N*K), colsT = new Float32Array(K*N), gyT = new Float32Array(N*co), WT = transpose(P.W, co, K), dcols = new Float32Array(N*K);
      for (let bt = 0; bt < B; bt++){
        const g = gy.subarray(bt*co*N, (bt + 1)*co*N);
        for (let o = 0; o < co; o++){ let s = 0; for (let q = 0; q < N; q++) s += g[o*N + q]; G.B[o] += s; }
        im2colT(X, bt*c*h*w, c, h, w, k, st, pd, ho, wo, cols); transpose(cols, N, K, colsT);
        matmulNT(g, co, colsT, K, N, G.W, K, 1, null, true);
        transpose(g, co, N, gyT);
        matmulNT(gyT, N, WT, K, co, dcols, K, 1, null, false);
        col2imT(dcols, gx, bt*c*h*w, c, h, w, k, st, pd, ho, wo);
      }
      return [gx];
    }
    case 'maxpool': case 'avgpool': {
      const [c,h,w] = x.shape, [,ho,wo] = n._out, k = +p.kernel, st = isAuto(p.stride) ? k : +p.stride, gx = new Float32Array(X.length);
      for (let bt = 0; bt < B; bt++){
        const xo = bt*c*h*w, yo = bt*c*ho*wo;
        for (let ci = 0; ci < c; ci++) for (let i = 0; i < ho; i++) for (let j = 0; j < wo; j++){
          const g = gy[yo + (ci*ho+i)*wo+j]; if (!g) continue;
          if (n.type === 'maxpool'){   // to the first maximum in the window, as PyTorch does
            let bi = -1, bv = -Infinity;
            for (let a = 0; a < k; a++) for (let b = 0; b < k; b++){ const xi = xo + (ci*h+i*st+a)*w+j*st+b; if (X[xi] > bv){ bv = X[xi]; bi = xi; } }
            gx[bi] += g;
          } else for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) gx[xo + (ci*h+i*st+a)*w+j*st+b] += g/(k*k);
        }
      }
      return [gx];
    }
    case 'gap': { const [c, h, w] = x.shape, hw = h*w; return [X.map((_, i) => { const bt = Math.floor(i/S), ci = Math.floor((i % S)/hw); return gy[bt*c + ci]/hw; })]; }
    case 'activation': { const k = presetOf(n); return k ? backNode({...n, type:k}, ins, y, gy, ctx) : [actCustomBack(n, x, gy)]; }
    case 'relu': return [gy.map((g, i) => X[i] > 0 ? g : 0)];
    case 'gelu': return [gy.map((g, i) => { const v = X[i], c = 0.7978845608, th = Math.tanh(c*(v + 0.044715*v*v*v)); return g*(0.5*(1 + th) + 0.5*v*(1 - th*th)*c*(1 + 3*0.044715*v*v)); })];
    case 'sigmoid': return [gy.map((g, i) => g*y.data[i]*(1 - y.data[i]))];
    case 'tanh': return [gy.map((g, i) => g*(1 - y.data[i]*y.data[i]))];
    case 'softmax': {
      const Y = y.data, C = x.shape.length === 1 ? S : x.shape[0], Gn = S/C, gx = new Float32Array(X.length);
      for (let bt = 0; bt < B; bt++) for (let q = 0; q < Gn; q++){
        const o = bt*S + q; let s = 0;
        for (let c = 0; c < C; c++) s += gy[o + c*Gn]*Y[o + c*Gn];
        for (let c = 0; c < C; c++) gx[o + c*Gn] = Y[o + c*Gn]*(gy[o + c*Gn] - s);
      }
      return [gx];
    }
    case 'batchnorm': {
      const C = x.shape[0], per = S/C, P = bnParams(n, C), G = pgrad(ctx, P), cache = ctx.bn.get(n.id), gx = new Float32Array(X.length);
      if (cache){   // batch statistics: the gradient also flows through the batch mean and variance
        const {xhat, inv} = cache, m = B*per, sg = new Float64Array(C), sgx = new Float64Array(C);
        for (let bt = 0; bt < B; bt++) for (let c = 0; c < C; c++){ const o = bt*S + c*per; for (let j = 0; j < per; j++){ const i = o + j; sg[c] += gy[i]; sgx[c] += gy[i]*xhat[i]; } }
        for (let c = 0; c < C; c++){ G.B[c] += sg[c]; G.W[c] += sgx[c]; }
        for (let bt = 0; bt < B; bt++) for (let c = 0; c < C; c++){
          const o = bt*S + c*per, f = P.W[c]*inv[c]/m;
          for (let j = 0; j < per; j++){ const i = o + j; gx[i] = f*(m*gy[i] - sg[c] - xhat[i]*sgx[c]); }
        }
      } else {      // running statistics: a fixed scale and shift per channel
        for (let c = 0; c < C; c++){
          const inv = 1/Math.sqrt(P.rv[c] + BN_EPS);
          for (let bt = 0; bt < B; bt++){ const o = bt*S + c*per; for (let j = 0; j < per; j++){ const i = o + j, g = gy[i]; G.W[c] += g*(X[i] - P.rm[c])*inv; G.B[c] += g; gx[i] = g*P.W[c]*inv; } }
        }
      }
      return [gx];
    }
    case 'dropout': { const m = ctx.masks.get(n.id); return [m ? gy.map((g, i) => g*m[i]) : gy]; }
    case 'add': return [gy, gy];
    case 'concat': {
      const Sb = ins[1].data.length/B, ga = new Float32Array(X.length), gb = new Float32Array(ins[1].data.length);
      for (let bt = 0; bt < B; bt++){ const o = bt*(S + Sb); ga.set(gy.subarray(o, o + S), bt*S); gb.set(gy.subarray(o + S, o + S + Sb), bt*Sb); }
      return [ga, gb];
    }
  }
  return [gy];   // flatten, unflatten, output, gout
}

// What a training run needs: the Loss block, the Data component that feeds it, the dataset and its split.
function makeCtx(lossId){
  infer();
  const n = findNode(lossId);
  if (!n || n.type !== 'loss') throw new Error('That Loss block is gone.');
  if (n._r && n._r.err) throw new Error(n._r.err);
  const d = n._srcData, ds = d && dsById(d.props.dataset), sp = d && splitOf(d);
  if (!sp) throw new Error('The Loss block needs a Data component upstream.');
  const ups = []; for (let g = n._g; g && g._blk; g = g._blk._g) ups.push(g._blk.id);   // the blocks around the Loss run backward
  return {src:d, loss:n, ds, sp, pg:new Map(), ups};
}
// One batch forward, and backward when training (or when `backward` asks for gradients without training mode).
// train switches BatchNorm to batch statistics and dropout on. Returns the mean loss and how many samples scored correct.
function runBatch(ctx, idxs, train, backward = train){
  const xT = batchOf(ctx.ds, idxs, ctx.src.props.flatten);
  Object.assign(ctx, {B:idxs.length, train, out:new Map(), ins:new Map(), grad:new Map(), masks:new Map(), bn:new Map(), seeded:new Set(ctx.ups || []), order:new Map(), L:null, correct:0, scored:0});
  fwdT(root, null, xT, ctx);
  if (ctx.L == null) throw new Error('the Loss block got no value.');
  if (backward) bwdT(root, null, ctx);
  return {L:ctx.L, correct:ctx.correct, scored:ctx.scored};
}
// Adam (PyTorch defaults: betas 0.9 and 0.999, eps 1e-8) or SGD with momentum 0.9. Gradients are already
// averaged over the batch.
function optStep(ctx, kind, lr, t){
  const c1 = 1 - Math.pow(.9, t), c2 = 1 - Math.pow(.999, t);
  for (const [P, G] of ctx.pg){
    for (const k of ['W', 'B']){
      const w = P[k], g = G[k]; if (!w.length) continue;
      if (kind === 'sgd'){
        const v = (P.v ||= {})[k] ||= new Float32Array(w.length);
        for (let i = 0; i < w.length; i++){ v[i] = .9*v[i] + g[i]; w[i] -= lr*v[i]; }
      } else {
        const m = (P.m ||= {})[k] ||= new Float32Array(w.length), v = (P.v ||= {})[k] ||= new Float32Array(w.length);
        for (let i = 0; i < w.length; i++){ const gi = g[i]; m[i] = .9*m[i] + .1*gi; v[i] = .999*v[i] + .001*gi*gi; w[i] -= lr*(m[i]/c1)/(Math.sqrt(v[i]/c2) + 1e-8); }
      }
      g.fill(0);
    }
    P.trained = true;
  }
}
const EVAL_N = 2000, EVAL_BATCH = 256;   // test samples scored after each epoch, in batches of this size
async function evaluateSet(ctx, idxs, max, hooks){
  const N = Math.min(max, idxs.length); let Ls = 0, correct = 0, scored = 0, last = now(), done = 0;
  for (let i = 0; i < N; i += EVAL_BATCH){
    const part = idxs.subarray(i, Math.min(N, i + EVAL_BATCH)), r = runBatch(ctx, part, false);
    Ls += r.L*part.length; correct += r.correct; scored += r.scored; done += part.length;
    if (hooks && now() - last > 30){ await hooks.yield(); last = now(); if (hooks.shouldStop()) break; }
  }
  return {testLoss: done ? Ls/done : null, testAcc: scored ? correct/scored : null, testN:done};
}
/* A training run, the same in the worker and (as a fallback) on the page. hooks: yield() lets other work run,
   shouldStop() asks to stop, progress(T) reports now and then, epochDone(T) after each epoch. */
async function trainLoop(lossId, opts, hooks){
  const ctx = makeCtx(lossId), trainIdx = ctx.sp.train, {lr, bs, epochs, optimizer} = opts;
  if (!trainIdx.length) throw new Error('The Data component has no training samples. Lower its test split.');
  const T = {id:lossId, epoch:0, epochs, step:0, steps:Math.ceil(trainIdx.length/bs), hist:[], raw:[], tests:[], ema:null,
    trainAcc:null, testAcc:null, testLoss:null, testN:0, done:false, evaluating:false};
  for (const P of wcache.values()){ delete P.m; delete P.v; }   // fresh optimizer state
  let t = 0, last = now(), lastEmit = 0;
  for (; T.epoch < epochs && !hooks.shouldStop(); T.epoch++){
    const order = Uint32Array.from(trainIdx);
    for (let i = order.length - 1; i > 0; i--){ const j = Math.floor(Math.random()*(i + 1)), v = order[i]; order[i] = order[j]; order[j] = v; }
    let correct = 0, scored = 0;
    for (T.step = 0; T.step < T.steps && !hooks.shouldStop(); T.step++){
      const r = runBatch(ctx, order.subarray(T.step*bs, Math.min(order.length, (T.step + 1)*bs)), true);
      if (!isFinite(r.L)) throw new Error('the loss stopped being a number. Try a smaller learning rate.');
      optStep(ctx, optimizer, lr, ++t);
      T.ema = T.ema == null ? r.L : .98*T.ema + .02*r.L; T.hist.push(T.ema); T.raw.push(r.L);
      correct += r.correct; scored += r.scored; T.trainAcc = scored ? correct/scored : null;
      if (now() - last > 30){ await hooks.yield(); last = now(); if (last - lastEmit > 300){ lastEmit = last; hooks.progress(T); } }
    }
    if (!hooks.shouldStop() && ctx.sp.test.length){
      T.evaluating = true; hooks.progress(T);
      Object.assign(T, await evaluateSet(ctx, ctx.sp.test, EVAL_N, hooks));
      T.evaluating = false; T.tests.push({step:T.hist.length - 1, L:T.testLoss});
    }
    if (hooks.epochDone) hooks.epochDone(T);
  }
  T.done = !hooks.shouldStop();
  return T;
}

/* ---------- collecting ---------- */
const COLLECT_BATCH = 256, COLLECT_MAX = 64e6;   // values one collecting component may keep (256 MB of float32)
// The samples an inference run takes from a Data component: all, its train split or its test split, maybe only the first n.
function runPool(d){
  const ds = dsById(d.props.dataset), sp = ds && splitOf(d); if (!sp) return null;
  const from = d.props.run_from, pool = from === 'train' ? sp.train : from === 'test' ? sp.test : Uint32Array.from({length:ds.n}, (_, i) => i);
  const want = String(d.props.run_n ?? '').trim().toLowerCase(), k = want === '' || want === 'all' ? pool.length : Math.floor(Number(want));
  if (!(k >= 1)) throw new Error('How many must be a whole number of at least 1, or "all".');
  return pool.subarray(0, Math.min(k, pool.length));
}
// The collecting Data components that a Data component's flow reaches, by wires or by the expressions that read it.
function collectorsOf(d){
  const out = [], seen = new Set();
  const walk = (g, id) => {
    for (const e of allEdges(g)) if (e.from === id){
      const m = byId(g, e.to); if (!m || seen.has(m.id)) continue; seen.add(m.id);
      if (m.type === 'data'){ if (m._collect) out.push(m); continue; }   // the flow stops at a Data component
      if (PLOTS.has(m.type)){ out.push(m); continue; }                 // a Plot 2D keeps what reaches it too
      if (m.type === 'group'){ const gin = m.graph.nodes.find(x => x.type === 'gin'); if (gin) walk(m.graph, gin.id); }   // into a block
      if (m.type === 'gout'){ const b = g._blk; if (b && b._g) walk(b._g, b.id); continue; }                            // out of one
      walk(g, m.id);
    }
  };
  walk(d._g || root, d.id);
  return out;
}
/* An inference run: the chosen samples of Data component srcId, in batches, through the network as it is now
   (BatchNorm with its running statistics, no dropout). Every collecting component it reaches keeps what arrives,
   with each sample's label. Returns {id → {shape, n, x, labels}}. hooks as for trainLoop, plus progress(done, N). */
async function collectLoop(srcId, hooks, opt = {}){   // opt.all: every sample, not the run settings; opt.only: just these collecting components
  infer();
  const d = findNode(srcId), sds = d && dsById(d.props.dataset);
  if (!d || d.type !== 'data' || !sds || (d._collect && !sds.x)) throw new Error('Start inference on a Data component that has samples.');
  if (d._r && d._r.err) throw new Error(d._r.err);
  const idxs = opt.all ? Uint32Array.from({length:sds.n}, (_, i) => i) : runPool(d); if (!idxs || !idxs.length) throw new Error('No samples to run. Check the split it runs on.');
  const targets = opt.only || collectorsOf(d); if (!targets.length) throw new Error('Nothing collects this flow. Wire a layer into another Data component\'s input.');
  const ctx = {src:d, loss:null, ds:dsById(d.props.dataset), pg:new Map()}, N = idxs.length, got = new Map();
  for (const m of targets) if (m._in[0] && N*sizeOf(m._in[0]) > COLLECT_MAX) throw new Error(`${m.name} would keep ${(N*sizeOf(m._in[0])/1e6).toFixed(0)} million values. Run fewer samples.`);
  let last = now();
  for (let i = 0; i < N && !hooks.shouldStop(); i += COLLECT_BATCH){
    const part = idxs.subarray(i, Math.min(N, i + COLLECT_BATCH)), xT = batchOf(ctx.ds, part, d.props.flatten);
    Object.assign(ctx, {B:part.length, train:false, out:new Map(), ins:new Map(), grad:new Map(), masks:new Map(), bn:new Map(), seeded:new Set(), order:new Map(), cap:new Map(), L:null});
    fwdT(root, null, xT, ctx);
    for (const [id, t] of ctx.cap){
      if (!targets.some(m => m.id === id)) continue;   // only the components this run collects into
      const S = t.data.length/t.B;
      let c = got.get(id); if (!c){ c = {shape:t.shape, n:0, x:new Float32Array(N*S), labels:new Int32Array(N)}; got.set(id, c); }
      c.x.set(t.data, c.n*S);
      for (let b = 0; b < t.B; b++) c.labels[c.n + b] = t.labels ? t.labels[b] : ctx.ds.labels[part[b]];
      c.n += t.B;
    }
    if (now() - last > 30){ await hooks.yield(); last = now(); hooks.progress(Math.min(N, i + COLLECT_BATCH), N); }
  }
  if (!got.size) throw new Error('Nothing reached the collecting components. Check that every layer before them works.');
  for (const c of got.values()) if (c.n < N){ c.x = c.x.slice(0, c.n*sizeOf(c.shape)); c.labels = c.labels.slice(0, c.n); }   // stopped early
  return Object.fromEntries(got);
}

/* ---------- PCA ---------- */
/* A PCA component's fit lives with the weights (so it is saved and exported like them): W holds the components
   (one row each), B the mean, rm each component's share of the variance. */
const pcaParams = (n, S, k) => { const P = wcache.get(n.id); return P && P.key === `pca:${S}:${k}` ? P : null; };
// Eigenvalues and eigenvectors of a symmetric matrix A (m×m, row-major, overwritten) by cyclic Jacobi rotations.
function jacobiEig(A, m){
  const V = new Float64Array(m*m); for (let i = 0; i < m; i++) V[i*m + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++){
    let off = 0; for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) off += A[i*m + j]*A[i*m + j];
    if (off < 1e-22) break;
    for (let p = 0; p < m; p++) for (let q = p + 1; q < m; q++){
      const apq = A[p*m + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q*m + q] - A[p*m + p])/(2*apq), t = Math.sign(th || 1)/(Math.abs(th) + Math.sqrt(th*th + 1)), c = 1/Math.sqrt(t*t + 1), s = t*c;
      for (let r = 0; r < m; r++){ const arp = A[r*m + p], arq = A[r*m + q]; A[r*m + p] = c*arp - s*arq; A[r*m + q] = s*arp + c*arq; }
      for (let r = 0; r < m; r++){ const apr = A[p*m + r], aqr = A[q*m + r]; A[p*m + r] = c*apr - s*aqr; A[q*m + r] = s*apr + c*aqr; }
      for (let r = 0; r < m; r++){ const vrp = V[r*m + p], vrq = V[r*m + q]; V[r*m + p] = c*vrp - s*vrq; V[r*m + q] = s*vrp + c*vrq; }
    }
  }
  return {vals:Array.from({length:m}, (_, i) => A[i*m + i]), V};   // eigenvector j is column j of V
}
// The top k eigenvectors of the covariance C (S×S): exactly for small S, otherwise by subspace iteration and a
// Rayleigh–Ritz step. Returns them sorted by eigenvalue, largest first.
async function topEig(C, S, k, hooks){
  if (S <= 160){ const {vals, V} = jacobiEig(C.slice(), S); const ord = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]).slice(0, k); return {vals:ord.map(i => vals[i]), vecs:ord.map(i => Float64Array.from({length:S}, (_, r) => V[r*S + i]))}; }
  const K = Math.min(S, k + 8), r = rng(12345), Q = Array.from({length:K}, () => Float64Array.from({length:S}, () => r() - .5));
  const orth = () => { for (let a = 0; a < K; a++){ const q = Q[a]; for (let b = 0; b < a; b++){ let d = 0; for (let i = 0; i < S; i++) d += q[i]*Q[b][i]; for (let i = 0; i < S; i++) q[i] -= d*Q[b][i]; } let nn = 0; for (let i = 0; i < S; i++) nn += q[i]*q[i]; nn = Math.sqrt(nn) || 1; for (let i = 0; i < S; i++) q[i] /= nn; } };
  const mul = q => { const z = new Float64Array(S); for (let i = 0; i < S; i++){ let s = 0; const o = i*S; for (let j = 0; j < S; j++) s += C[o + j]*q[j]; z[i] = s; } return z; };
  orth(); let prev = 0, last = now();
  for (let it = 0; it < 300; it++){
    for (let a = 0; a < K; a++) Q[a] = mul(Q[a]);
    orth();
    let ray = 0; for (let a = 0; a < k; a++){ const z = mul(Q[a]); for (let i = 0; i < S; i++) ray += Q[a][i]*z[i]; }
    if (it > 5 && Math.abs(ray - prev) <= 1e-10*Math.abs(ray)) break; prev = ray;
    if (hooks && now() - last > 30){ await hooks.yield(); last = now(); }
  }
  const CQ = Q.map(mul), T = new Float64Array(K*K);
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++){ let s = 0; for (let i = 0; i < S; i++) s += Q[a][i]*CQ[b][i]; T[a*K + b] = s; }
  for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) T[a*K + b] = T[b*K + a] = (T[a*K + b] + T[b*K + a])/2;
  const {vals, V} = jacobiEig(T, K), ord = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]).slice(0, k);
  return {vals:ord.map(i => vals[i]), vecs:ord.map(j => { const v = new Float64Array(S); for (let a = 0; a < K; a++){ const w = V[a*K + j]; for (let i = 0; i < S; i++) v[i] += w*Q[a][i]; } return v; })};
}
/* Fits a PCA component on what reaches it: the first fit_n samples of the train split of the Data component
   upstream (all of them when it has no train split), run through the network as it is now. */
async function fitPCA(id, hooks){
  infer();
  const n = findNode(id); if (!n || n.type !== 'pca') throw new Error('That PCA component is gone.');
  if (n._r && n._r.err) throw new Error(n._r.err);
  const d = upstreamData(n._g || root, n), ds = d && dsById(d.props.dataset);
  if (!ds || (d._collect && !ds.x)) throw new Error('No Data component with samples feeds it.');
  const sp = splitOf(d), pool = sp.train.length ? sp.train : sp.test, f = String(n.props.fit_n ?? 'all').trim().toLowerCase();
  const idxs = pool.subarray(0, f === 'all' ? pool.length : Math.min(pool.length, Math.floor(Number(f))));
  const S = sizeOf(n._in[0]), k = Number(n.props.components), N = idxs.length;
  if (N < 2) throw new Error('It needs at least 2 samples to fit on.');
  const mean = new Float64Array(S), C = new Float64Array(S*S), ctx = {src:d, loss:null, ds, pg:new Map()};
  let got = 0, last = now();
  for (let i = 0; i < N && !hooks.shouldStop(); i += 256){
    const part = idxs.subarray(i, Math.min(N, i + 256));
    Object.assign(ctx, {B:part.length, train:false, out:new Map(), ins:new Map(), grad:new Map(), masks:new Map(), bn:new Map(), seeded:new Set(), order:new Map(), cap:new Map(), L:null});
    fwdT(root, null, batchOf(ds, part, d.props.flatten), ctx);
    const t = ctx.cap.get(n.id); if (!t) throw new Error('Nothing reaches it from the Data component.');
    for (let b = 0; b < t.B; b++){
      const x = t.data.subarray(b*S, (b + 1)*S);
      for (let a = 0; a < S; a++){ const xa = x[a]; if (!xa) continue; mean[a] += xa; const o = a*S; for (let c = a; c < S; c++) C[o + c] += xa*x[c]; }
      got++;
      if (now() - last > 30){ await hooks.yield(); last = now(); if (hooks.progress) hooks.progress(got, N); }
    }
  }
  if (got < 2) throw new Error('Stopped before it had 2 samples.');
  for (let a = 0; a < S; a++) mean[a] /= got;
  let trace = 0;
  for (let a = 0; a < S; a++) for (let c = a; c < S; c++){ const v = (C[a*S + c] - got*mean[a]*mean[c])/(got - 1); C[a*S + c] = C[c*S + a] = v; if (a === c) trace += v; }
  const {vals, vecs} = await topEig(C, S, k, hooks);
  const W = new Float32Array(k*S), ratio = new Float32Array(k);
  vecs.forEach((v, j) => {
    let big = 0; for (let i = 0; i < S; i++) if (Math.abs(v[i]) > Math.abs(v[big])) big = i;   // a fixed sign: its largest entry positive
    const s = v[big] < 0 ? -1 : 1; for (let i = 0; i < S; i++) W[j*S + i] = s*v[i];
    ratio[j] = trace > 0 ? Math.max(0, vals[j])/trace : 0;
  });
  wcache.set(n.id, {key:`pca:${S}:${k}`, W, B:Float32Array.from(mean), rm:ratio, trained:true});
  return {n:got, k, S, ratio:Array.from(ratio), d};
}
