/* Netgraph engine: shape inference.
   inferGraph walks a graph in order and records on each node: _in (input shapes), _out (output shape),
   _r ({err | wait | out, sum, params, warn}). It also resolves variables in settings into n._p. */

// A layer setting may name a variable instead of a number (like n in Output features); the variable's slider
// supplies the value. n._p holds the settings with variables filled in; n._vars lists the variables used.
const NUMPROPS = {
  out_features:{min:1, max:512, step:1}, in_features:{min:1, max:4096, step:1},
  out_channels:{min:1, max:256, step:1}, in_channels:{min:1, max:256, step:1},
  kernel:{min:1, max:9, step:1}, stride:{min:1, max:4, step:1}, padding:{min:0, max:4, step:1},
  p:{min:0, max:0.95, step:0.05},
};
const VAR_NAME = /^[A-Za-z_]\w*$/;
function resolveProps(n, scope = {}){
  n._scope = scope;   // the variables of the blocks around this node (see blockScope)
  if (n.type === 'flatten' && typeof n.props.expr === 'string'){   // a Flatten's old Expression field is now its input cell
    if (n.props.expr.trim()) n.props.inputs = [n.props.expr];
    delete n.props.expr;
  }
  if (n.type === 'loss'){ n._p = n.props; n._vars = []; return; }   // a Loss binds its own variables
  const p = {...n.props}, used = [];
  for (const k of Object.keys(NUMPROPS)){
    const v = typeof p[k] === 'string' ? p[k].trim() : null;
    if (!v || !VAR_NAME.test(v) || v.toLowerCase() === 'auto' || v.toLowerCase() === 'same') continue;
    if (v in scope){ p[k] = scope[v]; continue; }   // a block variable: its slider lives on the block
    const vars = n.props.vars || (n.props.vars = {});
    if (!vars[v]){ const d = DEFS[n.type].props[k]; vars[v] = {value: typeof d === 'number' ? d : NUMPROPS[k].min, ...NUMPROPS[k]}; }
    p[k] = Number(vars[v].value);
    if (!used.includes(v)) used.push(v);
  }
  n._p = p; n._vars = used;
}

function inferNode(n, ins){
  const p = n._p || n.props, x = ins[0];
  const need3 = () => x.length !== 3 ? {err:`Needs a C×H×W tensor, but receives ${fmt(x)}. Remove the Flatten or feed an image-shaped input.`} : null;
  switch (n.type){
    case 'data': {
      /* A Data component with something reaching its input collects: during an inference run (started on the Data
         component the flow comes from) it keeps every sample that reaches it, and those become its dataset. */
      n._collect = x !== undefined;
      if (x === null) return {wait:'Waiting for upstream layers.'};
      const ds = dsById(p.dataset), got = ds && (!n._collect || ds.x) ? ds : null;
      if (n._collect && !got) return {out:x, sum:'Collects what reaches it', warn:'Nothing collected yet. Run inference on the Data component the flow starts from.'};
      if (!got) return {err: datasets.length ? 'Pick a dataset.' : 'Import a dataset first (Import data, top bar).'};
      const pct = Number(p.test ?? 20), seed = Number(p.seed ?? 1);
      if (!(pct >= 0 && pct <= 90)) return {err:'Test split must be a percentage from 0 to 90.'};
      if (!Number.isInteger(seed)) return {err:'Shuffle seed must be a whole number.'};
      const sp = splitOf(n), sh = dsShape(ds, p.flatten);
      const stale = n._collect && fmt(ds.shape) !== fmt(x) ? `Collected ${fmt(ds.shape)} samples, but its input now gives ${fmt(x)}. Run inference again to collect afresh.` : null;
      return {out:sh, sum:`${n._collect ? `${ds.n.toLocaleString()} collected` : ds.name} · ${sp.train.length.toLocaleString()} train / ${sp.test.length.toLocaleString()} test`, warn:stale};
    }
    case 'sliders': {   // a vector of hand-set values
      const N = (p.sliders || []).length;
      if (!N) return {err:'Add a slider.'};
      return {out:[N], sum:`${N} slider${N > 1 ? 's' : ''}`};
    }
    case 'input': {
      const s = String(p.shape).split(/[\s,x×]+/).filter(Boolean).map(Number);
      if (!s.length || s.some(v => !intOk(v))) return {err:'Shape must be positive whole numbers, like 1,28,28 or 784.'};
      return {out:s, sum: s.length===3 ? `${s[0]} channel${s[0]>1?'s':''}, ${s[1]}×${s[2]} pixels` : `${s.join('×')} features`};
    }
    case 'output': return {sum:'Network output'};
    case 'gout': return {sum:'Leaves the block'};
    case 'dense': {
      const fin = x[x.length-1];
      if (!isAuto(p.in_features) && Number(p.in_features) !== fin) return {err:`Pinned to ${p.in_features} input features, but receives ${fin}.`};
      const fo = Number(p.out_features);
      if (!intOk(fo)) return {err:'Output features must be a positive whole number.'};
      return {out:[...x.slice(0,-1), fo], params: fin*fo + (p.bias ? fo : 0), sum:`${fin} → ${fo} units`,
        warn: x.length > 1 ? 'Acts on the last dimension only. Add Flatten first for a classic MLP.' : null};
    }
    case 'conv2d': {
      const b = need3(); if (b) return b;
      const [c,h,w] = x;
      if (!isAuto(p.in_channels) && Number(p.in_channels) !== c) return {err:`Pinned to ${p.in_channels} input channels, but receives ${c}.`};
      const co = String(p.out_channels).trim().toLowerCase() === 'same' ? c : Number(p.out_channels);
      const k = Number(p.kernel), s = Number(p.stride), pd = Number(p.padding);
      if (!intOk(co)) return {err:'Output channels must be a positive whole number or "same".'};
      if (!intOk(k) || !intOk(s) || !(Number.isInteger(pd) && pd >= 0)) return {err:'Kernel size and stride must be at least 1; padding at least 0.'};
      const ho = Math.floor((h+2*pd-k)/s)+1, wo = Math.floor((w+2*pd-k)/s)+1;
      if (ho < 1 || wo < 1) return {err:`A ${k}×${k} kernel doesn't fit a ${h}×${w} input. Add padding or use a smaller kernel.`};
      return {out:[co,ho,wo], params:c*co*k*k+co, sum:`${co} filters, ${k}×${k}, stride ${s}, pad ${pd}`};
    }
    case 'maxpool': case 'avgpool': {
      const b = need3(); if (b) return b;
      const [c,h,w] = x, k = Number(p.kernel), s = isAuto(p.stride) ? k : Number(p.stride);
      if (!intOk(k) || !intOk(s)) return {err:'Kernel size and stride must be at least 1.'};
      const ho = Math.floor((h-k)/s)+1, wo = Math.floor((w-k)/s)+1;
      if (ho < 1 || wo < 1) return {err:`A ${k}×${k} window doesn't fit a ${h}×${w} input.`};
      return {out:[c,ho,wo], sum:`${k}×${k} window, stride ${s}`};
    }
    case 'pca': {   // projects each sample (flattened) onto its first principal components, once fitted
      const N = sizeOf(x), k = Number(p.components), f = String(p.fit_n ?? '').trim().toLowerCase();
      if (!(Number.isInteger(k) && k >= 1 && k <= N)) return {err:`Components must be a whole number from 1 to ${N.toLocaleString()}.`};
      if (!(f === 'all' || Number.isInteger(Number(f)) && Number(f) >= 2)) return {err:'Fit on must be a number of samples (at least 2) or "all".'};
      const P = pcaParams(n, N, k), pct = P ? Math.round(1000*P.rm.reduce((a, v) => a + v, 0))/10 : 0;
      return {out:[k], sum: P ? `${k} component${k > 1 ? 's' : ''} · ${pct}% of variance` : `${k} component${k > 1 ? 's' : ''}, not fitted`, warn: P ? null : 'Not fitted yet. Select it and press Fit.'};
    }
    case 'plot2d': case 'plot3d': {   // plots what reaches it: a vector as one point, a matrix as one point per row
      const D = n.type === 'plot3d' ? 3 : 2, N = x.length === 2 ? x[1] : sizeOf(x), ds = [p.x_dim, p.y_dim, p.z_dim].slice(0, D).map(Number);
      if (N < D) return {err:`Needs at least ${D} values per ${x.length === 2 ? 'row' : 'sample'} to plot, but gets ${fmt(x)}.`};
      if (!ds.every(v => Number.isInteger(v) && v >= 0 && v < N)) return {err:`The axes are value indices from 0 to ${N - 1}.`};
      return {sum: x.length === 2 ? `${x[0].toLocaleString()} rows as points` : N === D ? 'Points' : `Values ${ds.slice(0, -1).join(', ')} and ${ds[D - 1]} of ${N}`};
    }
    case 'ptrin': {   // passes its input on to the stretch its pair runs
      const t = String(p.target ?? '').trim();
      return {out:x, sum: t ? `Feeds ${t}` : 'Unlinked', warn: t ? null : 'Drag its output onto the input of the layer where the stretch starts.'};
    }
    case 'gap': { const b = need3(); if (b) return b; return {out:[x[0]], sum:`Averages each ${x[1]}×${x[2]} map`}; }
    case 'flatten': return {out:[x.reduce((a,b)=>a*b,1)], sum:`${x.length}D → 1D`, warn: x.length===1 ? 'Input is already flat.' : null};
    case 'unflatten': {   // reshape: same values, new shape
      const total = x.reduce((a,b)=>a*b,1);
      let s;
      if (isAuto(p.to)){ let k = Math.floor(Math.sqrt(total)); while (k > 1 && total % (k*k)) k--; s = [total/(k*k), k, k]; }
      else {
        s = String(p.to).split(/[\s,x×]+/).filter(Boolean).map(Number);
        const free = s.filter(v => v === -1).length;
        if (!s.length || free > 1 || s.some(v => v !== -1 && !intOk(v))) return {err:'Output shape must be positive whole numbers, like 32,7,7, with at most one -1.'};
        const known = s.reduce((a, v) => v === -1 ? a : a*v, 1);
        if (free){ if (total % known) return {err:`Can't split ${total.toLocaleString()} values into ${fmt(s.map(v => v === -1 ? '?' : v))}: ${total.toLocaleString()} isn't a multiple of ${known.toLocaleString()}.`}; s = s.map(v => v === -1 ? total/known : v); }
        else if (known !== total) return {err:`${fmt(s)} holds ${known.toLocaleString()} values, but the input has ${total.toLocaleString()}.`};
      }
      return {out:s, sum:`${x.length}D → ${s.length}D`, warn: x.length > 1 ? 'Input isn\'t flat; its values are reshaped in order.' : null};
    }
    case 'batchnorm':
      if (x.length !== 1 && x.length !== 3) return {err:`Needs features or C×H×W, but receives ${fmt(x)}.`};
      return {out:x, params:2*x[0], sum:`${x.length===3?'2D':'1D'}, ${x[0]} ${x.length===3?'channels':'features'}`};
    case 'dropout': { const q = Number(p.p); if (!(q >= 0 && q < 1)) return {err:'Drop probability must be between 0 and 1.'}; return {out:x, sum:`Drops ${Math.round(q*100)}% during training`}; }
    case 'activation': {
      const k = presetOf(n);
      if (k === 'softmax') return {out:x, sum: x.length === 1 ? 'Over features' : 'Over channels'};
      if (k) return {out:x, sum:'Element-wise'};
      let sh; try { sh = shapeOf(bindVars(parseLoss(p.fn), n, true), n, null, null, x); } catch(e){ return {err:`f(x): ${e.message}`}; }
      return {out: sizeOf(sh) === sizeOf(x) ? x : (sh.length ? sh : [1]), sum:`f(x) = ${String(p.fn).trim()}`};
    }
    case 'softmax': return {out:x, sum: x.length===1 ? 'Over features' : 'Over channels'};
    case 'relu': case 'gelu': case 'sigmoid': case 'tanh': return {out:x, sum:'Element-wise'};
    case 'add': {
      const [a,b] = ins;
      if (fmt(a) !== fmt(b)) return {err:`Shapes must match to add: ${fmt(a)} vs ${fmt(b)}.`};
      return {out:a, sum:'Element-wise sum'};
    }
    case 'concat': {
      const [a,b] = ins;
      if (a.length !== b.length || fmt(a.slice(1)) !== fmt(b.slice(1))) return {err:`Only the first dimension may differ: ${fmt(a)} vs ${fmt(b)}.`};
      return {out:[a[0]+b[0], ...a.slice(1)], sum: a.length===3 ? 'Along channels' : 'Along features'};
    }
  }
  return {out:x};
}

// Wires plus the links that expressions and Pointers make by naming components: those count as edges for ordering.
// A name inside a block that reaches out to a component of this graph links that component to the block.
function allEdges(g){
  const extra = [], link = (owner, m, to) => { const h = holderIn(g, m); if (h && h !== to) extra.push({from:h.id, to:to.id, port:-1}); };
  const scan = (m, to) => {   // m's own links, then those of everything inside it
    if (m.type === 'ptrout'){ const I = pairOf(m, m._g); if (I) link(m, I, to); }   // a Pointer out comes after its Pointer in
    for (const src of exprsOf(m)){
      let ast; try { ast = parseLoss(src); } catch(e){ continue; }
      for (const key of ast.refs){ const rr = lookup(key, m._g); if (rr.node) link(m, rr.node, to); }
      for (const key of ast.posts){ const rr = lookup(key, m._g); if (!rr.node) continue; const nx = nextDense(rr.graph, rr.node, ''); if (nx.node) link(m, nx.node, to); }
    }
    if (m.graph) for (const c of m.graph.nodes) scan(c, to);
  };
  for (const n of g.nodes) scan(n, n);
  return extra.length ? g.edges.concat(extra) : g.edges;
}
function topo(g){
  const E = allEdges(g), indeg = new Map(g.nodes.map(n => [n.id, 0]));
  for (const e of E) if (indeg.has(e.to)) indeg.set(e.to, indeg.get(e.to)+1);
  const q = g.nodes.filter(n => indeg.get(n.id) === 0), order = [];
  while (q.length){
    const n = q.shift(); order.push(n);
    for (const e of E) if (e.from === n.id && indeg.has(e.to)){ indeg.set(e.to, indeg.get(e.to)-1); if (indeg.get(e.to) === 0) q.push(byId(g, e.to)); }
  }
  return order;
}
/* Block variables: a block declares name = value rows (props.params), edited in its Block input's info window.
   A number becomes a slider on the block (its value lives in props.vars); an expression is worked out from the
   rows above it. Layers inside the block, and blocks inside it, can use the names. */
const PARAM_FNS = {abs:Math.abs, log:Math.log, exp:Math.exp, sqrt:Math.sqrt};
function evalConst(a, scope){
  switch (a.k){
    case 'num': return a.v;
    case 'ref':
      if (a.attr || a.idx || a.key0.includes(SEP) || !(a.key0 in scope)) throw new Error(`${a.label} isn't a variable declared above it.`);
      return scope[a.key0];
    case 'neg': return -evalConst(a.a, scope);
    case 'bin': return BIN[a.op](evalConst(a.a, scope), evalConst(a.b, scope));
    case 'fn':
      if (!PARAM_FNS[a.f]) throw new Error(`${a.f}() can't be used here; use abs, log, exp or sqrt.`);
      return PARAM_FNS[a.f](evalConst(a.args[0], scope));
  }
}
function paramSlider(v){
  return Number.isInteger(v) ? {value:v, min:v >= 1 ? 1 : 0, max:Math.max(10, v*4), step:1} : {value:v, min:0, max:Math.max(1, Math.abs(v)*4), step:0.01};
}
function blockScope(g, outer){
  const params = (g.props && g.props.params) || [], scope = {...outer}, sliders = [];
  g._paramErr = null;
  for (const prm of params){
    const name = String(prm.name ?? '').trim(), txt = String(prm.expr ?? '').trim();
    if (!name) continue;
    if (!VAR_NAME.test(name)){ g._paramErr ||= `"${name}" isn't a valid name; use letters, digits and _.`; continue; }
    if (txt !== '' && !isNaN(txt)){   // a number: a slider on the block
      const vars = g.props.vars || (g.props.vars = {});
      if (!vars[name]) vars[name] = paramSlider(Number(txt));
      scope[name] = Number(vars[name].value); sliders.push(name);
    } else {
      try { scope[name] = evalConst(parseLoss(txt || '?'), scope); }
      catch(e){ g._paramErr ||= `${name}: ${txt ? e.message : 'give it a number or an expression.'}`; }
    }
  }
  g._vars = sliders;
  return scope;
}
function inferGraph(g, ginShape, scope = {}){
  const res = {out:null, params:0, errs:0, ins:[], outs:[]};
  const map = new Map(g.nodes.map(n => [n.id, n]));
  g.nodes.forEach(n => { n._r = {err:'Part of a loop. Networks must flow one way.'}; n._in = []; n._out = null; n._src = false; n._g = g; if (n.graph) n.graph._blk = n; });
  for (const n of topo(g)){
    const d = DEFS[n.type];
    const ins = new Array(d.ins).fill(undefined);
    for (const e of g.edges) if (e.to === n.id && e.port < d.ins && wired(n, e.port)){ const s = map.get(e.from); ins[e.port] = s ? s._out : undefined; }
    n._in = ins;
    let r, exprErr = null;
    resolveProps(n, scope);
    for (let i = 0; i < d.ins; i++){   // inputs given by expressions take the shape of the expression's value
      const src = portExpr(n, i); if (!src) continue;
      try {
        const ast = parseLoss(src);
        if (ast.k === 'ref' && !ast.idx && !ast.attr){ const lk = lookup(ast.key0, g); if (!lk.node) throw new Error(lk.err); }   // a lone unknown name is a typo, not a variable
        const sh = shapeOf(bindVars(ast, n), n, g, null); ins[i] = sh.length ? sh : [1];
      }
      catch(e){ ins[i] = null; exprErr ||= `${d.ins > 1 ? 'Input ' + 'ab'[i] : 'Input'}: ${e.message}`; }
    }
    if (exprErr) r = {err:exprErr};
    else if (n.type === 'gin') r = ginShape ? {out:ginShape} : {wait:'Connect the block on the outside to see shapes.'};
    else if (n.type === 'group'){
      const sub = inferGraph(n.graph, ins[0] || null, blockScope(n, scope));
      n._sub = sub;
      if (n._paramErr) r = {err:`Block variable ${n._paramErr}`};
      else if (ins[0] === undefined && n.graph.edges.some(e => e.from === (n.graph.nodes.find(m => m.type === 'gin') || {}).id)) r = {err:'Connect an input.'};
      else if (ins[0] === undefined){   // a block with its own data needs no input, nor an output
        r = sub.errs ? {err:`${sub.errs} issue${sub.errs>1?'s':''} inside. Double-click to open the block.`} : {out:sub.out};
      }
      else if (sub.errs) r = {err:`${sub.errs} issue${sub.errs>1?'s':''} inside. Double-click to open the block.`};
      else if (ins[0] === null) r = {wait:'Waiting for upstream layers.'};
      else if (!sub.out) r = {err:'Nothing reaches the block output. Open it to connect.'};
      else r = {out:sub.out};
      r.params = sub.params;
      r.sum = `${countLayers(n.graph)} layers inside`;
    }
    else if (n.type === 'loss') r = checkLoss(n, g);
    else if (n.type === 'data') r = inferNode(n, ins);   // its input is optional: wired, it collects
    else if (n.type === 'dense' && ins[0] === undefined){
      const fi = Number(n.props.in_features);
      if (intOk(fi)){ ins[0] = [fi]; n._src = true; r = inferNode(n, ins); }
      else r = {wait:'Set Input features, or connect a layer.'};
    }
    else if (n.type === 'gout' && ins[0] === undefined) r = {wait:'Connect the last layer here.'};   // the block says so when it needs an output
    else if (ins.some(v => v === undefined)) r = {err: d.ins > 1 ? 'Connect both inputs.' : 'Connect an input.'};
    else if (ins.some(v => v === null)) r = {wait:'Waiting for upstream layers.'};
    else r = n.type === 'ptrout' ? pointerShape(n, g, ins[0]) : inferNode(n, ins);
    if (r.err){ r.out = null; res.errs++; }
    if (r.wait) r.out = null;
    n._r = r; n._out = r.out === undefined ? null : r.out;
    // Variables nothing uses any more (like the partial names met while typing one) are dropped, once all is well.
    if (!r.err && n.props.vars) for (const k of Object.keys(n.props.vars)) if (!(n._vars || []).includes(k)) delete n.props.vars[k];
    res.params += r.params || 0;
    if (n.type === 'gout') res.out = ins[0] || null;
    if ((n.type === 'input' || n.type === 'data' && !n._collect) && n._out) res.ins.push(n._out);
    if (n._src && n._out) res.ins.push(n._in[0]);
    if (n.type === 'output' && ins[0]) res.outs.push(ins[0]);
  }
  for (const n of g.nodes) if (n._r.err && n._r.err.startsWith('Part of a loop')) res.errs++;
  return res;
}
function infer(){ migrateActs(); linkGraphs(); stats = inferGraph(root, null); }
// The Data component whose samples reach n: back along wires and expression links, out through a block's input to
// what feeds the block, and into a block through its output.
function upstreamData(g, n){
  const seen = new Set(), st = [[g, n]], E = new Map(), edgesOf = gg => { let e = E.get(gg); if (!e){ e = allEdges(gg); E.set(gg, e); } return e; };
  while (st.length){
    const [gg, m] = st.pop(); if (!m || seen.has(m.id)) continue; seen.add(m.id);
    if (m.type === 'data') return m;
    if (m.type === 'gin'){ const b = gg._blk; if (b && b._g) for (const e of edgesOf(b._g)) if (e.to === b.id) st.push([b._g, byId(b._g, e.from)]); continue; }
    if (m.type === 'group'){ const go = m.graph.nodes.find(x => x.type === 'gout'); if (go) st.push([m.graph, go]); }
    for (const e of edgesOf(gg)) if (e.to === m.id) st.push([gg, byId(gg, e.from)]);
  }
  return null;
}
// The Dense layer that a layer's neurons feed, one-to-one through activations, Dropout or BatchNorm.
function nextDense(g, n, name){
  let c = n;
  for (;;){
    const nx = g.edges.filter(e => e.from === c.id).map(e => byId(g, e.to)).filter(m => m && m.type !== 'output' && m.type !== 'loss');
    if (!nx.length) return {err:`${name} has no Dense layer after it${g === root ? '' : ' inside its block'}.`};
    if (nx.length > 1) return {err:`${name} feeds ${nx.length} layers, so its outgoing weights are ambiguous.`};
    const m = nx[0];
    if (m.type === 'dense') return {node:m};
    if (!['activation', 'relu', 'gelu', 'sigmoid', 'tanh', 'softmax', 'dropout', 'batchnorm'].includes(m.type)) return {err:`Between ${name} and the next Dense layer there's a ${DEFS[m.type].label}, so its neurons don't map one-to-one.`};
    c = m;
  }
}

/* ---------- pointers ---------- */
/* Pointers come in pairs and run a stretch of the network again, with the same weights, on another input. A Pointer
   in takes the new input on its wire; its output is a link into the input of the layer where the stretch starts
   (props.target), next to that layer's own wire. A Pointer out is wired from the layer where the stretch ends and
   puts out what the stretch gives on the Pointer in's input. It names its Pointer in in props.pair. The stretch is
   every component on a wire path between the two ends; nothing in it may take an input from outside it, except the
   first one's own input. */
function ptrRef(txt, g){
  const s = String(txt ?? '').trim(); if (!s) return null;
  let a; try { a = parseLoss(s); } catch(e){ return {err:e.message}; }
  if (a.k !== 'ref' || a.idx || a.attr) return {err:`"${s}" isn't a component. Name one, like dec1.`};
  const rr = lookup(a.key0, g); return rr.err ? rr : {...rr, label:a.label};
}
function pairOf(n, g){ const r = ptrRef(n.props.pair, g); return r && !r.err && r.graph === (g || root) && r.node.type === 'ptrin' ? r.node : null; }
const PTR_BAD = new Set(['data', 'loss', 'sliders', 'input', 'output', 'gin', 'gout', 'ptrin', 'ptrout']);
function ptrPlan(n, g){
  const I = pairOf(n, g);
  if (!I) return {err: String(n.props.pair ?? '').trim() ? `${String(n.props.pair).trim()} isn't a Pointer in beside it.` : 'Pair it with a Pointer in.'};
  if (!String(I.props.target ?? '').trim()) return {err:`Drag ${I.name}'s output onto the input of the layer where the stretch starts.`};
  const A = ptrRef(I.props.target, g); if (A.err) return {err:`${I.name}: ${A.err}`};
  if (A.graph !== g) return {err:`${I.name} feeds ${A.label}, which isn't beside it. Put the pair next to the stretch.`};
  const ze = g.edges.find(e => e.to === n.id && e.port === 0), a = A.node, z = ze && byId(g, ze.from);
  if (!z) return {err:'Wire the layer where the stretch ends into it.'};
  const reach = (start, fwd) => { const seen = new Set(), st = [start]; while (st.length){ const id = st.pop(); if (seen.has(id)) continue; seen.add(id); for (const e of g.edges) if ((fwd ? e.from : e.to) === id) st.push(fwd ? e.to : e.from); } return seen; };
  const fw = reach(a.id, true), bw = reach(z.id, false), stretch = g.nodes.filter(m => fw.has(m.id) && bw.has(m.id)), ids = new Set(stretch.map(m => m.id));
  if (!ids.has(z.id)) return {err:`${z.name} doesn't come after ${a.name}, where ${I.name} feeds in.`};
  for (const m of stretch){
    if (PTR_BAD.has(m.type)) return {err:`A pointer pair runs layers and blocks; ${m.name} is a ${labelOf(m)}.`};
    for (let i = 0; i < DEFS[m.type].ins; i++){
      if (portExpr(m, i)) return {err:`${m.name}'s input is an expression, so it can't run on ${I.name}'s input.`};
      if (m === a) continue;
      const e = g.edges.find(x => x.to === m.id && x.port === i), src = e && byId(g, e.from);
      if (!src || !ids.has(src.id)) return {err:`${m.name} also takes ${src ? src.name : 'an input'} from outside the stretch, so the pair can't run it alone.`};
    }
  }
  if (DEFS[a.type].ins !== 1) return {err:`${a.name} takes ${DEFS[a.type].ins} inputs; feed a component with one.`};
  const gin = {id:'ptrin:' + n.id, type:'gin', name:'in', props:{}, _r:{}}, gout = {id:'ptrout:' + n.id, type:'gout', name:'out', props:{}, _r:{}};
  const edges = g.edges.filter(e => ids.has(e.from) && ids.has(e.to) && e.to !== a.id).concat({from:gin.id, to:a.id, port:0}, {from:z.id, to:gout.id, port:0});
  return {a, z, I, stretch, graph:{nodes:[gin, ...stretch, gout], edges}};
}
function pointerShape(n, g, x){
  const P = ptrPlan(n, g); n._ptr = P.err ? null : P; n._pg = P.err ? null : P.graph;
  if (P.err) return {err:P.err};
  const bad = P.stretch.find(m => m._r && m._r.err); if (bad) return {err:`${bad.name} has an issue: ${bad._r.err}`};
  const want = P.a._in[0], got = P.I._out;
  if (!want || !P.z._out) return {wait:`Waiting for ${P.a.name} to get its input.`};
  if (!got) return {wait:`Waiting for ${P.I.name}'s input.`};
  if (fmt(want) !== fmt(got)) return {err:`${P.a.name} takes ${fmt(want)}, but ${P.I.name} gets ${fmt(got)}. The pair reuses the weights, so the shapes must match.`};
  return {out:P.z._out, sum: P.a === P.z ? `${P.a.name} again, on ${P.I.name}` : `${P.a.name} … ${P.z.name} again, on ${P.I.name}`};
}
