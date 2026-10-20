/* Netgraph engine: expressions (Loss blocks and Flatten expressions).
   parseLoss turns text into a tree; bindVars fills in variables; checkLoss works out sizes;
   evalLoss computes one sample's value and, when asked, gradients for every component it reads. */

/* ---------- training ---------- */
/* Loss expressions: component names, numbers, + - * / ^ (or **), parentheses, and the functions below.
   ce(x): cross-entropy of raw scores x against the data labels. nll(p): the same for probabilities.
   mse(a, b): mean squared error. A result with many values is averaged. Names with spaces go in quotes. */
const LFN = {abs:1, log:1, exp:1, sqrt:1, tanh:1, sigmoid:1, relu:1, softmax:1, max:[1, 2], min:[1, 2], sum:1, mean:1, ce:1, nll:1, mse:2};   // max/min: of one input's values, or element-wise of two
const pcache = new Map();
function parseLoss(src){
  src = String(src ?? '').trim();
  if (!src) throw new Error('Write the loss as an expression, like ce(fc3) or mse(relu8, flatten1).');
  if (pcache.has(src)) return pcache.get(src);
  const toks = [], re = /\s*(?:(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+)|([A-Za-z_]\w*)|"([^"]*)"|'([^']*)'|(\*\*|[-+*/^(),.\[\]:]))/y;
  for (let i = 0; i < src.length;){
    re.lastIndex = i; const m = re.exec(src);
    if (!m){ if (!src.slice(i).trim()) break; throw new Error(`Unexpected "${src.slice(i).trim()[0]}" in the expression.`); }
    toks.push(m[1] != null ? {t:'num', v:Number(m[1])} : m[2] != null ? {t:'id', v:m[2]} : m[3] != null || m[4] != null ? {t:'id', v:m[3] ?? m[4], quoted:true} : {t:m[5] === '**' ? '^' : m[5]});
    i = re.lastIndex;
  }
  let p = 0;
  const take = t => toks[p] && toks[p].t === t ? (p++, true) : false;
  const show = t => t.t === 'num' || t.t === 'id' ? t.v : t.t;
  const close = () => { if (!take(')')) throw new Error(toks[p] ? `Expected ")" before "${show(toks[p])}".` : 'Missing ")".'); };
  const expr = () => { let a = term(); for (;;){ if (take('+')) a = {k:'bin', op:'+', a, b:term()}; else if (take('-')) a = {k:'bin', op:'-', a, b:term()}; else return a; } };
  const term = () => { let a = unary(); for (;;){ if (take('*')) a = {k:'bin', op:'*', a, b:unary()}; else if (take('/')) a = {k:'bin', op:'/', a, b:unary()}; else return a; } };
  const unary = () => take('-') ? {k:'neg', a:unary()} : take('+') ? unary() : power();
  const power = () => { const a = atom(); return take('^') ? {k:'bin', op:'^', a, b:unary()} : a; };
  const atom = () => {
    const t = toks[p++];
    if (!t) throw new Error('The expression ends too early.');
    if (t.t === 'num') return {k:'num', v:t.v};
    if (t.t === 'id'){
      if (!t.quoted && take('(')){
        if (!LFN[t.v]) throw new Error(`Unknown function ${t.v}(). Use ${Object.keys(LFN).join(', ')}.`);
        const args = [expr()]; while (take(',')) args.push(expr()); close();
        const ar = LFN[t.v];
        if (Array.isArray(ar) ? !ar.includes(args.length) : args.length !== ar) throw new Error(`${t.v}() takes ${Array.isArray(ar) ? ar.join(' or ') : ar} input${(Array.isArray(ar) ? ar[ar.length - 1] : ar) > 1 ? 's' : ''}.`);
        return {k:'fn', f:t.v, args};
      }
      const path = [t.v];   // Block.layer reaches inside blocks, as deep as needed
      while (take('.')){ const nx = toks[p++]; if (!nx || nx.t !== 'id') throw new Error('Expected a component name after ".".'); path.push(nx.v); }
      const index = () => {   // [3], [-1], [0:10], [2, 5:10, :]
        if (!take('[')) return null;
        const idx = [];
        const num = () => { const neg = take('-'), tk = toks[p]; if (tk && tk.t === 'num'){ p++; if (!Number.isInteger(tk.v)) throw new Error('Indices must be whole numbers.'); return neg ? -tk.v : tk.v; } if (tk && tk.t === 'id' && !tk.quoted){ p++; return {v:tk.v, neg}; } if (neg) throw new Error('Expected a number after "-".'); return null; };
        do { const a = num(); if (take(':')) idx.push({a, b:num()}); else if (a == null) throw new Error('Expected an index, like fc1[3] or fc1[0:10].'); else idx.push({one:a}); } while (take(','));
        if (!take(']')) throw new Error('Missing "]".');
        return idx;
      };
      // Weights around a Dense layer's neurons: fc1.pre_weights (into fc1, one row per neuron), fc1.post_weights (out to the next Dense layer).
      let attr = null;
      if (path.length > 1 && WATTR.has(path[path.length - 1])) attr = path.pop();
      let idx = index();
      if (!attr && take('.')){
        const at = toks[p++];
        if (!at || at.t !== 'id' || !WATTR.has(at.v)) throw new Error('After an index, use .pre_weights (weights coming in) or .post_weights (weights going out), like fc1[3].pre_weights.');
        attr = at.v;
        const more = index(); if (more) idx = (idx || []).concat(more);
      }
      const key0 = path.join(SEP), plabel = path.join('.');
      const ix = x => x == null ? '' : typeof x === 'number' ? x : (x.neg ? '-' : '') + x.v;
      const label = plabel + (attr ? '.' + attr : '') + (idx ? `[${idx.map(i => i.one != null ? ix(i.one) : `${ix(i.a)}:${ix(i.b)}`).join(', ')}]` : '');
      return {k:'ref', key0, key: attr ? `w${SEP}${label}` : key0, label, plabel, idx, attr, quoted:!!t.quoted};
    }
    if (t.t === '('){ const a = expr(); close(); return a; }
    throw new Error(`Unexpected "${show(t)}" in the expression.`);
  };
  const ast = expr();
  if (p < toks.length) throw new Error(`Unexpected "${show(toks[p])}" in the expression.`);
  const refs = new Set(), posts = new Set(), mark = a => { a.live = a.k === 'ref' ? (refs.add(a.key0), a.attr === 'post_weights' && posts.add(a.key0), true) : a.k === 'num' ? false : a.k === 'fn' ? a.args.map(mark).some(Boolean) : a.k === 'neg' ? mark(a.a) : [mark(a.a), mark(a.b)].some(Boolean); return a.live; };
  mark(ast); ast.refs = [...refs]; ast.posts = [...posts];
  if (pcache.size > 64) pcache.clear();
  pcache.set(src, ast);
  return ast;
}
const refName = name => /^[A-Za-z_]\w*$/.test(name) ? name : `"${name}"`;
// Which flat positions an index list picks from a tensor of this shape (row-major), with clear errors.
const selCache = new Map();
function selIndex(shape, idx, label){
  const ck = shape.join(',') + '|' + JSON.stringify(idx);
  if (selCache.has(ck)) return selCache.get(ck);
  if (idx.length > shape.length) throw new Error(`${label}: ${idx.length} indices, but the tensor has ${shape.length} dimension${shape.length > 1 ? 's' : ''} (${fmt(shape)}).`);
  const where = d => shape.length > 1 ? `dimension ${d + 1} has` : 'it has';
  const ranges = shape.map((n, d) => {
    const it = idx[d]; if (!it) return [0, n];
    const fix = (v, dflt) => v == null ? dflt : v < 0 ? n + v : v;
    if (it.one != null){ const v = fix(it.one); if (v < 0 || v >= n) throw new Error(`${label}: index ${it.one} is out of range; ${where(d)} ${n} value${n > 1 ? 's' : ''} (0 to ${n - 1}).`); return [v, v + 1]; }
    const a = fix(it.a, 0), b = fix(it.b, n);
    if (a < 0 || b > n || a >= b) throw new Error(`${label}: the slice ${it.a ?? ''}:${it.b ?? ''} doesn't fit; ${where(d)} ${n} value${n > 1 ? 's' : ''}.`);
    return [a, b];
  });
  let sel = [0];
  ranges.forEach(([a, b], d) => { const stride = shape.slice(d + 1).reduce((x, y) => x*y, 1), nx = []; for (const base of sel) for (let v = a; v < b; v++) nx.push(base + v*stride); sel = nx; });
  const out = Int32Array.from(sel);
  if (selCache.size > 200) selCache.clear();
  selCache.set(ck, out);
  return out;
}
const WATTR = new Set(['pre_weights', 'post_weights']);
/* Variables: a plain name in an expression that isn't a component (like k in fc1[k] or alpha in alpha*x)
   is a variable with a slider. Values live in n.props.vars; bindVars fills them into the tree before each
   use: a._var for scalar variables and a._idx for indices with the variables replaced by whole numbers. */
function refDims(a, g){   // the shape an index on this reference applies to, if it can be known yet
  const rr = lookup(a.key0, g); if (!rr.node) return null;
  const L = rr.node;
  if (a.attr === 'pre_weights' && L._in && L._in[0] && L._out) return [lastDim(L._out), lastDim(L._in[0])];
  if (a.attr === 'post_weights' && L._out) return [lastDim(L._out)];
  return L._out || null;
}
function bindVars(ast, n, fnOfX = false){   // fnOfX: an Activation's function, where x is its input and every other name a variable
  const vars = n.props.vars || (n.props.vars = {}), used = n._vars || (n._vars = []);   // adds to the settings' variables
  const val = (name, dimSize) => {
    if (n._scope && name in n._scope) return n._scope[name];   // a variable of a block around this node
    if (!vars[name]) vars[name] = dimSize != null ? {value:0, min:0, max:Math.max(1, dimSize - 1), step:1} : {value:1, min:0, max:1, step:0.01};
    if (!used.includes(name)) used.push(name);
    return Number(vars[name].value) || 0;
  };
  const walk = a => {
    if (a.k === 'ref'){
      a._var = null; a._idx = null; a._x = false; a._bad = false; a._g = n._g;
      if (fnOfX){
        if (a.key0 === 'x' && !a.attr && !a.idx) a._x = true;
        else if (!a.attr && !a.idx && !a.key0.includes(SEP)){ a._var = a.key0; a._varVal = val(a.key0); }
        else a._bad = true;
      }
      else if (!a.attr && !a.idx && !a.quoted && !a.key0.includes(SEP) && !isName(a.key0, n._g)){ a._var = a.key0; a._varVal = val(a.key0); }
      else if (a.idx){
        const dims = refDims(a, n._g);
        const num = (x, d) => x == null ? null : typeof x === 'number' ? x : (x.neg ? -1 : 1)*Math.round(val(x.v, dims ? dims[d] : 10));
        a._idx = a.idx.map((it, d) => it.one != null ? {one:num(it.one, d)} : {a:num(it.a, d), b:num(it.b, d)});
      }
    }
    if (a.a) walk(a.a); if (a.b) walk(a.b); if (a.args) a.args.forEach(walk);
  };
  walk(ast);
  return ast;
}
// The weights a .pre_weights/.post_weights reference reads: which entries of which weight array, as a matrix with one row per
// neuron of the named layer (then indexed). The positions are cached; the values are read fresh each time.
const wposCache = new Map();
function weightView(a, g=root){
  const rr = lookup(a.key0, g); if (rr.err) throw new Error(rr.err);
  const L = rr.node, gg = rr.graph;
  if (L.type !== 'dense') throw new Error(`.${a.attr} works on Dense layers, and ${a.plabel} is a ${L.type === 'group' ? 'block' : labelOf(L)}.`);
  if ((!L._in || !L._in[0] || !L._out) && L._r && L._r.wait) throw Object.assign(new Error(`Waiting for ${a.plabel}.`), {wait:true});   // only waiting, not broken
  if (!L._in || !L._in[0] || !L._out) throw new Error(`${a.plabel} has no shape yet. Fix the issues before it first.`);
  const fi = lastDim(L._in[0]), fo = lastDim(L._out);
  const j = a._idx && a._idx[0].one;
  if (j != null && (j >= fo || j < -fo)) throw new Error(`${a.plabel} has ${fo} neurons (0 to ${fo - 1}), so there's no neuron ${j}.`);
  let P, shape, at, M = null;
  if (a.attr === 'pre_weights'){ P = weightsOf(L, fi*fo, fo, fi); shape = [fo, fi]; at = (j, i) => j*fi + i; }
  else {
    const nx = nextDense(gg, L, a.plabel); if (nx.err) throw new Error(nx.err);
    M = nx.node;
    if (!M._in || !M._in[0] || !M._out) throw new Error(`${M.name} has no shape yet. Fix the issues around it first.`);
    const fiM = lastDim(M._in[0]), foM = lastDim(M._out);
    if (fiM !== fo) throw new Error(`${M.name} reads ${fiM} values, not ${a.plabel}'s ${fo} neurons.`);
    P = weightsOf(M, fiM*foM, foM, fiM); shape = [fo, foM]; at = (j, o) => o*fiM + j;
  }
  const ck = `${L.id}|${M ? M.id : ''}|${a.attr}|${shape}|${JSON.stringify(a._idx)}`;
  let pos = wposCache.get(ck);
  if (!pos){
    const sel = a._idx ? selIndex(shape, a._idx, a.label) : null, cols = shape[1], n = sel ? sel.length : shape[0]*cols;
    pos = new Int32Array(n);
    for (let k = 0; k < n; k++){ const flat = sel ? sel[k] : k; pos[k] = at(Math.floor(flat/cols), flat % cols); }
    if (wposCache.size > 100) wposCache.clear();
    wposCache.set(ck, pos);
  }
  return {P, pos, shape, M};
}
const SEP = '\u0001';
   // joins the parts of a dotted reference inside keys
const topName = key => key.split(SEP)[0];
// Follows Block.layer paths: returns the component and the blocks it sits in, or an error message.
function resolveRef(key, g=root){
  const parts = key.split(SEP), groups = []; let graph = g, node = null;
  for (let i = 0; i < parts.length; i++){
    node = graph.nodes.find(x => x.name === parts[i]);
    if (!node) return {err: i ? `${parts.slice(0, i).join('.')} has no component named "${parts[i]}".` : `There's no component named "${parts[0]}".`};
    if (i < parts.length - 1){
      if (node.type !== 'group') return {err: node.type === 'dense' ? `${parts.slice(0, i + 1).join('.')} is a Dense layer: after its name use .pre_weights or .post_weights, like ${parts.slice(0, i + 1).join('.')}[0].pre_weights.` : `${parts.slice(0, i + 1).join('.')} isn't a block, so there's nothing inside it.`};
      groups.push(node); graph = node.graph;
    }
  }
  return {node, groups, graph};
}
/* Names are looked up from where the expression sits: its own graph first, then each block around it out to the
   top level, so a component inside a block sees its neighbours by name and everything around it too. Every node
   knows its graph (n._g) and every block's graph its block (g._blk); linkGraphs sets them. */
function linkGraphs(g = root){ for (const n of g.nodes){ n._g = g; if (n.graph){ n.graph._blk = n; linkGraphs(n.graph); } } }
function graphChain(g){ const out = []; for (let x = g || root; x; x = x._blk && x._blk._g) out.push(x); if (out[out.length - 1] !== root) out.push(root); return out; }
function lookup(key, g){
  let first = null; for (const x of graphChain(g)){ const r = resolveRef(key, x); if (r.node) return r; first ||= r; }
  const dm = !key.includes(SEP) && didYouMean(key, g);
  return dm ? {...first, err:`${first.err} Did you mean ${dm}?`} : first;
}
// The component a mistyped name most likely meant: the same up to lookalike characters (l, 1 and I; O and 0) and
// case, or one character off.
function didYouMean(name, g){
  const norm = s => s.toLowerCase().replace(/[l1i|]/g, '1').replace(/o/g, '0'), want = norm(name);
  const off1 = (a, b) => { if (Math.abs(a.length - b.length) > 1) return false; let i = 0, j = 0, d = 0; while (i < a.length && j < b.length){ if (a[i] === b[j]){ i++; j++; continue; } if (++d > 1) return false; if (a.length > b.length) i++; else if (b.length > a.length) j++; else { i++; j++; } } return d + (a.length - i) + (b.length - j) <= 1; };
  let best = null;
  for (const x of graphChain(g)) for (const m of x.nodes){
    if (m.type === 'gin' || m.type === 'gout' || m.name === name) continue;
    if (norm(m.name) === want) return refName(m.name);
    if (!best && name.length > 2 && off1(m.name.toLowerCase(), name.toLowerCase())) best = refName(m.name);
  }
  return best;
}
const isName = (name, g) => graphChain(g).some(x => x.nodes.some(m => m.name === name));
// The component in graph g that holds m (m itself, or the block in g that it sits inside), or null.
function holderIn(g, m){ while (m && m._g !== g){ const b = m._g && m._g._blk; if (!b) return null; m = b; } return m || null; }
// Shape check for a Loss block; also finds the Data component that supplies its samples and labels.
// The shape of an expression's value for one sample: [] for a single number. A reference keeps its component's
// shape; an index keeps the dimensions it slices and drops the ones it picks; element-wise maths keeps the shape
// (or becomes a flat vector when two different shapes of the same size meet); sum, mean, mse, ce and nll give one number.
function selShape(shape, idx){   // a single index drops its dimension; a slice keeps its length
  const out = [];
  shape.forEach((n, d) => {
    const it = idx[d]; if (!it){ out.push(n); return; }
    if (it.one != null) return;
    const fix = (v, dflt) => v == null ? dflt : v < 0 ? n + v : v;
    out.push(fix(it.b, n) - fix(it.a, 0));
  });
  return out;
}
function shapeOf(ast, n, g, ds, xShape){
  const shp = a => {
    switch (a.k){
      case 'num': return [];
      case 'ref': {
        if (a._var != null) return [];
        if (a._x) return xShape;
        if (a._bad) throw new Error(`${a.label}: a function of x can use x, numbers and variables.`);
        if (a.attr){ const wv = weightView(a, g); return a._idx ? selShape(wv.shape, a._idx) : wv.shape; }
        const rr = lookup(a.key, g), m = rr.node;
        if (rr.err) throw new Error(rr.err);
        if (m === n) throw new Error(`${n.type === 'loss' ? 'A loss' : 'A component'} can't use its own value.`);
        const sh = m._out || ((m.type === 'output' || m.type === 'gout') && m._in && m._in[0]);
        if (!sh && m._r && m._r.wait) throw Object.assign(new Error(`Waiting for ${a.label}.`), {wait:true});   // it's only waiting (for data, say), not broken
        if (!sh) throw new Error(`${a.label} has no output yet. Fix the issues before it first.`);
        if (!a._idx) return sh;
        selIndex(sh, a._idx, a.label);   // checks the indices
        return selShape(sh, a._idx);
      }
      case 'neg': return shp(a.a);
      case 'bin': {
        const x = shp(a.a), y = shp(a.b), sx = sizeOf(x), sy = sizeOf(y);
        if (sx === 1) return y.length ? y : x;
        if (sy === 1) return x;
        if (sx !== sy) throw new Error(`Sizes don't match: ${sx.toLocaleString()} vs ${sy.toLocaleString()} values.`);
        return fmt(x) === fmt(y) ? x : [sx];
      }
      case 'fn': {
        const sv = a.args.map(shp), s0 = sizeOf(sv[0]);
        if (a.f === 'mse'){ const s1 = sizeOf(sv[1]); if (s0 !== 1 && s1 !== 1 && s0 !== s1) throw new Error(`mse() needs two inputs of the same size, but gets ${s0.toLocaleString()} and ${s1.toLocaleString()} values.`); return []; }
        if (a.f === 'ce' || a.f === 'nll'){ if (ds && s0 !== ds.classes) throw new Error(`${a.f}() needs one score per class (${ds.classes} for ${ds.name}), but gets ${s0.toLocaleString()} values.`); return []; }
        if (a.f === 'max' || a.f === 'min'){
          if (sv.length === 1) return [];
          const s1 = sizeOf(sv[1]); if (s0 === 1) return sv[1]; if (s1 === 1) return sv[0];
          if (s0 !== s1) throw new Error(`${a.f}() needs two inputs of the same size, but gets ${s0.toLocaleString()} and ${s1.toLocaleString()} values.`);
          return fmt(sv[0]) === fmt(sv[1]) ? sv[0] : [s0];
        }
        return a.f === 'sum' || a.f === 'mean' ? [] : sv[0];
      }
    }
  };
  return shp(ast);
}
// Shape check for a Loss block; also finds the Data component that supplies its samples and labels.
function checkLoss(n, g){
  if (n.props.expr == null){   // Loss blocks from before expressions: turn their wire into ce()/nll()
    const e = g.edges.find(e => e.to === n.id), s = e && byId(g, e.from);
    n.props.expr = s ? `${s.type === 'softmax' || presetOf(s) === 'softmax' ? 'nll' : 'ce'}(${refName(s.name)})` : '';
    delete n.props.fn; g.edges = g.edges.filter(e => e.to !== n.id);
  }
  let ast; try { ast = bindVars(parseLoss(n.props.expr), n); } catch(e){ n._srcData = null; return {err:e.message}; }
  n._srcData = null;
  for (const key of ast.refs){ const rr = lookup(key, g), d = rr.node && upstreamData(rr.graph, rr.node); if (d){ n._srcData = d; break; } }
  let size; try { size = sizeOf(shapeOf(ast, n, g, n._srcData && dsById(n._srcData.props.dataset))); } catch(e){ return e.wait ? {wait:e.message} : {err:e.message}; }
  if (!n._srcData) return {err:'None of the components it uses is fed by a Data component.'};
  return {out:[1], sum: size > 1 ? `Mean of ${size.toLocaleString()} values` : 'Single value'};
}
const mapv = (x, f) => typeof x === 'number' ? f(x) : x.map(v => f(v));
// Element-wise op with broadcasting of plain numbers and 1-value tensors.
function ew(x, y, f){
  const xn = typeof x === 'number', yn = typeof y === 'number';
  if (xn && yn) return f(x, y);
  if (xn || (!yn && x.length === 1 && y.length > 1)){ const c = xn ? x : x[0]; return y.map(v => f(c, v)); }
  if (yn || (y.length === 1 && x.length > 1)){ const c = yn ? y : y[0]; return x.map(v => f(v, c)); }
  if (x.length !== y.length) throw new Error(`Sizes don't match: ${x.length} vs ${y.length} values.`);
  return x.map((v, i) => f(v, y[i]));
}
const BIN = {'+':(a, b) => a + b, '-':(a, b) => a - b, '*':(a, b) => a*b, '/':(a, b) => a/b, '^':Math.pow};
const total = v => { let s = 0; for (const x of v) s += x; return s; };
// Evaluates an expression; get(name) gives a tensor, label() the sample's label.
// With wantGrad, also returns d(scale·loss)/d(each named tensor).
function evalLoss(ast, get, label, wantGrad, scale=1, seed=null){
  let acc = null;
  const ev = a => {
    switch (a.k){
      case 'num': return a._v = a.v;
      case 'ref': {
        if (a._var != null) return a._v = a._varVal;
        if (a.attr){ const wv = weightView(a, a._g); a._P = wv.P; a._pos = wv.pos; return a._v = Float32Array.from(wv.pos, q => wv.P.W[q]); }
        const t = get(a.key); if (!a._idx){ a._sel = null; return a._v = t.data; } a._full = t.data.length; a._sel = selIndex(t.shape, a._idx, a.label); return a._v = Float32Array.from(a._sel, i => t.data[i]); }
      case 'neg': return a._v = mapv(ev(a.a), v => -v);
      case 'bin': return a._v = ew(ev(a.a), ev(a.b), BIN[a.op]);
      case 'fn': {
        const xs = a.args.map(ev), x = xs[0];
        switch (a.f){
          case 'abs': return a._v = mapv(x, Math.abs);
          case 'log': return a._v = mapv(x, v => Math.log(Math.max(v, 1e-12)));
          case 'exp': return a._v = mapv(x, Math.exp);
          case 'sqrt': return a._v = mapv(x, v => Math.sqrt(Math.max(v, 0)));
          case 'tanh': return a._v = mapv(x, Math.tanh);
          case 'sigmoid': return a._v = mapv(x, v => 1/(1 + Math.exp(-v)));
          case 'relu': return a._v = mapv(x, v => v > 0 ? v : 0);
          case 'softmax': {   // over all the values it gets
            if (typeof x === 'number') return a._v = 1;
            let m = -Infinity; for (const v of x) m = Math.max(m, v);
            const e = x.map(v => Math.exp(v - m)), s = total(e); return a._v = e.map(v => v/s);
          }
          case 'max': case 'min': {
            if (xs.length === 2) return a._v = ew(x, xs[1], a.f === 'max' ? Math.max : Math.min);
            if (typeof x === 'number') return a._v = x;
            let k = 0; for (let i = 1; i < x.length; i++) if (a.f === 'max' ? x[i] > x[k] : x[i] < x[k]) k = i;
            a._k = k; return a._v = x[k];
          }
          case 'sum': return a._v = typeof x === 'number' ? x : total(x);
          case 'mean': return a._v = typeof x === 'number' ? x : total(x)/x.length;
          case 'mse': { const d = ew(x, xs[1], (p, q) => p - q); a._d = d; return a._v = typeof d === 'number' ? d*d : total(d.map(v => v*v))/d.length; }
          case 'ce': case 'nll': {
            const y = label(), v = typeof x === 'number' ? [x] : x;
            if (y === undefined) throw new Error(`${a.f}() needs labels from a Data component.`);
            a._y = y;
            if (acc === null) acc = argmax(v) === y ? 1 : 0;
            if (a.f === 'nll') return a._v = -Math.log(Math.max(v[y], 1e-7));
            let m = -Infinity, s = 0; for (const q of v) m = Math.max(m, q); for (const q of v) s += Math.exp(q - m);
            a._p = Float32Array.from(v, q => Math.exp(q - m)/s);
            return a._v = -(v[y] - m - Math.log(s));
          }
        }
      }
    }
  };
  const r = ev(ast), L = typeof r === 'number' ? r : total(r)/r.length;
  if (!wantGrad) return {L, acc, v:r};
  const grads = new Map();
  const fit = (g, like) => {   // shape a gradient like the value it belongs to (undo broadcasting)
    if (typeof like === 'number') return typeof g === 'number' ? g : total(g);
    if (typeof g === 'number') return new Float32Array(like.length).fill(g);
    return g.length !== like.length && like.length === 1 ? Float32Array.of(total(g)) : g;
  };
  const mul = (p, q) => p*q;
  const bk = (a, g) => {
    if (!a.live) return;
    g = fit(g, a._v);
    switch (a.k){
      case 'ref': {
        if (a._var != null) return;   // variables are settings, not trained
        if (a.attr){ const o = grads.get(a.key); if (o) o.g = o.g.map((v, i) => v + g[i]); else grads.set(a.key, {P:a._P, pos:a._pos, g:Float32Array.from(g)}); return; }   // straight into the weights
        if (a._sel){ const full = new Float32Array(a._full); a._sel.forEach((i, k) => full[i] += g[k]); g = full; }   // only the picked neurons get gradient
        const o = grads.get(a.key); grads.set(a.key, o ? o.map((v, i) => v + g[i]) : Float32Array.from(g)); return;
      }
      case 'neg': return bk(a.a, mapv(g, v => -v));
      case 'bin': {
        const x = a.a._v, y = a.b._v;
        if (a.op === '+'){ bk(a.a, g); bk(a.b, g); }
        else if (a.op === '-'){ bk(a.a, g); bk(a.b, mapv(g, v => -v)); }
        else if (a.op === '*'){ bk(a.a, ew(g, y, mul)); bk(a.b, ew(g, x, mul)); }
        else if (a.op === '/'){ bk(a.a, ew(g, y, (p, q) => p/q)); bk(a.b, ew(ew(g, x, mul), y, (p, q) => -p/(q*q))); }
        else { bk(a.a, ew(g, ew(x, y, (p, q) => q*Math.pow(p, q - 1)), mul)); bk(a.b, ew(g, ew(x, y, (p, q) => p > 0 ? Math.pow(p, q)*Math.log(p) : 0), mul)); }
        return;
      }
      case 'fn': {
        const c = a.args[0], x = c._v, v = a._v;
        switch (a.f){
          case 'abs': return bk(c, ew(g, x, (p, q) => p*Math.sign(q)));
          case 'log': return bk(c, ew(g, x, (p, q) => p/Math.max(q, 1e-12)));
          case 'exp': return bk(c, ew(g, v, mul));
          case 'sqrt': return bk(c, ew(g, v, (p, q) => q > 0 ? p/(2*q) : 0));
          case 'tanh': return bk(c, ew(g, v, (p, q) => p*(1 - q*q)));
          case 'sigmoid': return bk(c, ew(g, v, (p, q) => p*q*(1 - q)));
          case 'relu': return bk(c, ew(g, x, (p, q) => q > 0 ? p : 0));
          case 'softmax': {
            if (typeof x === 'number') return;
            const gv = fit(g, v); let s = 0; for (let i = 0; i < v.length; i++) s += gv[i]*v[i];
            return bk(c, v.map((q, i) => q*(gv[i] - s)));
          }
          case 'max': case 'min': {
            if (a.args.length === 2){   // the larger (smaller) side gets the gradient; a tie splits it, as torch.maximum does
              const y = a.args[1]._v, w = (p, q) => a.f === 'max' ? (p > q ? 1 : p < q ? 0 : 0.5) : (p < q ? 1 : p > q ? 0 : 0.5);
              bk(c, ew(g, ew(x, y, w), mul)); bk(a.args[1], ew(g, ew(x, y, (p, q) => 1 - w(p, q)), mul)); return;
            }
            if (typeof x === 'number') return bk(c, g);
            const gx = new Float32Array(x.length); gx[a._k] = g; return bk(c, gx);
          }
          case 'sum': return bk(c, g);
          case 'mean': return bk(c, typeof x === 'number' ? g : g/x.length);
          case 'mse': { const d = a._d, k = typeof d === 'number' ? 1 : d.length, gd = mapv(d, q => 2*q*g/k); bk(c, gd); bk(a.args[1], mapv(gd, q => -q)); return; }
          case 'ce': { const gx = a._p.map((p, i) => g*(p - (i === a._y ? 1 : 0))); return bk(c, typeof x === 'number' ? gx[0] : gx); }
          case 'nll': { const xv = typeof x === 'number' ? [x] : x, gx = new Float32Array(xv.length); gx[a._y] = -g/Math.max(xv[a._y], 1e-7); return bk(c, typeof x === 'number' ? gx[0] : gx); }
        }
      }
    }
  };
  bk(ast, seed || (typeof r === 'number' ? scale : new Float32Array(r.length).fill(scale/r.length)));
  return {L, acc, grads, v:r};
}
const FN_HELP = {abs:'absolute value', log:'natural log', exp:'e to the power', sqrt:'square root', sum:'sum of values', mean:'mean of values', ce:'cross-entropy of scores vs labels', nll:'cross-entropy of probabilities', mse:'mean squared error (a, b)'};

/* Keeping expressions pointed at the right components when names change. refSpans finds where each
   component reference sits in the text; rewriteRefs swaps in the components' new paths after a rename,
   group or ungroup, using their paths from before the change. */
function refSpans(src){
  src = String(src ?? '');
  const re = /(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+)|([A-Za-z_]\w*)|"([^"]*)"|'([^']*)'|(\*\*|[-+*/^(),.\[\]:])|\s+/y, toks = [];
  for (let i = 0; i < src.length;){
    re.lastIndex = i; const m = re.exec(src);
    if (!m){ i++; continue; }
    if (m[2] != null || m[3] != null || m[4] != null) toks.push({t:'id', v:m[2] ?? m[3] ?? m[4], s:i, e:re.lastIndex, quoted:m[2] == null});
    else if (m[5]) toks.push({t:m[5], s:i, e:re.lastIndex});
    else if (m[1]) toks.push({t:'num', s:i, e:re.lastIndex});
    i = re.lastIndex;
  }
  const spans = []; let depth = 0;
  for (let k = 0; k < toks.length; k++){
    const t = toks[k];
    if (t.t === '[') depth++; else if (t.t === ']') depth--;
    if (t.t !== 'id' || depth > 0) continue;                              // names inside [] are variables
    if (k > 0 && toks[k - 1].t === '.') continue;                          // part of a path, or .pre_weights
    if (!t.quoted && toks[k + 1] && toks[k + 1].t === '(') continue;      // a function
    const path = [t.v]; let end = t.e, j = k;
    while (toks[j + 1] && toks[j + 1].t === '.' && toks[j + 2] && toks[j + 2].t === 'id' && !WATTR.has(toks[j + 2].v)){ path.push(toks[j + 2].v); end = toks[j + 2].e; j += 2; }
    spans.push({start:t.s, end, path});
  }
  return spans;
}
function pathMap(g=root, pre=[], out=new Map()){ for (const n of g.nodes){ const p = [...pre, n.name]; out.set(n.id, p); if (n.graph) pathMap(n.graph, p, out); } return out; }
function exprNodes(g=root, out=[]){ for (const n of g.nodes){ if (n.props && (n.type === 'ptrin' || n.type === 'ptrout' || (typeof n.props.expr === 'string' && n.props.expr.trim()) || (n.props.inputs || []).some(Boolean))) out.push(n); if (n.graph) exprNodes(n.graph, out); } return out; }
function rewriteRefs(before){
  const after = pathMap(), byBefore = new Map(), byAfter = new Map(); let changed = 0;
  for (const [id, p] of before) byBefore.set(p.join(SEP), id);
  for (const [id, p] of after) byAfter.set(p.join(SEP), id);
  // What a path names from inside the graph at prefix pre: the innermost enclosing graph that has it.
  const seek = (map, pre, rel) => { for (let k = pre.length; k >= 0; k--){ const id = map.get([...pre.slice(0, k), ...rel].join(SEP)); if (id) return id; } return null; };
  const fix = (src, oldPre, newPre) => {
    const spans = refSpans(src);
    for (let i = spans.length - 1; i >= 0; i--){
      const sp = spans[i], id = seek(byBefore, oldPre, sp.path), abs = id && after.get(id); if (!abs) continue;
      let np = null;   // the shortest path that names it from the expression's new place
      for (let k = newPre.length; k >= 0 && !np; k--) if (abs.slice(0, k).join(SEP) === newPre.slice(0, k).join(SEP) && seek(byAfter, newPre, abs.slice(k)) === id) np = abs.slice(k);
      if (np && np.join(SEP) !== sp.path.join(SEP)){ src = src.slice(0, sp.start) + np.map(refName).join('.') + src.slice(sp.end); changed++; }
    }
    return src;
  };
  for (const n of exprNodes()){
    const oldPre = (before.get(n.id) || after.get(n.id)).slice(0, -1), newPre = after.get(n.id).slice(0, -1), f = s => fix(s, oldPre, newPre);
    if (typeof n.props.expr === 'string') n.props.expr = f(n.props.expr);
    if (n.props.inputs) n.props.inputs = n.props.inputs.map(s => typeof s === 'string' && s ? f(s) : s);
    for (const k of n.type === 'ptrin' ? ['target'] : n.type === 'ptrout' ? ['pair'] : []) if (typeof n.props[k] === 'string' && n.props[k]) n.props[k] = f(n.props[k]);
  }
  return changed;
}
