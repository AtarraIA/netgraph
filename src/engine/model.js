/* Netgraph engine: the graph model.
   The engine has no DOM: it runs in the page, in the training worker and in the Node tests.
   Graphs are {nodes, edges}; a node is {id, type, name, x, y, props} (plus .graph for a block). */
const uid = () => Math.random().toString(36).slice(2, 10);
const DEFS = {
  input:    {label:'Input', fam:'io', ins:0, outs:1, props:{shape:'1,28,28'}},
  output:   {label:'Output', fam:'io', ins:1, outs:0},
  dense:    {label:'Dense', fam:'dense', ins:1, outs:1, props:{in_features:'auto', out_features:128, bias:true}},
  conv2d:   {label:'Conv2D', fam:'conv', ins:1, outs:1, props:{in_channels:'auto', out_channels:32, kernel:3, stride:1, padding:1}},
  maxpool:  {label:'MaxPool2D', fam:'pool', ins:1, outs:1, props:{kernel:2, stride:'auto'}},
  avgpool:  {label:'AvgPool2D', fam:'pool', ins:1, outs:1, props:{kernel:2, stride:'auto'}},
  gap:      {label:'Global Avg Pool', fam:'pool', ins:1, outs:1},
  relu:     {label:'ReLU', fam:'act', ins:1, outs:1},
  gelu:     {label:'GELU', fam:'act', ins:1, outs:1},
  sigmoid:  {label:'Sigmoid', fam:'act', ins:1, outs:1},
  tanh:     {label:'Tanh', fam:'act', ins:1, outs:1},
  softmax:  {label:'Softmax', fam:'act', ins:1, outs:1},
  batchnorm:{label:'BatchNorm', fam:'norm', ins:1, outs:1},
  dropout:  {label:'Dropout', fam:'norm', ins:1, outs:1, props:{p:0.5}},
  flatten:  {label:'Flatten', fam:'shape', ins:1, outs:1},
  unflatten:{label:'Unflatten', fam:'shape', ins:1, outs:1, props:{to:'auto'}},
  add:      {label:'Add', fam:'merge', ins:2, outs:1},
  concat:   {label:'Concat', fam:'merge', ins:2, outs:1},
  group:    {label:'Block', fam:'block', ins:1, outs:1},
  gin:      {label:'Block input', fam:'io', ins:0, outs:1},
  gout:     {label:'Block output', fam:'io', ins:1, outs:0},
  data:     {label:'Data', fam:'data', ins:1, outs:1, props:{dataset:'', flatten:false, index:0, test:20, seed:1, run_from:'test', run_n:'all'}},
  sliders:  {label:'Sliders', fam:'io', ins:0, outs:1, props:{sliders:[{name:'a', value:0, min:-1, max:1, step:0.01}, {name:'b', value:0, min:-1, max:1, step:0.01}]}},
  pca:      {label:'PCA', fam:'shape', ins:1, outs:1, props:{components:2, fit_n:5000}},
  plot2d:   {label:'Plot 2D', fam:'io', ins:1, outs:0, props:{x_dim:0, y_dim:1}},
  plot3d:   {label:'Plot 3D', fam:'io', ins:1, outs:0, props:{x_dim:0, y_dim:1, z_dim:2}},
  ptrin:    {label:'Pointer in', fam:'ptr', ins:1, outs:1, props:{target:''}},
  ptrout:   {label:'Pointer out', fam:'ptr', ins:1, outs:1, props:{pair:''}},
  activation:{label:'Activation', fam:'act', ins:1, outs:1, props:{fn:'x'}},
  loss:     {label:'Loss', fam:'loss', ins:0, outs:1, props:{expr:'', optimizer:'adam', lr:0.001, batch:32, epochs:1}},
};
const SHORT = {input:'x', output:'out', dense:'fc', conv2d:'conv', maxpool:'pool', avgpool:'avgpool', gap:'gap', relu:'relu', gelu:'gelu', sigmoid:'sigmoid', tanh:'tanh', softmax:'softmax', batchnorm:'bn', dropout:'drop', flatten:'flatten', unflatten:'unflatten', add:'add', concat:'cat', gin:'in', gout:'out', data:'data', loss:'loss', activation:'act', sliders:'sliders', ptrin:'pin', ptrout:'pout', plot2d:'plot', plot3d:'plot3d', pca:'pca'};
const PLOTS = new Set(['plot2d', 'plot3d']);   // components that show what reaches them and keep what inference runs send

/* ---------- state ---------- */
let root = {nodes:[], edges:[]};
let stats = {params:0, errs:0, ins:[], outs:[]};
const byId = (g, id) => g.nodes.find(n => n.id === id);
function findNode(id, g = root){ for (const n of g.nodes){ if (n.id === id) return n; if (n.graph){ const m = findNode(id, n.graph); if (m) return m; } } return null; }   // anywhere, inside blocks too
const edge = (from, to, port=0) => ({id:uid(), from, to, port});
function allNames(g=root, set=new Set()){ for (const n of g.nodes){ set.add(n.name); if (n.graph) allNames(n.graph, set); } return set; }
function nextName(type){ const b = SHORT[type]; const s = allNames(); if (type==='input' && !s.has('x')) return 'x'; let i=1; while (s.has(b+i)) i++; return b+i; }
function localName(g, type){ const b = SHORT[type]; let i=1; while (g.nodes.some(n => n.name === b+i)) i++; return b+i; }
function nextBlockName(base){ const s = allNames(); if (!s.has(base)) return base; let i=2; while (s.has(base+' '+i)) i++; return base+' '+i; }
function mk(type, x, y, props={}, name){
  if (ACT_PRESETS[type]) return mk('activation', x, y, {fn:ACT_PRESETS[type].fn, ...props}, name || nextName(type));   // ReLU & co. are Activations
  const d = DEFS[type];
  return {id:uid(), type, x, y, name: name || nextName(type), props:{...JSON.parse(JSON.stringify(d.props||{})), ...props}};
}

/* ---------- blocks ---------- */
function blockFrom(name, specs){
  const g = {nodes:[], edges:[]};
  const gin = mk('gin', 0, 0, {}, 'in'); g.nodes.push(gin); let prev = gin;
  specs.forEach(([t,p], i) => { const n = mk(t, (i+1)*240, 0, p||{}, localName(g,t)); g.nodes.push(n); g.edges.push(edge(prev.id, n.id)); prev = n; });
  const go = mk('gout', (specs.length+1)*240, 0, {}, 'out'); g.nodes.push(go); g.edges.push(edge(prev.id, go.id));
  return {id:uid(), type:'group', x:0, y:0, name, props:{}, graph:g};
}
function residualBlock(){
  const g = {nodes:[], edges:[]};
  const add = (t,x,y,p) => { const n = mk(t,x,y,p||{}, t==='gin'?'in':t==='gout'?'out':localName(g,t)); g.nodes.push(n); return n; };
  const gin = add('gin', 0, 150);
  const c1 = add('conv2d', 240, 0, {out_channels:'same'});
  const b1 = add('batchnorm', 480, 0);
  const r1 = add('relu', 720, 0);
  const c2 = add('conv2d', 960, 0, {out_channels:'same'});
  const b2 = add('batchnorm', 1200, 0);
  const sum = add('add', 1440, 60);
  const r2 = add('relu', 1680, 60);
  const go = add('gout', 1920, 60);
  g.edges.push(edge(gin.id,c1.id), edge(c1.id,b1.id), edge(b1.id,r1.id), edge(r1.id,c2.id), edge(c2.id,b2.id),
    edge(b2.id,sum.id,0), edge(gin.id,sum.id,1), edge(sum.id,r2.id), edge(r2.id,go.id));
  return {id:uid(), type:'group', x:0, y:0, name:'Residual block', props:{}, graph:g};
}
const BUILTIN = [
  {label:'MLP', fam:'dense', make:() => blockFrom('MLP', [['dense',{out_features:128}],['relu'],['dense',{out_features:64}],['relu']])},
  {label:'Conv block', fam:'conv', make:() => blockFrom('Conv block', [['conv2d',{out_channels:32}],['batchnorm'],['relu'],['maxpool']])},
  {label:'Residual block', fam:'merge', make:residualBlock},
];
function strip(o){ return JSON.parse(JSON.stringify(o, (k,v) => k.startsWith('_') ? undefined : v)); }
function remapGraph(g){
  const m = {};
  for (const n of g.nodes){ const nid = uid(); m[n.id] = nid; n.id = nid; if (n.graph) remapGraph(n.graph); }
  g.edges = g.edges.filter(e => m[e.from] && m[e.to]).map(e => ({id:uid(), from:m[e.from], to:m[e.to], port:e.port}));
}
function cloneFresh(node){ const c = strip(node); c.id = uid(); if (c.graph) remapGraph(c.graph); return c; }
function countLayers(g){ let c=0; for (const n of g.nodes){ if (n.type==='gin'||n.type==='gout') continue; c += n.graph ? countLayers(n.graph) : 1; } return c; }

/* ---------- shape inference ---------- */
const intOk = v => Number.isInteger(v) && v >= 1;
const isAuto = v => v === '' || v == null || String(v).trim().toLowerCase() === 'auto';
const fmt = s => Array.isArray(s) ? s.join('×') : '—';
const fmtN = n => n >= 1e6 ? (n/1e6).toFixed(2)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'k' : String(n);

/* ---------- inference ---------- */
function rng(seed){ let a = seed >>> 0; return () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function hashStr(str){ let h = 2166136261; for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 16777619); return h >>> 0; }
function range(a){ let lo = Infinity, hi = -Infinity; for (const v of a){ if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }
const argmax = a => { let m = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[m]) m = i; return m; };
const ELEM = new Set(['activation','relu','gelu','sigmoid','tanh','softmax','dropout','batchnorm','flatten']);
const ACT = new Set(['activation','relu','gelu','sigmoid','tanh','softmax']);
/* Activations are one component whose function of x is typed in (props.fn). ReLU, GELU, Sigmoid, Tanh and Softmax
   are just these functions: while a component's function matches one exactly it runs the fast built-in code, shows
   that name and exports as the matching PyTorch module. Other names in a function become variables (sliders). */
const ACT_PRESETS = {
  relu:{label:'ReLU', fn:'max(x, 0)'},
  gelu:{label:'GELU', fn:'0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x^3)))'},
  sigmoid:{label:'Sigmoid', fn:'1/(1 + exp(-x))'},
  tanh:{label:'Tanh', fn:'tanh(x)'},
  softmax:{label:'Softmax', fn:'softmax(x)'},
};
const normFn = s => String(s ?? '').replace(/\s+/g, '');
const presetOf = n => { if (!n || n.type !== 'activation') return null; const f = normFn(n.props.fn); for (const k in ACT_PRESETS) if (normFn(ACT_PRESETS[k].fn) === f) return k; return null; };
const labelOf = n => n.type === 'activation' ? (ACT_PRESETS[presetOf(n)] || {label:'Activation'}).label : DEFS[n.type].label;
function migrateActs(g = root){   // ReLU, GELU, ... components saved before Activations existed
  for (const n of g.nodes){ if (n.type === 'pointer'){ n.type = 'ptrout'; n.props = {pair:''}; }   // single Pointers came before pairs
    if (ACT_PRESETS[n.type]){ n.props = {...(n.props || {}), fn:ACT_PRESETS[n.type].fn}; n.type = 'activation'; } if (n.graph) migrateActs(n.graph); }
}
function mlpLike(g){
  let dense = 0;
  for (const m of g.nodes){
    if (m.type === 'gin' || m.type === 'gout') continue;
    if (m.type === 'dense') dense++;
    else if (m.type === 'group'){ if (!mlpLike(m.graph)) return false; dense++; }
    else if (!ELEM.has(m.type)) return false;
  }
  return dense > 0;
}
// Components whose input is an expression instead of a wire: every Loss, and a Flatten with an expression.
// Every input of a component is either a wire or an expression: props.inputs[i] holds the expression for input i
// ('' or missing means the wire). A Loss block's input is always its expression (props.expr).
const portExpr = (n, i) => {
  const p = n.props || {}, s = p.inputs ? p.inputs[i] : n.type === 'flatten' && i === 0 ? p.expr : undefined;   // a Flatten's old Expression field counts as its input
  return typeof s === 'string' && s.trim() ? s.trim() : null;
};
const wired = (n, i) => !portExpr(n, i);
const exprsOf = n => { if (n.type === 'loss') return [String(n.props.expr ?? '')]; const out = []; for (let i = 0; i < DEFS[n.type].ins; i++){ const s = portExpr(n, i); if (s) out.push(s); } return out; };
const hasExpr = n => n.type === 'loss' || exprsOf(n).length > 0;
const lastDim = s => Array.isArray(s) ? s[s.length-1] : null;
