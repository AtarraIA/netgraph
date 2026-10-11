function render(){ infer(); forward(root, null); renderNeuro(); renderNodes(); renderWires(); renderInspector(); renderCrumbs(); save(); }
function soft(){ infer(); forward(root, null); renderNeuro(); renderNodes(); renderWires(); renderInspector(); renderCrumbs(); save(); }

/* ---------- persistence ---------- */
const KEY = 'netgraph.v1';
let saveT;
function save(){ clearTimeout(saveT); saveT = setTimeout(persist, 250); }
function persist(){ saveT = null; commit(); try { localStorage.setItem(KEY, JSON.stringify({root:strip(root), lib})); } catch(e){} }
function flushSave(){ if (saveT){ clearTimeout(saveT); persist(); } }

/* ---------- undo ---------- */
// Every save compares the graph with the last snapshot; a real change becomes an undo step.
// Edits within one save delay (like typing a name) land in a single step. Browsing samples isn't a step.
const hist = {past:[], future:[], cur:null};
function commit(){
  const o = strip(root), s = JSON.stringify(o), k = JSON.stringify(o, (key, v) => key === 'index' ? undefined : v);
  if (hist.cur && hist.cur.k === k){ hist.cur.s = s; return; }
  if (hist.cur){ hist.past.push(hist.cur); if (hist.past.length > 200) hist.past.shift(); hist.future = []; }
  hist.cur = {s, k, path:[...path]};
  undoBtns();
}
function undoBtns(){ $('#bUndo').disabled = !hist.past.length; $('#bRedo').disabled = !hist.future.length; }
function stepHistory(back){
  flushSave();
  if (TR && TR.running){ toast('Stop training before undoing changes.'); return; }
  const from = back ? hist.past : hist.future, to = back ? hist.future : hist.past;
  if (!from.length){ toast(back ? 'Nothing to undo.' : 'Nothing to redo.'); return; }
  to.push(hist.cur); hist.cur = from.pop();
  root = JSON.parse(hist.cur.s); path = [...hist.cur.path]; cur();   // cur() drops a path that no longer exists
  nview = null; sel.clear(); selWire = null;
  renderPalette(); render(); undoBtns();
}
function load(){ try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && s.root && Array.isArray(s.root.nodes)){ root = s.root; lib = Array.isArray(s.lib) ? s.lib : []; return true; } } catch(e){} return false; }

/* ---------- toast ---------- */
let toastT;
function toast(m){ const t = $('#toast'); t.textContent = m; t.classList.add('on'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('on'), 3200); }

/* ---------- view ---------- */
function toWorld(cx, cy){ const r = cv.getBoundingClientRect(); return {x:(cx - r.left - view.x)/view.k, y:(cy - r.top - view.y)/view.k}; }
function fit(){
  const g = cur(), r = cv.getBoundingClientRect();
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  if (nview && nbounds) ({x0, y0, x1, y1} = nbounds);
  else {
    if (!g.nodes.length){ view = {x:r.width/2-100, y:r.height/2-60, k:1}; applyView(); return; }
    for (const n of g.nodes){ const b = boxes.get(n.id); x0=Math.min(x0,n.x); y0=Math.min(y0,n.y); x1=Math.max(x1,n.x+(b ? b.w : n.w || W)); y1=Math.max(y1,n.y+(b ? b.h : 150)); }
  }
  const side = insp.offsetWidth && window.innerWidth > 820 ? insp.offsetWidth + 24 : 0, w = r.width - side;
  const k = Math.max(0.3, Math.min(1.1, (w-80)/(x1-x0), (r.height-110)/(y1-y0)));
  view = {k, x:(w - (x1-x0)*k)/2 - x0*k, y:(r.height - (y1-y0)*k)/2 - y0*k + 14};
  applyView();
}
const KMIN = 0.25, KMAX = 2.2;
function zoomAt(f, cx, cy){
  const r = cv.getBoundingClientRect();
  if (cx == null){ cx = r.left + r.width/2; cy = r.top + r.height/2; }
  const mx = cx - r.left, my = cy - r.top;
  const k = Math.max(KMIN, Math.min(KMAX, view.k * f));
  view.x = mx - (mx - view.x) * k / view.k; view.y = my - (my - view.y) * k / view.k; view.k = k;
  applyView();
}
let hinted = false;
function arrive(){
  renderPalette(); render();
  const v = views[vkey()]; if (v){ view = {...v}; applyView(); } else fit();
  cv.classList.remove('enter'); void cv.offsetWidth; cv.classList.add('enter');
  if (nview && !nview.id && !hinted){ hinted = true; toast('Neuron view. Switch to Layers at the top left to edit.'); }
}
function pickMode(){
  if (!neuronModel(cur())) return null;
  const m = modes[path.join('/')];
  if (m) return m === 'neurons' ? {id:null} : null;
  return path.length && mlpLike(cur()) ? {id:null} : null;
}
function enter(id){
  const n = byId(cur(), id); if (!n) return;
  if (n.type === 'dense') return enterLayer(id);
  if (n.type !== 'group') return;
  views[vkey()] = {...view}; path.push(id); nview = null; sel.clear(); selWire = null;
  infer(); nview = pickMode(); arrive();
}
function enterLayer(id){
  const n = byId(cur(), id);
  if (!n || !n._in || !n._in[0] || !n._out){ toast('Connect this layer, or set its Input features, to see its neurons.'); return; }
  views[vkey()] = {...view}; nview = {id}; sel.clear(); selWire = null; arrive();
}
function goDepth(d){
  const back = nview && nview.id, same = d === path.length;
  if (same && !back) return;
  views[vkey()] = {...view};
  path = path.slice(0, d); sel.clear(); selWire = null;
  if (same){ nview = null; sel = new Set([back]); } else { infer(); nview = pickMode(); }
  arrive();
}
function goUp(){
  if (nview && nview.id) goDepth(path.length);
  else if (path.length) goDepth(path.length - 1);
  else if (nview) setMode('layers');
}
function setMode(m){
  if ((m === 'neurons') === !!(nview && !nview.id)) return;
  modes[path.join('/')] = m; views[vkey()] = {...view};
  nview = m === 'neurons' ? {id:null} : null; sel.clear(); selWire = null; arrive();
}

/* ---------- graph edits ---------- */
function reaches(g, a, b){ const E = allEdges(g), st=[a], seen=new Set(); while (st.length){ const c = st.pop(); if (c === b) return true; if (seen.has(c)) continue; seen.add(c); for (const e of E) if (e.from === c) st.push(e.to); } return false; }
// A Pointer in's output dropped on an input links it there: that layer keeps its own wire and the pair runs the stretch
// from it on the Pointer in's input.
function pointAt(ptrId, toId){
  const g = cur(), p = byId(g, ptrId), t = byId(g, toId);
  if (!p || p.type !== 'ptrin' || !t || p === t) return false;
  if (DEFS[t.type].ins !== 1){ toast(`A pointer feeds components with one input; ${t.name} takes ${DEFS[t.type].ins}.`); return true; }
  p.props.target = refName(t.name);
  toast(`${p.name} now feeds ${t.name}, beside its own input.`);
  return true;
}
// Adds a Pointer in and its Pointer out, side by side.
function addPair(pos){
  const g = cur(), pin = mk('ptrin', 0, 0), k = pin.name.slice(3), pout = mk('ptrout', 0, 0, {pair:pin.name}, allNames().has('pout' + k) ? undefined : 'pout' + k);
  if (pos){ pin.x = Math.round(pos.x - W/2); pin.y = Math.round(pos.y - 30); }
  else { const r = cv.getBoundingClientRect(), c = toWorld(r.left + r.width/2, r.top + r.height/2); pin.x = Math.round(c.x - W - 60); pin.y = Math.round(c.y - 40); while (overlaps(g, pin.x, pin.y)) pin.y += 120; }
  pout.x = pin.x + W + 120; pout.y = pin.y;
  g.nodes.push(pin, pout); sel = new Set([pin.id]); selWire = null; render();
  toast(`Wire the new input into ${pin.name} and drag its output onto the layer where the stretch starts. Wire the layer where it ends into ${pout.name}.`);
}
function connect(fromId, toId, port){
  const g = cur();
  if (fromId === toId) return false;
  if (reaches(g, toId, fromId)){ toast('That connection would create a loop.'); return false; }
  const old = g.edges.filter(e => e.to === toId && e.port === port), ne = edge(fromId, toId, port);
  g.edges = g.edges.filter(e => !old.includes(e)).concat(ne);
  const src = byId(g, fromId);
  if (src && src.type === 'data'){
    infer();
    const why = misfit(src, byId(g, toId));
    if (why){ g.edges = g.edges.filter(e => e !== ne).concat(old); toast(why); return false; }
  }
  return true;
}
function firstErr(g){
  for (const m of g.nodes){
    if (m._r && m._r.err && !m._r.err.startsWith('Connect') && m.type !== 'group') return m;
    if (m.graph){ const x = firstErr(m.graph); if (x) return x; }
  }
  return null;
}
// Why a Data component can't feed this layer, or null when the shapes fit.
function misfit(src, t){
  const s = src._out;
  if (!s) return 'This Data component has no dataset yet. Select it to import one.';
  const label = t.type === 'group' ? t.name : labelOf(t);
  let entry = [t];
  if (t.type === 'group'){ const gin = t.graph.nodes.find(m => m.type === 'gin'); entry = gin ? t.graph.edges.filter(e => e.from === gin.id).map(e => byId(t.graph, e.to)).filter(Boolean) : []; }
  if (s.length > 1 && entry.some(m => m.type === 'dense')) return `${label} needs flat features, but the data is ${fmt(s)}. Turn on Flatten in the Data component, or add a Flatten layer.`;
  const r = t._r || {};
  if (r.err && !r.err.startsWith('Connect')){ const deep = t.type === 'group' && firstErr(t.graph); return `${fmt(s)} data doesn't fit ${label}: ${deep ? deep._r.err : r.err}`; }
  return null;
}
function create(kind){
  const [k, v] = kind.split(':');
  if (k === 'layer'){ const n = mk(v, 0, 0); if (v === 'data' && datasets.length) n.props.dataset = datasets[0].id; return n; }
  if (k === 'block'){ const n = BUILTIN[+v].make(); n.name = nextBlockName(n.name); return n; }
  if (k === 'lib'){ const it = lib.find(l => l.id === v); if (!it) return null; const n = cloneFresh(it.node); n.name = nextBlockName(it.label); return n; }
}
// A component's drawn width (some are wider than W, or resized).
const widthOf = n => { const el = nodesEl.querySelector(`.node[data-id="${n.id}"]`); return el && el.offsetWidth || n.w || W; };
function overlaps(g, x, y){ return g.nodes.some(n => Math.abs(n.x - x) < W && Math.abs(n.y - y) < 100); }
function addNode(kind, pos){
  const [k, v] = kind.split(':');
  if (nview){ if (nview.id) goDepth(path.length); else setMode('layers'); pos = null; }
  if (k === 'layer' && v === 'pointer') return addPair(pos);
  const g = cur(), n = create(kind); if (!n) return;
  let src = null;
  if (pos){ n.x = Math.round(pos.x - W/2); n.y = Math.round(pos.y - 30); }
  else {
    const only = sel.size === 1 ? byId(g, [...sel][0]) : null;
    if (n.type === 'loss' && only && DEFS[only.type].outs && !n.props.expr) n.props.expr = `ce(${refName(only.name)})`;
    if (only && DEFS[only.type].outs && (DEFS[n.type].ins || n.type === 'loss')){ src = DEFS[n.type].ins ? only : null; n.x = only.x + widthOf(only) + 40; n.y = only.y; while (overlaps(g, n.x, n.y)) n.y += 120; }
    else { const r = cv.getBoundingClientRect(); const c = toWorld(r.left + r.width/2, r.top + r.height/2); n.x = Math.round(c.x - W/2); n.y = Math.round(c.y - 40); while (overlaps(g, n.x, n.y)) { n.x += 30; n.y += 30; } }
  }
  g.nodes.push(n);
  if (src) connect(src.id, n.id, 0);
  sel = new Set([n.id]); selWire = null;
  render();
}
function deleteSel(){
  const g = cur();
  if (selWire && selWire.startsWith('ptr:')){ const p = byId(g, selWire.slice(4)); if (p) p.props.target = ''; selWire = null; render(); return; }   // a Pointer in stops feeding
  if (selWire){ g.edges = g.edges.filter(e => e.id !== selWire); selWire = null; render(); return; }
  const ids = new Set([...sel].filter(id => { const n = byId(g,id); return n && n.type !== 'gin' && n.type !== 'gout'; }));
  if (!ids.size) return;
  for (const n of g.nodes) if (n.type === 'ptrout'){ const I = pairOf(n, g); if (I && (ids.has(I.id) || ids.has(n.id))){ ids.add(I.id); ids.add(n.id); } }   // pointers go in pairs
  g.nodes = g.nodes.filter(n => !ids.has(n.id));
  g.edges = g.edges.filter(e => !ids.has(e.from) && !ids.has(e.to));
  sel.clear(); render();
}
function duplicate(){
  const g = cur(), n = byId(g, [...sel][0]); if (!n) return;
  const c = cloneFresh(n); c.x += 30; c.y += 120; c.name = n.type === 'group' ? nextBlockName(n.name) : nextName(n.type);
  g.nodes.push(c); sel = new Set([c.id]); render();
}
function groupSelectionRaw(){
  const g = cur();
  const S = new Set([...sel].filter(id => { const n = byId(g,id); return n && n.type !== 'gin' && n.type !== 'gout'; }));
  if (!S.size){ toast('Select one or more components to group.'); return; }
  const inE = g.edges.filter(e => !S.has(e.from) && S.has(e.to));
  const outE = g.edges.filter(e => S.has(e.from) && !S.has(e.to));
  const srcs = new Set(inE.map(e => e.from)), exits = new Set(outE.map(e => e.from));
  if (srcs.size > 1){ toast(`A block takes one input tensor, but this selection receives ${srcs.size}. Include the branching layers too.`); return; }
  if (exits.size > 1){ toast(`A block returns one tensor, but ${exits.size} selected layers feed layers outside. Include the merge layer too.`); return; }
  const nodes = g.nodes.filter(n => S.has(n.id));
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const gin = mk('gin', minx - 240, miny, {}, 'in'), gout = mk('gout', maxx + 240, miny, {}, 'out');
  const inner = {nodes:[gin, ...nodes, gout], edges:g.edges.filter(e => S.has(e.from) && S.has(e.to))};
  for (const e of inE) inner.edges.push(edge(gin.id, e.to, e.port));
  if (!inE.length){
    const free = [];
    for (const n of nodes) if (n.type !== 'data') for (let i = 0; i < DEFS[n.type].ins; i++) if (!portExpr(n, i) && !inner.edges.some(e => e.to === n.id && e.port === i)) free.push([n.id, i]);   // a Data component's input is optional
    if (free.length === 1) inner.edges.push(edge(gin.id, free[0][0], free[0][1]));
  }
  let exit = [...exits][0];
  if (!exit){ const sinks = nodes.filter(n => DEFS[n.type].outs && !inner.edges.some(e => e.from === n.id)); if (sinks.length === 1) exit = sinks[0].id; }
  if (exit) inner.edges.push(edge(exit, gout.id, 0));
  const grp = {id:uid(), type:'group', x:Math.round((minx+maxx)/2), y:Math.round((miny+maxy)/2), name:nextBlockName('Block'), props:{}, graph:inner};
  g.nodes = g.nodes.filter(n => !S.has(n.id)); g.nodes.push(grp);
  g.edges = g.edges.filter(e => !S.has(e.from) && !S.has(e.to));
  if (inE.length) g.edges.push(edge([...srcs][0], grp.id, 0));
  for (const e of outE) g.edges.push(edge(grp.id, e.to, e.port));
  sel = new Set([grp.id]); selWire = null; render();
  toast('Grouped into a block. Rename it on the right, or double-click to open it.');
}
function ungroupRaw(id){
  const g = cur(), grp = byId(g, id);
  if (!grp || grp.type !== 'group'){ toast('Select a block to ungroup.'); return; }
  const inner = grp.graph;
  const gin = inner.nodes.find(n => n.type === 'gin'), gout = inner.nodes.find(n => n.type === 'gout');
  const body = inner.nodes.filter(n => n !== gin && n !== gout);
  if (!body.length){ g.nodes = g.nodes.filter(n => n !== grp); g.edges = g.edges.filter(e => e.from !== grp.id && e.to !== grp.id); render(); return; }
  const cx = body.reduce((s,n) => s+n.x, 0)/body.length, cy = body.reduce((s,n) => s+n.y, 0)/body.length;
  const outerNames = new Set(); for (const n of g.nodes) if (n !== grp) outerNames.add(n.name);
  for (const n of body){
    n.x = Math.round(n.x - cx + grp.x); n.y = Math.round(n.y - cy + grp.y);
    if (outerNames.has(n.name)) n.name = n.type === 'group' ? nextBlockName(n.name) : nextName(n.type);
    outerNames.add(n.name);
  }
  const extIn = g.edges.find(e => e.to === grp.id), extOut = g.edges.filter(e => e.from === grp.id);
  const toGout = gout && inner.edges.find(e => e.to === gout.id);
  const newE = inner.edges.filter(e => e.from !== gin?.id && e.to !== gout?.id);
  if (extIn) for (const e of inner.edges) if (gin && e.from === gin.id && e.to !== gout?.id) newE.push(edge(extIn.from, e.to, e.port));
  const srcOut = toGout ? (toGout.from === gin?.id ? extIn?.from : toGout.from) : null;
  if (srcOut) for (const e of extOut) newE.push(edge(srcOut, e.to, e.port));
  g.nodes = g.nodes.filter(n => n !== grp).concat(body);
  g.edges = g.edges.filter(e => e.from !== grp.id && e.to !== grp.id).concat(newE);
  sel = new Set(body.map(n => n.id)); render();
}
/* ---------- copy & paste ---------- */
let clip = null, lastPtr = null;
function copySel(quiet){
  const g = cur(), ids = new Set([...sel].filter(id => { const n = byId(g, id); return n && n.type !== 'gin' && n.type !== 'gout'; }));
  if (!ids.size){ if (!quiet) toast('Select components to copy.'); return false; }
  clip = {nodes:g.nodes.filter(n => ids.has(n.id)).map(strip), edges:g.edges.filter(e => ids.has(e.from) && ids.has(e.to)).map(e => ({...e})), pastes:0};
  try { localStorage.setItem('netgraph.clip', JSON.stringify(clip)); } catch(e){}
  if (!quiet) toast(`Copied ${ids.size} component${ids.size > 1 ? 's' : ''}.`);
  return true;
}
// Gives a copied node (and everything inside a block) fresh ids, recording old → new in map.
function freshIds(n, map){
  map[n.id] = uid(); n.id = map[n.id];
  if (n.graph){ n.graph.nodes.forEach(m => freshIds(m, map)); n.graph.edges = n.graph.edges.map(e => ({id:uid(), from:map[e.from], to:map[e.to], port:e.port})); }
}
function pasteClip(){
  if (!clip) try { clip = JSON.parse(localStorage.getItem('netgraph.clip')); } catch(e){}
  if (!clip || !clip.nodes || !clip.nodes.length){ toast(`Nothing to paste. Select components and press ${isMac ? '⌘C' : 'Ctrl+C'} first.`); return; }
  const g = cur(), keep = clip.nodes;
  const map = {}, copies = keep.map(n => { const c = JSON.parse(JSON.stringify(n)); freshIds(c, map); return c; });
  // Pasted layers keep their trained weights.
  for (const [o, nw] of Object.entries(map)){ const P = wcache.get(o); if (P) wcache.set(nw, {key:P.key, W:P.W.slice(), B:P.B.slice(), trained:P.trained}); }
  const x0 = Math.min(...copies.map(c => c.x)), y0 = Math.min(...copies.map(c => c.y)), x1 = Math.max(...copies.map(c => c.x + (c.w || W))), y1 = Math.max(...copies.map(c => c.y + 110));
  let dx, dy;
  if (lastPtr){ const p = toWorld(lastPtr.x, lastPtr.y); dx = Math.round(p.x - (x0 + x1)/2); dy = Math.round(p.y - (y0 + y1)/2); }
  else { clip.pastes++; dx = dy = 40*clip.pastes; }
  for (const c of copies){
    c.x += dx; c.y += dy;
    if (allNames().has(c.name)) c.name = c.type === 'group' ? nextBlockName(c.name) : nextName(c.type);
    g.nodes.push(c);
  }
  for (const e of clip.edges) if (map[e.from] && map[e.to]){ const ne = edge(map[e.from], map[e.to], e.port); if (e.bend) ne.bend = {...e.bend}; g.edges.push(ne); }
  sel = new Set(copies.map(c => c.id)); selWire = null;
  render();
  if (keep.length < clip.nodes.length) toast('Input, Output, Data and Loss stay in the top-level network, so they weren\'t pasted here.');
}
function saveToLib(id){
  const n = byId(cur(), id); if (!n || n.type !== 'group') return;
  const label = n.name.replace(/\s+\d+$/, '');
  lib.push({id:uid(), label, node:strip(n)});
  renderPalette(); save(); toast(`Saved “${label}” to My blocks.`);
}

/* ---------- example ---------- */
function example(){
  root = {nodes:[], edges:[]}; path = []; nview = null; sel.clear(); selWire = null;
  const put = (n, x) => { n.x = x; n.y = 40; root.nodes.push(n); return n; };
  const inp = put(mk('data', 0, 0, {dataset:'mnist'}), 0);
  const cb = c => blockFrom('Conv block', [['conv2d',{out_channels:c}],['batchnorm'],['relu'],['maxpool']]);
  const b1 = cb(16); b1.name = nextBlockName('Conv block'); put(b1, 250);
  const b2 = cb(32); b2.name = nextBlockName('Conv block'); put(b2, 500);
  const fl = put(mk('flatten'), 750);
  const mlp = BUILTIN[0].make(); mlp.name = nextBlockName('MLP'); put(mlp, 1000);
  const fc = put(mk('dense', 0, 0, {out_features:10}), 1250);
  const sm = put(mk('softmax'), 1500);
  const ls = mk('loss', 1500, 300, {expr:`ce(${fc.name})`}); root.nodes.push(ls);
  const out = put(mk('output'), 1750);
  const chain = [inp, b1, b2, fl, mlp, fc, sm, out];
  for (let i = 0; i < chain.length - 1; i++) root.edges.push(edge(chain[i].id, chain[i+1].id));
}

// Grouping and ungrouping change components' dotted names (fc1 becomes Block.fc1 and back); expressions follow.
function groupSelection(){ const before = pathMap(); groupSelectionRaw(); if (rewriteRefs(before)) render(); }
function ungroup(id){ const before = pathMap(); ungroupRaw(id); if (rewriteRefs(before)) render(); }
