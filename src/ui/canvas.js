
/* ---------- rendering ---------- */
const cv = $('#cv'), world = $('#world'), nodesEl = $('#nodes'), wg = $('#wg'), insp = $('#insp');
function applyView(){
  world.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.k})`;
  cv.style.backgroundSize = `${22*view.k}px ${22*view.k}px`;
  cv.style.backgroundPosition = `${view.x}px ${view.y}px`;
}
const isNet = n => n.type === 'dense' || (n.type === 'group' && mlpLike(n.graph));
function denseNodes(g, out=[]){ for (const m of topo(g)){ if (m.type === 'dense') out.push([g, m]); else if (m.type === 'group') denseNodes(m.graph, out); } return out; }
function postAct(g, m){ let c = m; for (;;){ const o = g.edges.filter(e => e.from === c.id), nx = o.length === 1 && byId(g, o[0].to); if (!nx || !ACT.has(nx.type) || !nx._act) return c._act; c = nx; } }
function miniNet(n){
  const vec = t => t && t.shape.length === 1 ? t.data : null;
  let cols = [{n:lastDim(n._in && n._in[0]), v:vec(n._actIn && n._actIn[0])},
    ...(n.type === 'dense' ? [{n:lastDim(n._out), v:vec(n._act)}] : denseNodes(n.graph).map(([g, m]) => ({n:lastDim(m._out), v:vec(postAct(g, m))})))];
  if (cols.length > 5) cols = [cols[0], cols[1], cols[cols.length >> 1], cols[cols.length-2], cols[cols.length-1]];
  const x0 = 28, x1 = 172, cy = PY0, step = (x1 - x0)/(cols.length - 1);
  const C = cols.map((c, i) => {
    const x = x0 + i*step;
    if (c.n && c.n < 10){ const sp = Math.min(9, 76/c.n), idx = [...Array(c.n).keys()]; return {...c, x, idx, ys:idx.map(k => cy + (k - (c.n-1)/2)*sp)}; }
    return {...c, x, idx: c.n ? [0, 1, c.n-2, c.n-1] : [], ys:[cy-30, cy-18, cy+18, cy+30], more:true};
  });
  let l = '', d = '';
  for (let i = 1; i < C.length; i++) for (const ya of C[i-1].ys) for (const yb of C[i].ys) l += `M${C[i-1].x},${ya}L${C[i].x},${yb}`;
  for (const c of C){
    const m = c.v ? amax(c.v) : 1;
    c.ys.forEach((y, k) => {
      const j = c.idx[k], a = c.v && j != null && j < c.v.length ? c.v[j]/m : null;
      d += `<circle class="mn${c.more?' g':''}" cx="${c.x}" cy="${y}" r="3.6"${a != null ? ` style="fill:${divCSS(a)}"` : ''}/>`;
    });
    if (c.more) d += `<text class="md" x="${c.x}" y="${cy+4}">⋮</text>`;
  }
  return `<svg class="mini" viewBox="0 0 200 92" aria-hidden="true"><g><path class="ml" d="${l}"/>${d}</g></svg>`;
}
function nodeHTML(n){
  const d = DEFS[n.type], r = n._r || {}, isG = n.type === 'group';
  const title = isG ? esc(n.name) : labelOf(n);
  const sub = isG ? 'block' : (n.type==='gin' || n.type==='gout') ? '' : esc(n.name);
  const ins = (n._in || []).map(fmt);
  let shape;
  if (d.ins === 0) shape = `<span class="k">out</span> ${fmt(n._out)}`;
  else if (d.outs === 0) shape = `<span class="k">in</span> ${ins.join(', ')}`;
  if (n.type === 'loss') shape = `<span class="k">=</span> ${esc(n.props.expr || '…')}`;
  else if (n.type === 'ptrin') shape = `${ins.join('')} <span class="k">↪</span> ${esc(String(n.props.target || '…').trim())}`;
  else if (n.type === 'ptrout') shape = `<span class="k">↩</span> ${esc(String(n.props.pair || '…').trim())} <span class="ar">→</span> ${fmt(n._out)}`;
  else if (n.type === 'data' && !r.err && dsById(n.props.dataset) && (!n._collect || dsById(n.props.dataset).x)){ const ds = dsById(n.props.dataset); shape = `<span class="k">holds</span> ${ds.n.toLocaleString()} × ${fmt(dsShape(ds))}`; }   // what it holds: how many, and each one's shape
  else if (hasExpr(n)) shape = `<span class="k">←</span> ${esc(exprsOf(n).join(', '))} <span class="ar">→</span> ${fmt(n._out)}`;   // inputs given by expressions
  else if (d.outs === 0) shape = `<span class="k">in</span> ${ins.join(', ')}`;
  else shape = `${ins.join(' + ')} <span class="ar">→</span> ${fmt(n._out)}`;
  const msg = r.err ? `<div class="msg err">${esc(r.err)}</div>` : r.warn ? `<div class="msg warn">${esc(r.warn)}</div>` : r.wait ? `<div class="msg wait">${esc(r.wait)}</div>` : '';
  let ports = '';
  for (let i = 0; i < d.ins; i++) if (wired(n, i)) ports += `<div class="port in" data-node="${n.id}" data-port="${i}" data-dir="in" style="left:0;top:${PY0+i*PSTEP}px" title="Input${d.ins>1?' '+'ab'[i]:''}">${d.ins>1?`<span class="pl">${'ab'[i]}</span>`:''}</div>`;
  if (d.outs) ports += `<div class="port out" data-node="${n.id}" data-dir="out" style="left:${W}px;top:${PY0}px" title="Output"></div>`;
  let extra = '';
  if (n.type === 'sliders') extra = slidersRows(n);
  else if (PLOTS.has(n.type)) extra = plotHTML(n);
  else if (n.type === 'data'){ const s = !r.err && sampleOf(n); if (s) extra = `<div class="dn${s.pixels ? '' : ' dc'}">${sampleHTML(s)}<div class="dl">#${s.i} · ${splitOf(n).isTest[s.i] ? 'test' : 'train'}<br>label<b>${s.label}</b></div></div>`; }
  else if (n.type === 'loss') extra = lossBody(n);
  else if (n._act) extra = n.type === 'output' && isPred(n._act) ? predHTML(n._act) : n.type === 'output' ? '' : actHTML(n._act);
  const vt = `<button class="vt" data-view="${n.id}" aria-pressed="${!!n.view}" aria-label="${n.view ? 'Show the layer' : 'Show activations'}" title="View activations (V)"></button>`;
  const sz = sizeStyle(n), rz = '<div class="rz" title="Drag to resize · double-click to reset"></div>';
  if (n.view) return `<div class="node view${n.type === 'loss' ? ' lv' : n.type === 'sliders' ? ' sld' : PLOTS.has(n.type) ? ' plt' : ''}${sel.has(n.id)?' sel':''}${r.err?' bad':''}" data-id="${n.id}" style="left:${n.x}px;top:${n.y}px;--fam:${FAM[d.fam]};${sz}">
    <div class="vw">${viewPanels(n)}</div>
    <div class="mcap"><div class="mt"><span class="mk"></span>${title}<small>${sub}</small></div><div class="nshape">${shape}</div>${msg}</div>
    ${n._vars && n._vars.length ? nodeVarsHTML(n) : ''}${n.type === 'sliders' ? slidersRows(n) : ''}${n.type === 'data' && !r.err ? stepperHTML(n) : ''}${n.type === 'loss' ? `<div class="lrow">${lossBody(n, true)}</div>` : ''}${ports}${vt}${rz}</div>`;
  if (isNet(n)) return `<div class="node net${isG?' grp':''}${sel.has(n.id)?' sel':''}${r.err?' bad':''}" data-id="${n.id}" style="left:${n.x}px;top:${n.y}px;--fam:${FAM[d.fam]};${sz}" title="Double-click to see its neurons">
    ${miniNet(n)}<div class="mcap"><div class="mt"><span class="mk"></span>${title}<small>${sub}</small></div><div class="nshape">${shape}</div>${r.params ? `<div class="np">${fmtN(r.params)} params</div>` : ''}${msg}</div>
    ${n._vars && n._vars.length ? nodeVarsHTML(n) : ''}${ports}${vt}${rz}</div>`;
  return `<div class="node${isG?' grp':''}${n.type === 'sliders' ? ' sld' : PLOTS.has(n.type) ? ' plt' : ''}${sel.has(n.id)?' sel':''}${r.err?' bad':''}" data-id="${n.id}" style="left:${n.x}px;top:${n.y}px;--fam:${FAM[d.fam]};min-height:${Math.max(86, PY0+(d.ins-1)*PSTEP+26)}px;${sz}">
    <div class="nh"><span class="nt">${title}</span><span class="nn">${sub}</span></div>
    <div class="nb">${r.sum ? `<div class="ns">${esc(r.sum)}</div>` : ''}<div class="nshape">${shape}</div>${r.params ? `<div class="np">${fmtN(r.params)} params</div>` : ''}${extra}${msg}</div>
    ${n._vars && n._vars.length ? nodeVarsHTML(n) : ''}${ports}${vt}${rz}</div>`;
}
// A resized component: its width, and the size of its view panels so they grow with it.
const psOf = n => Math.floor(Math.min(n.w - 24, 600));
const sizeStyle = n => n.w ? `width:${n.w}px;--ps:${psOf(n)}px;` : '';
function viewPanels(n){
  const d = DEFS[n.type];
  // Only what a component puts out; the Output component, which has nothing to put out, shows what it receives.
  if (PLOTS.has(n.type)) return `<div class="pn plp">${plotHTML(n)}</div>`;
  if (n.type === 'data'){   // the sample itself, in grayscale, not activation colors
    const s = !(n._r && n._r.err) && sampleOf(n);
    if (s && !s.pixels) return `<div class="pn"><span class="pl">out</span>${pic(tensorURL(s.t), s.t.shape.length === 3 ? '' : 'strip')}</div>`;   // collected values, never read as class scores
    return `<div class="pn${s ? '' : ' empty'}"><span class="pl">out</span>${s ? pic(sampleURL(s), '', `A handwritten ${s.label}`) : ''}</div>`;
  }
  return n.type === 'loss' ? chartHTML(n) : d.outs ? panel(n._act, 'out') : panel((n._actIn || [])[0], 'in');
}
// A Sliders component's sliders, on the component itself.
function slidersRows(n){
  return `<div class="nvars sls">${(n.props.sliders || []).map((s, i) => `<div class="nvar"><b title="${esc(n.name)}[${i}]">${esc(s.name)}</b><button class="tb" data-slstep="${i}" data-node="${n.id}" data-d="-1" aria-label="Decrease ${esc(s.name)}">‹</button><input type="range" data-sl="${i}" data-node="${n.id}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}" aria-label="${esc(s.name)}"><button class="tb" data-slstep="${i}" data-node="${n.id}" data-d="1" aria-label="Increase ${esc(s.name)}">›</button><output>${fmtV(s.value)}</output></div>`).join('')}</div>`;
}
// Variable sliders on the component itself.
function nodeVarsHTML(n){
  if (!n._vars || !n._vars.length) return '';
  return `<div class="nvars">${n._vars.map(name => { const v = n.props.vars[name], nm = esc(name);
    return `<div class="nvar"><b>${nm}</b><button class="tb" data-nvstep="${nm}" data-node="${n.id}" data-d="-1" aria-label="Decrease ${nm}">‹</button><input type="range" data-nvar="${nm}" data-node="${n.id}" min="${v.min}" max="${v.max}" step="${v.step}" value="${v.value}" aria-label="${nm}"><button class="tb" data-nvstep="${nm}" data-node="${n.id}" data-d="1" aria-label="Increase ${nm}">›</button><output>${fmtV(v.value)}</output></div>`;
  }).join('')}</div>`;
}
function renderNodes(){ nodesEl.innerHTML = cur().nodes.map(nodeHTML).join(''); paintViz(nodesEl); layoutPorts(); }
/* Ports slide around their component's border to face what they connect to: an output faces the
   average direction of its targets, an input faces its source. Wires leave each port at a right angle,
   so facing ports give short, clean curves. Unconnected ports keep the usual left/right spots. */
let anch = new Map(), boxes = new Map();
   // "id:out" / "id:in0" → {x, y, nx, ny}; id → box, in world coordinates
function perim(b, dx, dy){   // where a ray from the box centre in direction (dx, dy) leaves the box, kept off the corners
  const cx = b.x + b.w/2, cy = b.y + b.h/2;
  if (!dx && !dy) dx = 1;
  const sx = dx ? (b.w/2)/Math.abs(dx) : Infinity, sy = dy ? (b.h/2)/Math.abs(dy) : Infinity;
  if (sx <= sy) return {x:cx + Math.sign(dx)*b.w/2, y:Math.max(b.y + 12, Math.min(b.y + b.h - 12, cy + dy*sx)), nx:Math.sign(dx), ny:0};
  return {x:Math.max(b.x + 14, Math.min(b.x + b.w - 14, cx + dx*sy)), y:cy + Math.sign(dy)*b.h/2, nx:0, ny:Math.sign(dy)};
}
const toward = (b, o) => perim(b, (o.x + o.w/2) - (b.x + b.w/2), (o.y + o.h/2) - (b.y + b.h/2));
function layoutPorts(){
  const g = cur(), els = new Map(); boxes = new Map(); anch = new Map();
  for (const n of g.nodes){ const el = nodesEl.querySelector(`.node[data-id="${n.id}"]`); els.set(n.id, el); boxes.set(n.id, {x:n.x, y:n.y, w:el && el.offsetWidth || W, h:el && el.offsetHeight || 120}); }
  const dirs = new Map(), pull = (k, from, to) => {
    const a = boxes.get(from), b = boxes.get(to), dx = (b.x + b.w/2) - (a.x + a.w/2), dy = (b.y + b.h/2) - (a.y + a.h/2), l = Math.hypot(dx, dy) || 1, v = dirs.get(k) || [0, 0];
    dirs.set(k, [v[0] + dx/l, v[1] + dy/l]);
  };
  for (const e of g.edges) if (boxes.has(e.from) && boxes.has(e.to)){ pull(`${e.from}:out`, e.from, e.to); pull(`${e.to}:in${e.port}`, e.to, e.from); }
  for (const n of g.nodes){ const t = n.type === 'ptrin' && ptrTarget(n, g); if (t && boxes.has(t.id)) pull(`${n.id}:out`, n.id, t.id); }   // a Pointer faces what it points to
  for (const n of g.nodes){
    const b = boxes.get(n.id), d = DEFS[n.type], list = [];
    const at = (k, fallback) => { const v = dirs.get(k); return v && (Math.abs(v[0]) > 1e-6 || Math.abs(v[1]) > 1e-6) ? perim(b, v[0], v[1]) : fallback; };
    for (let i = 0; i < d.ins; i++) if (wired(n, i)) list.push([`${n.id}:in${i}`, at(`${n.id}:in${i}`, {x:b.x, y:b.y + PY0 + i*PSTEP, nx:-1, ny:0})]);
    if (d.outs) list.push([`${n.id}:out`, at(`${n.id}:out`, {x:b.x + b.w, y:b.y + PY0, nx:1, ny:0})]);
    for (let i = 0; i < list.length; i++) for (let j = 0; j < i; j++){   // ports on one component never overlap
      const p = list[i][1], q = list[j][1];
      if (Math.hypot(p.x - q.x, p.y - q.y) >= 20) continue;
      if (p.nx){ p.y = q.y + 22 > b.y + b.h - 10 ? q.y - 22 : q.y + 22; } else { p.x = q.x + 22 > b.x + b.w - 10 ? q.x - 22 : q.x + 22; }
    }
    const el = els.get(n.id);
    for (const [k, a] of list){
      anch.set(k, a);
      const pe = el && el.querySelector(`:scope > .port[data-dir="${k.endsWith(':out') ? 'out' : 'in'}"]${k.endsWith(':out') ? '' : `[data-port="${k.split(':in')[1]}"]`}`);
      if (pe){ pe.style.left = (a.x - b.x) + 'px'; pe.style.top = (a.y - b.y) + 'px'; }
    }
  }
}
// A wire between two anchors. Without a bend it is one cubic leaving and entering at right angles to the
// border; with a bend it is two cubics joined smoothly at the bend point.
function wireGeom(p, q, bend){
  const lead = d => Math.max(20, Math.min(140, d*0.4));
  if (bend){
    const bx = (p.x + q.x)/2 + bend.x, by = (p.y + q.y)/2 + bend.y;
    let tx = q.x - p.x, ty = q.y - p.y; const tl = Math.hypot(tx, ty) || 1; tx /= tl; ty /= tl;
    const d1 = Math.hypot(bx - p.x, by - p.y), d2 = Math.hypot(q.x - bx, q.y - by), k1 = lead(d1), k2 = lead(d2), l1 = d1/3, l2 = d2/3;
    return {d:`M${p.x},${p.y} C${p.x + p.nx*k1},${p.y + p.ny*k1} ${bx - tx*l1},${by - ty*l1} ${bx},${by} C${bx + tx*l2},${by + ty*l2} ${q.x + q.nx*k2},${q.y + q.ny*k2} ${q.x},${q.y}`, mx:bx, my:by};
  }
  const k = lead(Math.hypot(q.x - p.x, q.y - p.y)), c1x = p.x + p.nx*k, c1y = p.y + p.ny*k, c2x = q.x + q.nx*k, c2y = q.y + q.ny*k;
  return {d:`M${p.x},${p.y} C${c1x},${c1y} ${c2x},${c2y} ${q.x},${q.y}`, mx:(p.x + 3*c1x + 3*c2x + q.x)/8, my:(p.y + 3*c1y + 3*c2y + q.y)/8};
}
function renderWires(){
  layoutPorts();
  const g = cur(); let h = '';
  for (const e of g.edges){
    const a = byId(g, e.from), b = byId(g, e.to), p = anch.get(`${e.from}:out`), q = anch.get(`${e.to}:in${e.port}`);
    if (!a || !b || !p || !q) continue;
    const w = wireGeom(p, q, e.bend), bad = b._r && b._r.err && !(b._r.err.startsWith('Connect'));
    h += `<path class="w${bad?' bad':''}${selWire===e.id?' on':''}" d="${w.d}"/><path class="whit" data-edge="${e.id}" d="${w.d}"/>`;
    if (a._out && Math.hypot(q.x - p.x, q.y - p.y) > 70) h += `<text class="wl" x="${w.mx}" y="${w.my - 8}">${fmt(a._out)}</text>`;
    if (selWire === e.id) h += `<circle class="wh" cx="${w.mx}" cy="${w.my}" r="5" data-bend="${e.id}"><title>Drag to bend; double-click to straighten</title></circle>`;
  }
  for (const n of g.nodes) for (const src of exprsOf(n)){   // dashed: the components an expression reads, in this graph
    let ast; try { ast = parseLoss(src); } catch(e){ continue; }
    for (const a of new Set(ast.refs.map(key => holderIn(g, lookup(key, g).node)))){
      const ba = a && boxes.get(a.id), bl = boxes.get(n.id);
      if (!a || a === n || !ba || !bl) continue;
      h += `<path class="w ref" d="${wireGeom(toward(ba, bl), toward(bl, ba)).d}"/>`;
    }
  }
  for (const n of g.nodes) if (n.type === 'ptrin'){   // a Pointer in's link: a wire into the input of the layer it feeds, beside that input's own wire
    const t = ptrTarget(n, g), p = t && anch.get(`${n.id}:out`), q = t && anch.get(`${t.id}:in0`); if (!p || !q) continue;
    const w = wireGeom(p, q), id = 'ptr:' + n.id;
    h += `<path class="w pw${selWire === id ? ' on' : ''}" d="${w.d}"/><path class="whit" data-edge="${id}" d="${w.d}"/>`;
  }
  if (tempWire) h += `<path class="w tmp" d="${wireGeom(tempWire.p, tempWire.q).d}"/>`;
  wg.innerHTML = h;
}
// Dragging a wire bends it through the pointer; the bend is kept relative to the wire's midpoint.
// The layer a Pointer in feeds, when it sits beside it (its link is drawn as a wire into that layer's input).
function ptrTarget(n, g){ const r = ptrRef(n.props.target, g); return r && !r.err && r.graph === g && r.node !== n && DEFS[r.node.type].ins ? r.node : null; }
function startBend(e, id){
  if (id.startsWith('ptr:')){ selWire = id; sel.clear(); renderNodes(); renderWires(); renderInspector(); return; }   // a Pointer's link: select it
  const g = cur(), ed = g.edges.find(x => x.id === id); if (!ed) return;
  if (selWire !== id || sel.size){ selWire = id; sel.clear(); renderNodes(); renderWires(); renderInspector(); }
  const sx = e.clientX, sy = e.clientY; let moved = false;
  track(m => {
    if (!moved && Math.hypot(m.clientX - sx, m.clientY - sy) < 4) return;
    moved = true;
    const p = anch.get(`${ed.from}:out`), q = anch.get(`${ed.to}:in${ed.port}`), w = toWorld(m.clientX, m.clientY);
    if (p && q){ ed.bend = {x:Math.round(w.x - (p.x + q.x)/2), y:Math.round(w.y - (p.y + q.y)/2)}; renderWires(); }
  }, () => { if (moved){ save(); renderInspector(); } });
}
function renderCrumbs(){
  const lay = nview && nview.id ? byId(cur(), nview.id) : null;
  let g = root, h = `<button data-depth="0"${path.length||lay?'':' aria-current="page"'}>Network</button>`;
  path.forEach((id, i) => { const n = byId(g, id); g = n.graph; h += `<span class="sep">›</span><button data-depth="${i+1}"${i===path.length-1&&!lay?' aria-current="page"':''}>${esc(n.name)}</button>`; });
  if (lay) h += `<span class="sep">›</span><button aria-current="page">${esc(lay.name)}</button>`;
  else if (neuronModel(cur())) h += `<span class="seg" role="group" aria-label="View"><button data-mode="layers" aria-pressed="${!nview}">Layers</button><button data-mode="neurons" aria-pressed="${!!nview}">Neurons</button></span>`;
  $('#crumbs').innerHTML = h;
}
function renderPalette(){
  let h = '';
  for (const [title, types] of PALETTE){
    h += `<h3>${title}</h3>`;
    for (const t of types){
      const D = t === 'pointer' ? {label:'Pointer pair', fam:'ptr'} : DEFS[t];
      h += `<div class="pi" data-kind="layer:${t}" style="--fam:${FAM[D.fam]}" title="Drag onto the canvas, or click to add"><span class="sw"></span>${D.label}</div>`;
    }
  }
  h += `<h3>Blocks</h3>`;
  BUILTIN.forEach((b,i) => h += `<div class="pi blk" data-kind="block:${i}" style="--fam:${FAM[b.fam]}"><span class="sw"></span>${b.label}</div>`);
  h += `<h3>My blocks</h3>`;
  if (!lib.length) h += `<p class="empty">Select a block and choose “Save to my blocks” to reuse it here.</p>`;
  lib.forEach(l => h += `<div class="pi blk" data-kind="lib:${l.id}" style="--fam:${FAM.block}"><span class="sw"></span>${esc(l.label)}<button class="x" data-dellib="${l.id}" aria-label="Remove ${esc(l.label)}">×</button></div>`);
  $('#pal').innerHTML = h;
}
