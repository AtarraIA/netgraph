
/* ---------- canvas interaction ---------- */
function track(onMove, onUp){
  const mv = e => onMove(e), up = e => { window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up); onUp(e); };
  window.addEventListener('pointermove', mv); window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
}
let lastTap = {id:null, t:0};
cv.addEventListener('pointerdown', e => {
  if (e.target.closest('.zoom,.crumbs,.insp')) return;
  const orb = orbitTarget(e); if (orb && e.button === 0) return startOrbit(e, orb);   // a 3D plot in view mode turns
  if (e.target.closest('.nvars input, .nvars button')){   // sliders on a component work like normal sliders; the rest of the row is the component
    const b = e.target.closest('[data-nvstep]'); if (b){ e.preventDefault(); stepVar(byId(cur(), b.dataset.node), b.dataset.nvstep, +b.dataset.d); }
    const sb = e.target.closest('[data-slstep]'); if (sb){ e.preventDefault(); stepSlider(byId(cur(), sb.dataset.node), +sb.dataset.slstep, +sb.dataset.d); }
    return;
  }
  const grip = e.target.closest('.rz');
  if (grip){ e.preventDefault(); return startResize(e, grip.closest('.node').dataset.id); }
  if (e.button !== 0 && e.pointerType === 'mouse') { if (e.button === 1) startPan(e); return; }
  const port = e.target.closest('.port'), node = e.target.closest('.node'), wire = e.target.closest('.whit');
  e.preventDefault();
  if (e.target.closest('.nidx')) return;   // let the sample field take focus
  // A loss chart pans only once its Loss block is selected and has a history; otherwise a press is on the component.
  const lch = chartTarget(e);
  if (lch){ e.preventDefault(); startChartPan(e, lch); return; }
  const stp = e.target.closest('[data-step]');
  if (stp){ const n = byId(cur(), stp.dataset.node); if (n) stepSample(n, stp.dataset.step); return; }
  const trb = e.target.closest('[data-train]');
  if (trb){ if (!trb.disabled) toggleTrain(trb.dataset.train); return; }
  const vt = e.target.closest('.vt');
  if (vt){ toggleView([vt.dataset.view]); return; }
  if (port) return startWire(e, port);
  if (node) return startNodeDrag(e, node);
  const bh = e.target.closest('[data-bend]');
  if (bh || wire) return startBend(e, bh ? bh.dataset.bend : wire.dataset.edge);
  if (e.shiftKey || e.ctrlKey || e.metaKey || boxMode) startBox(e); else startPan(e, true);   // Shift-, Ctrl- or ⌘-drag draws a selection box
});
function startPan(e, clearOnTap){
  const sx = e.clientX, sy = e.clientY, vx = view.x, vy = view.y; let moved = false;
  cv.style.cursor = 'grabbing';
  track(m => { const dx = m.clientX - sx, dy = m.clientY - sy; if (Math.abs(dx)+Math.abs(dy) > 3) moved = true; view.x = vx + dx; view.y = vy + dy; applyView(); },
    () => { cv.style.cursor = ''; if (!moved && clearOnTap && (sel.size || selWire)){ sel.clear(); selWire = null; renderNodes(); renderWires(); renderInspector(); } });
}
function startBox(e){
  const r = cv.getBoundingClientRect(), box = $('#box'), sx = e.clientX, sy = e.clientY, add = e.shiftKey || e.ctrlKey || e.metaKey;
  const base = add ? new Set(sel) : new Set();
  box.hidden = false;
  const upd = m => {
    const x0 = Math.min(sx, m.clientX), y0 = Math.min(sy, m.clientY), x1 = Math.max(sx, m.clientX), y1 = Math.max(sy, m.clientY);
    Object.assign(box.style, {left:(x0-r.left)+'px', top:(y0-r.top)+'px', width:(x1-x0)+'px', height:(y1-y0)+'px'});
    const a = toWorld(x0, y0), b = toWorld(x1, y1);
    sel = new Set(base);
    for (const n of cur().nodes){ const el = nodesEl.querySelector(`[data-id="${n.id}"]`); const h = el ? el.offsetHeight : 90, w = el ? el.offsetWidth : W; if (n.x < b.x && n.x+w > a.x && n.y < b.y && n.y+h > a.y) sel.add(n.id); }
    nodesEl.querySelectorAll('.node').forEach(el => el.classList.toggle('sel', sel.has(el.dataset.id)));
  };
  upd(e);
  track(upd, () => { box.hidden = true; selWire = null; renderInspector(); });
}
// Dragging the bottom-right grip resizes a component; its left edge stays put.
function startResize(e, id){
  const n = byId(cur(), id), el = nodesEl.querySelector(`.node[data-id="${id}"]`); if (!n || !el) return;
  const w0 = el.offsetWidth, sx = e.clientX, sy = e.clientY;
  track(m => {
    const grow = Math.max(m.clientX - sx, m.clientY - sy)/view.k;   // out to the right or down both make it bigger
    const w = Math.round(Math.max(150, Math.min(1000, w0 + grow)));
    n.w = w;
    el.style.width = w + 'px'; el.style.setProperty('--ps', psOf(n) + 'px');
    renderWires();
  }, () => { renderNodes(); renderWires(); save(); });
}
function startNodeDrag(e, el){
  const g = cur(), id = el.dataset.id, n = byId(g, id); if (!n) return;
  const now = performance.now();
  const multi = e.shiftKey || e.ctrlKey || e.metaKey;   // Shift-, Ctrl- or ⌘-click adds to or removes from the selection
  if (!multi && lastTap.id === id && now - lastTap.t < 380 && (n.type === 'group' || n.type === 'dense')){ lastTap = {id:null, t:0}; enter(id); return; }
  lastTap = {id, t:now};
  let changed = false;
  if (multi){ sel.has(id) ? sel.delete(id) : sel.add(id); changed = true; }
  else if (!sel.has(id)){ sel = new Set([id]); changed = true; }
  if (selWire){ selWire = null; changed = true; }
  if (changed){ renderNodes(); renderWires(); renderInspector(); }
  const start = toWorld(e.clientX, e.clientY);
  const moving = [...sel].map(i => byId(g, i)).filter(Boolean).map(m => ({m, x:m.x, y:m.y}));
  let moved = false;
  track(m => {
    const p = toWorld(m.clientX, m.clientY), dx = p.x - start.x, dy = p.y - start.y;
    if (!moved && Math.abs(dx)+Math.abs(dy) < 3/view.k) return;
    moved = true;
    for (const o of moving){ o.m.x = Math.round(o.x + dx); o.m.y = Math.round(o.y + dy); const ne = nodesEl.querySelector(`[data-id="${o.m.id}"]`); if (ne){ ne.style.left = o.m.x+'px'; ne.style.top = o.m.y+'px'; } }
    renderWires();
  }, () => {
    if (moved) save();
    else if (!multi && sel.size > 1){ sel = new Set([id]); renderNodes(); renderInspector(); }
  });
}
function startWire(e, port){
  const g = cur(), nid = port.dataset.node, dir = port.dataset.dir, pi = +port.dataset.port || 0, n = byId(g, nid);
  let fromId = null, toEnd = null;
  if (dir === 'out') fromId = nid;
  else {
    const ex = g.edges.find(x => x.to === nid && x.port === pi);
    if (ex){ fromId = ex.from; g.edges = g.edges.filter(x => x !== ex); soft(); }
    else toEnd = {id:nid, port:pi};
  }
  const A = anch.get(fromId ? `${fromId}:out` : `${nid}:in${pi}`) || (fromId ? (() => { const a = byId(g, fromId); return {x:a.x+W, y:a.y+PY0, nx:1, ny:0}; })() : {x:n.x, y:n.y+PY0+pi*PSTEP, nx:-1, ny:0});
  const upd = m => { const p = toWorld(m.clientX, m.clientY), c = {x:p.x, y:p.y, nx:-A.nx, ny:-A.ny}; tempWire = fromId ? {p:A, q:c} : {p:c, q:A}; renderWires(); };
  upd(e);
  track(upd, m => {
    tempWire = null;
    const t = document.elementFromPoint(m.clientX, m.clientY);
    const tp = t && t.closest('.port');
    if (tp){
      if (fromId && tp.dataset.dir === 'in'){ if (!pointAt(fromId, tp.dataset.node)) connect(fromId, tp.dataset.node, +tp.dataset.port); }
      else if (toEnd && tp.dataset.dir === 'out') connect(tp.dataset.node, toEnd.id, toEnd.port);
    }
    soft();
  });
}
/* Zooming past the limit dives into the layer under the cursor, or surfaces to the parent level. */
let dive = {acc:0, el:null, t:0};
function diveReset(){
  clearTimeout(dive.t);
  if (dive.el){ dive.el.classList.remove('dive'); dive.el.style.removeProperty('--d'); }
  $('#crumbs').classList.remove('surf');
  dive = {acc:0, el:null, t:0};
}
cv.addEventListener('pointermove', e => { lastPtr = {x:e.clientX, y:e.clientY}; });
cv.addEventListener('contextmenu', e => { if (e.ctrlKey) e.preventDefault(); });
   // Ctrl-drag on a Mac shouldn't open the menu
cv.addEventListener('pointerleave', () => { lastPtr = null; });
function chartTarget(e){ const el = e.target.closest('.lchart'); return el && sel.size === 1 && sel.has(el.dataset.chart) && chartSpan(el.dataset.chart) ? el : null; }
cv.addEventListener('wheel', e => {
  const orb = orbitTarget(e); if (orb) return zoomOrbit(e, orb);
  if (e.target.closest('.insp')) return;
  const lch = chartTarget(e);
  if (lch){ e.preventDefault(); chartZoom(e, lch); return; }
  e.preventDefault();
  const f = -e.deltaY * (e.ctrlKey ? 0.01 : 0.0015);
  const inward = f > 0 && view.k >= KMAX - 1e-6, outward = f < 0 && view.k <= KMIN + 1e-6;
  if (inward || outward){
    let target = null;
    if (inward){
      if (nview) return;
      const el = document.elementFromPoint(e.clientX, e.clientY)?.closest('.node'), n = el && byId(cur(), el.dataset.id);
      if (!n || (n.type !== 'group' && n.type !== 'dense')){ diveReset(); return; }
      target = el;
    } else if (!path.length && !nview) return;
    if (dive.el !== target || Math.sign(dive.acc) !== Math.sign(f)){ diveReset(); dive.el = target; }
    clearTimeout(dive.t);
    dive.acc += f;
    const d = Math.min(1, Math.abs(dive.acc) / 0.6), mark = target || $('#crumbs');
    mark.classList.add(target ? 'dive' : 'surf'); mark.style.setProperty('--d', d);
    if (d >= 1){ const id = target && target.dataset.id; diveReset(); if (id) enter(id); else goUp(); return; }
    dive.t = setTimeout(diveReset, 450);
    return;
  }
  diveReset();
  zoomAt(Math.exp(f), e.clientX, e.clientY);
}, {passive:false});
cv.addEventListener('dblclick', e => { const orb = orbitTarget(e); if (orb){ resetOrbit(orb); return; } if (e.target.closest('.nstep,.nvars input,.nvars button')) return; const grip = e.target.closest('.rz'); if (grip){ const n = byId(cur(), grip.closest('.node').dataset.id); if (n && n.w){ delete n.w; render(); } return; } const lch = chartTarget(e); if (lch){ chartView.delete(lch.dataset.chart); renderChart(lch.dataset.chart); return; } const wb = e.target.closest('[data-bend], .whit'); if (wb){ const ed = cur().edges.find(x => x.id === (wb.dataset.bend || wb.dataset.edge)); if (ed && ed.bend){ delete ed.bend; renderWires(); renderInspector(); save(); } return; } if (nview || e.shiftKey || e.ctrlKey || e.metaKey || e.target.closest('.vt,[data-train]')) return; const node = e.target.closest('.node'); if (node){ const n = byId(cur(), node.dataset.id); if (n && (n.type === 'group' || n.type === 'dense') && path[path.length-1] !== n.id) enter(n.id); } });
$('#zin').onclick = () => zoomAt(1.2);
$('#zout').onclick = () => zoomAt(1/1.2);
$('#zfit').onclick = fit;
$('#crumbs').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.mode) setMode(b.dataset.mode); else if (b.dataset.depth) goDepth(+b.dataset.depth); });
nodesEl.addEventListener('click', e => {   // keyboard activation of buttons on components
  if (e.detail !== 0) return;
  const sb = e.target.closest('[data-slstep]'); if (sb){ stepSlider(byId(cur(), sb.dataset.node), +sb.dataset.slstep, +sb.dataset.d); return; }
  const b = e.target.closest('.vt'), t = e.target.closest('[data-train]'), st = e.target.closest('[data-step]');
  if (b) toggleView([b.dataset.view]); else if (t && !t.disabled) toggleTrain(t.dataset.train); else if (st){ const n = byId(cur(), st.dataset.node); if (n) stepSample(n, st.dataset.step); }
});
// The sample field on a Data component applies on Enter or when it loses focus.
nodesEl.addEventListener('change', e => { const f = e.target.closest('.nidx'); if (!f) return; const n = byId(cur(), f.dataset.node), v = Math.floor(Number(f.value)); if (n && Number.isFinite(v)){ n.props.index = v; soft(); } });
nodesEl.addEventListener('keydown', e => { if (e.target.closest('.nidx') && e.key === 'Enter') e.target.blur(); });

/* ---------- palette drag ---------- */
$('#pal').addEventListener('pointerdown', e => {
  const del = e.target.closest('[data-dellib]');
  if (del){ e.stopPropagation(); return; }
  const it = e.target.closest('.pi'); if (!it || (e.button !== 0 && e.pointerType === 'mouse')) return;
  const kind = it.dataset.kind, ghost = $('#ghost'), sx = e.clientX, sy = e.clientY;
  let dragging = false;
  track(m => {
    if (!dragging && Math.hypot(m.clientX - sx, m.clientY - sy) > 6 && !(m.pointerType === 'touch' && Math.abs(m.clientX - sx) > Math.abs(m.clientY - sy) && window.innerWidth <= 820)){
      dragging = true; ghost.textContent = it.textContent.replace('×','').trim(); ghost.hidden = false;
    }
    if (dragging){ ghost.style.left = (m.clientX + 10)+'px'; ghost.style.top = (m.clientY + 8)+'px'; }
  }, m => {
    ghost.hidden = true;
    if (m.type === 'pointercancel') return;
    if (dragging){
      const r = cv.getBoundingClientRect();
      if (m.clientX >= r.left && m.clientX <= r.right && m.clientY >= r.top && m.clientY <= r.bottom) addNode(kind, toWorld(m.clientX, m.clientY));
    } else addNode(kind, null);
  });
});
$('#pal').addEventListener('click', e => {
  const del = e.target.closest('[data-dellib]');
  if (del){ lib = lib.filter(l => l.id !== del.dataset.dellib); renderPalette(); save(); }
});

/* ---------- inspector events ---------- */
/* ---------- suggestions while typing an expression ---------- */
const acEl = document.createElement('div');
acEl.className = 'ac';
acEl.setAttribute('role', 'listbox');
acEl.hidden = true;
document.body.appendChild(acEl);
const acMeasure = document.createElement('canvas').getContext('2d');
let ac = {open:false, items:[], i:0, from:0, to:0};
function acContext(el){
  const pos = el.selectionStart, before = el.value.slice(0, pos);
  if ((before.match(/"/g) || []).length % 2) return null;                                       // inside quotes
  if ((before.match(/\[/g) || []).length > (before.match(/\]/g) || []).length) return null;      // inside an index
  const m = before.match(/((?:(?:[A-Za-z_]\w*|"[^"]*")(?:\[[^\]]*\])?\.)*)([A-Za-z_]\w*)?$/);
  if (!m || (m.index > 0 && /[\w.\]]/.test(before[m.index - 1]) && !m[1])) return null;
  const segs = [...(m[1] || '').matchAll(/([A-Za-z_]\w*|"[^"]*")(\[[^\]]*\])?\./g)].map(x => ({name:x[1].startsWith('"') ? x[1].slice(1, -1) : x[1], idx:!!x[2]}));
  const word = m[2] || '';
  return {segs, path:segs.map(x => x.name), word, start:pos - word.length, end:pos + el.value.slice(pos).match(/^\w*/)[0].length};
}
function acItems(ctx, self){
  const w = ctx.word, wl = w.toLowerCase(), out = [];
  const score = name => name.startsWith(w) ? 0 : name.toLowerCase().startsWith(wl) ? 1 : wl && name.toLowerCase().includes(wl) ? 2 : -1;
  const add = (name, text, kind, detail, s) => { if (s >= 0) out.push({name, text, kind, detail, s}); };
  if (!ctx.path.length){
    const seenN = new Set(); for (const gg of graphChain(cur())) for (const n of gg.nodes) if (n !== self && !seenN.has(n.name)){ seenN.add(n.name); const sh = n._out || (n.type === 'output' && n._in && n._in[0]); add(n.name, refName(n.name), n.type === 'group' ? 'block' : labelOf(n), sh ? fmt(sh) : '', score(n.name)); }
    for (const fn of Object.keys(LFN)) add(fn + '()', fn + '(', 'function', FN_HELP[fn], score(fn) + .5);
  } else {
    const rr = lookup(ctx.path.join(SEP), cur()), last = ctx.segs[ctx.segs.length - 1];
    if (!rr.node) return [];
    if (rr.node.type === 'group' && !last.idx){
      for (const n of rr.node.graph.nodes){ const sh = n._out || (n.type === 'gout' && n._in && n._in[0]); add(n.name, refName(n.name), n.type === 'gin' ? 'block input' : n.type === 'gout' ? 'block output' : n.type === 'group' ? 'block' : labelOf(n), sh ? fmt(sh) : '', score(n.name)); }
    } else if (rr.node.type === 'dense' && rr.node._in && rr.node._in[0] && rr.node._out){
      const fi = lastDim(rr.node._in[0]), fo = lastDim(rr.node._out), gg = rr.graph, nx = nextDense(gg, rr.node, '');
      add('pre_weights', 'pre_weights', last.idx ? 'weights into this neuron' : 'all incoming weights', last.idx ? `${fi}` : `${fo}×${fi}`, score('pre_weights'));
      if (nx.node && nx.node._out) add('post_weights', 'post_weights', `${last.idx ? 'weights from this neuron' : 'all weights'} into ${nx.node.name}`, last.idx ? `${lastDim(nx.node._out)}` : `${fo}×${lastDim(nx.node._out)}`, score('post_weights') + .1);
    }
  }
  out.sort((a, b) => a.s - b.s);   // stable: same-quality matches keep the network's order
  return out.slice(0, 12);
}
function acPaint(){
  acEl.innerHTML = ac.items.map((it, k) => `<div class="aci${k === ac.i ? ' on' : ''}" role="option" aria-selected="${k === ac.i}" data-k="${k}"><b>${esc(it.name)}</b><span class="ack">${esc(it.kind)}</span><span class="acd">${esc(it.detail)}</span></div>`).join('');
  const on = acEl.querySelector('.on'); if (on) on.scrollIntoView({block:'nearest'});
}
function acHide(){ ac.open = false; acEl.hidden = true; }
function acUpdate(el, force){
  const ctx = acContext(el);
  if (!ctx || (!force && !ctx.word && !ctx.path.length)) return acHide();
  let items = acItems(ctx, byId(cur(), [...sel][0]));
  if (!force && items.length === 1 && items[0].text === ctx.word) items = [];
  if (!items.length) return acHide();
  const keep = ac.open && ac.items[ac.i] ? ac.items[ac.i].name : null;
  ac = {open:true, items, i:Math.max(0, items.findIndex(it => it.name === keep)), from:ctx.start, to:ctx.end};
  acPaint();
  const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
  acMeasure.font = `${cs.fontSize} ${cs.fontFamily}`;
  const x = r.left + parseFloat(cs.paddingLeft) + acMeasure.measureText(el.value.slice(0, ctx.start)).width - el.scrollLeft;
  acEl.hidden = false;
  acEl.style.left = Math.max(8, Math.min(x - 8, innerWidth - acEl.offsetWidth - 8)) + 'px';
  acEl.style.top = (r.bottom + acEl.offsetHeight + 4 > innerHeight ? r.top - acEl.offsetHeight - 4 : r.bottom + 4) + 'px';
}
function acAccept(el, k){
  const it = ac.items[k]; if (!it) return;
  const v = el.value; el.value = v.slice(0, ac.from) + it.text + v.slice(ac.to);
  const p = ac.from + it.text.length; el.setSelectionRange(p, p);
  acHide();
  el.dispatchEvent(new Event('input', {bubbles:true}));
}
const isExprKey = k => k === 'expr' || k === 'target' || k === 'pair' || /^in\d$/.test(k || '');   // a Loss expression, an input cell or a Pointer's ends
const exprField = () => { const el = document.activeElement; return el && insp.contains(el) && isExprKey(el.dataset.k) ? el : null; };
insp.addEventListener('keydown', e => {
  if (!isExprKey(e.target.dataset.k)) return;
  if (e.key === ' ' && e.ctrlKey){ e.preventDefault(); acUpdate(e.target, true); return; }
  if (!ac.open) return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp'){ e.preventDefault(); ac.i = (ac.i + (e.key === 'ArrowDown' ? 1 : -1) + ac.items.length) % ac.items.length; acPaint(); }
  else if (e.key === 'Enter' || e.key === 'Tab'){ e.preventDefault(); acAccept(e.target, ac.i); }
  else if (e.key === 'Escape'){ e.preventDefault(); e.stopPropagation(); acHide(); }
});
insp.addEventListener('keyup', e => { if (isExprKey(e.target.dataset.k) && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Home' || e.key === 'End')) acUpdate(e.target); });
insp.addEventListener('focusout', () => setTimeout(() => { if (!exprField()) acHide(); }, 0));
acEl.addEventListener('mousedown', e => { e.preventDefault(); const it = e.target.closest('.aci'), el = exprField(); if (it && el) acAccept(el, +it.dataset.k); });
function varOf(el){ const n = byId(cur(), [...sel][0]); return n && n.props.vars && n.props.vars[el.dataset.var] ? [n, n.props.vars[el.dataset.var]] : [null, null]; }
// Refresh while a slider is being dragged. The component holding that slider keeps its elements (so the drag
// continues) and only has its view panels redrawn; everything else is rebuilt.
function refreshCanvas(holdId){
  infer(); forward(root, null); renderNeuro();
  if (!holdId) renderNodes();
  else {
    for (const n of cur().nodes){
      const el = nodesEl.querySelector(`.node[data-id="${n.id}"]`); if (!el) continue;
      if (n.id !== holdId){ el.outerHTML = nodeHTML(n); continue; }
      const vw = el.querySelector('.vw'); if (vw) vw.innerHTML = viewPanels(n);
    }
    paintViz(nodesEl); layoutPorts();
  }
  renderWires(); save();
}
function stepVar(n, name, dir){ const v = n && n.props.vars && n.props.vars[name]; if (!v) return; v.value = Math.min(v.max, Math.max(v.min, +(v.value + dir*v.step).toFixed(10))); soft(); }
// A slider of a Sliders component, dragged on the component or in the info window.
function slideTo(el, keepInspector){
  const n = byId(cur(), el.dataset.node), s = n && (n.props.sliders || [])[+el.dataset.sl]; if (!s) return;
  s.value = Number(el.value);
  const o = el.closest('.nvar, .slrow'); if (o){ const out = o.querySelector('output'); if (out) out.textContent = fmtV(s.value); }
  refreshCanvas(keepInspector ? null : n.id); if (!keepInspector) renderInspector();
}
function stepSlider(n, i, dir){ const s = n && (n.props.sliders || [])[i]; if (!s) return; s.value = Math.min(s.max, Math.max(s.min, +(s.value + dir*s.step).toFixed(10))); soft(); }
nodesEl.addEventListener('input', e => { const sl = e.target.closest('[data-sl]'); if (sl) slideTo(sl, false); });
nodesEl.addEventListener('change', e => { if (e.target.closest('[data-sl]')) soft(); });
nodesEl.addEventListener('input', e => {
  const r = e.target.closest('[data-nvar]'); if (!r) return;
  const n = byId(cur(), r.dataset.node), v = n && n.props.vars && n.props.vars[r.dataset.nvar]; if (!v) return;
  v.value = Number(r.value); r.parentNode.querySelector('output').textContent = fmtV(v.value);
  refreshCanvas(n.id); renderInspector();
});
nodesEl.addEventListener('change', e => { if (e.target.closest('[data-nvar]')) soft(); });
insp.addEventListener('change', e => {
  if (e.target.classList.contains('runr')){ save(); return; }
  if (e.target.dataset.sli != null){   // a Sliders component's slider: name, min, step or max
    const n = byId(cur(), [...sel][0]), s = n && (n.props.sliders || [])[+e.target.dataset.sli]; if (!s) return;
    const k = e.target.dataset.slk, v = e.target.value.trim();
    if (k === 'name') s.name = v || s.name;
    else { const x = Number(v); if (Number.isFinite(x) && (k !== 'step' || x > 0)) s[k] = x; if (s.min > s.max) [s.min, s.max] = [s.max, s.min]; s.value = Math.min(s.max, Math.max(s.min, s.value)); }
    render(); return;
  }
  if (e.target.dataset.param != null){   // a block variable row
    const blk = curBlock(), p = blk && (blk.props.params || [])[+e.target.dataset.param]; if (!p) return;
    const v = e.target.value.trim(), vars = blk.props.vars ||= {};
    if (e.target.dataset.pk === 'name'){ if (vars[p.name] && v && !vars[v]){ vars[v] = vars[p.name]; delete vars[p.name]; } p.name = v; }
    else {
      p.expr = v;
      if (v !== '' && !isNaN(v)){   // a typed number sets the slider; a range that doesn't hold it is redrawn around it
        const x = Number(v), sl = vars[p.name];
        if (!sl) vars[p.name] = paramSlider(x);
        else { sl.value = x; if (x > sl.max || x < sl.min){ const f = paramSlider(x); sl.min = Math.min(sl.min, f.min); sl.max = Math.max(sl.max, f.max); sl.step = Number.isInteger(x) && Number.isInteger(sl.step) ? sl.step : Math.min(sl.step, f.step); } }
      } else delete vars[p.name];   // an expression has no slider
    }
    render(); return;
  }
  if (!e.target.dataset.var) return;
  const [n, v] = varOf(e.target); if (!n) return;
  if (e.target.dataset.vk){
    const x = Number(e.target.value);
    if (Number.isFinite(x) && (e.target.dataset.vk !== 'step' || x > 0)) v[e.target.dataset.vk] = x;
    if (v.min > v.max) [v.min, v.max] = [v.max, v.min];
    v.value = Math.min(v.max, Math.max(v.min, v.value));
  }
  soft();
});
insp.addEventListener('input', e => {
  if (e.target.classList.contains('runr')){   // how many samples an inference run takes: the field and the button follow the slider
    const n = byId(cur(), e.target.dataset.node); if (!n) return;
    const v = Math.round(Number(e.target.value)), all = v >= Number(e.target.max);
    n.props.run_n = all ? 'all' : v;
    const f = e.target.parentNode.querySelector('[data-k="run_n"]'); if (f) f.value = n.props.run_n;
    const b = insp.querySelector('[data-act="collect"]'); if (b && !isCollecting(n)) b.textContent = `▶ Run inference on ${v.toLocaleString()}`;
    return;
  }
  if (e.target.dataset.param != null || e.target.dataset.sli != null) return;   // these apply on Enter or leaving the field
  if (e.target.dataset.sl != null){ slideTo(e.target, true); return; }   // block variables apply on Enter or leaving the field
  if (e.target.dataset.var){
    if (e.target.type !== 'range') return;   // min, step and max apply on Enter or when the field loses focus
    const [n, v] = varOf(e.target); if (!n) return;
    v.value = Number(e.target.value);
    e.target.closest('.var').querySelector('output').textContent = fmtV(v.value);
    refreshCanvas();
    return;
  }
  const k = e.target.dataset.k; if (!k) return;
  const n = byId(cur(), [...sel][0]); if (!n) return;
  if (k === 'name'){ const before = pathMap(); n.name = e.target.value; rewriteRefs(before); renderCrumbs(); }   // expressions follow the rename
  else if (/^in\d$/.test(k)){   // an input cell: a wire or an expression
    const i = +k.slice(2), g = cur(), v = e.target.value.trim(), ins = n.props.inputs || (n.props.inputs = []);
    const wire = g.edges.find(x => x.to === n.id && x.port === i), target = v && g.nodes.find(m => m.name === v && m !== n && DEFS[m.type].outs);
    if (target && (wire && wire.from === target.id || connect(target.id, n.id, i))) ins[i] = '';   // a component's name: a wire
    else { ins[i] = v; g.edges = g.edges.filter(x => !(x.to === n.id && x.port === i)); }          // an expression (or nothing: a free port)
    if (!ins.some(Boolean)) delete n.props.inputs;
  }
  else if (e.target.type === 'checkbox') n.props[k] = e.target.checked;
  // A complete number is stored as a number; anything else ("0.", "0.50", "-", a variable name) exactly as typed,
  // so the field doesn't rewrite what's being typed. The engine reads numbers out of either.
  else { const v = e.target.value.trim(); n.props[k] = v !== '' && !isNaN(v) && String(Number(v)) === v ? Number(v) : v; }
  soft();
  if (isExprKey(k)){ const el = exprField(); if (el) acUpdate(el); }
});
insp.addEventListener('click', e => {
  const vs = e.target.closest('[data-vstep]');
  if (vs){ stepVar(byId(cur(), [...sel][0]), vs.dataset.vstep, +vs.dataset.d); return; }
  const b = e.target.closest('[data-act]'); if (!b) return;
  const id = [...sel][0], a = b.dataset.act;
  if (a === 'addslider' || a === 'delslider'){
    const n = byId(cur(), id); if (!n) return;
    const ss = n.props.sliders ||= [];
    if (a === 'addslider'){ const taken = new Set(ss.map(s => s.name)), name = 'abcdefghijklmnopqrstuvwxyz'.split('').find(c => !taken.has(c)) || `s${ss.length + 1}`; ss.push({name, value:0, min:-1, max:1, step:0.01}); }
    else ss.splice(+b.dataset.i, 1);
    render(); return;
  }
  if (a === 'addparam' || a === 'delparam'){
    const blk = curBlock(); if (!blk) return;
    const ps = blk.props.params ||= [];
    if (a === 'addparam'){
      const taken = new Set(ps.map(p => p.name)), name = ['n', 'm', 'k', 'width', 'depth'].find(x => !taken.has(x)) || `v${ps.length + 1}`;
      ps.push({name, expr:'8'}); render();
      const f = insp.querySelector(`[data-param="${ps.length - 1}"][data-pk="name"]`); if (f){ f.focus(); f.select(); }
    } else { const [p] = ps.splice(+b.dataset.i, 1); if (p && blk.props.vars) delete blk.props.vars[p.name]; render(); }
    return;
  }
  if (a === 'delete' || a === 'delwire') deleteSel();
  else if (a === 'unbend'){ const ed = cur().edges.find(x => x.id === selWire); if (ed){ delete ed.bend; renderWires(); renderInspector(); save(); } }
  else if (a === 'group') groupSelection();
  else if (a === 'ungroup') ungroup(id);
  else if (a === 'open') enter(id);
  else if (a === 'savelib') saveToLib(id);
  else if (a === 'dup') duplicate();
  else if (a === 'view') toggleView([id]);
  else if (a === 'train') toggleTrain(id);
  else if (a === 'collect') toggleCollect(id);
  else if (a === 'fitpca') toggleFitPca(id);
  else if (a === 'evaluate') evalFull(byId(cur(), id));
  else if (a === 'resetw') resetWeights();
  else if (a === 'prev' || a === 'next' || a === 'rand'){ const n = byId(cur(), id); if (n) stepSample(n, a); }
  else if (a === 'import') $('#fileIn').click();
  else if (a === 'up') goUp();
  else if (a === 'neurons') enterLayer(id);
  else if (a === 'layers'){ if (nview && nview.id) goDepth(path.length); else setMode('layers'); }
});

/* ---------- toolbar & keys ---------- */
$('#bGroup').onclick = groupSelection;
$('#bUngroup').onclick = () => ungroup([...sel][0]);
$('#bDelete').onclick = deleteSel;
$('#bBox').onclick = () => { boxMode = !boxMode; $('#bBox').setAttribute('aria-pressed', boxMode); };
$('#bExample').onclick = () => { example(); renderPalette(); render(); fit(); flushSave(); toast(`Example loaded. Press ${isMac ? '⌘Z' : 'Ctrl+Z'} to get your network back.`); };
$('#bClear').onclick = () => { root = {nodes:[], edges:[]}; path = []; nview = null; sel.clear(); selWire = null; const i = mk('data', 0, 40, {dataset: datasets.length ? datasets[0].id : 'mnist'}), o = mk('output', 600, 40); root.nodes.push(i); root.nodes.push(o); renderPalette(); render(); fit(); flushSave(); toast(`Canvas cleared. Press ${isMac ? '⌘Z' : 'Ctrl+Z'} to undo.`); };
$('#bUndo').onclick = () => stepHistory(true);
$('#bRedo').onclick = () => stepHistory(false);
$('#bImport').onclick = () => $('#fileIn').click();
/* Project files: the network, My blocks and the trained weights in one .json file (see engine/project.js). */
function saveProject(){
  const name = `netgraph-${new Date().toISOString().slice(0, 10)}.json`, text = projectJSON(lib);
  download(name, text, 'application/json');
  toast(`Saved ${name} (${(text.length/1024).toFixed(0)} KB). Open it again with Open or by dropping it on the canvas.`);
}
async function openProject(file){
  if (TR && TR.running){ toast('Stop training before opening another project.'); return; }
  let p; try { p = readProject(await file.text()); } catch(err){ toast(`Couldn't open ${file.name}: ${err.message}`); return; }
  root = p.graph; lib = p.blocks; path = []; nview = null; sel.clear(); selWire = null;
  importWeights(p.weights); trainLog.clear(); saveWeights();
  renderPalette(); render(); fit(); flushSave();   // its own undo step
  toast(`Opened ${file.name}.${p.missing.length ? ` It uses the dataset "${p.missing.join('", "')}": import it to run the network.` : ''} Press ${isMac ? '⌘Z' : 'Ctrl+Z'} to go back.`);
}
$('#bSave').onclick = saveProject;
$('#bOpen').onclick = () => $('#projIn').click();
$('#projIn').onchange = e => { if (e.target.files[0]) openProject(e.target.files[0]); e.target.value = ''; };
$('#fileIn').onchange = e => { importFiles(e.target.files); e.target.value = ''; };
cv.addEventListener('dragover', e => { if ([...e.dataTransfer.types].includes('Files')){ e.preventDefault(); cv.classList.add('drop'); } });
cv.addEventListener('dragleave', e => { if (!cv.contains(e.relatedTarget)) cv.classList.remove('drop'); });
cv.addEventListener('drop', e => {   // a project file opens; anything else is a dataset
  cv.classList.remove('drop'); if (!e.dataTransfer.files.length) return;
  e.preventDefault();
  const files = [...e.dataTransfer.files], proj = files.find(f => /\.json$/i.test(f.name));
  if (proj) openProject(proj); else importFiles(files);
});
$('#bExport').onclick = () => { $('#dCode').textContent = codegen(); $('#dlg').showModal(); };
function download(name, data, type){
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], {type})); a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$('#dPy').onclick = () => download('netgraph_model.py', $('#dCode').textContent, 'text/x-python');
$('#dWeights').onclick = () => { const t = stateDict(); download('netgraph_weights.safetensors', safetensors(t), 'application/octet-stream'); toast(`Saved ${t.length} tensors. Load them with model.load_state_dict(load_file("netgraph_weights.safetensors")).`); };
$('#dClose').onclick = () => $('#dlg').close();
$('#dCopy').onclick = async () => {
  const txt = $('#dCode').textContent;
  try { await navigator.clipboard.writeText(txt); $('#dCopy').textContent = 'Copied'; }
  catch(err){ const r = document.createRange(); r.selectNodeContents($('#dCode')); const s = getSelection(); s.removeAllRanges(); s.addRange(r); $('#dCopy').textContent = 'Press ⌘C / Ctrl+C'; }
  setTimeout(() => $('#dCopy').textContent = 'Copy code', 1800);
};
window.addEventListener('keydown', e => {
  const typing = e.target.closest && e.target.closest('input,textarea,select');
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'g'){ e.preventDefault(); e.shiftKey ? ungroup([...sel][0]) : groupSelection(); return; }
  if ((e.metaKey || e.ctrlKey) && ['s', 'o'].includes(e.key.toLowerCase())){ e.preventDefault(); if (e.key.toLowerCase() === 's') saveProject(); else $('#projIn').click(); return; }
  if (typing || $('#dlg').open) return;   // text fields keep their own undo
  if ((e.metaKey || e.ctrlKey) && (e.key.toLowerCase() === 'z' || e.key.toLowerCase() === 'y')){ e.preventDefault(); stepHistory(e.key.toLowerCase() === 'z' && !e.shiftKey); return; }
  if ((e.metaKey || e.ctrlKey) && !nview && ['c', 'x', 'v'].includes(e.key.toLowerCase()) && !String(getSelection())){
    e.preventDefault();
    const k = e.key.toLowerCase();
    if (k === 'v') pasteClip(); else if (copySel(k === 'x')) { if (k === 'x') deleteSel(); }
    return;
  }
  const one = sel.size === 1 ? byId(cur(), [...sel][0]) : null;
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && one && one.type === 'data'){ e.preventDefault(); stepSample(one, e.key === 'ArrowRight' ? 'next' : 'prev'); }
  else if (e.key === 'Delete' || e.key === 'Backspace'){ e.preventDefault(); deleteSel(); }
  else if (e.key.toLowerCase() === 'g' && !e.metaKey && !e.ctrlKey && !e.altKey && !nview){ e.preventDefault(); e.shiftKey ? ungroup([...sel][0]) : groupSelection(); }
  else if (e.key.toLowerCase() === 'v' && !e.metaKey && !e.ctrlKey && !e.altKey && sel.size && !nview){ e.preventDefault(); toggleView([...sel]); }
  else if (e.key === 'Escape'){ if (sel.size || selWire){ sel.clear(); selWire = null; renderNodes(); renderWires(); renderInspector(); } else goUp(); }
  else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd' && sel.size === 1){ e.preventDefault(); duplicate(); }
  else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'a' && !nview){ e.preventDefault(); sel = new Set(cur().nodes.map(n => n.id)); renderNodes(); renderInspector(); }
});
const isMac = /Mac|iPhone|iPad/.test(navigator.platform || '');
if (!isMac){ document.querySelector('#bOpen kbd').textContent = 'Ctrl+O'; document.querySelector('#bSave kbd').textContent = 'Ctrl+S'; document.querySelector('#bUndo kbd').textContent = 'Ctrl+Z'; document.querySelector('#bRedo kbd').textContent = 'Ctrl+Y'; }

// A 3D plot in the info window turns, zooms and resets the same way.
insp.addEventListener('pointerdown', e => { const orb = orbitTarget(e); if (orb && e.button === 0) startOrbit(e, orb); });
insp.addEventListener('wheel', e => { const orb = orbitTarget(e); if (orb) zoomOrbit(e, orb); }, {passive:false});
insp.addEventListener('dblclick', e => { const orb = orbitTarget(e); if (orb) resetOrbit(orb); });
