function propField(n, k){
  const v = n.props[k], def = DEFS[n.type].props[k];
  if (typeof def === 'boolean') return `<label class="chk"><input type="checkbox" data-k="${k}"${v?' checked':''}> ${LABELS[k]||k}</label>`;
  let hint = HINTS[k] || '';
  if (k==='in_features' && isAuto(v) && n._in[0]) hint = `Currently ${n._in[0][n._in[0].length-1]}, read from the incoming tensor.`;
  if (k==='in_channels' && isAuto(v) && n._in[0]) hint = `Currently ${n._in[0][0]}, read from the incoming tensor.`;
  if (k==='to' && isAuto(v) && n._out) hint = `Currently ${fmt(n._out)}. Type a shape to choose another, like 16,7,7 or -1,7,7.`;
  return `<label class="f"><span>${LABELS[k]||k}</span><input type="text" data-k="${k}" value="${esc(v)}" autocomplete="off" spellcheck="false">${hint?`<small>${esc(hint)}</small>`:''}</label>`;
}
// Sample browser on a Data component in view mode.
function stepperHTML(n){
  const s = sampleOf(n); if (!s) return '';
  return `<div class="nstep"><button class="tb" data-step="prev" data-node="${n.id}" aria-label="Previous sample">‹</button><input type="text" class="nidx" data-node="${n.id}" value="${s.i}" aria-label="Sample number" autocomplete="off" spellcheck="false"><button class="tb" data-step="next" data-node="${n.id}" aria-label="Next sample">›</button><button class="tb" data-step="rand" data-node="${n.id}">Random</button></div>`;
}
function dataPanel(n){
  const coll = n._collect, r = n._r || {};
  let h = `<h2>Data <small>${esc(n.name)}</small></h2>`;
  if (!datasets.length && !coll) return h + `<p class="lead">No datasets yet. Import the MNIST files from the data/mnist folder, or drop them on the canvas.</p>
    <div class="acts"><button class="tb primary" data-act="import">Import data…</button><button class="tb" data-act="delete">Delete</button></div>`;
  const s = !r.err && sampleOf(n), ds0 = dsById(n.props.dataset), ds = ds0 && (!coll || ds0.x) ? ds0 : null;
  if (s) h += sampleHTML(s, true);
  const sp = ds && splitOf(n);
  h += dimsRows([...(coll ? [['Input', fmt(n._in[0])]] : []), ['Output', fmt(n._out)], ...(s ? [['Label', String(s.label)], ['Sample', `${s.i.toLocaleString()} of ${s.ds.n.toLocaleString()}`], ['Split', sp.isTest[s.i] ? 'Test' : 'Train']] : [])]);
  h += r.err ? `<p class="err">${esc(r.err)}</p>` : r.warn ? `<p class="warn">${esc(r.warn)}</p>` : r.wait ? `<p class="lead">${esc(r.wait)}</p>` : '';
  if (s && !n.view) h += `<div class="step"><button class="tb" data-act="prev" aria-label="Previous sample">‹</button><input type="text" data-k="index" value="${s.i}" aria-label="Sample number" autocomplete="off" spellcheck="false"><button class="tb" data-act="next" aria-label="Next sample">›</button><button class="tb" data-act="rand">Random</button></div>`;
  if (coll){ if (ds) h += dimsRows([['Holds', `${ds.n.toLocaleString()} × ${fmt(dsShape(ds))}`], ['From', ds.from || '—']]); }
  else if (ds) h += dimsRows([['Holds', `${ds.n.toLocaleString()} × ${fmt(dsShape(ds))}`]]);
  else h += `<label class="f"><span>Dataset</span><select data-k="dataset">${ds ? '' : '<option value="">Pick one</option>'}${datasets.map(d => `<option value="${esc(d.id)}"${d === ds ? ' selected' : ''}>${esc(d.name)}</option>`).join('')}</select></label>`;
  if (ds){
    const sh = dsShape(ds);
    if (sh.length > 1) h += `<label class="chk"><input type="checkbox" data-k="flatten"${n.props.flatten ? ' checked' : ''}> Flatten to ${sizeOf(sh).toLocaleString()} features</label>`;
    h += `<label class="f"><span>Test split (%)</span><input type="text" data-k="test" value="${esc(n.props.test ?? 20)}" autocomplete="off" spellcheck="false"><small>${sp ? `${sp.train.length.toLocaleString()} train · ${sp.test.length.toLocaleString()} test. Samples are shuffled with the seed below, then the test share is held out.` : ''}</small></label>
      <label class="f"><span>Shuffle seed</span><input type="text" data-k="seed" value="${esc(n.props.seed ?? 1)}" autocomplete="off" spellcheck="false"></label>`;
  }
  if (ds && !r.err) h += inferHTML(n);   // collected samples can be sent on too
  h += `<label class="f"><span>Name</span><input type="text" data-k="name" value="${esc(n.name)}" autocomplete="off" spellcheck="false"></label>`;
  h += inputCells(n, coll ? '' : 'Wire a layer here, or name one, and this component collects what reaches it during an inference run.');
  h += `<div class="acts">${viewBtn(n)}${coll ? '' : '<button class="tb" data-act="import">Import data…</button>'}<button class="tb" data-act="dup">Duplicate</button><button class="tb" data-act="delete">Delete</button></div>`;
  return h;
}
// Inference runs: send samples of this Data component through the network; the Data components their flow reaches keep it.
function inferHTML(n){
  const on = isCollecting(n), cs = collectorsOf(n), p = n.props, ds = dsById(p.dataset), sp = splitOf(n);
  const total = p.run_from === 'train' ? sp.train.length : p.run_from === 'all' ? ds.n : sp.test.length;
  let pool = null, perr = null; try { pool = runPool(n); } catch(e){ perr = e.message; }
  const k = pool ? pool.length : total;
  let h = `<div class="infer"><span class="cap">Inference</span>
    <label class="f"><span>Samples from</span><select data-k="run_from">${[['test', 'Test split'], ['train', 'Train split'], ['all', 'Whole dataset']].map(([v, l]) => `<option value="${v}"${(p.run_from || 'test') === v ? ' selected' : ''}>${l}</option>`).join('')}</select></label>
    <div class="f"><span>How many</span><div class="rn"><input type="range" class="runr" data-node="${n.id}" min="1" max="${total}" step="1" value="${k}" aria-label="How many samples"><input type="text" data-k="run_n" value="${esc(p.run_n ?? 'all')}" placeholder="all" aria-label="How many samples" autocomplete="off" spellcheck="false"></div><small>Of ${total.toLocaleString()}. Drag, or type a number or all.</small></div>`;
  if (perr) h += `<p class="err">${esc(perr)}</p>`;
  h += `<div class="acts"><button class="tb${on ? '' : ' primary'}" data-act="collect"${(!cs.length || perr || !k) && !on ? ' disabled' : ''}>${on ? '■ Stop' : `▶ Run inference on ${k.toLocaleString()}`}</button>${on ? `<span class="iprog">${CR.done.toLocaleString()} / ${CR.N.toLocaleString()}</span>` : ''}</div>`;
  h += `<small class="vnote">${cs.length ? `Collects into ${cs.map(m => `<b>${esc(m.name)}</b>`).join(', ')}.` : 'Wire a layer into another Data component\'s input to collect what reaches it.'}</small></div>`;
  return h;
}
// A cell per input: the wired component's name, or an expression. Typing a component's name (in this graph) wires
// it; anything else is an expression and replaces the wire; an empty cell is an unconnected port.
function inputCells(n, note){
  const d = DEFS[n.type], g = cur(); let h = '';
  for (let i = 0; i < d.ins; i++){
    const e = g.edges.find(x => x.to === n.id && x.port === i), src = e && byId(g, e.from);
    const val = portExpr(n, i) ?? (src ? src.name : '');
    h += `<label class="f"><span>${d.ins > 1 ? `Input ${'ab'[i]}` : 'Input'}</span><input type="text" data-k="in${i}" value="${esc(val)}" placeholder="a component, or an expression" autocomplete="off" spellcheck="false">${i === d.ins - 1 && note !== '' ? `<small>${note || 'The wired component, or an expression like fc1[0:10] * 2 or relu1 + Block.fc2. A component\'s name wires it; an empty cell is a free port.'}</small>` : ''}</label>`;
  }
  return h;
}
// A PCA component: how much variance each component explains, and the Fit button.
function pcaHTML(n){
  const S = n._in[0] && sizeOf(n._in[0]), k = Number(n.props.components), P = S && pcaParams(n, S, k), on = isCollecting(n), outs = collectorsOf(n), r = n._r || {};
  let h = `<div class="infer"><span class="cap">Principal components</span>`;
  if (P){
    const tot = 100*P.rm.reduce((a, v) => a + v, 0);
    h += `<div class="evr">${Array.from(P.rm).slice(0, 12).map((v, j) => `<div><span>PC${j + 1}</span><s><i style="width:${Math.max(1, 100*v/P.rm[0])}%"></i></s><b>${(100*v).toFixed(1)}%</b></div>`).join('')}${k > 12 ? `<div><span>…</span></div>` : ''}<div class="tot"><span>Total</span><b>${tot.toFixed(1)}%</b></div></div>`;
  }
  const bad = r.err && !r.err.startsWith('Not fitted');
  h += `<div class="acts"><button class="tb${on ? '' : ' primary'}" data-act="fitpca"${bad && !on ? ' disabled' : ''}>${on ? '■ Stop' : `▶ ${P ? 'Fit again' : 'Fit'}${outs.length ? ' and project' : ''}`}</button>${on ? `<span class="iprog">${CR.done.toLocaleString()} / ${CR.N.toLocaleString()}</span>` : ''}</div>`;
  h += `<small class="vnote">Fits on the first samples of the train split of the Data component upstream, as they reach it. ${outs.length ? `Then runs the whole dataset through it into ${outs.map(m => `<b>${esc(m.name)}</b>`).join(', ')}.` : 'Wire its output into a Data component to keep the projected data.'}</small></div>`;
  return h;
}
// A Plot 2D: the plot, big, with its settings.
function plotPanel(n){
  const r = n._r || {}, p = n.props;
  let h = `<h2>${labelOf(n)} <small>${esc(n.name)}</small></h2>${plotHTML(n, true)}<div class="plotn" data-plotn="${n.id}">${plotNote(n)}</div>`;
  h += dimsRows([['Input', fmt(n._in[0])], ['Kept', n._pts && n._pts.n ? `${n._pts.n.toLocaleString()} points` : '—']]);
  h += r.err ? `<p class="err">${esc(r.err)}</p>` : r.warn ? `<p class="warn">${esc(r.warn)}</p>` : '';
  h += `<label class="f"><span>Name</span><input type="text" data-k="name" value="${esc(n.name)}" autocomplete="off" spellcheck="false"></label>`;
  h += inputCells(n);
  for (const k of n.type === 'plot3d' ? ['x_dim', 'y_dim', 'z_dim'] : ['x_dim', 'y_dim']) h += propField(n, k);
  h += `<div class="acts"><button class="tb" data-act="dup">Duplicate</button><button class="tb" data-act="delete">Delete</button></div>`;
  return h;
}
// A Sliders component: each slider is one value of its output vector. Names, ranges and values apply on Enter or leaving a field.
function slidersPanel(n){
  const ss = n.props.sliders ||= [], r = n._r || {};
  let h = `<h2>Sliders <small>${esc(n.name)}</small></h2>${dimsRows([['Output', fmt(n._out)]])}${r.err ? `<p class="err">${esc(r.err)}</p>` : ''}<div class="sliders">`;
  ss.forEach((s, i) => {
    const f = (k, label) => `<label>${label}<input type="text" data-sli="${i}" data-slk="${k}" value="${esc(s[k])}" autocomplete="off" spellcheck="false"></label>`;
    h += `<div class="slrow"><div class="slh"><input type="text" class="sln" data-sli="${i}" data-slk="name" value="${esc(s.name)}" aria-label="Name of ${esc(n.name)}[${i}]" autocomplete="off" spellcheck="false"><span class="sli">${esc(n.name)}[${i}]</span><output>${fmtV(s.value)}</output><button class="x" data-act="delslider" data-i="${i}" aria-label="Remove ${esc(s.name)}">×</button></div>
      <input type="range" data-sl="${i}" data-node="${n.id}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}" aria-label="${esc(s.name)}">
      <div class="vr">${f('min', 'min')}${f('step', 'step')}${f('max', 'max')}</div></div>`;
  });
  h += `</div><button class="tb" data-act="addslider">+ Add slider</button><small class="vnote">The output is a vector of these values, in order. Use it like any output: wire it, or name it in an expression (${esc(n.name)}[0]).</small>
    <label class="f"><span>Name</span><input type="text" data-k="name" value="${esc(n.name)}" autocomplete="off" spellcheck="false"></label>
    <div class="acts">${viewBtn(n)}<button class="tb" data-act="dup">Duplicate</button><button class="tb" data-act="delete">Delete</button></div>`;
  return h;
}
// The block whose inside is on screen, if any.
const curBlock = () => path.length ? byId(parentGraph(), path[path.length - 1]) : null;
// Block variables, edited in the Block input's info window: name = value rows. Applied on Enter or leaving a field.
function paramsHTML(blk){
  const ps = blk.props.params ||= [], vars = blk.props.vars || {};
  const rows = ps.map((p, i) => {
    const txt = String(p.expr ?? '').trim(), num = txt !== '' && !isNaN(txt), shown = num && vars[p.name] ? fmtV(vars[p.name].value) : txt;
    return `<div class="prow"><input type="text" data-param="${i}" data-pk="name" value="${esc(p.name)}" placeholder="name" aria-label="Variable name" autocomplete="off" spellcheck="false"><span>=</span><input type="text" data-param="${i}" data-pk="expr" value="${esc(shown)}" placeholder="64 or 2*n" aria-label="Value of ${esc(p.name)}" autocomplete="off" spellcheck="false"><button class="x" data-act="delparam" data-i="${i}" aria-label="Remove ${esc(p.name)}">×</button></div>`;
  }).join('');
  return `<div class="params"><span class="cap">Block variables</span>${rows}<button class="tb" data-act="addparam">+ Add variable</button>${blk._paramErr ? `<p class="err">${esc(blk._paramErr)}</p>` : ''}<small>A number becomes a slider on the block; an expression can use the variables above it. Layers inside the block can use these names in their settings and expressions.</small></div>`;
}
function parentGraph(){ let gg = root; for (const id of path.slice(0,-1)) gg = byId(gg,id).graph; return gg; }
function dimsRows(rows){ return `<div class="dims">${rows.map(([a,b]) => `<span>${a}</span><b>${b}</b>`).join('')}</div>`; }
function renderInspector(){
  const ae = document.activeElement;
  const fk = ae && insp.contains(ae) && ae.dataset.k;
  let ss = null, se = null; try { ss = ae.selectionStart; se = ae.selectionEnd; } catch(e){}
  const g = cur();
  let h = '';
  if (nview && nmodel){
    if (nview.id){
      const n = byId(g, nview.id), fi = nmodel.cols[0].shape[0], fo = nmodel.cols[1].shape[0];
      h = `<h2>Dense <small>${esc(n.name)}</small></h2>
        ${dimsRows([['Inputs', fi.toLocaleString()], ['Neurons', fo.toLocaleString()], ['Weights', (fi*fo).toLocaleString()], ['Biases', n.props.bias ? fo.toLocaleString() : '0']])}`;
    } else {
      const gn = path.length ? byId(parentGraph(), path[path.length-1]) : null;
      h = `<h2>${gn ? esc(gn.name) : 'Network'} <small>neurons</small></h2>
        ${dimsRows([['Parameters', (gn ? (gn._r && gn._r.params) || 0 : stats.params).toLocaleString()]])}
        <ul class="layers">${nmodel.cols.map(c => `<li><span>${c.via ? esc(c.name) : c.label}${c.tags.length ? ' · ' + c.tags.map(t => t.label).join(', ') : ''}</span><code>${fmt(c.shape)}</code></li>`).join('')}</ul>`;
    }
    h += `<div class="acts"><button class="tb" data-act="layers">${nview.id ? 'Back to layers' : 'Edit layers'}</button></div>`;
  } else if (selWire && selWire.startsWith('ptr:')){
    const p = byId(g, selWire.slice(4)), t = p && ptrTarget(p, g);
    if (!t){ selWire = null; return renderInspector(); }
    h = `<h2>Pointer link</h2><p class="lead">${esc(p.name)} feeds ${esc(t.name)}, with the same weights; its Pointer out gives the result. ${esc(t.name)}'s own input stays as it is.</p>
      <div class="acts"><button class="tb" data-act="delwire">Delete link</button></div>`;
  } else if (selWire){
    const e = g.edges.find(e => e.id === selWire);
    if (!e){ selWire = null; return renderInspector(); }
    const a = byId(g, e.from), b = byId(g, e.to);
    h = `<h2>Connection</h2><p class="lead">${esc(a.name)} → ${esc(b.name)}${DEFS[b.type].ins>1?` (input ${'ab'[e.port]})`:''}</p>
      ${dimsRows([['Tensor', fmt(a._out)]])}
      <div class="acts">${e.bend ? '<button class="tb" data-act="unbend">Straighten</button>' : ''}<button class="tb" data-act="delwire">Delete connection</button></div>`;
  } else if (sel.size > 1){
    const ns = [...sel].map(id => byId(g,id)).filter(Boolean);
    const p = ns.reduce((s,n) => s + ((n._r && n._r.params) || 0), 0);
    h = `<h2>${ns.length} layers selected</h2>
      ${dimsRows([['Parameters', p.toLocaleString()]])}
      <div class="acts"><button class="tb primary" data-act="group">Group into block</button><button class="tb" data-act="delete">Delete</button></div>`;
  } else if (sel.size === 1 && byId(g, [...sel][0]) && PLOTS.has(byId(g, [...sel][0]).type)){
    h = plotPanel(byId(g, [...sel][0]));
  } else if (sel.size === 1 && byId(g, [...sel][0]) && byId(g, [...sel][0]).type === 'sliders'){
    h = slidersPanel(byId(g, [...sel][0]));
  } else if (sel.size === 1 && byId(g, [...sel][0]) && byId(g, [...sel][0]).type === 'data'){
    h = dataPanel(byId(g, [...sel][0]));
  } else if (sel.size === 1 && byId(g, [...sel][0]) && byId(g, [...sel][0]).type === 'loss'){
    h = lossPanel(byId(g, [...sel][0]));
  } else if (sel.size === 1 && byId(g, [...sel][0])){
    const n = byId(g, [...sel][0]), d = DEFS[n.type], r = n._r || {};
    const isG = n.type === 'group', fixed = n.type==='gin' || n.type==='gout';
    const inRows = d.ins === 0 ? [] : d.ins === 1 ? [['Input dimension', fmt(n._in[0])]] : [['Input a', fmt(n._in[0])], ['Input b', fmt(n._in[1])]];
    const outRows = d.outs ? [['Output dimension', fmt(n._out)]] : [];
    h = `<h2>${isG ? esc(n.name) : labelOf(n)} <small>${isG ? 'block' : fixed ? '' : esc(n.name)}</small></h2>
      ${dimsRows([...inRows, ...outRows, ...(r.params || isG || d.props ? [['Parameters', (r.params||0).toLocaleString()]] : [])])}
      ${r.err ? `<p class="err">${esc(r.err)}</p>` : r.warn ? `<p class="warn">${esc(r.warn)}</p>` : ''}${actPanel(n)}`;
    if (n.type === 'gin'){ const blk = curBlock(); if (blk) h += paramsHTML(blk); }
    else if (fixed) h += `<p class="lead">Whatever reaches here becomes the block's output.</p>`;
    else h += `<label class="f"><span>${isG ? 'Block name' : 'Name'}</span><input type="text" data-k="name" value="${esc(n.name)}" autocomplete="off" spellcheck="false"></label>`;
    if (n.type !== 'gout') h += inputCells(n);
    for (const k of Object.keys(d.props || {})) h += propField(n, k);
    if (n.type === 'pca') h += pcaHTML(n);
    if (n._vars && n._vars.length) h += varsHTML(n);
    if (isG){
      h += `<ul class="layers">${topo(n.graph).filter(m => m.type!=='gin' && m.type!=='gout').map(m => `<li><span>${m.type==='group'?esc(m.name):labelOf(m)}</span><code>${fmt(m._out)}</code></li>`).join('')}</ul>
        <div class="acts"><button class="tb primary" data-act="open">Open block</button>${viewBtn(n)}<button class="tb" data-act="ungroup">Ungroup</button><button class="tb" data-act="savelib">Save to my blocks</button><button class="tb" data-act="delete">Delete</button></div>`;
    } else if (!fixed) h += `<div class="acts">${viewBtn(n)}${n.type==='dense' ? `<button class="tb" data-act="neurons">View neurons</button>` : ''}<button class="tb" data-act="dup">Duplicate</button><button class="tb" data-act="delete">Delete</button></div>`;
  } else {
    sel.clear();
    if (path.length){
      const gn = byId(parentGraph(), path[path.length-1]);
      h = `<h2>${esc(gn.name)} <small>block</small></h2>
        ${dimsRows([['Input dimension', fmt(gn._in[0])], ['Output dimension', fmt(gn._out)], ['Parameters', ((gn._r&&gn._r.params)||0).toLocaleString()]])}
        <div class="acts"><button class="tb" data-act="up">Back to ${path.length>1?'parent block':'network'}</button></div>`;
    } else {
      const hasOut = root.nodes.some(n => n.type==='output');
      h = `<h2>Network</h2>
        ${dimsRows([['Input', stats.ins.map(fmt).join(', ') || '—'], ['Output', stats.outs.map(fmt).join(', ') || '—'], ['Parameters', stats.params.toLocaleString()], ['Issues', String(stats.errs)]])}
        ${!hasOut ? `<p class="warn">Add an Output layer to mark what the network returns.</p>` : ''}`;
      const on = root.nodes.find(n => n.type === 'output' && isPred(n._act));
      if (on) h += `<div class="actp"><span class="cap">Prediction</span>${predHTML(on._act, true)}</div>`;
    }
  }
  insp.innerHTML = h; paintViz(insp);
  if (fk){ const ne = insp.querySelector(`[data-k="${fk}"]`); if (ne){ ne.focus(); try { if (ss != null) ne.setSelectionRange(ss, se); } catch(e){} } }
  if (ac.open){ const el = exprField(); if (el) acUpdate(el); else acHide(); }
}
