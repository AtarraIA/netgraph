
/* ---------- neuron view ---------- */
const NR = 8, NGAP = 26, CGAP = 230, NMAX = 12;
const ng = $('#ng');
function walkChain(g, n, cols, spans){
  const seen = new Set();
  while (true){
    if (seen.has(n.id)) return false; seen.add(n.id);
    // Side branches into Output or Loss don't break the chain; follow the one path that continues.
    const outs = g.edges.filter(e => e.from === n.id && !['output', 'loss'].includes((byId(g, e.to) || {}).type));
    if (!outs.length || n.type === 'gout' || n.type === 'output') return true;
    if (outs.length > 1) return false;
    const m = byId(g, outs[0].to);
    if (!m || DEFS[m.type].ins > 1) return false;
    if (m.type === 'gout' || m.type === 'output') return true;
    if (!m._out) return false;
    if (m.type === 'group'){
      const gin = m.graph.nodes.find(x => x.type === 'gin'), c0 = cols.length;
      if (!gin || !walkChain(m.graph, gin, cols, spans)) return false;
      if (cols.length > c0) spans.push({a:c0, b:cols.length-1, name:m.name});
    } else if (ELEM.has(m.type) && !(m.type === 'flatten' && m._in[0].length > 1)){
      cols[cols.length-1].tags.push({t:m.type, label: m.type === 'dropout' ? `Dropout ${m.props.p}` : labelOf(m)});
      cols[cols.length-1].t = m._act;
    } else cols.push({shape:m._out, name:m.name, label:labelOf(m), via:m.type, fam:DEFS[m.type].fam, params:(m._r && m._r.params) || 0, tags:[], t:m._act});
    n = m;
  }
}
function neuronModel(g, onlyId){
  if (onlyId){
    const n = byId(g, onlyId);
    if (!n || n.type !== 'dense' || !n._in || !n._in[0] || !n._out) return null;
    return {cols:[{shape:[lastDim(n._in[0])], name:'in', label:'Inputs', via:null, tags:[], t:n._actIn && n._actIn[0]},
      {shape:[lastDim(n._out)], name:n.name, label:'Dense', via:'dense', fam:'dense', params:n._r.params, tags:[], t:n._act}], spans:[]};
  }
  const starts = g.nodes.filter(n => n.type === 'gin' || n.type === 'input' || n.type === 'data' || n._src);
  if (starts.length !== 1 || !starts[0]._out) return null;
  const st = starts[0], cols = [], spans = [];
  if (st._src){
    cols.push({shape:[lastDim(st._in[0])], name:'in', label:'Inputs', via:null, tags:[]});
    cols.push({shape:st._out, name:st.name, label:'Dense', via:'dense', fam:'dense', params:st._r.params || 0, tags:[], t:st._act});
  } else cols.push({shape:st._out, name: st.type === 'gin' ? 'in' : st.name, label: st.type === 'gin' ? 'Block input' : st.type === 'data' ? 'Data' : 'Input', via:null, tags:[], t:st._act});
  if (!walkChain(g, st, cols, spans) || cols.length < 2) return null;
  return {cols, spans};
}
const rowsOf = n => n <= NMAX ? [...Array(n).keys()] : [0,1,2,3,4,5,null,n-2,n-1];
function renderNeuro(){
  nmodel = nview ? neuronModel(cur(), nview.id) : null;
  if (nview && !nmodel) nview = null;
  cv.classList.toggle('neuro', !!nview);
  if (!nview){ ng.innerHTML = ''; nbounds = null; return; }
  const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
  const cols = nmodel.cols.map((c, ci) => {
    const x = ci * CGAP, s = c.shape;
    if (s.length === 1){
      const rows = rowsOf(s[0]), y0 = -(rows.length-1)/2*NGAP;
      return {...c, x, vec:true, n:s[0], rows, ys:rows.map((_, k) => y0 + k*NGAP), top:y0-NR, bot:-y0+NR, left:x-NR, right:x+NR};
    }
    const hh = s[s.length-2], ww = s[s.length-1], ch = s.slice(0,-2).reduce((a,b) => a*b, 1);
    const hp = clamp(24+hh*2.2, 30, 120), wp = clamp(24+ww*2.2, 30, 120), L = Math.min(6, ch), off = 7;
    return {...c, x, vec:false, hp, wp, L, off, top:-hp/2-(L-1)*off, bot:hp/2, left:x-wp/2, right:x+wp/2+(L-1)*off};
  });
  let lines = '', shapes = '', labels = '';
  for (let ci = 1; ci < cols.length; ci++){
    const A = cols[ci-1], B = cols[ci];
    if (A.vec && B.vec){
      const mesh = B.via === 'dense' || A.n !== B.n;
      A.rows.forEach((ra, a) => { if (ra === null) return; B.rows.forEach((rb, b) => { if (rb === null || (!mesh && a !== b)) return;
        lines += `<line class="nl" x1="${A.x}" y1="${A.ys[a]}" x2="${B.x}" y2="${B.ys[b]}" data-a="${ci-1}:${a}" data-b="${ci}:${b}"/>`; }); });
    } else if (!A.vec && B.vec){
      const span = Math.max(1, B.bot - B.top);
      B.rows.forEach((rb, b) => { if (rb === null) return; const y = clamp(B.ys[b]*A.hp/span, -A.hp/2, A.hp/2);
        lines += `<line class="nl" x1="${A.x+A.wp/2}" y1="${y}" x2="${B.x}" y2="${B.ys[b]}" data-b="${ci}:${b}"/>`; });
    } else if (!A.vec && !B.vec){
      for (const [sx, sy] of [[-1,-1],[1,-1],[-1,1],[1,1]])
        lines += `<line class="nl frus" x1="${A.x+sx*A.wp/2}" y1="${sy*A.hp/2}" x2="${B.x+sx*B.wp/2}" y2="${sy*B.hp/2}"/>`;
    }
    if (B.params) labels += `<text class="nw" x="${(A.x+B.x)/2}" y="${Math.max(A.bot, B.bot)+24}">${fmtN(B.params)} params</text>`;
  }
  cols.forEach((c, ci) => {
    const act = c.tags.some(t => ACT.has(t.t));
    const vals = c.vec && c.t && c.t.data.length === c.n ? c.t.data : null, vm = vals ? amax(vals) : 1;
    if (c.vec) c.rows.forEach((r, k) => {
      const fill = vals && r !== null ? ` style="fill:${divCSS(vals[r]/vm)}"` : '';
      shapes += r === null ? `<text class="ndots" x="${c.x}" y="${c.ys[k]+5}">⋮</text>`
        : `<circle class="nu${act?' act':''}" cx="${c.x}" cy="${c.ys[k]}" r="${NR}" data-k="${ci}:${k}"${fill}><title>${esc(c.via ? c.name : c.label)}[${r}] · neuron ${r} of 0–${(c.n - 1).toLocaleString()}${vals ? ' = ' + vals[r].toFixed(3) : ''}</title></circle>`;
    });
    else for (let i = c.L-1; i >= 0; i--){
      const x = c.x-c.wp/2+i*c.off, y = -c.hp/2-i*c.off, dim = `x="${x}" y="${y}" width="${c.wp}" height="${c.hp}"`;
      shapes += `<rect class="nmap" ${dim} rx="2"/>`;
      if (c.t && c.t.shape.length === 3 && i < c.t.shape[0]) shapes += `<foreignObject ${dim}>${pic(c.label === 'Data' ? grayURL(c.t, i) : mapURL(c.t, i), 'nimg')}</foreignObject>`;
      if (i === 0) shapes += `<rect class="nmapo" ${dim} rx="2"/>`;
    }
    if (vals && ci === cols.length-1 && c.n <= NMAX && c.t.label !== undefined){
      const am = argmax(vals);
      c.rows.forEach((r, k) => { shapes += `<text class="nlab${r === am ? ' hi' : ''}" x="${c.x+NR+10}" y="${c.ys[k]+4}">${r}${r === c.t.label ? '  ← label' : ''}</text>`; });
    }
    labels += `<text class="nkind" x="${c.x}" y="${c.top-34}" fill="${FAM[c.fam || 'io']}">${esc(c.label.toUpperCase())}</text>`;
    if (c.via) labels += `<text class="nname" x="${c.x}" y="${c.top-17}">${esc(c.name)}</text>`;
    labels += `<text class="ncount" x="${c.x}" y="${c.bot+24}">${c.vec ? c.n.toLocaleString() : fmt(c.shape)}</text>`;
    if (c.tags.length) labels += `<text class="ntag" x="${c.x}" y="${c.bot+42}">${esc(c.tags.map(t => t.label).join(' · '))}</text>`;
  });
  const top = Math.min(...cols.map(c => c.top)) - 60;
  let ymin = top + 10;
  for (const sp of nmodel.spans){
    const depth = nmodel.spans.filter(o => o !== sp && o.a >= sp.a && o.b <= sp.b).length, y = top - depth*24;
    const x0 = cols[sp.a].left - 34, x1 = cols[sp.b].right + 34;
    labels += `<path class="nbr" d="M${x0},${y+7}V${y}H${x1}V${y+7}"/><text class="nbrt" x="${(x0+x1)/2}" y="${y-8}">${esc(sp.name.toUpperCase())}</text>`;
    ymin = Math.min(ymin, y - 24);
  }
  ng.innerHTML = `<g>${lines}</g><g>${shapes}</g><g>${labels}</g>`; paintViz(ng);
  nbounds = {x0:Math.min(...cols.map(c => c.left)) - 70, x1:Math.max(...cols.map(c => c.right)) + 70, y0:ymin - 20, y1:Math.max(...cols.map(c => c.bot)) + 60};
}
function hot(k){
  ng.querySelectorAll('.hot').forEach(x => x.classList.remove('hot'));
  ng.classList.toggle('focus', !!k);
  if (!k) return;
  ng.querySelector(`.nu[data-k="${k}"]`)?.classList.add('hot');
  ng.querySelectorAll(`[data-a="${k}"],[data-b="${k}"]`).forEach(x => x.classList.add('hot'));
}
$('#neuro').addEventListener('pointerover', e => { const c = e.target.closest('.nu'); if (c) hot(c.dataset.k); });
$('#neuro').addEventListener('pointerout', e => { if (e.target.closest('.nu')) hot(null); });
