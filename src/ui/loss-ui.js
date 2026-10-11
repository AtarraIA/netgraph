const pctTxt = v => v == null ? '—' : `${(100*v).toFixed(1)}%`;
function sparkSVG(h, w=176, ht=30){
  if (h.length < 2) return '';
  const step = Math.max(1, Math.ceil(h.length/200)), v = [];
  for (let i = 0; i < h.length; i += step) v.push(h[i]);
  v.push(h[h.length-1]);
  const [lo, hi] = range(v), pts = v.map((y, i) => `${(i/(v.length-1)*w).toFixed(1)},${(ht - 2 - (hi > lo ? (y - lo)/(hi - lo) : .5)*(ht - 4)).toFixed(1)}`).join(' ');
  return `<svg class="spark" viewBox="0 0 ${w} ${ht}" preserveAspectRatio="none" role="img" aria-label="Training loss, from ${v[0].toFixed(3)} to ${v[v.length-1].toFixed(3)}"><polyline points="${pts}"/></svg>`;
}
function lossBody(n, compact){
  const T = trainLog.get(n.id), on = isTraining(n), bad = n._r && n._r.err;
  let h = `<div class="trn"><button class="tb${on ? '' : ' primary'} trb" data-train="${n.id}"${bad ? ' disabled' : ''}>${on ? '■ Stop' : '▶ Train'}</button><span class="tl">${n._lossNow != null ? `loss ${n._lossNow.toFixed(3)}` : ''}</span></div>`;
  if (T && T.steps){
    const frac = Math.min(1, (T.epoch*T.steps + T.step)/(T.epochs*T.steps));
    h += `<div class="tp"><i style="width:${(100*frac).toFixed(1)}%"></i></div>
      <div class="ts">${on ? (T.evaluating ? `Epoch ${T.epoch+1}/${T.epochs} · testing…` : `Epoch ${T.epoch+1}/${T.epochs} · ${Math.round(100*frac)}%`) : T.done ? 'Done' : 'Stopped'}</div>${compact ? '' : sparkSVG(T.hist)}`;
  }
  if (T && (T.trainAcc != null || T.testAcc != null)) h += `<div class="ts">train <b>${pctTxt(T.trainAcc)}</b> · test <b>${T.evaluating && !on ? '…' : pctTxt(T.testAcc)}</b></div>`;
  else if (T && (T.ema != null || T.testLoss != null)) h += `<div class="ts">train <b>${T.ema != null ? T.ema.toFixed(4) : '—'}</b> · test <b>${T.evaluating && !on ? '…' : T.testLoss != null ? T.testLoss.toFixed(4) : '—'}</b></div>`;
  return h;
}
/* Loss chart for a Loss block in view mode: the whole run at once. Drag pans, scrolling zooms
   around the pointer, double-click shows everything again. The vertical scale fits what's visible. */
const chartView = new Map();
   // loss id → {x0, x1} in training steps; absent = the whole run
const CW = 300, CH = 140, CP = {l:40, r:8, t:14, b:18};
const fmtL = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
function chartHTML(n){ return `<div class="lchart" data-chart="${n.id}" title="Select the Loss block, then drag to pan · scroll to zoom · double-click to see it all">${chartSVG(n)}</div>`; }
function chartSVG(n){
  const T = trainLog.get(n.id);
  if (!T || !T.hist || T.hist.length < 2) return `<svg class="lc" viewBox="0 0 ${CW} ${CH}" role="img" aria-label="Loss chart, empty"><text class="lc-empty" x="${CW/2}" y="${CH/2 - 4}">The loss curve appears here</text><text class="lc-empty" x="${CW/2}" y="${CH/2 + 12}">${n._lossNow != null ? `this sample: ${n._lossNow.toFixed(4)}` : 'while the network trains'}</text></svg>`;
  const N = T.hist.length, raw = T.raw && T.raw.length === N ? T.raw : T.hist, v = chartView.get(n.id);
  const x0 = v ? Math.max(0, v.x0) : 0, x1 = v ? Math.min(N - 1, v.x1) : N - 1, i0 = Math.floor(x0), i1 = Math.ceil(x1);
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i <= i1; i++){ lo = Math.min(lo, raw[i], T.hist[i]); hi = Math.max(hi, raw[i], T.hist[i]); }
  for (const t of T.tests || []) if (t.step >= i0 && t.step <= i1){ lo = Math.min(lo, t.L); hi = Math.max(hi, t.L); }
  if (!(hi > lo)) hi = lo + 1e-3;
  const pad = (hi - lo)*0.08; lo = lo >= 0 ? Math.max(0, lo - pad) : lo - pad; hi += pad;
  const W2 = CW - CP.l - CP.r, H2 = CH - CP.t - CP.b, X = i => CP.l + (i - x0)/((x1 - x0) || 1)*W2, Y = y => CP.t + (1 - (y - lo)/(hi - lo))*H2;
  const trace = (arr, band) => {   // at most a couple of points per pixel column; the raw losses become a min–max band
    const cols = Math.ceil(W2), per = (i1 - i0 + 1)/cols, pts = [];
    if (per <= 1) for (let i = i0; i <= i1; i++) pts.push(`${X(i).toFixed(1)},${Y(arr[i]).toFixed(1)}`);
    else for (let c = 0; c < cols; c++){
      const a = Math.floor(i0 + c*per), b = Math.min(i1, Math.floor(i0 + (c + 1)*per) - 1); if (b < a) continue;
      let mn = Infinity, mx = -Infinity; for (let i = a; i <= b; i++){ mn = Math.min(mn, arr[i]); mx = Math.max(mx, arr[i]); }
      const x = X((a + b)/2).toFixed(1);
      if (band) pts.push(`${x},${Y(mx).toFixed(1)}`, `${x},${Y(mn).toFixed(1)}`); else pts.push(`${x},${Y(arr[b]).toFixed(1)}`);
    }
    return pts.join(' ');
  };
  let g = '';
  for (const f of [0, .5, 1]){ const y = CP.t + f*H2, val = hi - f*(hi - lo); g += `<line class="lc-grid" x1="${CP.l}" x2="${CW - CP.r}" y1="${y}" y2="${y}"/><text class="lc-ax" x="${CP.l - 4}" y="${y + 3}" text-anchor="end">${fmtL(val)}</text>`; }
  if (T.steps) for (let k = 1; k*T.steps < N; k++){ const s = k*T.steps; if (s < x0 || s > x1) continue; g += `<line class="lc-ep" x1="${X(s)}" x2="${X(s)}" y1="${CP.t}" y2="${CP.t + H2}"/><text class="lc-ax" x="${X(s) + 3}" y="${CP.t + 8}">epoch ${k + 1}</text>`; }
  g += `<text class="lc-ax" x="${CP.l}" y="${CH - 4}">step ${Math.round(x0).toLocaleString()}</text><text class="lc-ax" x="${CW - CP.r}" y="${CH - 4}" text-anchor="end">${Math.round(x1).toLocaleString()}${v ? '' : ' (all)'}</text>`;
  let tests = ''; for (const t of T.tests || []) if (t.step >= i0 && t.step <= i1) tests += `<circle class="lc-test" cx="${X(t.step)}" cy="${Y(t.L)}" r="3"><title>Test loss ${t.L.toFixed(4)}</title></circle>`;
  const last = T.hist[N - 1];
  return `<svg class="lc" viewBox="0 0 ${CW} ${CH}" role="img" aria-label="Training loss over ${N.toLocaleString()} steps, now ${last.toFixed(4)}">
    <defs><clipPath id="lcc-${n.id}"><rect x="${CP.l}" y="${CP.t - 2}" width="${W2}" height="${H2 + 4}"/></clipPath></defs>${g}
    <g clip-path="url(#lcc-${n.id})"><polyline class="lc-raw" points="${trace(raw, true)}"/><polyline class="lc-ema" points="${trace(T.hist)}"/>${tests}</g>
    <text class="lc-now" x="${CW - CP.r}" y="${CP.t - 3}" text-anchor="end">${last.toFixed(4)}</text>
    <text class="lc-ax" x="${CP.l}" y="${CP.t - 4}">— smoothed · ▒ per step${(T.tests || []).length ? ' · ● test' : ''}</text></svg>`;
}
function renderChart(id){ const el = nodesEl.querySelector(`.lchart[data-chart="${id}"]`), n = findNode(id); if (el && n) el.innerHTML = chartSVG(n); }
function chartSpan(id){ const T = trainLog.get(id); if (!T || !T.hist || T.hist.length < 3) return null; const N = T.hist.length, v = chartView.get(id) || {x0:0, x1:N - 1}; return {N, x0:v.x0, x1:Math.min(N - 1, v.x1)}; }
function startChartPan(e, el){
  const id = el.dataset.chart, c = chartSpan(id); if (!c) return;
  const span = c.x1 - c.x0, sx = e.clientX, px = el.getBoundingClientRect().width*(CW - CP.l - CP.r)/CW;
  track(m => { let x0 = c.x0 - (m.clientX - sx)/px*span; x0 = Math.max(0, Math.min(c.N - 1 - span, x0)); chartView.set(id, {x0, x1:x0 + span}); renderChart(id); }, () => {});
}
function chartZoom(e, el){
  const id = el.dataset.chart, c = chartSpan(id); if (!c) return;
  const r = el.getBoundingClientRect(), fx = Math.max(0, Math.min(1, ((e.clientX - r.left)/r.width*CW - CP.l)/(CW - CP.l - CP.r)));
  const at = c.x0 + fx*(c.x1 - c.x0), span = Math.max(8, Math.min(c.N - 1, (c.x1 - c.x0)*Math.exp(e.deltaY*(e.ctrlKey ? 0.01 : 0.0015))));
  if (span >= c.N - 1.5) chartView.delete(id);
  else { const x0 = Math.max(0, Math.min(c.N - 1 - span, at - fx*span)); chartView.set(id, {x0, x1:x0 + span}); }
  renderChart(id);
}
const fmtV = v => Number.isInteger(+v) ? String(+v) : (+v).toFixed(Math.min(6, Math.max(2, -Math.floor(Math.log10(Math.abs(+v) || 1)) + 2)));
function varsHTML(n){
  if (!n._vars || !n._vars.length) return '';
  return `<div class="vars"><span class="cap">Variables</span>${n._vars.map(name => {
    const v = n.props.vars[name], nm = esc(name);
    return `<div class="var">
      <div class="vh"><b>${nm}</b><output>${fmtV(v.value)}</output></div>
      <div class="vs"><button class="tb" data-vstep="${nm}" data-d="-1" aria-label="Decrease ${nm}">‹</button><input type="range" data-var="${nm}" min="${v.min}" max="${v.max}" step="${v.step}" value="${v.value}" aria-label="${nm}"><button class="tb" data-vstep="${nm}" data-d="1" aria-label="Increase ${nm}">›</button></div>
      <div class="vr"><label>min<input type="text" data-var="${nm}" data-vk="min" value="${v.min}" autocomplete="off" spellcheck="false"></label><label>step<input type="text" data-var="${nm}" data-vk="step" value="${v.step}" autocomplete="off" spellcheck="false"></label><label>max<input type="text" data-var="${nm}" data-vk="max" value="${v.max}" autocomplete="off" spellcheck="false"></label></div>
    </div>`;
  }).join('')}<small class="vnote">Names that aren't components become variables.</small></div>`;
}
function lossPanel(n){
  const r = n._r || {}, T = trainLog.get(n.id), on = isTraining(n), p = n.props;
  const txt = (k, label, hint, ph) => `<label class="f"><span>${label}</span><input type="text" data-k="${k}" value="${esc(p[k])}"${ph ? ` placeholder="${esc(ph)}"` : ''} autocomplete="off" spellcheck="false">${hint ? `<small>${hint}</small>` : ''}</label>`;
  const testTxt = !T ? '—' : T.evaluating ? 'testing…' : T.testN ? `${T.testAcc != null ? pctTxt(T.testAcc) + ', ' : ''}loss ${T.testLoss.toFixed(4)} (${T.testN.toLocaleString()})` : '—';
  let h = `<h2>Loss <small>${esc(n.name)}</small></h2>
    ${txt('expr', 'Expression', 'Training minimizes this value. Use component names, numbers, + − * / ^ and abs, log, exp, sqrt, sum, mean. <b>ce(x)</b> is cross-entropy of raw scores against the data labels, <b>nll(p)</b> the same for probabilities, <b>mse(a, b)</b> the mean squared error. A result with many values is averaged. Reach inside blocks with dots, like MLP.fc1, and pick neurons with brackets, like fc1[3] or fc1[0:10] (counting from 0). fc1[3].pre_weights are the weights into neuron 3 and fc1[3].post_weights the weights from it into the next Dense layer; without a neuron index they are whole matrices (fc1.pre_weights is neurons × inputs). Type a name or a dot for suggestions, or press Ctrl+Space. Put names with spaces in quotes, like "MLP 2".fc1.', 'ce(fc3)')}
    ${varsHTML(n)}
    ${dimsRows([['This sample', n._lossNow != null ? n._lossNow.toFixed(4) : '—'], ...(T ? [['Train', T.trainAcc != null ? pctTxt(T.trainAcc) : T.ema != null ? `loss ${T.ema.toFixed(4)}` : '—'], ['Test', testTxt]] : [])])}`;
  if (r.err) h += `<p class="err">${esc(r.err)}</p>`;
  if (T && T.hist.length > 1) h += `<div class="actp"><span class="cap">Training loss</span>${sparkSVG(T.hist, 228, 60)}</div>`;
  h += `<div class="acts"><button class="tb primary" data-act="train"${r.err ? ' disabled' : ''}>${on ? 'Stop training' : 'Train'}</button><button class="tb" data-act="evaluate"${r.err || on ? ' disabled' : ''}>Evaluate on test set</button><button class="tb" data-act="resetw"${on ? ' disabled' : ''}>Reset weights</button></div>
    <label class="f" style="margin-top:12px"><span>Optimizer</span><select data-k="optimizer">${[['adam', 'Adam'], ['sgd', 'SGD with momentum']].map(([v, l]) => `<option value="${v}"${p.optimizer === v ? ' selected' : ''}>${l}</option>`).join('')}</select></label>
    ${txt('lr', 'Learning rate', 'Adam works well around 0.001; SGD around 0.01.')}${txt('batch', 'Batch size')}${txt('epochs', 'Epochs', 'One epoch is one pass over the training split.')}
    <label class="f"><span>Name</span><input type="text" data-k="name" value="${esc(n.name)}" autocomplete="off" spellcheck="false"></label>
    <div class="acts">${viewBtn(n)}<button class="tb" data-act="delete">Delete</button></div>`;
  return h;
}
