/* ---------- training (page side) ---------- */
/* Training runs in a Web Worker made from the engine's own source (the <script id="engine"> text), so the page
   stays responsive. The worker gets a copy of the graph, the weights and (once) the dataset, and streams back
   progress with the weights so far. If a worker can't be started, the same loop runs here on the page. */
const yieldNow = () => new Promise(r => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });
let TR = null;                 // the running job: {id, running, worker?}
const trainLog = new Map();    // loss block id → progress, loss history, accuracy (the T objects from trainLoop)
const isTraining = n => !!(TR && TR.running && TR.id === n.id);
let worker = null, workerFailed = false, workerJob = null;
const workerData = new Set();  // dataset ids the worker already has
function getWorker(){
  if (worker || workerFailed) return worker;
  try {
    const src = document.getElementById('engine').textContent;
    worker = new Worker(URL.createObjectURL(new Blob([src], {type:'text/javascript'})));
    worker.onmessage = e => workerJob && workerJob(e.data);
    worker.onerror = e => { e.preventDefault(); workerFailed = true; worker = null; workerData.clear(); if (workerJob) workerJob({type:'error', message:'the training worker stopped unexpectedly.', fatal:true}); };
  } catch(e){ workerFailed = true; worker = null; }
  return worker;
}
function sendDataset(w, ds){ if (!workerData.has(ds.id)){ w.postMessage({type:'dataset', ds}); workerData.add(ds.id); } }
function toggleTrain(id){
  const n = findNode(id); if (!n) return;
  if (isTraining(n)){ TR.running = false; if (TR.worker) TR.worker.postMessage({type:'stop'}); render(); }
  else startTrain(n);
}
function trainOpts(n){
  const p = n.props, lr = Number(p.lr), bs = Math.floor(Number(p.batch)), epochs = Math.floor(Number(p.epochs));
  if (!(lr > 0)) return {err:'Learning rate must be a positive number, like 0.001.'};
  if (!(bs >= 1)) return {err:'Batch size must be a whole number of at least 1.'};
  if (!(epochs >= 1)) return {err:'Epochs must be a whole number of at least 1.'};
  return {lr, bs, epochs, optimizer:p.optimizer === 'sgd' ? 'sgd' : 'adam'};
}
// Checks everything the page can before starting; returns the dataset in use, or null after saying why not.
function trainReady(n){
  let ctx; try { ctx = makeCtx(n.id); } catch(err){ toast(err.message); return null; }
  return ctx;
}
async function startTrain(n){
  const ctx = trainReady(n); if (!ctx) return;
  if (CR && CR.running){ toast('Wait for the inference run to finish, or stop it first.'); return; }
  const opts = trainOpts(n); if (opts.err){ toast(opts.err); return; }
  if (!ctx.sp.train.length){ toast('The Data component has no training samples. Lower its test split.'); return; }
  if (TR && TR.running) toggleTrain(TR.id);
  const job = TR = {id:n.id, running:true, worker:null};
  const T0 = {id:n.id, epoch:0, epochs:opts.epochs, step:0, steps:Math.ceil(ctx.sp.train.length/opts.bs), hist:[], raw:[], tests:[], ema:null, trainAcc:null, testAcc:null, testN:0};
  trainLog.set(n.id, T0);
  render();
  const finish = async (T, err) => {
    job.running = false; if (TR === job) TR = null;
    if (T) trainLog.set(n.id, {...T, running:false});
    await saveWeights(); render();
    if (err) toast(`Training stopped: ${err}`);
    else if (T && T.done) toast(`Training finished.${T.testAcc != null ? ` Test accuracy ${(100*T.testAcc).toFixed(1)}% on ${T.testN.toLocaleString()} samples.` : T.testLoss != null ? ` Test loss ${T.testLoss.toFixed(4)} on ${T.testN.toLocaleString()} samples.` : ''}`);
  };
  const w = getWorker();
  if (w){
    job.worker = w; let lastDraw = 0;
    sendDataset(w, ctx.ds);
    workerJob = async msg => {
      if (TR !== job && msg.type !== 'done' && msg.type !== 'error') return;
      if (msg.weights) importWeights(msg.weights);
      if (msg.type === 'progress' || msg.type === 'epoch'){
        trainLog.set(n.id, msg.T);
        if (msg.type === 'epoch') saveWeights();
        if (now() - lastDraw > 250){ lastDraw = now(); soft(); }
      } else if (msg.type === 'done'){ workerJob = null; await finish(msg.T); }
      else if (msg.type === 'error'){ workerJob = null; await finish(trainLog.get(n.id), msg.message); }
    };
    w.postMessage({type:'train', root:strip(root), weights:exportWeights(), lossId:n.id, opts});
    return;
  }
  // No worker: the same loop on the page, pausing now and then so the page can draw.
  let lastDraw = 0;
  try {
    const T = await trainLoop(n.id, opts, {
      yield: yieldNow,
      shouldStop: () => !job.running,
      progress: T => { trainLog.set(n.id, T); if (now() - lastDraw > 400){ lastDraw = now(); soft(); } },
      epochDone: () => saveWeights(),
    });
    await finish(T);
  } catch(err){ await finish(trainLog.get(n.id), err.message); }
}
async function evalFull(n){
  if (TR && TR.running){ toast('Wait for training to finish, or stop it first.'); return; }
  const ctx = trainReady(n); if (!ctx) return;
  if (!ctx.sp.test.length){ toast('The Data component has no test samples. Raise its test split.'); return; }
  const T = trainLog.get(n.id) || {id:n.id, hist:[], epoch:0, epochs:0, step:0, steps:0};
  trainLog.set(n.id, T); T.evaluating = true; render();
  const done = r => { Object.assign(T, r || {}); T.evaluating = false; render(); if (T.testN) toast(`${T.testAcc != null ? `Test accuracy ${(100*T.testAcc).toFixed(1)}%, l` : 'L'}oss ${T.testLoss.toFixed(4)} on all ${T.testN.toLocaleString()} test samples.`); };
  const w = getWorker();
  if (w){
    sendDataset(w, ctx.ds);
    workerJob = msg => {
      if (msg.type === 'evaluated'){ workerJob = null; done(msg.r); }
      else if (msg.type === 'error'){ workerJob = null; T.evaluating = false; render(); toast(`Evaluation stopped: ${msg.message}`); }
    };
    w.postMessage({type:'evaluate', root:strip(root), weights:exportWeights(), lossId:n.id});
    return;
  }
  try { done(await evaluateSet(ctx, ctx.sp.test, Infinity, {yield:yieldNow, shouldStop:() => false})); }
  catch(err){ T.evaluating = false; render(); toast(`Evaluation stopped: ${err.message}`); }
}
function resetWeights(){
  if (TR && TR.running){ toast('Stop training before resetting the weights.'); return; }
  (function walk(g){ for (const m of g.nodes){ wcache.delete(m.id); if (m.graph) walk(m.graph); } })(root);
  trainLog.clear();
  dbTx('readwrite', st => st.clear(), 'weights').catch(() => {});
  render(); toast('Weights reset to fresh random values.');
}

/* ---------- inference runs (page side) ---------- */
/* A Data component sends its samples through the network; every collecting Data component the flow reaches keeps
   what arrives as a dataset of its own (stored like imported ones). Runs in the worker when there is one. */
let CR = null;   // the running inference run: {id, running, done, N, worker?}
const isCollecting = n => !!(CR && CR.running && CR.id === n.id);
function toggleCollect(id){
  const n = findNode(id); if (!n) return;
  if (isCollecting(n)){ CR.running = false; if (CR.worker) CR.worker.postMessage({type:'stop'}); render(); }
  else startCollect(n);
}
function keepCollected(src, caps, how0){
  const sds = dsById(src.props.dataset), how = how0 || {test:'test split', train:'train split'}[src.props.run_from] || 'whole dataset', names = [];
  for (const [id, c] of Object.entries(caps)){
    const m = findNode(id); if (!m) continue;
    if (PLOTS.has(m.type)){ keepPlot(m, c, sds.classes, `${src.name}: ${sds.name}, ${how}`); names.push(`${m.name} ${c.n.toLocaleString()} points`); continue; }
    const old = dsById(m.props.dataset), dsid = old && old.owner === m.id ? old.id : 'cap-' + uid();
    const ds = {id:dsid, name:`${m.name} (collected)`, shape:c.shape, classes:sds.classes, n:c.n, x:c.x, labels:c.labels, owner:m.id, from:`${src.name}: ${sds.name}, ${how}`};
    datasets = datasets.filter(d => d.id !== dsid).concat(ds); workerData.delete(dsid);
    m.props.dataset = dsid; m.props.index = Math.min(Number(m.props.index) || 0, c.n - 1);
    dbTx('readwrite', st => st.put(ds)).catch(() => {});
    names.push(`${m.name} ${c.n.toLocaleString()} × ${fmt(c.shape)}`);
  }
  return names;
}
async function startCollect(n){
  if (TR && TR.running){ toast('Wait for training to finish, or stop it first.'); return; }
  if (CR && CR.running) return;
  infer();
  let N; try { N = runPool(n).length; } catch(e){ toast(e.message); return; }
  const job = CR = {id:n.id, running:true, done:0, N, worker:null};
  render();
  const finish = (caps, err) => {
    job.running = false; if (CR === job) CR = null;
    if (caps){ const got = keepCollected(n, caps); render(); save(); toast(`Collected ${got.join(' · ')}.`); }
    else { render(); if (err) toast(`Inference stopped: ${err}`); }
  };
  const progress = (done, total) => { job.done = done; job.N = total; const el = insp.querySelector('.iprog'); if (el) el.textContent = `${done.toLocaleString()} / ${total.toLocaleString()}`; };
  const w = getWorker();
  if (w){
    job.worker = w;
    sendDataset(w, dsById(n.props.dataset));
    workerJob = msg => {
      if (msg.type === 'cprogress') progress(msg.done, msg.N);
      else if (msg.type === 'collected'){ workerJob = null; finish(msg.caps); }
      else if (msg.type === 'error'){ workerJob = null; finish(null, msg.message); }
    };
    w.postMessage({type:'collect', root:strip(root), weights:exportWeights(), srcId:n.id});
    return;
  }
  try { finish(await collectLoop(n.id, {yield:yieldNow, shouldStop:() => !job.running, progress})); }
  catch(err){ finish(null, err.message); }
}

/* ---------- PCA (page side) ---------- */
// Fits a PCA component, then runs the whole dataset through it into the Data components (and plots) after it.
async function toggleFitPca(id){
  const n = findNode(id); if (!n) return;
  if (isCollecting(n)){ CR.running = false; render(); return; }
  if (TR && TR.running){ toast('Wait for training to finish, or stop it first.'); return; }
  if (CR && CR.running) return;
  const job = CR = {id:n.id, running:true, done:0, N:0};
  render();
  const progress = (done, total) => { job.done = done; job.N = total; const el = insp.querySelector('.iprog'); if (el) el.textContent = `${done.toLocaleString()} / ${total.toLocaleString()}`; };
  const hooks = {yield:yieldNow, shouldStop:() => !job.running, progress};
  let msg;
  try {
    const r = await fitPCA(n.id, hooks), pct = (100*r.ratio.reduce((a, v) => a + v, 0)).toFixed(1);
    await saveWeights(); infer();
    msg = `Fitted ${r.k} component${r.k > 1 ? 's' : ''} on ${r.n.toLocaleString()} samples: ${pct}% of the variance.`;
    const outs = collectorsOf(n);
    if (outs.length && job.running){
      const caps = await collectLoop(r.d.id, hooks, {all:true, only:outs});
      msg += ` Projected into ${keepCollected(r.d, caps, `whole dataset, through ${n.name}`).join(' · ')}.`;
    } else if (!outs.length) msg += ' Wire its output into a Data component to keep the projected data.';
  } catch(err){ msg = `PCA stopped: ${err.message}`; }
  job.running = false; if (CR === job) CR = null;
  render(); save(); toast(msg);
}
