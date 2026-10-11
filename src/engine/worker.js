/* Netgraph engine: the training worker.
   The page starts a Web Worker from this same engine code, so training runs off the main thread. Messages in:
   {type:'dataset', ds} once per dataset; {type:'train', root, weights, lossId, opts}; {type:'evaluate', root,
   weights, lossId}; {type:'collect', root, weights, srcId}; {type:'stop'}. Messages out: 'progress' and 'epoch'
   (with T and the weights so far), 'done', 'evaluated', 'cprogress', 'collected' and 'error'. This block only runs inside a worker. */
if (typeof document === 'undefined' && typeof importScripts === 'function' && typeof self !== 'undefined'){
  let stopFlag = false;
  const hooks = {
    yield: () => new Promise(r => setTimeout(r, 0)),   // lets a 'stop' message in
    shouldStop: () => stopFlag,
    progress: T => self.postMessage({type:'progress', T, weights:exportWeights()}),
    epochDone: T => self.postMessage({type:'epoch', T, weights:exportWeights()}),
  };
  self.onmessage = async e => {
    const m = e.data;
    if (m.type === 'stop'){ stopFlag = true; return; }
    if (m.type === 'dataset'){ datasets = datasets.filter(d => d.id !== m.ds.id).concat(m.ds); splitCache.clear(); return; }
    root = m.root; wcache.clear(); importWeights(m.weights); stopFlag = false;
    try {
      if (m.type === 'train'){ const T = await trainLoop(m.lossId, m.opts, hooks); self.postMessage({type:'done', T, weights:exportWeights()}); }
      else if (m.type === 'collect'){
        const caps = await collectLoop(m.srcId, {...hooks, progress:(done, N) => self.postMessage({type:'cprogress', done, N})});
        self.postMessage({type:'collected', caps}, Object.values(caps).flatMap(c => [c.x.buffer, c.labels.buffer]));
      }
      else if (m.type === 'evaluate'){ const ctx = makeCtx(m.lossId); self.postMessage({type:'evaluated', r:await evaluateSet(ctx, ctx.sp.test, Infinity, hooks)}); }
    } catch(err){ self.postMessage({type:'error', message:err.message}); }
  };
}
