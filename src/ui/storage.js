function idb(){
  return idb.p ||= new Promise((res, rej) => {
    const r = indexedDB.open('netgraph', 2);
    r.onupgradeneeded = () => { for (const st of ['datasets', 'weights']) if (!r.result.objectStoreNames.contains(st)) r.result.createObjectStore(st, {keyPath:'id'}); };
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
async function dbTx(mode, fn, store='datasets'){
  const db = await idb();
  return new Promise((res, rej) => { const t = db.transaction(store, mode), q = fn(t.objectStore(store)); t.oncomplete = () => res(q && q.result); t.onerror = () => rej(t.error); });
}
async function loadDatasets(){
  try { datasets = (await dbTx('readonly', st => st.getAll())) || []; } catch(e){ datasets = []; }
  // Datasets stored by earlier versions were kept as train/test splits; merge them into one pool.
  datasets = datasets.map(d => d.splits ? joinParts(d, ['train', 'test'].concat(Object.keys(d.splits)).filter((k, i, a) => d.splits[k] && a.indexOf(k) === i).map(k => d.splits[k])) : d);
}
async function readIdx(file){
  let buf = await file.arrayBuffer();
  const u = new Uint8Array(buf, 0, 2);
  if (u[0] === 0x1f && u[1] === 0x8b) buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const dv = new DataView(buf), magic = dv.getUint32(0), nd = magic & 0xff;
  if ((magic >>> 8) !== 0x08) throw new Error(`${file.name} isn't an IDX file of unsigned bytes.`);
  const dims = []; for (let i = 0; i < nd; i++) dims.push(dv.getUint32(4 + 4*i));
  return {dims, data:new Uint8Array(buf, 4 + 4*nd)};
}
async function importFiles(list){
  const files = [...list];
  if (!files.length) return;
  toast(`Reading ${files.length} file${files.length>1?'s':''}…`);
  // Pair each images file with its labels file by name, then pool everything into one dataset.
  const parts = {};
  try {
    for (const f of files){
      const key = f.name.toLowerCase().replace(/\.gz$/, '').replace(/images|labels|idx\d/g, ''), r = await readIdx(f);
      parts[key] ||= {key};
      if (r.dims.length === 3) parts[key].img = r; else if (r.dims.length === 1) parts[key].lab = r;
    }
  } catch(err){ toast(err.message || 'Could not read those files.'); return; }
  const pairs = Object.values(parts).filter(p => p.img && p.lab).sort((x, y) => (/train/.test(y.key) - /train/.test(x.key)) || x.key.localeCompare(y.key));
  if (!pairs.length){ toast('Pick an images file together with its labels file, like train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz.'); return; }
  const [, h, w] = pairs[0].img.dims;
  for (const p of pairs){
    if (p.img.dims[1] !== h || p.img.dims[2] !== w){ toast('Those files have different image sizes.'); return; }
    if (p.lab.dims[0] !== p.img.dims[0]){ toast('An images file and its labels file have different counts.'); return; }
  }
  let classes = 0; for (const p of pairs) for (const v of p.lab.data) if (v + 1 > classes) classes = v + 1;
  const name = h === 28 && w === 28 && classes === 10 ? 'MNIST' : `Dataset ${h}×${w}`;
  const id = name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const ds = joinParts({id, name, h, w, classes}, pairs.map(p => ({n:p.img.dims[0], images:p.img.data, labels:p.lab.data})));
  datasets = datasets.filter(d => d.id !== id).concat(ds);
  let kept = true; try { await dbTx('readwrite', st => st.put(ds)); } catch(e){ kept = false; }
  (function claim(g){ for (const n of g.nodes){ if (n.type === 'data' && !n._collect && !dsById(n.props.dataset)) n.props.dataset = id; if (n.graph) claim(n.graph); } })(root);
  render();
  toast(`${name} ready: ${ds.n.toLocaleString()} samples.${kept ? '' : ' (Your browser won\'t store it, so import again after reloading.)'}`);
}
async function autoFetch(){
  if (!/^https?:$/.test(location.protocol) || dsById('mnist')) return;
  const names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'];
  try {
    const files = await Promise.all(names.map(async nm => { const r = await fetch('data/mnist/' + nm); if (!r.ok) throw new Error(nm); return new File([await r.blob()], nm); }));
    await importFiles(files);
  } catch(e){}
}
function stepSample(n, how){
  const s = sampleOf(n); if (!s) return;
  n.props.index = how === 'rand' ? Math.floor(Math.random()*s.ds.n) : (s.i + (how === 'next' ? 1 : -1) + s.ds.n) % s.ds.n;
  soft();
}
async function saveWeights(){   // trained weights, with BatchNorm's running statistics
  const recs = exportWeights(true);
  try { await dbTx('readwrite', st => { recs.forEach(r => st.put(r)); return null; }, 'weights'); } catch(e){}
}
async function loadWeights(){
  try { importWeights((await dbTx('readonly', st => st.getAll(), 'weights')) || []); } catch(e){}
}
