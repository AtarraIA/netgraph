/* Netgraph engine: project files.
   A project file is JSON: {format:'netgraph', version, saved, graph, blocks, weights}. graph is the network as the
   app stores it; blocks are the saved "My blocks"; weights are the trained layers' parameters (with BatchNorm's
   running statistics) as base64 float32. Datasets are not included, only referred to by id. */
const PROJECT_VERSION = 1;

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function toBase64(bytes){
  let s = '';
  for (let i = 0; i < bytes.length; i += 3){
    const a = bytes[i], b = bytes[i + 1], c = bytes[i + 2], n = (a << 16) | ((b ?? 0) << 8) | (c ?? 0);
    s += B64[n >> 18 & 63] + B64[n >> 12 & 63] + (b === undefined ? '=' : B64[n >> 6 & 63]) + (c === undefined ? '=' : B64[n & 63]);
  }
  return s;
}
function fromBase64(s){
  const clean = String(s).replace(/[^A-Za-z0-9+/]/g, ''), out = new Uint8Array(Math.floor(clean.length*3/4)), map = {};
  for (let i = 0; i < 64; i++) map[B64[i]] = i;
  let o = 0;
  for (let i = 0; i < clean.length; i += 4){
    const n = (map[clean[i]] << 18) | (map[clean[i + 1]] << 12) | ((map[clean[i + 2]] ?? 0) << 6) | (map[clean[i + 3]] ?? 0);
    if (o < out.length) out[o++] = n >> 16 & 255;
    if (o < out.length && i + 2 < clean.length) out[o++] = n >> 8 & 255;
    if (o < out.length && i + 3 < clean.length) out[o++] = n & 255;
  }
  return out;
}
const f32ToB64 = a => a ? toBase64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)) : null;
const b64ToF32 = s => { if (s == null) return null; const b = fromBase64(s); return new Float32Array(b.buffer, 0, b.byteLength >> 2); };

function projectJSON(blocks = []){
  const ids = new Set(); (function walk(g){ for (const n of g.nodes){ ids.add(n.id); if (n.graph) walk(n.graph); } })(root);
  const weights = exportWeights(true).filter(r => ids.has(r.id))
    .map(r => ({id:r.id, key:r.key, W:f32ToB64(r.W), B:f32ToB64(r.B), rm:f32ToB64(r.rm), rv:f32ToB64(r.rv), nbt:r.nbt}));
  return JSON.stringify({format:'netgraph', version:PROJECT_VERSION, saved:new Date().toISOString(), graph:strip(root), blocks, weights});
}

// Reads a project file. Returns {graph, blocks, weights, missing} (missing: dataset ids the graph uses that aren't
// loaded), or throws an Error saying what's wrong. Does not change anything yet.
function readProject(text){
  let p;
  try { p = JSON.parse(text); } catch(e){ throw new Error('This file isn\'t valid JSON.'); }
  if (!p || p.format !== 'netgraph') throw new Error('This isn\'t a Netgraph project file.');
  if (!(p.version >= 1)) throw new Error('This project file has no version.');
  if (p.version > PROJECT_VERSION) throw new Error(`This project file is from a newer Netgraph (version ${p.version}).`);
  const g = p.graph;
  if (!g || !Array.isArray(g.nodes) || !Array.isArray(g.edges)) throw new Error('The project file has no network in it.');
  const missing = new Set(), badType = [];
  (function check(gr){
    for (const n of gr.nodes){
      if (!n || !n.id || !DEFS[n.type]) { badType.push(n && n.type); continue; }
      n.props ||= {};
      const collects = (gr.edges || []).some(e => e.to === n.id) || (n.props.inputs && n.props.inputs[0]);   // what it collected isn't in the file
      if (n.type === 'data' && n.props.dataset && !collects && !dsById(n.props.dataset)) missing.add(n.props.dataset);
      if (n.type === 'group'){ if (!n.graph || !Array.isArray(n.graph.nodes)) throw new Error(`The block "${n.name}" has no inside.`); check(n.graph); }
    }
  })(g);
  if (badType.length) throw new Error(`Unknown component type${badType.length > 1 ? 's' : ''}: ${[...new Set(badType)].join(', ')}.`);
  const weights = (p.weights || []).map(r => ({id:r.id, key:r.key, W:b64ToF32(r.W), B:b64ToF32(r.B), rm:b64ToF32(r.rm), rv:b64ToF32(r.rv), nbt:r.nbt || 0, trained:true}));
  return {graph:g, blocks:Array.isArray(p.blocks) ? p.blocks : [], weights, missing:[...missing]};
}
