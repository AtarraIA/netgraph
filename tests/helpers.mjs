// Shared test helpers: load the engine (the same files the page uses) into a fresh VM context, build graphs,
// make a small synthetic dataset, read safetensors files, and check gradients numerically.
import vm from 'node:vm';
import fs from 'node:fs';
import {ENGINE} from '../src/sources.mjs';

export function loadEngine(){
  const ctx = vm.createContext({console, performance, setTimeout, TextEncoder, TextDecoder, structuredClone});
  for (const f of ENGINE) vm.runInContext(fs.readFileSync(new URL(`../src/${f}`, import.meta.url), 'utf8'), ctx, {filename:f});
  const E = code => vm.runInContext(code, ctx);
  E.ctx = ctx;
  // Seeded randomness (the VM has its own Math): ids, fresh weights, shuffles and dropout repeat run to run, so a
  // numerical gradient check can't land on a ReLU or max-pool switch point by bad luck on some runs.
  E('Math.random = rng(20261010)');
  // Small graph-building helpers, evaluated inside the engine's context.
  E(`var P = (t, p, name) => { const n = mk(t, 0, 0, p || {}, name); root.nodes.push(n); return n; };
     var wire = (...ns) => { for (let i = 0; i < ns.length - 1; i++) root.edges.push(edge(ns[i].id, ns[i + 1].id)); };
     var fresh = () => { root = {nodes:[], edges:[]}; wcache.clear(); };`);
  return E;
}

// A tiny dataset whose class is written into the image (a bright square in one of `classes` positions, plus
// noise), so a network can learn it in a few hundred steps.
export function toyDataset(E, {n = 240, h = 8, w = 8, classes = 3, seed = 7} = {}){
  E(`(() => {
    const r = rng(${seed}), n = ${n}, h = ${h}, w = ${w}, C = ${classes}, images = new Uint8Array(n*h*w), labels = new Uint8Array(n);
    for (let i = 0; i < n; i++){
      const c = i % C; labels[i] = c;
      for (let k = 0; k < h*w; k++) images[i*h*w + k] = Math.floor(r()*60);
      const x0 = Math.floor(c*w/C);
      for (let y = 2; y < 5; y++) for (let x = x0; x < Math.min(w, x0 + 2); x++) images[i*h*w + y*w + x] = 200 + Math.floor(r()*55);
    }
    datasets = [{id:'toy', name:'Toy', h, w, classes:C, n, images, labels}]; splitCache.clear();
  })()`);
}

export function readSafetensors(buf){
  const len = Number(new DataView(buf.buffer, buf.byteOffset).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + len))), out = {};
  for (const [k, v] of Object.entries(header)){
    if (k === '__metadata__') continue;
    const [a, b] = v.data_offsets, bytes = new Uint8Array(buf.subarray(8 + len + a, 8 + len + b));   // a copy (a Buffer slice is a view into a shared pool)
    out[k] = {shape:v.shape, dtype:v.dtype, data: v.dtype === 'F32' ? new Float32Array(bytes.buffer) : v.dtype === 'I64' ? new BigInt64Array(bytes.buffer) : bytes};
  }
  return out;
}

export const maxRel = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])/Math.max(1e-3, Math.abs(a[i]) + Math.abs(b[i]))); return m; };
export const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

// Compares the engine's analytic gradients with central differences on the weights with the largest gradients.
// BatchNorm uses running statistics unless `train` asks for batch statistics; dropout must be off (p = 0).
export function gradCheck(E, lossName, idxs, {train = false, per = 4, h = 1e-3} = {}){
  return JSON.parse(E(`(() => {
    infer();
    const L = (function find(g){ for (const n of g.nodes){ if (n.name === ${JSON.stringify(lossName)} && n.type === 'loss') return n; const m = n.graph && find(n.graph); if (m) return m; } return null; })(root), ctx = makeCtx(L.id), idxs = ${JSON.stringify(idxs)};
    const run = () => runBatch({...ctx, pg:new Map()}, idxs, ${train}, false).L;
    ctx.pg = new Map();
    const r0 = runBatch(ctx, idxs, ${train}, true);   // analytic gradients
    let worst = 0, checks = 0, sets = 0;
    for (const [Pm, G] of ctx.pg){
      sets++;
      for (const k of ['W', 'B']){
        const g = G[k]; if (!g.length) continue;
        const order = [...g.keys()].sort((a, b) => Math.abs(g[b]) - Math.abs(g[a])).slice(0, ${per});
        for (const i of order){
          if (Math.abs(g[i]) < 1e-6) continue;
          const old = Pm[k][i];
          Pm[k][i] = old + ${h}; const a = run();
          Pm[k][i] = old - ${h}; const b = run();
          Pm[k][i] = old;
          const num = (a - b)/(2*${h});
          worst = Math.max(worst, Math.abs(num - g[i])/Math.max(1e-4, Math.abs(num) + Math.abs(g[i]))); checks++;
        }
      }
    }
    return JSON.stringify({worst, checks, sets, L:r0.L});
  })()`));
}
