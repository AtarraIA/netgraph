// PyTorch parity: export a network from the engine, run the same batch through the exported model in real
// PyTorch, and compare outputs, loss, every parameter's gradient, BatchNorm's running statistics and the
// weights after one optimizer step. Needs python3 with torch and safetensors; skipped otherwise.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {loadEngine, toyDataset, readSafetensors} from './helpers.mjs';

const harness = fileURLToPath(new URL('./parity_harness.py', import.meta.url));
const hasTorch = spawnSync('python3', ['-c', 'import torch, safetensors'], {encoding:'utf8'}).status === 0;

// Runs one scenario: the engine side (forward, backward, optimizer step) and the PyTorch side, then compares.
function parity(E, {loss, idxs, train, opt = 'adam', lr = 0.01}){
  const js = JSON.parse(E(`(() => {
    infer();
    const L = root.nodes.find(n => n.name === ${JSON.stringify(loss)}), ctx = makeCtx(L.id), idxs = ${JSON.stringify(idxs)};
    for (const P of wcache.values()){ delete P.m; delete P.v; }
    const weights = Array.from(safetensors(stateDict().map(t => ({...t, data:t.data.slice()}))));   // before anything changes
    const xb = batchOf(ctx.ds, idxs, ctx.src.props.flatten);
    const batch = Array.from(safetensors([{name:'x', shape:[idxs.length, ...xb.shape], data:xb.data}, {name:'y', shape:[idxs.length], dtype:'I64', data:BigInt64Array.from(xb.labels, v => BigInt(v))}]));
    const r = runBatch(ctx, idxs, ${train}, true);
    const outNode = root.nodes.find(n => n.type === 'output'), out = outNode && ctx.out.get(outNode.id);
    const plan = exportPlan(), names = new Map(); for (const t of stateDict(plan)) names.set(t.data, t.name);
    const grads = {}; for (const [P, G] of ctx.pg) for (const k of ['W', 'B']){ const nm = names.get(P[k]); if (nm) grads[nm] = Array.from(G[k]); }
    const bufs = {}; for (const t of stateDict(plan)) if (t.name.includes('running')) bufs[t.name] = Array.from(t.data);
    optStep(ctx, ${JSON.stringify(opt)}, ${lr}, 1);
    const step = {}; for (const t of stateDict(plan)) if (t.dtype !== 'I64' && !t.name.includes('running')) step[t.name] = Array.from(t.data);
    return JSON.stringify({weights, batch, L:r.L, out: out ? Array.from(out.data) : null, grads, bufs, step, code:codegen(), lossFn:'loss_' + pyName(L.name)});
  })()`));
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'netgraph-parity-'));
  fs.writeFileSync(path.join(dir, 'model.py'), js.code);
  fs.writeFileSync(path.join(dir, 'weights.safetensors'), Uint8Array.from(js.weights));
  fs.writeFileSync(path.join(dir, 'batch.safetensors'), Uint8Array.from(js.batch));
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify({train, loss:js.lossFn, opt, lr}));
  const py = spawnSync('python3', [harness, dir], {encoding:'utf8'});
  assert.equal(py.status, 0, `PyTorch run failed:\n${py.stderr}\n--- model.py ---\n${js.code}`);
  const res = readSafetensors(fs.readFileSync(path.join(dir, 'result.safetensors')));
  fs.rmSync(dir, {recursive:true});

  // Float32 sums run in a different order in the two engines, so compare like torch.allclose: a tolerance relative to
  // each tensor's scale, plus a small absolute floor (relative to the model's largest gradient) for values that are
  // zero in exact arithmetic, like the gradient of a bias that BatchNorm subtracts away.
  const close = (what, a, b, rtol = 2e-4, atol = 0) => {
    assert.equal(a.length, b.length, `${what}: ${a.length} vs ${b.length} values`);
    const scale = Math.max(1e-12, ...Array.from(b, Math.abs));
    let worst = 0;
    for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.max(0, Math.abs(a[i] - b[i]) - atol)/scale);
    assert.ok(worst < rtol, `${what}: differs by ${worst.toExponential(2)} of its largest value`);
    return worst;
  };
  const report = {loss:close('loss', [js.L], res.loss.data)};
  if (js.out) report.out = close('output', js.out, res.out.data);
  const torchGrads = Object.keys(res).filter(k => k.startsWith('grad.')).map(k => k.slice(5));
  assert.deepEqual(Object.keys(js.grads).sort(), torchGrads.sort(), 'the same parameters get gradients');
  const gmax = Math.max(...torchGrads.map(k => Math.max(...Array.from(res['grad.' + k].data, Math.abs))));
  report.grads = Math.max(...torchGrads.map(k => close(`gradient of ${k}`, js.grads[k], res['grad.' + k].data, 2e-4, 1e-5*gmax)));
  const bufs = Object.keys(js.bufs);
  if (bufs.length) report.bufs = Math.max(...bufs.map(k => close(`buffer ${k}`, js.bufs[k], res['buf.' + k].data)));
  // Adam's first step moves each weight by about lr·sign(gradient), so a weight whose gradient is only rounding noise
  // (zero in exact arithmetic) moves by ±lr at random in either engine. Compare the step where the gradient is real.
  report.step = Math.max(...Object.keys(js.step).map(k => {
    const g = res['grad.' + k] ? res['grad.' + k].data : null, keep = i => !g || Math.abs(g[i]) > 1e-5*gmax;
    const idx = [...js.step[k].keys()].filter(keep);
    return idx.length ? close(`${k} after one ${opt} step`, idx.map(i => js.step[k][i]), idx.map(i => res['step.' + k].data[i]), 2e-4) : 0;
  }));
  return {report, params:torchGrads.length};
}

function convNet(E){
  toyDataset(E);
  E(`fresh();
     var d = P('data', {dataset:'toy'}), c = P('conv2d', {out_channels:3, kernel:3, padding:1}, 'conv1'), bn = P('batchnorm', {}, 'bn1'), ge = P('gelu'),
         ap = P('avgpool'), c2 = P('conv2d', {out_channels:4, kernel:3, stride:2, padding:1}, 'conv2'), mp = P('maxpool', {kernel:2}), f = P('flatten', {}, 'flat'),
         fc = P('dense', {out_features:6}, 'fc1'), bn2 = P('batchnorm', {}, 'bn2'), th = P('tanh'), dr = P('dropout', {p:0}), head = P('dense', {out_features:3}, 'head'),
         sm = P('softmax', {}, 'probs'), sg = P('sigmoid', {}, 'sig'), out = P('output', {}, 'out1');
     wire(d, c, bn, ge, ap, c2, mp, f, fc, bn2, th, dr, head, sm, out); wire(head, sg);
     P('loss', {expr:'ce(head) + 0.05*sum(fc1[2].pre_weights^2) + 0.1*mse(probs, probs*0.5) + 0.02*abs(sum(fc1.post_weights)) + mean(sqrt(sig + 1)) + k*0'}, 'L1');`);
}

test('PyTorch parity: conv net, BatchNorm with batch statistics (train mode), Adam', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, () => {
  const E = loadEngine(); convNet(E);
  const {report, params} = parity(E, {loss:'L1', idxs:[0, 1, 2, 5, 7, 11], train:true});
  assert.ok(params >= 10);
  console.log('  conv net, train mode:', JSON.stringify(report));
});

test('PyTorch parity: after training, BatchNorm with running statistics (eval mode), SGD', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, async () => {
  const E = loadEngine(); convNet(E);
  await E(`trainLoop(root.nodes.find(n => n.name === 'L1').id, {lr:0.01, bs:16, epochs:2, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  const {report} = parity(E, {loss:'L1', idxs:[3, 4, 9, 20], train:false, opt:'sgd', lr:0.05});
  console.log('  conv net after training, eval mode:', JSON.stringify(report));
});

test('PyTorch parity: blocks, residual add, concat, Unflatten, expression Flatten, losses using losses', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh();
     var d = P('data', {dataset:'toy', flatten:true}), b = BUILTIN[0].make(); b.name = 'MLP'; root.nodes.push(b);
     var a = P('dense', {out_features:64}, 'fa'), s = P('add'), u = P('unflatten', {to:'4,4,4'}), cv = P('conv2d', {out_channels:2, kernel:3, padding:1}), f = P('flatten'),
         cat = P('concat'), head = P('dense', {out_features:3}, 'head');
     wire(d, b); wire(d, a);
     root.edges.push(edge(a.id, s.id, 0), edge(d.id, s.id, 1)); wire(s, u, cv, f);
     root.edges.push(edge(f.id, cat.id, 0), edge(b.id, cat.id, 1)); wire(cat, head);
     var fx = P('flatten', {expr:'MLP.relu1[0:8]*2 - fa[k] + MLP.fc1[1].pre_weights[0:8]'}, 'fx'), side = P('dense', {out_features:3}, 'side'); wire(fx, side);
     P('loss', {expr:'ce(head)'}, 'La'); P('loss', {expr:'mse(side, MLP.fc2[0:3])'}, 'Lb');
     P('loss', {expr:'0.5*La + Lb/2 + 0.01*sum(MLP.fc1[1].post_weights^2)'}, 'total');`);
  E(`infer(); root.nodes.find(n => n.name === 'fx').props.vars.k.value = 7;`);
  const {report, params} = parity(E, {loss:'total', idxs:[0, 3, 4, 8, 13], train:true});
  assert.ok(params >= 8);
  console.log('  blocks and expressions:', JSON.stringify(report));
});

test('PyTorch parity: input cells with expressions feeding conv, add and dense layers', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy'}), c1 = P('conv2d', {out_channels:4, kernel:3, padding:1}, 'c1'), bn = P('batchnorm', {}, 'bn1'), r1 = P('relu', {}, 'r1'); wire(d, c1, bn, r1);
     var c2 = P('conv2d', {out_channels:2, kernel:3, padding:1}, 'c2'); c2.props.inputs = ['c1[0:2] * 0.5 + r1[2:4]'];
     var f = P('flatten'); wire(c2, f); var fc = P('dense', {out_features:5}, 'fc'); wire(f, fc);
     var mix = P('add', {}, 'mix'); mix.props.inputs = ['fc', 'fc * k + fc.pre_weights[1, 0:5]'];
     var head = P('dense', {out_features:3}, 'head'); head.props.inputs = ['mix + 0.1*fc[0]'];
     P('loss', {expr:'ce(head)'}, 'L'); infer(); mix.props.vars.k.value = 0.3;`);
  const {report, params} = parity(E, {loss:'L', idxs:[0, 2, 5, 9], train:true});
  assert.ok(params >= 6);
  console.log('  input cells:', JSON.stringify(report));
});

test('PyTorch parity: activations with formulas of their own', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), f1 = P('dense', {out_features:8}, 'fc1'), leaky = P('activation', {fn:'max(x, a*x)'}, 'leaky'),
     f2 = P('dense', {out_features:6}, 'fc2'), swish = P('activation', {fn:'x*sigmoid(x) + 0.1*tanh(x)'}, 'swish'), f3 = P('dense', {out_features:3}, 'head'),
     sm = P('activation', {fn:'softmax(2*x)'}, 'probs'), out = P('output', {}, 'out1');
     wire(d, f1, leaky, f2, swish, f3, sm, out); P('loss', {expr:'nll(probs)'}, 'L'); infer(); leaky.props.vars.a.value = 0.2;`);
  const {report, params} = parity(E, {loss:'L', idxs:[0, 1, 4, 7], train:true});
  assert.ok(params >= 6);
  console.log('  custom activations:', JSON.stringify(report));
});

test('PyTorch parity: a Sliders component feeding a layer and an expression', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), fc = P('dense', {out_features:3}, 'fc'), sl = P('sliders', {}, 'sl1');
     sl.props.sliders = [{name:'a', value:0.5, min:-1, max:1, step:0.01}, {name:'b', value:-0.25, min:-1, max:1, step:0.01}, {name:'c', value:1.5, min:0, max:2, step:0.1}];
     var dec = P('dense', {out_features:3}, 'dec'); wire(sl, dec); wire(d, fc);
     var head = P('dense', {out_features:3}, 'head'); head.props.inputs = ['fc*sl1[2] + dec'];
     P('loss', {expr:'ce(head)'}, 'L');`);
  const {report, params} = parity(E, {loss:'L', idxs:[0, 1, 4], train:true});
  assert.ok(params >= 5);
  console.log('  sliders:', JSON.stringify(report));
});

test('PyTorch parity: a pointer pair running a stretch again (BatchNorm inside) with shared weights', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), enc = P('dense', {out_features:6}, 'enc'), r = P('relu', {}, 'r1'), dec1 = P('dense', {out_features:6}, 'dec1'),
       bn = P('batchnorm', {}, 'bn1'), r2 = P('relu', {}, 'r2'), dec2 = P('dense', {out_features:3}, 'dec2'); wire(d, enc, r, dec1, bn, r2, dec2);
     var side = P('dense', {out_features:6}, 'side'), pin = P('ptrin', {target:'dec1'}, 'pin1'), pout = P('ptrout', {pair:'pin1'}, 'pout1'); wire(d, side, pin); wire(dec2, pout);
     P('loss', {expr:'ce(dec2) + ce(pout1)'}, 'L');`);
  const {report} = parity(E, {loss:'L', idxs:[0, 1, 4, 7], train:true});
  console.log('  pointer:', JSON.stringify(report));
});

test('PyTorch parity: a fitted PCA (fixed) between trainable layers', {skip: !hasTorch && 'needs python3 with torch and safetensors'}, async () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), fc = P('dense', {out_features:12}, 'fc'), pca = P('pca', {components:5, fit_n:'all'}, 'pca1'), head = P('dense', {out_features:3}, 'head'); wire(d, fc, pca, head);
     P('loss', {expr:'ce(head)'}, 'L'); infer();`);
  await E(`fitPCA(pca.id, {yield:async () => {}, shouldStop:() => false})`);
  const {report} = parity(E, {loss:'L', idxs:[0, 1, 4, 7], train:true});
  console.log('  pca:', JSON.stringify(report));
});
