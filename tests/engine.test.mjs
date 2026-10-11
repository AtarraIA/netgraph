// Engine tests: shapes, variables, expressions, gradients (every layer type, both BatchNorm modes), renames,
// training and the safetensors export. Run with: npm test
import test from 'node:test';
import assert from 'node:assert/strict';
import {loadEngine, toyDataset, gradCheck, readSafetensors} from './helpers.mjs';

test('shapes: layers, unflatten and errors', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy'}), c = P('conv2d', {out_channels:4, kernel:3, padding:1}), m = P('maxpool'), f = P('flatten'),
     u = P('unflatten', {to:'-1,2,2'}), g = P('gap'), fc = P('dense', {out_features:5}); wire(d, c, m, f, u, g, fc); infer();`);
  assert.equal(E('fmt(c._out)'), '4×8×8');
  assert.equal(E('fmt(m._out)'), '4×4×4');
  assert.equal(E('fmt(u._out)'), '16×2×2');
  assert.equal(E('fmt(fc._out)'), '5');
  E(`u.props.to = '3,3'; infer();`);
  assert.match(E('u._r.err'), /holds 9 values, but the input has 64/);
});

test('variables in layer settings', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), fc = P('dense', {out_features:'n'}); wire(d, fc); infer();`);
  assert.deepEqual(E('JSON.stringify(fc._vars)'), '["n"]');
  assert.equal(E('fmt(fc._out)'), '128');          // the default value comes from the setting's default
  E(`fc.props.vars.n.value = 12; infer();`);
  assert.equal(E('fmt(fc._out)'), '12');
  E(`fc.props.out_features = 'auto'; infer();`);    // "auto" is never a variable
  assert.ok(E('!!fc._r.err'));
  E(`var dr = P('dropout', {p:'q'}); wire(fc, dr); fc.props.out_features = 6; infer();`);
  assert.equal(E('dr._vars[0]'), 'q');
  assert.equal(E('dr._p.p'), 0.5);
});

test('expressions: indices, weights around a neuron, errors', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), f1 = P('dense', {out_features:6}, 'fc1'), r = P('relu', {}, 'relu1'), f2 = P('dense', {out_features:3}, 'fc2');
     wire(d, f1, r, f2); var fx = P('flatten', {expr:'fc1[2].pre_weights'}, 'probe'); infer(); forward(root, null);`);
  assert.equal(E('fmt(fx._out)'), '64');
  assert.ok(E('[...fx._act.data].every((v, i) => v === weightsOf(f1, 64*6, 6, 64).W[2*64 + i])'));
  E(`fx.props.inputs = ['fc1[4].post_weights']; infer(); forward(root, null);`);
  assert.equal(E('fmt(fx._out)'), '3');
  assert.ok(E('[...fx._act.data].every((v, o) => v === weightsOf(f2, 6*3, 3, 6).W[o*6 + 4])'));
  const err = ex => E(`fx.props.inputs = [${JSON.stringify(ex)}]; infer(); fx._r.err`);
  assert.match(err('fc2[1].post_weights'), /no Dense layer after it/);
  assert.match(err('fc1[9].pre_weights'), /has 6 neurons/);
  assert.match(err('relu1.pre_weights'), /works on Dense layers/);
  assert.match(err('fc1[0:9]'), /doesn't fit/);
  assert.match(err('(fc1'), /Missing "\)"/);
  // a name that isn't a component is a variable
  E(`fx.props.inputs = ['fc1[k]']; infer();`);
  assert.equal(E('fx.props.vars.k.max'), 5);
});

test('gradients: every layer type, BatchNorm with running and with batch statistics', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh();
     var d = P('data', {dataset:'toy'}), c = P('conv2d', {out_channels:3, kernel:3, padding:1}, 'conv1'), bn = P('batchnorm', {}, 'bn1'), ge = P('gelu'),
         ap = P('avgpool'), c2 = P('conv2d', {out_channels:4, kernel:3, stride:2, padding:1}), mp = P('maxpool', {kernel:2}), f = P('flatten', {}, 'flat'),
         fc = P('dense', {out_features:6}, 'fc1'), bn2 = P('batchnorm', {}, 'bn2'), th = P('tanh'), dr = P('dropout', {p:0}), head = P('dense', {out_features:3}, 'head'),
         sm = P('softmax', {}, 'probs'), sg = P('sigmoid');
     wire(d, c, bn, ge, ap, c2, mp, f, fc, bn2, th, dr, head, sm); wire(head, sg);
     P('loss', {expr:'ce(head) + 0.05*sum(fc1[2].pre_weights^2) + 0.1*mse(probs, probs*0.5) + 0.02*abs(sum(fc1.post_weights)) + mean(sqrt(sg + 1))'}, 'L1');`);
  for (const train of [false, true]){
    const r = gradCheck(E, 'L1', [0, 1, 2, 5, 7], {train});
    assert.ok(r.checks >= 20, `only ${r.checks} checks`);
    assert.ok(r.worst < 2e-2, `${train ? 'batch' : 'running'} statistics: worst relative error ${r.worst}`);
  }
});

test('gradients: blocks, add, concat, Unflatten, expression Flatten, losses using losses', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh();
     var d = P('data', {dataset:'toy', flatten:true}), b = BUILTIN[0].make(); b.name = 'MLP'; root.nodes.push(b);
     var a = P('dense', {out_features:64}, 'fa'), s = P('add'), u = P('unflatten', {to:'4,4,4'}), cv = P('conv2d', {out_channels:2, kernel:3, padding:1}), f = P('flatten'),
         cat = P('concat'), head = P('dense', {out_features:3}, 'head');
     wire(d, b); wire(d, a);
     root.edges.push(edge(a.id, s.id, 0), edge(d.id, s.id, 1)); wire(s, u, cv, f);
     root.edges.push(edge(f.id, cat.id, 0), edge(b.id, cat.id, 1)); wire(cat, head);
     var fx = P('flatten', {expr:'MLP.relu1[0:8]*2 - fa[k]'}, 'fx'), side = P('dense', {out_features:3}, 'side'); wire(fx, side);
     P('loss', {expr:'ce(head)'}, 'La'); P('loss', {expr:'nll(side) * 0 + mse(side, MLP.fc2[0:3])'}, 'Lb');
     P('loss', {expr:'0.5*La + Lb/2 + 0.01*sum(MLP.fc1[1].post_weights^2)'}, 'total');`);
  E(`root.nodes.find(n => n.name === 'Lb').props.expr = 'mse(side, MLP.fc2[0:3])';`);
  const r = gradCheck(E, 'total', [0, 3, 4, 8]);
  assert.ok(r.sets >= 6, `only ${r.sets} weight sets got gradients`);
  assert.ok(r.worst < 2e-2, `worst relative error ${r.worst}`);
});

test('expressions follow renames, grouping and ungrouping', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), f1 = P('dense', {out_features:6}, 'fc1'), r1 = P('relu', {}, 'relu1'), f2 = P('dense', {out_features:3}, 'fc2');
     wire(d, f1, r1, f2); var L = P('loss', {expr:'ce(fc2) + 0.1*sum(fc1[k].pre_weights^2) + mean(relu1)'}, 'L');`);
  E(`var before = pathMap(); f1.name = 'enc'; rewriteRefs(before);`);
  assert.equal(E('L.props.expr'), 'ce(fc2) + 0.1*sum(enc[k].pre_weights^2) + mean(relu1)');
  E(`before = pathMap(); f2.name = 'my head'; rewriteRefs(before);`);
  assert.equal(E('L.props.expr'), 'ce("my head") + 0.1*sum(enc[k].pre_weights^2) + mean(relu1)');
  // move enc and relu1 into a block, as grouping does
  E(`before = pathMap();
     var blk = {id:uid(), type:'group', x:0, y:0, name:'Block', props:{}, graph:{nodes:[f1, r1], edges:[]}};
     root.nodes = root.nodes.filter(n => n !== f1 && n !== r1).concat(blk); rewriteRefs(before);`);
  assert.equal(E('L.props.expr'), 'ce("my head") + 0.1*sum(Block.enc[k].pre_weights^2) + mean(Block.relu1)');
});

test('training learns the toy task (batched, Adam, BatchNorm with batch statistics)', async () => {
  const E = loadEngine(); toyDataset(E, {n:300});
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true, test:20}), f1 = P('dense', {out_features:16}), bn = P('batchnorm'), r = P('relu'), f2 = P('dense', {out_features:3}, 'head');
     wire(d, f1, bn, r, f2); var L = P('loss', {expr:'ce(head)'}, 'L');`);
  const T = JSON.parse(await E(`trainLoop(L.id, {lr:0.01, bs:16, epochs:8, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}}).then(T => JSON.stringify(T))`));
  assert.ok(T.done);
  assert.ok(T.hist[T.hist.length - 1] < T.hist[0]*0.5, `loss went ${T.hist[0]} → ${T.hist[T.hist.length - 1]}`);
  assert.ok(T.testAcc >= 0.9, `test accuracy ${T.testAcc}`);
  assert.ok(E('wcache.get(bn.id).nbt') > 0, 'BatchNorm running statistics were updated');
});

test('safetensors export round-trips the state dict', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy'}), c = P('conv2d', {out_channels:2}), bn = P('batchnorm'), f = P('flatten'), fc = P('dense', {out_features:3, bias:false}); wire(d, c, bn, f, fc); infer();`);
  const names = JSON.parse(E('JSON.stringify(stateDict().map(t => t.name))'));
  assert.deepEqual(names, ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'fc1.weight']);
  const bytes = Uint8Array.from(E('Array.from(safetensors(stateDict()))'));
  const st = readSafetensors(bytes);
  assert.deepEqual(st['conv1.weight'].shape, [2, 1, 3, 3]);
  assert.deepEqual(st['fc1.weight'].shape, [3, 128]);
  assert.equal(st['bn1.num_batches_tracked'].dtype, 'I64');
  assert.ok(E(`(() => { const w = stateDict().find(t => t.name === 'fc1.weight').data; return w === weightsOf(fc, 128*3, 3, 128).W; })()`));
});

test('block variables: sliders on the block, derived values, scope', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true});
     var blk = blockFrom('Block', [['dense', {out_features:'width'}], ['relu'], ['dense', {out_features:'half'}]]); root.nodes.push(blk);
     blk.props.params = [{name:'width', expr:'32'}, {name:'half', expr:'width/2'}]; wire(d, blk); infer();`);
  assert.equal(E('fmt(blk._out)'), '16');
  assert.equal(E('JSON.stringify(blk._vars)'), '["width"]');            // only the number is a slider
  assert.equal(E('blk.graph.nodes[1]._vars.length'), 0);                 // the layer inside has no slider of its own
  E(`blk.props.vars.width.value = 20; infer();`);
  assert.equal(E('fmt(blk.graph.nodes[1]._out)') + ' ' + E('fmt(blk._out)'), '20 10');
  E(`blk.props.params[1].expr = 'width/ nope'; infer();`);
  assert.match(E('blk._r.err'), /nope isn't a variable declared above it/);
  // names declared in a block are not visible outside it
  E(`blk.props.params[1].expr = '4'; var out = P('dense', {out_features:'width'}); wire(blk, out); infer();`);
  assert.equal(E('JSON.stringify(out._vars)'), '["width"]');             // a variable of its own, not the block's
  // a nested block sees its parent's variables
  E(`var inner = blockFrom('Inner', [['dense', {out_features:'width'}]]); blk.graph.nodes.push(inner);
     var g = blk.graph, relu = g.nodes.find(n => n.type === 'activation'); g.edges.push(edge(relu.id, inner.id)); infer();`);
  assert.equal(E('fmt(inner._out)'), '20');
});

test('input cells: expressions keep shapes, feed any layer, and train', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy'}), c1 = P('conv2d', {out_channels:4, kernel:3, padding:1}, 'c1'), r1 = P('relu', {}, 'r1'); wire(d, c1, r1);
     var c2 = P('conv2d', {out_channels:2, kernel:3, padding:1}, 'c2'); c2.props.inputs = ['c1[0:2] * 0.5 + r1[2:4]'];
     var f = P('flatten'); wire(c2, f); var fc = P('dense', {out_features:5}, 'fc'); wire(f, fc);
     var mix = P('add', {}, 'mix'); mix.props.inputs = ['fc', 'fc * k'];
     var head = P('dense', {out_features:3}, 'head'); head.props.inputs = ['mix + 0.1*fc[0]'];
     P('loss', {expr:'ce(head)'}, 'L'); infer();`);
  assert.equal(E('fmt(c2._in[0])'), '2×8×8');          // a slice of channels stays a map
  assert.equal(E('fmt(mix._out)') + ' ' + E('fmt(head._out)'), '5 3');
  assert.equal(E('mix._vars.join()'), 'k');
  assert.equal(E('c2._r.err || "ok"'), 'ok');
  const r = gradCheck(E, 'L', [0, 1, 4, 6]);
  assert.ok(r.sets >= 4, `only ${r.sets} weight sets got gradients`);   // c1 trains through the expressions
  assert.ok(r.worst < 2e-2, `worst relative error ${r.worst}`);
  // old Flatten expressions become input cells
  E(`var old = P('flatten', {expr:'r1[0]'}, 'old'); infer();`);
  assert.equal(E('JSON.stringify(old.props.inputs)') + ' ' + E('fmt(old._out)'), '["r1[0]"] 64');
});

test('activations: built-ins are formulas, custom formulas train, old types convert', () => {
  const E = loadEngine(); toyDataset(E);
  // the built-ins and equivalent formulas that take the general path give the same values and gradients
  const pairs = [['relu', 'max(0, x)'], ['gelu', '0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x*x*x)))'], ['sigmoid', '1/(exp(-x) + 1)'], ['tanh', 'tanh(x)*1'], ['softmax', 'exp(x - max(x))/sum(exp(x - max(x)))']];
  for (const [kind, fn] of pairs){
    const out = JSON.parse(E(`(() => {
      fresh(); const d = P('data', {dataset:'toy', flatten:true}), fc = P('dense', {out_features:5}, 'fc'), a = P(${JSON.stringify(kind)}, {}, 'a1'), b = P('activation', {fn:${JSON.stringify(fn)}}, 'a2');
      wire(d, fc, a); wire(fc, b); infer();
      const x = runNode(fc, [batchOf(dsById('toy'), [0, 1, 2], true)]), ya = runNode(a, [x]), yb = runNode(b, [x]), gy = Float32Array.from(ya.data, (_, i) => Math.sin(i));
      const c = {pg:new Map(), bn:new Map(), masks:new Map()}, ga = backNode(a, [x], ya, gy, c)[0], gb = backNode(b, [x], yb, gy, c)[0];
      return JSON.stringify({label:labelOf(a), b:labelOf(b), dy:Math.max(...ya.data.map((v, i) => Math.abs(v - yb.data[i]))), dg:Math.max(...ga.map((v, i) => Math.abs(v - gb[i])))});
    })()`));
    assert.ok(out.dy < 1e-6 && out.dg < 1e-6, `${kind}: values differ by ${out.dy}, gradients by ${out.dg}`);
    assert.equal(out.b, 'Activation');
  }
  // custom formulas with a variable: shapes, sliders and gradients
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), f1 = P('dense', {out_features:8}, 'fc1'), leaky = P('activation', {fn:'max(x, a*x)'}, 'leaky'),
     f2 = P('dense', {out_features:6}, 'fc2'), swish = P('activation', {fn:'x*sigmoid(x)'}, 'swish'), head = P('dense', {out_features:3}, 'head');
     wire(d, f1, leaky, f2, swish, head); P('loss', {expr:'ce(head)'}, 'L'); infer(); leaky.props.vars.a.value = 0.1; infer();`);
  assert.equal(E('labelOf(leaky) + " " + fmt(leaky._out) + " " + leaky._vars.join()'), 'Activation 8 a');
  const r = gradCheck(E, 'L', [0, 3, 5]);
  assert.ok(r.worst < 2e-2 && r.sets >= 3, `worst ${r.worst}, ${r.sets} weight sets`);
  // a ReLU made the old way, and one saved as an old type, are both Activations showing "ReLU"
  E(`var r1 = P('relu', {}, 'r1'); var old = {id:uid(), type:'relu', x:0, y:0, name:'oldrelu', props:{}}; root.nodes.push(old); infer();`);
  assert.equal(E('[r1.type, labelOf(r1), old.type, labelOf(old)].join()'), 'activation,ReLU,activation,ReLU');
  assert.match(E(`(() => { const a = P('activation', {fn:'x + y.z'}); wire(f1, a); infer(); return a._r.err; })()`), /can use x, numbers and variables/);
});

test('project files: a round trip reproduces the network, blocks and trained weights', async () => {
  const E = loadEngine(); toyDataset(E, {n:120});
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true, index:5}), blk = blockFrom('Block', [['dense', {out_features:'width'}], ['batchnorm'], ['relu']]); root.nodes.push(blk);
     blk.props.params = [{name:'width', expr:'12'}]; var head = P('dense', {out_features:3}, 'head'); head.props.inputs = ['Block'];
     var leaky = P('activation', {fn:'max(x, a*x)'}, 'leaky'); wire(d, blk); wire(head, leaky); var L = P('loss', {expr:'ce(head) + 0*sum(leaky)'}, 'L'); infer();`);
  await E(`trainLoop(L.id, {lr:0.01, bs:16, epochs:1, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  const before = E(`infer(); forward(root, null); JSON.stringify(Array.from(leaky._act.data))`);
  const file = E(`projectJSON([{id:'b1', label:'My MLP', node:strip(blk)}])`);
  assert.ok(JSON.parse(file).weights.length >= 3, 'trained weights are in the file');
  // a fresh app opens it
  const F = loadEngine(); toyDataset(F, {n:120});
  F(`var p = readProject(${JSON.stringify(file)}); root = p.graph; importWeights(p.weights); infer(); forward(root, null);`);
  const leakyAfter = F(`JSON.stringify(Array.from(root.nodes.find(n => n.name === 'leaky')._act.data))`);
  assert.equal(leakyAfter, before, 'the same output for the same sample');
  assert.equal(F('p.blocks[0].label'), 'My MLP');
  assert.equal(F(`fmt(root.nodes.find(n => n.name === 'Block')._out)`), '12');
  // errors say what's wrong
  assert.throws(() => F(`readProject('{"hello":1}')`), /isn't a Netgraph project file/);
  assert.throws(() => F(`readProject('not json')`), /isn't valid JSON/);
  const G = loadEngine();
  assert.deepEqual(JSON.parse(G(`JSON.stringify(readProject(${JSON.stringify(file)}).missing)`)), ['toy']);
});

test('sliders: a vector of hand-set values, the same for every sample, untouched by training', async () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), fc = P('dense', {out_features:3}, 'fc'), sl = P('sliders', {}, 'sl1');
     sl.props.sliders = [{name:'a', value:0.5, min:-1, max:1, step:0.01}, {name:'b', value:-0.25, min:-1, max:1, step:0.01}, {name:'c', value:1, min:0, max:2, step:0.1}];
     var dec = P('dense', {out_features:3}, 'dec'); wire(sl, dec); wire(d, fc);
     var head = P('dense', {out_features:3}, 'head'); head.props.inputs = ['fc*sl1[2] + dec'];
     var L = P('loss', {expr:'ce(head)'}, 'L'); infer(); forward(root, null);`);
  assert.equal(E('fmt(sl._out)'), '3');
  assert.equal(E('Array.from(sl._act.data).join()'), '0.5,-0.25,1');
  const r = gradCheck(E, 'L', [0, 1, 2]);
  assert.ok(r.worst < 2e-2 && r.sets >= 3, `worst ${r.worst}, ${r.sets} weight sets`);   // fc, dec and head all learn
  await E(`trainLoop(L.id, {lr:0.01, bs:8, epochs:1, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.equal(E('sl.props.sliders.map(s => s.value).join()'), '0.5,-0.25,1');
  E(`sl.props.sliders = []; infer();`);
  assert.equal(E('sl._r.err'), 'Add a slider.');
});

test('collecting: a wired Data component keeps what reaches it during an inference run, then feeds a network', async () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true, run_from:'test', run_n:10}), enc = P('dense', {out_features:4}, 'enc'), dec = P('dense', {out_features:64}, 'dec');
     var c = P('data', {}, 'codes'), c2 = P('data', {}, 'part'); wire(d, enc, dec); wire(enc, c); c2.props.inputs = ['enc[0:2] * 2'];
     var ae = P('loss', {expr:'mse(dec, data1)'}, 'L'); d.name = 'data1'; infer();`);
  assert.equal(E('c._collect && c2._collect && !d._collect'), true);
  assert.equal(E('fmt(c._out) + " " + fmt(c2._out)'), '4 2');                 // shaped like its input before anything is collected
  assert.match(E('c._r.warn'), /Nothing collected yet/);
  assert.equal(E('d._r.err || null'), null);                                     // a Data component without input still reads its dataset
  assert.equal(E('collectorsOf(d).map(m => m.name).sort().join()'), 'codes,part');
  const caps = await E(`collectLoop(d.id, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.equal(Object.keys(caps).sort().join(), E('[c.id, c2.id].sort().join()'));
  const cc = caps[E('c.id')], idx = E('Array.from(splitOf(d).test.subarray(0, 10))');
  assert.equal(cc.n, 10); assert.equal(cc.x.length, 40); assert.deepEqual([...cc.shape], [4]);
  assert.equal([...cc.labels].join(), idx.map(i => E(`datasets[0].labels[${i}]`)).join());   // labels come along
  // The same values the network gives that sample on its own.
  E(`d.props.index = ${idx[3]}; forward(root, null);`);
  const one = E('Array.from(enc._act.data)');
  [...cc.x.subarray(12, 16)].forEach((v, k) => assert.ok(Math.abs(v - one[k]) < 1e-5));
  const p2 = caps[E('c2.id')]; [...p2.x.subarray(6, 8)].forEach((v, k) => assert.ok(Math.abs(v - 2*one[k]) < 1e-5));
  // Kept as a dataset, the collected codes are the component's output and can train a classifier.
  E(`datasets.push({id:'cap-1', name:'codes (collected)', shape:[4], classes:3, n:${cc.n}, x:new Float32Array([${[...cc.x]}]), labels:new Int32Array([${[...cc.labels]}]), owner:c.id});
     c.props.dataset = 'cap-1'; c.props.test = 20; var head = P('dense', {out_features:3}, 'head'); wire(c, head); var cl = P('loss', {expr:'ce(head)'}, 'CL'); infer(); forward(root, null);`);
  assert.equal(E('fmt(c._out) + " " + (c._r.err || c._r.warn || "ok")'), '4 ok');
  assert.equal(E('cl._srcData === c'), true);
  assert.equal(E('Array.from(c._act.data).map(v => v.toFixed(5)).join()'), [...cc.x.subarray(0, 4)].map(v => v.toFixed(5)).join());
  const T = await E(`trainLoop(cl.id, {lr:0.01, bs:4, epochs:2, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.ok(T.done && T.hist.length === 4 && T.hist.every(Number.isFinite));
  E(`enc.props.out_features = 5; infer();`);
  assert.match(E('c._r.warn'), /Collected 4 samples, but its input now gives 5/);
  assert.match(E('codegen()'), /# codes collects enc/);                         // in PyTorch it passes the flow on
  E(`root.edges = root.edges.filter(e => e.to !== c.id && e.to !== head.id); c2.props.inputs = []; infer();`);
  await assert.rejects(E(`collectLoop(d.id, {yield:async () => {}, shouldStop:() => false, progress(){}})`), /Nothing collects this flow/);
});

test('pointer pairs: a stretch runs again on another input with the same weights, trains both ways, and checks its ends', async () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}, 'data1'), enc = P('dense', {out_features:4}, 'enc'), r = P('relu', {}, 'r1'), dec1 = P('dense', {out_features:8}, 'dec1'),
       r2 = P('activation', {fn:'tanh(x)'}, 'r2'), dec2 = P('dense', {out_features:64}, 'dec2'); wire(d, enc, r, dec1, r2, dec2);
     var sl = P('sliders', {}, 'sl1'); sl.props.sliders = [0.3, -0.2, 0.9, 0.1].map((v, i) => ({name:'abcd'[i], value:v, min:-1, max:1, step:0.01}));
     var pin = P('ptrin', {target:'dec1'}, 'pin1'), pout = P('ptrout', {pair:'pin1'}, 'pout1'); wire(sl, pin); wire(dec2, pout); infer(); forward(root, null);`);
  assert.equal(E('pout._r.err || null'), null);
  assert.equal(E('fmt(pout._out) + " " + fmt(pin._out)'), '64 4');
  assert.equal(E('stats.params'), E('enc._r.params + dec1._r.params + dec2._r.params'));     // no weights of their own
  assert.equal(E('dec1._in[0].join()'), '4');                                                  // dec1 keeps its own wire
  const by = E(`(() => { const x = tensor([4], 1, Float32Array.from(sl.props.sliders.map(s => s.value))); return Array.from(runNode(dec2, [runNode(r2, [runNode(dec1, [x])])]).data); })()`);
  const got = E('Array.from(pout._act.data)');
  by.forEach((v, i) => assert.ok(Math.abs(v - got[i]) < 1e-6));
  // Fed what the main flow feeds dec1, it gives the main flow's output.
  E(`root.edges = root.edges.filter(e => e.to !== pin.id); wire(r, pin); infer(); forward(root, null);`);
  assert.equal(E('Array.from(pout._act.data).join()'), E('Array.from(dec2._act.data).join()'));
  // Gradients reach the shared weights through both runs, and flow back out of the Pointer in.
  E(`root.edges = root.edges.filter(e => e.to !== pin.id); var side = P('dense', {out_features:4}, 'side'); wire(d, side, pin);
     var L = P('loss', {expr:'mse(dec2, data1) + mse(pout1, data1)'}, 'L'); infer();`);
  assert.equal(E('L._r.err || null'), null);
  const g = gradCheck(E, 'L', [0, 1, 2]);
  assert.ok(g.worst < 2e-2 && g.sets === 4, `worst ${g.worst}, ${g.sets} weight sets`);   // enc, dec1, dec2 and side
  const T = await E(`trainLoop(L.id, {lr:0.01, bs:8, epochs:1, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.ok(T.done && T.hist.every(Number.isFinite));
  const code = E('codegen()');
  assert.match(code, /# pout1: dec1 … dec2 again on pin1, same weights/);
  assert.equal((code.match(/self\.dec1 = nn\.Linear/g) || []).length, 1);
  assert.equal((code.match(/= self\.dec1\(/g) || []).length, 2);
  // Ends that don't work.
  const err = (target, pair) => E(`pin.props.target = ${JSON.stringify(target)}; pout.props.pair = ${JSON.stringify(pair)}; infer(); pout._r.err || ''`);
  assert.match(err('enc', 'pin1'), /enc takes 64, but pin1 gets 4/);
  assert.match(err('dec1', ''), /Pair it with a Pointer in/);
  assert.match(err('dec1', 'dec1'), /dec1 isn't a Pointer in/);
  assert.match(err('', 'pin1'), /Drag pin1's output onto the input/);
  assert.match(err('nope', 'pin1'), /no component named "nope"/);
  E(`root.edges = root.edges.filter(e => e.to !== pout.id); wire(enc, pout);`);
  assert.match(err('dec1', 'pin1'), /enc doesn't come after dec1/);
  E(`root.edges = root.edges.filter(e => e.to !== pout.id); wire(dec2, pout); var add = P('add', {}, 'add1'); root.edges = root.edges.filter(e => e.to !== dec2.id); root.edges.push(edge(r2.id, add.id, 0), edge(side.id, add.id, 1)); wire(add, dec2); side.props.out_features = 8; infer();`);
  assert.match(err('dec1', 'pin1'), /add1 also takes side from outside the stretch/);
  // Renames follow.
  E(`root.edges = root.edges.filter(e => e.to !== add.id && e.from !== add.id); root.nodes = root.nodes.filter(n => n !== add); wire(r2, dec2); side.props.out_features = 4; infer();`);
  E(`(() => { const before = pathMap(); dec1.name = 'D1'; pin.name = 'IN'; rewriteRefs(before); })(); infer();`);
  assert.equal(E('pin.props.target + " " + pout.props.pair + " " + (pout._r.err || "ok")'), 'D1 IN ok');
  // Single Pointers from before pairs become an unpaired Pointer out.
  E(`root.nodes.push({id:'old', type:'pointer', name:'ptr9', x:0, y:0, props:{from:'D1', until:''}}); infer();`);
  assert.equal(E('byId(root, "old").type + " " + byId(root, "old")._r.err'), 'ptrout Connect an input.');
});

test('blocks hold anything: Data and Loss inside, names looked up from inside out, grouping keeps them working', async () => {
  const E = loadEngine(); toyDataset(E);
  // A whole training setup inside a block: its own Data, layers and a Loss naming its neighbours.
  E(`fresh(); var B = blockFrom('Trainer', [['dense', {out_features:3}]]); root.nodes.push(B);
     var g = B.graph, gin = g.nodes.find(n => n.type === 'gin'), fc = g.nodes.find(n => n.type === 'dense');
     g.edges = g.edges.filter(e => e.from !== gin.id);
     var d = mk('data', 0, 0, {dataset:'toy', flatten:true}, 'data1'); g.nodes.push(d); g.edges.push(edge(d.id, fc.id));
     var L = mk('loss', 0, 0, {expr:'ce(fc1)'}, 'L'); g.nodes.push(L); infer(); forward(root, null);`);
  assert.equal(E('B._r.err || "ok"'), 'ok');                        // no input needed: it has its own data
  assert.equal(E('L._r.err || "ok"'), 'ok');
  assert.equal(E('L._srcData === d'), true);
  assert.equal(E('typeof L._lossNow'), 'number');
  const gc = gradCheck(E, 'L', [0, 1, 2]);
  assert.ok(gc.worst < 2e-2 && gc.sets === 1, `worst ${gc.worst}`);
  const T = await E(`trainLoop(L.id, {lr:0.05, bs:16, epochs:3, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.ok(T.done && T.hist[T.hist.length - 1] < T.hist[0], 'the loss inside the block goes down');
  // A name inside a block sees its neighbours first, then the graphs around it.
  E(`var top = P('dense', {out_features:3}, 'fc1'); var top2 = P('data', {dataset:'toy', flatten:true}, 'dtop'); wire(top2, top);
     var inner2 = mk('dense', 0, 0, {out_features:2}, 'fc9'); inner2.props.inputs = ['fc1 + dtop[0:3]']; g.nodes.push(inner2); infer(); forward(root, null);`);
  assert.equal(E('inner2._r.err || "ok"'), 'ok');
  const v = E(`(() => { const a = fc._act.data, b = top2._act.data; return [inner2._actIn[0].data[1], a[1] + b[1]]; })()`);
  assert.ok(Math.abs(v[0] - v[1]) < 1e-6);                          // fc1 is the block's own fc1, dtop the top-level Data
  // A top-level Loss can read a layer inside a block that has its own Data.
  E(`var L2 = P('loss', {expr:'ce(Trainer.fc1)'}, 'L2'); infer();`);
  assert.equal(E('(L2._r.err || "ok") + " " + (L2._srcData === d)'), 'ok true');
  // Grouping a Loss with what it reads keeps it working; names inside become relative, names outside stay reachable.
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}, 'data1'), a = P('dense', {out_features:4}, 'fc1'), b = P('dense', {out_features:3}, 'fc2'); wire(d, a, b);
     var L = P('loss', {expr:'ce(fc2) + mean(data1) * 0'}, 'L'); sel = new Set([a.id, b.id, L.id]);`);
  E(`(() => { const before = pathMap(); const g = root, S = new Set([a.id, b.id, L.id]);
     const blk = {id:uid(), type:'group', x:0, y:0, name:'Net', props:{}, graph:{nodes:[mk('gin', 0, 0, {}, 'in'), a, b, L, mk('gout', 0, 0, {}, 'out')], edges:[]}};
     const gi = blk.graph.nodes[0], go = blk.graph.nodes[4];
     blk.graph.edges = [edge(gi.id, a.id), edge(a.id, b.id), edge(b.id, go.id)];
     root.nodes = root.nodes.filter(n => !S.has(n.id)).concat(blk); root.edges = [edge(d.id, blk.id)];
     rewriteRefs(before); })(); infer();`);
  assert.equal(E('L.props.expr'), 'ce(fc2) + mean(data1) * 0');      // fc2 is its neighbour, data1 still found outside
  assert.equal(E('(L._r.err || "ok") + " " + (L._srcData === d)'), 'ok true');   // found through the block's input
  const T2 = await E(`trainLoop(L.id, {lr:0.01, bs:16, epochs:1, optimizer:'adam'}, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.ok(T2.done && T2.hist.every(Number.isFinite));
  // A collecting Data component inside a block gets the flow from outside.
  E(`var c = mk('data', 0, 0, {}, 'codes'); L.props.expr = 'ce(fc2)'; root.nodes.find(n => n.name === 'Net').graph.nodes.push(c); root.nodes.find(n => n.name === 'Net').graph.edges.push(edge(a.id, c.id));
     d.props.run_n = 5; infer();`);
  assert.equal(E('collectorsOf(d).map(m => m.name).join()'), 'codes');
  const caps = await E(`collectLoop(d.id, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.equal(caps[E('c.id')].n, 5);
  assert.match(E('codegen()'), /Net/);
});

test('names: a lone unknown name in an input cell is an error with a suggestion, not a variable', () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var sl = P('sliders', {}, 'sl2'), fc = P('dense', {out_features:3}, 'fc'); fc.props.inputs = ['s12']; infer();`);
  assert.equal(E('fc._r.err'), 'Input: There\'s no component named "s12". Did you mean sl2?');
  assert.equal(E('(fc._vars || []).length'), 0);
  E(`fc.props.inputs = ['sl2 * k']; infer();`);                                       // inside a bigger expression, k is still a variable
  assert.equal(E('(fc._r.err || "ok") + " " + fmt(fc._in[0]) + " " + fc._vars.join()'), 'ok 2 k');
  E(`fc.props.inputs = ['fcx']; infer();`);
  assert.match(E('fc._r.err'), /Did you mean fc\?/);
  assert.equal(E('nextName("sliders")'), 'sliders1');
  // A lone name reaching into a block is fine.
  E(`var B = blockFrom('Block', [['dense', {out_features:5}]]); root.nodes.push(B); wire(sl, B); fc.props.inputs = ['Block.fc1']; infer();`);
  assert.equal(E('(fc._r.err || "ok") + " " + fmt(fc._in[0])'), 'ok 5');
  fc_missing: {
    E(`fc.props.inputs = ['Block.fc9']; infer();`);
    assert.equal(E('fc._r.err'), 'Input: Block has no component named "fc9".');
  }
});

test('PCA: components are the covariance\'s top eigenvectors (exactly and by iteration), project, pass gradients, fill a Data component', async () => {
  for (const [h, w] of [[8, 8], [16, 16]]){   // 64 inputs: Jacobi on the covariance; 256: subspace iteration
    const E = loadEngine(); toyDataset(E, {h, w, n:300});
    E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}, 'data1'), pca = P('pca', {components:3, fit_n:'all'}, 'pca1'), out = P('data', {}, 'proj'); wire(d, pca, out); infer();`);
    assert.match(E('pca._r.warn'), /Not fitted/);
    const r = await E(`fitPCA(pca.id, {yield:async () => {}, shouldStop:() => false})`);
    assert.equal(r.n, E('splitOf(d).train.length'));
    // Check C v = λ v against a covariance worked out here.
    const ok = E(`(() => {
      const S = ${h*w}, idx = splitOf(d).train, X = batchOf(datasets[0], idx, true).data, N = idx.length, P = pcaParams(pca, S, 3), mu = new Float64Array(S);
      for (let b = 0; b < N; b++) for (let i = 0; i < S; i++) mu[i] += X[b*S + i]/N;
      const Cv = v => { const out = new Float64Array(S); for (let b = 0; b < N; b++){ let s = 0; for (let i = 0; i < S; i++) s += (X[b*S + i] - mu[i])*v[i]; for (let i = 0; i < S; i++) out[i] += s*(X[b*S + i] - mu[i])/(N - 1); } return out; };
      let worst = 0, ortho = 0, meanErr = 0; const lams = [];
      for (let j = 0; j < 3; j++){
        const v = P.W.subarray(j*S, (j + 1)*S), c = Cv(v); let lam = 0; for (let i = 0; i < S; i++) lam += v[i]*c[i]; lams.push(lam);
        let e = 0, nrm = 0; for (let i = 0; i < S; i++){ e += (c[i] - lam*v[i])**2; nrm += c[i]**2; } worst = Math.max(worst, Math.sqrt(e/nrm));
        for (let k = 0; k < 3; k++){ let dot = 0; for (let i = 0; i < S; i++) dot += v[i]*P.W[k*S + i]; ortho = Math.max(ortho, Math.abs(dot - (j === k ? 1 : 0))); }
      }
      for (let i = 0; i < S; i++) meanErr = Math.max(meanErr, Math.abs(mu[i] - P.B[i]));
      return {worst, ortho, meanErr, sorted: lams[0] >= lams[1] && lams[1] >= lams[2]};
    })()`);
    assert.ok(ok.worst < 1e-3 && ok.ortho < 1e-4 && ok.meanErr < 1e-5 && ok.sorted, JSON.stringify(ok));
    E(`infer(); forward(root, null);`);
    assert.equal(E('fmt(pca._out) + " " + (pca._r.warn || "ok")'), '3 ok');
    assert.match(E('pca._r.sum'), /3 components · [\d.]+% of variance/);
    // Projecting the whole dataset into the Data component after it.
    const caps = await E(`collectLoop(d.id, {yield:async () => {}, shouldStop:() => false, progress(){}}, {all:true, only:collectorsOf(pca)})`);
    const c = caps[E('out.id')];
    assert.equal(c.n, 300); assert.equal([...c.shape].join(), '3');
    const y0 = E(`(() => { d.props.index = 0; forward(root, null); return Array.from(pca._act.data); })()`);
    y0.forEach((v, j) => assert.ok(Math.abs(v - c.x[j]) < 1e-4));
  }
  // Gradients pass back through it (the components stay fixed).
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true}), fc = P('dense', {out_features:16}, 'fc'), pca = P('pca', {components:4, fit_n:100}, 'pca1'), head = P('dense', {out_features:3}, 'head'); wire(d, fc, pca, head); P('loss', {expr:'ce(head)'}, 'L'); infer();`);
  await E(`fitPCA(pca.id, {yield:async () => {}, shouldStop:() => false})`);
  const g = gradCheck(E, 'L', [0, 1, 2]);
  assert.ok(g.worst < 2e-2 && g.sets === 2, `worst ${g.worst}, ${g.sets} sets`);   // fc and head; not the PCA
  assert.match(E('codegen()'), /nn\.Linear\(16, 4\)\.requires_grad_\(False\)/);
});

test('Plot 2D: shows what reaches it, keeps what an inference run sends, plots matrix rows', async () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true, run_from:'test', run_n:20}, 'data1'), fc = P('dense', {out_features:3}, 'fc'), pl = P('plot2d', {}, 'plot1'); wire(d, fc, pl); infer(); forward(root, null);`);
  assert.equal(E('(pl._r.err || "ok") + " " + pl._r.sum'), 'ok Values 0 and 1 of 3');
  assert.equal(E('pl._actIn[0].data.length'), 3);
  assert.equal(E('collectorsOf(d).map(m => m.name).join()'), 'plot1');
  const caps = await E(`collectLoop(d.id, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.equal(caps[E('pl.id')].n, 20);
  E(`pl.props.inputs = ['fc.pre_weights']; infer();`);   // 3 neurons × 64 inputs: one point per row
  assert.equal(E('pl._r.sum'), '3 rows as points');
  E(`pl.props.inputs = ['fc[0]']; infer();`);
  assert.match(E('pl._r.err'), /at least 2 values/);
});

test('Plot 3D: three values per point, kept from inference runs', async () => {
  const E = loadEngine(); toyDataset(E);
  E(`fresh(); var d = P('data', {dataset:'toy', flatten:true, run_from:'test', run_n:10}, 'data1'), fc = P('dense', {out_features:4}, 'fc'), pl = P('plot3d', {}, 'p3'); wire(d, fc, pl); infer(); forward(root, null);`);
  assert.equal(E('(pl._r.err || "ok") + " " + pl._r.sum'), 'ok Values 0, 1 and 2 of 4');
  const caps = await E(`collectLoop(d.id, {yield:async () => {}, shouldStop:() => false, progress(){}})`);
  assert.equal(caps[E('pl.id')].n, 10);
  E(`pl.props.z_dim = 4; infer();`);
  assert.match(E('pl._r.err'), /value indices from 0 to 3/);
  E(`fc.props.out_features = 2; pl.props.z_dim = 1; infer();`);
  assert.match(E('pl._r.err'), /Needs at least 3 values/);
});
