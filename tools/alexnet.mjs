// Builds torchvision's ImageNet-trained AlexNet as a Netgraph project file, with its weights:
//   python3 tools/alexnet_weights.py <dir>     # downloads the weights (needs torch and torchvision)
//   node tools/alexnet.mjs <dir> [out.json]    # writes the project file (default: <dir>/alexnet.netgraph.json)
// The network starts with a Data component for the ImageNet sample that tools/imagenet_sample.py makes (import it
// with Import data). The app feeds pixels as 0..1, so a fixed 1×1 convolution first applies the normalization
// torchvision's AlexNet was trained with: (pixel − mean)/std for each channel.
// The project is checked before it's written: it's read back the way the app opens it, and its output for the
// test input in <dir>/check.safetensors must match PyTorch's.
import fs from 'node:fs';
import path from 'node:path';
import {loadEngine, readSafetensors} from '../tests/helpers.mjs';

const dir = process.argv[2], out = process.argv[3] || path.join(dir || '', 'alexnet.netgraph.json');
if (!dir){ console.error('Usage: node tools/alexnet.mjs <dir> [out.json]'); process.exit(1); }
const sd = readSafetensors(fs.readFileSync(path.join(dir, 'alexnet.safetensors')));
const check = readSafetensors(fs.readFileSync(path.join(dir, 'check.safetensors')));

// torchvision's layers in order: [type, props, name, state_dict prefix]. Its AdaptiveAvgPool2d((6, 6)) is left out:
// a 224×224 input already reaches it as 256×6×6, so it changes nothing.
const DATASET = 'dataset-3-224-224';   // the id the app gives an imported 3×224×224 dataset
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
sd['normalize.weight'] = {data:Float32Array.from({length:9}, (_, i) => i % 4 ? 0 : 1/STD[i/4])};   // 3×3×1×1, diagonal
sd['normalize.bias'] = {data:Float32Array.from(MEAN, (m, c) => -m/STD[c])};
const LAYERS = [
  ['data', {dataset:DATASET, run_from:'all'}, 'imagenet'],
  ['conv2d', {out_channels:3, kernel:1, stride:1, padding:0}, 'normalize', 'normalize'],
  ['conv2d', {out_channels:64, kernel:11, stride:4, padding:2}, 'conv1', 'features.0'], ['relu', {}, 'relu1'], ['maxpool', {kernel:3, stride:2}, 'pool1'],
  ['conv2d', {out_channels:192, kernel:5, stride:1, padding:2}, 'conv2', 'features.3'], ['relu', {}, 'relu2'], ['maxpool', {kernel:3, stride:2}, 'pool2'],
  ['conv2d', {out_channels:384, kernel:3, stride:1, padding:1}, 'conv3', 'features.6'], ['relu', {}, 'relu3'],
  ['conv2d', {out_channels:256, kernel:3, stride:1, padding:1}, 'conv4', 'features.8'], ['relu', {}, 'relu4'],
  ['conv2d', {out_channels:256, kernel:3, stride:1, padding:1}, 'conv5', 'features.10'], ['relu', {}, 'relu5'], ['maxpool', {kernel:3, stride:2}, 'pool5'],
  ['flatten', {}, 'flatten'],
  ['dropout', {p:0.5}, 'drop6'], ['dense', {out_features:4096}, 'fc6', 'classifier.1'], ['relu', {}, 'relu6'],
  ['dropout', {p:0.5}, 'drop7'], ['dense', {out_features:4096}, 'fc7', 'classifier.4'], ['relu', {}, 'relu7'],
  ['dense', {out_features:1000}, 'fc8', 'classifier.6'],
  ['output', {}, 'logits'],
];

// A stand-in for the dataset, so the network's shapes work out here.
const STAND_IN = `datasets = [{id:'${DATASET}', name:'Dataset 3×224×224', c:3, h:224, w:224, classes:1000, n:1, images:new Uint8Array(3*224*224), labels:new Uint16Array(1)}];`;
const E = loadEngine();
E('fresh();' + STAND_IN);
const ids = JSON.parse(E(`JSON.stringify(${JSON.stringify(LAYERS)}.map(([t, p, name], i) => {
  const n = mk(t, (i % 8)*240, Math.floor(i/8)*220, p, name); root.nodes.push(n);
  if (i) root.edges.push(edge(root.nodes[i - 1].id, n.id));
  return n.id;
}))`));
E.weights = {};
LAYERS.forEach(([, , , pre], i) => {
  if (!pre) return;
  const W = sd[`${pre}.weight`], B = sd[`${pre}.bias`];
  E.ctx.__w = {id:ids[i], key:`${W.data.length}:${B.data.length}:`, W:W.data, B:B.data};
  E(`importWeights([__w]); delete globalThis.__w;`);
});
// The keys must be the ones the engine would make for these shapes, or it would replace the weights with fresh ones.
const bad = E(`infer(); JSON.stringify(root.nodes.filter(n => n._r && n._r.err).map(n => n.name + ': ' + n._r.err))`);
if (bad !== '[]') throw new Error(`The graph has errors: ${bad}`);
const params = E('stats.params');

const text = E('projectJSON([])');
fs.writeFileSync(out, text);

// Read it back as the app does and run the test input through it.
const F = loadEngine();
F.ctx.__text = fs.readFileSync(out, 'utf8');
F(STAND_IN + `(() => { const p = readProject(__text); delete globalThis.__text; root = p.graph; importWeights(p.weights); infer(); })()`);
F.ctx.__x = check.x.data.map((v, i) => v*STD[Math.floor(i/(224*224))] + MEAN[Math.floor(i/(224*224))]);   // the test input as 0..1 pixels
const y = Float32Array.from(JSON.parse(F(`(() => {
  let t = null;
  for (const n of topo(root)){
    if (n.type === 'data') t = tensor(n._out, 1, __x);
    else if (n.type !== 'output') t = runNode(n, [t], 'eval');
  }
  return JSON.stringify(Array.from(t.data));
})()`)));
const ref = check.y.data, scale = Math.max(...Array.from(ref, Math.abs));
let worst = 0; for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(y[i] - ref[i])/scale);
const top = a => Array.from(a.keys()).sort((i, j) => a[j] - a[i]).slice(0, 5).join(', ');
console.log(`Top-5 classes: Netgraph ${top(y)} · PyTorch ${top(ref)} · largest difference ${worst.toExponential(2)} of the largest logit`);
if (worst > 1e-4 || top(y) !== top(ref)) throw new Error('The project\'s output doesn\'t match PyTorch.');
console.log(`Wrote ${out}: ${params.toLocaleString()} parameters, ${(fs.statSync(out).size/2**20).toFixed(0)} MB`);
