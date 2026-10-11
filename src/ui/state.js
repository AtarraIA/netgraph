const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const W = 200, PY0 = 46, PSTEP = 24;
const FAM = {io:'#D9DEE7', dense:'#7CC4FF', conv:'#FFC46B', pool:'#C3A2FF', act:'#FF7FB0', norm:'#6FF0C4', shape:'#D8EC7A', merge:'#6FE8F5', block:'#F2F4F8', data:'#FFFFFF', loss:'#FFE066', ptr:'#9DB4FF'};
const LABELS = {z_dim:'Third axis', components:'Components', fit_n:'Fit on (samples)', x_dim:'Horizontal axis', y_dim:'Vertical axis', target:'Feeds', pair:'Pointer in', fn:'Function of x', to:'Output shape', shape:'Input shape', in_features:'Input features', out_features:'Output features', bias:'Use bias', in_channels:'Input channels', out_channels:'Output channels', kernel:'Kernel size', stride:'Stride', padding:'Padding', p:'Drop probability'};
const HINTS = {
  fn:'Any formula in x, applied to the input: max(x, 0), x*sigmoid(x), max(x, a*x)… Other names become sliders. Functions: abs, log, exp, sqrt, tanh, sigmoid, relu, softmax, max, min, sum, mean. ReLU, GELU, Sigmoid, Tanh and Softmax are these formulas.',
  components:'How many principal components each sample is projected onto: its output size.',
  fit_n:'How many train samples to fit on, or all. Each costs about (input size)² operations.',
  z_dim:'Which value of each sample goes along the third axis, counting from 0.',
  x_dim:'Which value of each sample goes across, counting from 0.',
  y_dim:'Which value of each sample goes up, counting from 0.',
  target:'The layer where the stretch starts. Drag this pointer\'s output onto its input (its own wire stays), or type its name.',
  pair:'The Pointer in whose input the stretch runs on. Wire the layer where the stretch ends into this one.',
  shape:'Channels, height, width (e.g. 3,32,32) or a feature count (e.g. 784).',
  in_features:'"auto" reads it from the incoming tensor. With nothing connected, set a number and the layer takes its input directly.',
  in_channels:'"auto" reads it from the incoming tensor. Pin a number to assert it.',
  out_channels:'A number, or "same" to match the input channels.',
  stride:'"auto" uses the kernel size.',
  to:'"auto" picks channels × height × width with the largest square image that fits. Or type a shape like 32,7,7; one -1 is worked out for you.'
};
const PALETTE = [
  ['Data', ['data','sliders','input','output','plot2d','plot3d']],
  ['Linear', ['dense']],
  ['Convolution', ['conv2d','maxpool','avgpool','gap']],
  ['Activation', ['activation','relu','gelu','sigmoid','tanh','softmax']],
  ['Normalize and regularize', ['batchnorm','dropout']],
  ['Shape and merge', ['flatten','unflatten','add','concat','pointer','pca']],
  ['Training', ['loss']],
];
let lib = [];
let path = [];
let sel = new Set();
let selWire = null;
let boxMode = false;
let view = {x:40, y:80, k:1};
const views = {};
let tempWire = null;
let nview = null;
   // null: layer graph · {id:null}: current graph as neurons · {id}: one Dense layer as neurons
let nmodel = null, nbounds = null;
const modes = {};
const vkey = () => path.join('/') + (nview ? '#n' + (nview.id || '') : '');
function cur(){ let g = root; for (const id of path){ const n = g.nodes.find(n => n.id === id); if (!n) { path = []; return root; } g = n.graph; } return g; }
