/* Netgraph engine: PyTorch export.
   exportPlan() lays out a PyTorch module for the graph: the same layers under readable names, a forward() that
   also keeps every intermediate value in self.acts, and Loss and Flatten expressions translated to torch code.
   codegen() writes it as Python; stateDict() gives the current weights under the module's parameter names;
   safetensors() packs those into a file that model.load_state_dict(load_file(...)) reads. */
const pyName = s => { let n = String(s).toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'layer'; if (/^\d/.test(n)) n = 'l_' + n; return n; };
function moduleOf(n){
  const p = n._p || n.props, x = (n._in && n._in[0]) || [], o = n._out || [], q = v => v ?? '?';
  switch (n.type){
    case 'dense': return `nn.Linear(${q(x[x.length-1])}, ${q(o[o.length-1])}${p.bias ? '' : ', bias=False'})`;
    case 'conv2d': return `nn.Conv2d(${q(x[0])}, ${q(o[0])}, kernel_size=${p.kernel}, stride=${p.stride}, padding=${p.padding})`;
    case 'maxpool': return `nn.MaxPool2d(${p.kernel}, stride=${isAuto(p.stride) ? p.kernel : p.stride})`;
    case 'avgpool': return `nn.AvgPool2d(${p.kernel}, stride=${isAuto(p.stride) ? p.kernel : p.stride})`;
    case 'gap': return `nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())`;
    case 'flatten': return `nn.Flatten()`;
    case 'unflatten': return x.length > 1 ? `nn.Sequential(nn.Flatten(), nn.Unflatten(1, (${o.join(', ')})))` : `nn.Unflatten(1, (${o.join(', ')}${o.length === 1 ? ',' : ''}))`;
    case 'batchnorm': return `nn.BatchNorm${x.length === 3 ? '2d' : '1d'}(${q(x[0])})`;
    case 'dropout': return `nn.Dropout(p=${p.p})`;
    case 'activation': { const k = presetOf(n); return k ? moduleOf({...n, type:k}) : 'nn.Identity()'; }
    case 'relu': return `nn.ReLU()`;
    case 'gelu': return `nn.GELU(approximate="tanh")`;   // the same curve Netgraph uses
    case 'sigmoid': return `nn.Sigmoid()`;
    case 'tanh': return `nn.Tanh()`;
    case 'softmax': return `nn.Softmax(dim=${x.length === 1 ? -1 : 1})`;
    case 'pca': { const lin = `nn.Linear(${q(x.reduce((a, b) => a*b, 1))}, ${q(o[0])}).requires_grad_(False)`; return x.length > 1 ? `nn.Sequential(nn.Flatten(), ${lin})` : lin; }   // fixed: components · (x − mean)
  }
  return 'nn.Identity()';
}
// Torch code for an expression's value, laid out per sample in the given shape.
const shapeCode = (r, sh) => r.k === 'b' ? `${r.c}.reshape(-1, ${sh.join(', ')})` : r.k === 'w' ? `${r.c}.reshape(1, ${sh.join(', ')}).expand(_B, ${sh.map(() => -1).join(', ')})` : `torch.full((_B, ${sh.join(', ')}), ${r.c})`;
const pyNum = v => Number.isInteger(v) ? `${v}.0` : String(v);

function exportPlan(){
  infer();
  const used = new Set(['self', 'torch', 'nn', 'F', 'None', 'model', 'y', 'load_file', 'train', 'Net']);
  const un = b => { const base = pyName(b); let k = base, i = 2; while (used.has(k)) k = base + '_' + (i++); used.add(k); return k; };
  const plan = {init:[], fwd:[], args:[], rets:[], consts:[], losses:[], varOf:new Map(), mods:new Map(), pathOf:pathMap(), needB:false};
  // Picks positions out of a flattened tensor: a slice when they are contiguous, an index tensor otherwise.
  const pick = pos => {
    let run = true; for (let i = 1; i < pos.length; i++) if (pos[i] !== pos[0] + i){ run = false; break; }
    if (run) return `[:, ${pos[0]}:${pos[0] + pos.length}]`;
    const name = `_IDX${plan.consts.length}`; plan.consts.push(`${name} = torch.tensor([${Array.from(pos).join(', ')}])`);
    return `[:, ${name}]`;
  };
  // An expression in torch code. Values are [B, N] per sample ('b'), [1, N] for weights ('w') or plain numbers.
  plan.exprPy = (n, src, where, xVar) => {   // xVar: an Activation's input, for its function of x
    const ast = bindVars(parseLoss(src), n, !!xVar), owner = where === 'forward' ? 'self' : 'model';
    const kindOf = (...xs) => xs.some(x => x.k === 'b') ? 'b' : xs.some(x => x.k === 'w') ? 'w' : 'num';
    const tr = a => {
      switch (a.k){
        case 'num': return {c:pyNum(a.v), k:'num'};
        case 'ref': {
          if (a._var != null) return {c:pyNum(a._varVal), k:'num'};
          if (a._x) return {c:`${xVar}.flatten(1)`, k:'b'};
          if (a.attr){
            const wv = weightView(a, n._g), layer = a.attr === 'pre_weights' ? lookup(a.key0, n._g).node : wv.M, mod = plan.mods.get(layer.id);
            return {c:`${owner}.${mod.name}.weight.reshape(1, -1)${pick(wv.pos)}`, k:'w'};
          }
          const node = lookup(a.key, n._g).node;
          if (node.type === 'loss'){
            if (where === 'forward') throw new Error('an input cell can\'t use a Loss block in the export.');
            return {c:`loss_${pyName(node.name)}_per_sample(model, y)`, k:'b'};
          }
          const base = where === 'forward' ? plan.varOf.get(node.id) : `model.acts[${JSON.stringify(plan.pathOf.get(node.id).join('.'))}]`;
          const sh = node._out || (node._in && node._in[0]);
          return {c:`${base}.flatten(1)${a._idx ? pick(selIndex(sh, a._idx, a.label)) : ''}`, k:'b'};
        }
        case 'neg': { const x = tr(a.a); return {c:`(-${x.c})`, k:x.k}; }
        case 'bin': { const x = tr(a.a), y = tr(a.b); return {c:`(${x.c} ${a.op === '^' ? '**' : a.op} ${y.c})`, k:kindOf(x, y)}; }
        case 'fn': {
          const xs = a.args.map(tr), x = xs[0], t = s => x.k === 'num' ? `torch.as_tensor(${s})` : s;
          switch (a.f){
            case 'abs': return {c:`torch.abs(${t(x.c)})`, k:x.k};
            case 'log': return {c:`torch.log(torch.clamp(${t(x.c)}, min=1e-12))`, k:x.k};
            case 'exp': return {c:`torch.exp(${t(x.c)})`, k:x.k};
            case 'sqrt': return {c:`torch.sqrt(torch.clamp(${t(x.c)}, min=0))`, k:x.k};
            case 'tanh': return {c:`torch.tanh(${t(x.c)})`, k:x.k};
            case 'sigmoid': return {c:`torch.sigmoid(${t(x.c)})`, k:x.k};
            case 'relu': return {c:`torch.relu(${t(x.c)})`, k:x.k};
            case 'softmax': return x.k === 'num' ? {c:'1.0', k:'num'} : {c:`torch.softmax(${x.c}, dim=1)`, k:x.k};
            case 'max': case 'min': {
              if (xs.length === 1) return x.k === 'num' ? x : {c:`${x.c}.${a.f === 'max' ? 'amax' : 'amin'}(1, keepdim=True)`, k:x.k};
              const y = xs[1], tt = z => z.k === 'num' ? `torch.as_tensor(${z.c})` : z.c;
              return {c:`torch.${a.f === 'max' ? 'maximum' : 'minimum'}(${tt(x)}, ${tt(y)})`, k:kindOf(x, y)};
            }
            case 'sum': return x.k === 'num' ? x : {c:`${x.c}.sum(1, keepdim=True)`, k:x.k};
            case 'mean': return x.k === 'num' ? x : {c:`${x.c}.mean(1, keepdim=True)`, k:x.k};
            case 'mse': { const k = kindOf(x, xs[1]); return {c:`((${x.c} - ${xs[1].c}) ** 2)${k === 'num' ? '' : '.mean(1, keepdim=True)'}`, k}; }
            case 'ce': case 'nll':
              if (where === 'forward') throw new Error(`${a.f}() needs labels, which an input cell doesn't have in the export.`);
              return a.f === 'ce' ? {c:`F.cross_entropy(${x.c}, y, reduction="none").unsqueeze(1)`, k:'b'}
                : {c:`F.nll_loss(torch.log(torch.clamp(${x.c}, min=1e-7)), y, reduction="none").unsqueeze(1)`, k:'b'};
          }
        }
      }
    };
    return tr(ast);
  };
  const emit = (g, ginVar, prefix) => {
    let goutVar = 'None';
    for (const n of topo(g)){
      const ins = []; for (const e of g.edges) if (e.to === n.id && e.port < DEFS[n.type].ins && wired(n, e.port)) ins[e.port] = plan.varOf.get(e.from) ?? 'None';
      for (let i = 0; i < DEFS[n.type].ins; i++){   // inputs given by expressions, shaped like the input they feed
        const src = portExpr(n, i); if (!src) continue;
        const v = un(prefix + n.name + '_in' + (DEFS[n.type].ins > 1 ? 'ab'[i] : '')), sh = (n._in && n._in[i]) || [1];
        try {
          const r = plan.exprPy(n, src, 'forward'); if (r.k !== 'b') plan.needB = true;
          plan.fwd.push(`${v} = ${shapeCode(r, sh)}   # ${src}`);
        } catch(err){ plan.fwd.push(`${v} = None   # ${src}: ${err.message}`); }
        ins[i] = v;
      }
      const a = ins[0] ?? 'None', b = ins[1] ?? 'None', set = v => plan.varOf.set(n.id, v);
      switch (n.type){
        case 'data': if (ins[0] != null){ plan.fwd.push(`# ${n.name} collects ${ins[0]}`); set(ins[0]); break; }   // a collecting Data component passes the flow on
        // falls through
        case 'input': { const v = un(n.name); plan.args.push(v); set(v); break; }
        case 'sliders': {   // their current values, as a constant
          const v = un(prefix + n.name); plan.needB = true;
          plan.fwd.push(`${v} = torch.tensor([${(n.props.sliders || []).map(s => pyNum(Number(s.value) || 0)).join(', ')}]).expand(_B, -1)   # ${(n.props.sliders || []).map(s => s.name).join(', ')}`);
          set(v); break;
        }
        case 'loss': plan.losses.push(n); break;
        case 'output': plan.rets.push(a); set(a); break;
        case 'gin': set(ginVar); break;
        case 'gout': goutVar = a; set(a); break;
        case 'group': plan.fwd.push(`# ${n.name}`); set(emit(n.graph, a, prefix + pyName(n.name) + '_')); break;
        case 'ptrin': set(a); break;
        case 'plot2d': case 'plot3d': break;   // a view, not part of the model   // its input goes to the stretch its pair runs
        case 'ptrout': {   // the stretch again, calling the same modules
          if (!n._pg){ set('None'); break; }
          const keep = new Map(plan.varOf), x = plan.varOf.get(n._ptr.I.id) ?? 'None';
          plan.fwd.push(`# ${n.name}: ${n._ptr.a.name} … ${n._ptr.z.name} again on ${n._ptr.I.name}, same weights`);
          plan.reuse = (plan.reuse || 0) + 1; const v = emit(n._pg, x, prefix + pyName(n.name) + '_'); plan.reuse--;
          plan.varOf = keep; set(v); break;
        }
        case 'add': { const v = un(prefix + n.name); plan.fwd.push(`${v} = ${a} + ${b}`); set(v); break; }
        case 'concat': { const v = un(prefix + n.name); plan.fwd.push(`${v} = torch.cat([${a}, ${b}], dim=1)`); set(v); break; }
        case 'activation':
          if (!presetOf(n)){   // a function of its own: inline torch code
            const v = un(prefix + n.name), fn = String((n._p || n.props).fn);
            try { const r = plan.exprPy(n, fn, 'forward', a); if (r.k !== 'b') plan.needB = true; plan.fwd.push(`${v} = ${shapeCode(r, n._out || [1])}   # f(x) = ${fn}`); }
            catch(err){ plan.fwd.push(`${v} = None   # f(x) = ${fn}: ${err.message}`); }
            set(v); break;
          }   // falls through: a built-in becomes its PyTorch module
        default: {
          if (plan.reuse && plan.mods.has(n.id)){ const v = un(prefix + n.name); plan.fwd.push(`${v} = self.${plan.mods.get(n.id).name}(${a})`); set(v); break; }   // inside a Pointer
          let src = a; if (n._src){ src = un(prefix + n.name + '_in'); plan.args.push(src); }
          const m = un(prefix + n.name); plan.init.push(`self.${m} = ${moduleOf(n)}`); plan.fwd.push(`${m} = self.${m}(${src})`); set(m);
          plan.mods.set(n.id, {name:m, n});
        }
      }
    }
    return goutVar;
  };
  emit(root, null, '');
  return plan;
}

function codegen(){
  const plan = exportPlan(), L = ['import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', 'from safetensors.torch import load_file', ''];
  if (stats.errs) L.push(`# Note: the graph has ${stats.errs} unresolved issue${stats.errs>1?'s':''}; values marked "?" or None need fixing.`, '');
  const lossCode = plan.losses.map(n => { try { return {n, r:plan.exprPy(n, n.props.expr, 'loss')}; } catch(err){ return {n, err:err.message}; } });
  const vars = [];   // variables are exported with their current slider values
  (function walk(g){ for (const n of g.nodes){ for (const v of n._vars || []) vars.push(`#   ${n.name}: ${v} = ${n.props.vars[v].value}`); if (n.graph) walk(n.graph); } })(root);
  if (vars.length) L.push('# Variables, at their current values:', ...vars, '');
  if (plan.consts.length) L.push(...plan.consts, '');
  L.push('', 'class Net(nn.Module):', '    def __init__(self):', '        super().__init__()');
  if (!plan.init.length) L.push('        pass');
  plan.init.forEach(s => L.push('        ' + s));
  L.push('', `    def forward(self${plan.args.length ? ', ' + plan.args.join(', ') : ''}):`);
  if (plan.needB) L.push(`        _B = ${plan.args.length ? plan.args[0] + '.shape[0]' : '1'}`);
  plan.fwd.forEach(s => L.push('        ' + s));
  const acts = []; for (const [id, v] of plan.varOf) if (v && v !== 'None') acts.push(`${JSON.stringify(plan.pathOf.get(id).join('.'))}: ${v}`);
  L.push('        # every component\'s output, by the name Netgraph shows (Loss functions read these)');
  L.push(`        self.acts = {${acts.join(', ')}}`);
  L.push('        return ' + (plan.rets.length ? plan.rets.join(', ') : 'None  # add an Output layer'));
  for (const {n, r, err} of lossCode){
    const f = pyName(n.name);
    L.push('', '');
    if (err){ L.push(`# ${n.name} = ${n.props.expr}`, `# Not exported: ${err}`); continue; }
    L.push(`def loss_${f}_per_sample(model, y):`, `    """${n.name} = ${n.props.expr.replace(/"""/g, '')}, for each sample. Call model(x) first; y holds the labels."""`);
    L.push(`    return ${r.k === 'num' ? `torch.full((y.shape[0], 1), ${r.c})` : `${r.c}.mean(1, keepdim=True)`}`);
    L.push('', '', `def loss_${f}(model, y):`, `    return loss_${f}_per_sample(model, y).mean()`);
  }
  const first = lossCode.find(l => !l.err);
  if (first){
    const p = first.n.props, opt = p.optimizer === 'sgd' ? `torch.optim.SGD(model.parameters(), lr=${p.lr}, momentum=0.9)` : `torch.optim.Adam(model.parameters(), lr=${p.lr})`;
    L.push('', '', `def train(model, loader, epochs=${p.epochs}):`, `    """Trains like Netgraph's Train button on ${first.n.name}. loader gives batches (x, y) shaped like the Data component."""`,
      `    opt = ${opt}`, '    for _ in range(epochs):', '        model.train()', '        for x, y in loader:', '            opt.zero_grad()', '            model(x)',
      `            loss_${pyName(first.n.name)}(model, y).backward()`, '            opt.step()');
  }
  L.push('', '', 'if __name__ == "__main__":', '    model = Net()',
    '    # The weights from Netgraph: use "Download weights" in the export window, next to this file.',
    '    # model.load_state_dict(load_file("netgraph_weights.safetensors"))', '    model.eval()',
    `    print(sum(p.numel() for p in model.parameters()))  # ${stats.params.toLocaleString()} expected`);
  if (stats.ins.length) L.push(`    out = model(${stats.ins.map(s => `torch.randn(2, ${s.join(', ')})`).join(', ')})`, `    print(out.shape if torch.is_tensor(out) else [t.shape for t in out])`);
  return L.join('\n') + '\n';
}

// The current weights under the exported module's parameter names (the order model.state_dict() uses).
function stateDict(plan = exportPlan()){
  const out = [];
  for (const {name:m, n} of plan.mods.values()){
    const x = n._in && n._in[0], o = n._out, p = n._p || n.props; if (!x || !o) continue;
    if (n.type === 'dense'){
      const fi = lastDim(x), fo = lastDim(o), P = weightsOf(n, fi*fo, fo, fi);
      out.push({name:`${m}.weight`, shape:[fo, fi], data:P.W}); if (p.bias) out.push({name:`${m}.bias`, shape:[fo], data:P.B});
    } else if (n.type === 'conv2d'){
      const c = x[0], co = o[0], k = +p.kernel, P = weightsOf(n, co*c*k*k, co, c*k*k);
      out.push({name:`${m}.weight`, shape:[co, c, k, k], data:P.W}, {name:`${m}.bias`, shape:[co], data:P.B});
    } else if (n.type === 'pca'){   // as a Linear layer: weight = components, bias = −components · mean
      const S = sizeOf(x), k = o[0], P = pcaParams(n, S, k); if (!P) continue;
      const bias = new Float32Array(k); for (let j = 0; j < k; j++){ let s = 0; for (let i = 0; i < S; i++) s += P.W[j*S + i]*P.B[i]; bias[j] = -s; }
      const pre = x.length > 1 ? `${m}.1` : m; out.push({name:`${pre}.weight`, shape:[k, S], data:P.W}, {name:`${pre}.bias`, shape:[k], data:bias});
    } else if (n.type === 'batchnorm'){
      const C = x[0], P = bnParams(n, C);
      out.push({name:`${m}.weight`, shape:[C], data:P.W}, {name:`${m}.bias`, shape:[C], data:P.B}, {name:`${m}.running_mean`, shape:[C], data:P.rm},
        {name:`${m}.running_var`, shape:[C], data:P.rv}, {name:`${m}.num_batches_tracked`, shape:[], dtype:'I64', data:BigInt64Array.of(BigInt(P.nbt || 0))});
    }
  }
  return out;
}
// The safetensors format: an 8-byte little-endian header length, a JSON header, then the raw tensor bytes.
function safetensors(tensors){
  const header = {}; let off = 0;
  for (const t of tensors){ const bytes = t.data.byteLength; header[t.name] = {dtype:t.dtype || 'F32', shape:t.shape, data_offsets:[off, off + bytes]}; off += bytes; }
  let hb = new TextEncoder().encode(JSON.stringify(header));
  const pad = (8 - hb.length % 8) % 8; if (pad) hb = new TextEncoder().encode(JSON.stringify(header) + ' '.repeat(pad));
  const buf = new Uint8Array(8 + hb.length + off), dv = new DataView(buf.buffer);
  dv.setBigUint64(0, BigInt(hb.length), true); buf.set(hb, 8);
  let o = 8 + hb.length;
  for (const t of tensors){ buf.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), o); o += t.data.byteLength; }
  return buf;
}
