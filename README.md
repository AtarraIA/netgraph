# Netgraph

Compose neural networks as a graph, train them on MNIST in the browser, look inside every layer, and export them to PyTorch.

**Try it: [atarraia.github.io/netgraph/app](https://atarraia.github.io/netgraph/app/)** · Docs: [English](https://atarraia.github.io/netgraph/) · [Español](https://atarraia.github.io/netgraph/es/)

Or open `Netgraph_ compose neural networks.html` in a browser; it needs no server. To load MNIST, use **Import data** and pick the four files in `data/mnist/`. They are kept in the browser afterwards.

## Working on it

The HTML file is built from `src/`. Edit the files there, then rebuild:

```sh
node build.mjs     # rebuild the HTML file
npm test           # rebuild, then run every test
```

- `src/engine/` is the DOM-free core: graph model, shape inference, datasets, expressions, weights, batched forward and backward passes, training, and PyTorch export. The same code runs on the page, in the training Web Worker, and in the tests.
- `src/ui/` holds the page: canvas, components, info window, neuron view, editing, undo, and the training controls.
- `src/sources.mjs` lists the files in load order.

## Tests

- `tests/engine.test.mjs`: shapes, variables, expressions and their errors, numerical gradient checks for every layer type (BatchNorm in both modes), reference rewriting on rename/group, a training run, and the safetensors format.
- `tests/parity.test.mjs`: exports networks, runs the same batch in real PyTorch (`tests/parity_harness.py`), and compares outputs, loss, every gradient, BatchNorm running statistics and one optimizer step. Needs `python3` with `torch` and `safetensors`; skipped without them.
