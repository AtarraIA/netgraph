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
- `src/examples/autoencoder.json` is the project a first visit opens (an MNIST autoencoder, trained). To change it, save a project from the app over that file and rebuild.

## Tests

- `tests/engine.test.mjs`: shapes, variables, expressions and their errors, numerical gradient checks for every layer type (BatchNorm in both modes), reference rewriting on rename/group, a training run, and the safetensors format.
- `tests/parity.test.mjs`: exports networks, runs the same batch in real PyTorch (`tests/parity_harness.py`), and compares outputs, loss, every gradient, BatchNorm running statistics and one optimizer step. Needs `python3` with `torch` and `safetensors`; skipped without them.

## Pretrained models

`tools/` turns torchvision's ImageNet-trained AlexNet into a project file you can open in the app (**Open**, or drop it on the canvas), with its trained weights, and makes a 1000-image ImageNet sample (one image per class) to run it on:

```sh
python3 tools/alexnet_weights.py models                   # downloads the weights (needs torch, torchvision, safetensors)
node --max-old-space-size=8192 tools/alexnet.mjs models   # writes models/alexnet.netgraph.json (about 310 MB)
git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images /tmp/imagenet-sample-images
python3 tools/imagenet_sample.py /tmp/imagenet-sample-images models/imagenet-sample   # the two IDX files (about 115 MB) and classes.txt
```

In the app, **Import data** with the two files in `models/imagenet-sample/`, then open the project. The images are 3×224×224 (RGB), labelled with ImageNet's class indices 0 to 999; `classes.txt` gives each index's name. The project starts with ImageNet's normalization as a fixed 1×1 convolution, since the app feeds pixels as 0 to 1. `tools/alexnet.mjs` reads the project back the way the app does and checks that its output for a test input matches PyTorch's.

Import data takes IDX files: images as N×H×W (grayscale) or N×C×H×W (C channels) unsigned bytes, labels as unsigned bytes or, past 255 classes, 16-bit integers. `models/` isn't committed: the files are too big for GitHub.
