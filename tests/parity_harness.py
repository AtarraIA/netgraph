"""Runs an exported Netgraph model in PyTorch for the parity tests.

Reads model.py, weights.safetensors, batch.safetensors and config.json from the directory given, then:
one forward pass (train or eval mode), the loss, its backward pass, and one optimizer step. Writes the
results to result.safetensors so the test can compare them with Netgraph's own engine."""
import importlib.util
import json
import sys

import torch
from safetensors.torch import load_file, save_file

d = sys.argv[1]
cfg = json.load(open(f"{d}/config.json"))
spec = importlib.util.spec_from_file_location("netgraph_model", f"{d}/model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Net()
model.load_state_dict(load_file(f"{d}/weights.safetensors"))
batch = load_file(f"{d}/batch.safetensors")
x, y = batch["x"], batch["y"]
model.train(cfg["train"])

out = model(x)
loss = getattr(mod, cfg["loss"])(model, y)
loss.backward()

res = {"loss": loss.detach().reshape(1)}
if torch.is_tensor(out):
    res["out"] = out.detach().contiguous()
for k, p in model.named_parameters():
    if p.grad is not None:
        res["grad." + k] = p.grad.detach().contiguous()
for k, v in model.state_dict().items():
    if "running" in k:
        res["buf." + k] = v.detach().float().contiguous()

if cfg["opt"] == "sgd":
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
else:
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
opt.step()
for k, p in model.named_parameters():
    res["step." + k] = p.detach().contiguous()

save_file(res, f"{d}/result.safetensors")
