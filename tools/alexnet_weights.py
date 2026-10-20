# Downloads torchvision's ImageNet-trained AlexNet and writes its weights, plus one test input and the logits
# PyTorch gives for it, as safetensors files for alexnet.mjs:
#   python3 tools/alexnet_weights.py <dir>
import sys, os, torch, torchvision
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).eval()
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(out, 'alexnet.safetensors'))
torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad(): y = model(x)
save_file({'x': x, 'y': y}, os.path.join(out, 'check.safetensors'))
print('saved', sum(p.numel() for p in model.parameters()), 'parameters')
