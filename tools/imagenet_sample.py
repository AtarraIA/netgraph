# Turns a 1000-image ImageNet sample (one image per class, https://github.com/EliSchwartz/imagenet-sample-images)
# into IDX files the app imports with Import data:
#   git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images <src>
#   python3 tools/imagenet_sample.py <src> <out>
# Each image is prepared the way torchvision's AlexNet expects (shorter side to 256, center 224×224) and stored as
# 3×224×224 bytes, channel by channel. Labels are ImageNet class indices 0..999 (16-bit, since they pass 255), in the
# order of the sorted WordNet ids, the order torchvision's models use. Also writes classes.txt (index, id, name).
import sys, os, gzip, struct, numpy as np
from PIL import Image
from torchvision import transforms

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
files = sorted(f for f in os.listdir(src) if f.lower().endswith(('.jpeg', '.jpg')))
wnids = [f.split('_')[0] for f in files]
assert len(files) == 1000 and len(set(wnids)) == 1000, 'expected one image for each of the 1000 classes'
prep = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
images = np.stack([np.asarray(prep(Image.open(os.path.join(src, f)).convert('RGB'))).transpose(2, 0, 1) for f in files])
labels = np.arange(1000, dtype='>i2')

def idx(path, arr, code):   # IDX: magic (0, 0, type, number of dims), each dim as a big-endian uint32, then the data
    with gzip.open(path, 'wb') as f:
        f.write(struct.pack('>BBBB', 0, 0, code, arr.ndim) + struct.pack(f'>{arr.ndim}I', *arr.shape) + arr.tobytes())
idx(os.path.join(out, 'imagenet-sample-images-idx4-ubyte.gz'), images.astype(np.uint8), 0x08)
idx(os.path.join(out, 'imagenet-sample-labels-idx1-short.gz'), labels, 0x0B)
with open(os.path.join(out, 'classes.txt'), 'w') as f:
    for i, fn in enumerate(files): f.write(f"{i}\t{wnids[i]}\t{fn.split('_', 1)[1].rsplit('.', 1)[0].replace('_', ' ')}\n")
print(f'{len(files)} images, {images.shape[1]}×{images.shape[2]}×{images.shape[3]}, written to {out}')
