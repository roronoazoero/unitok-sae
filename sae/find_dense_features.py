import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from sae.model import TopKSAE
from sae.unitok_loader import build_unitok

def make_transform(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2 - 1),
    ])

@torch.no_grad()
def get_active_features(img_tensor, unitok, sae, device):
    buf = []
    def _hook(m, i, o): buf.append(o.float())
    h = unitok.encoder.blocks[15].register_forward_hook(_hook)
    try:
        unitok.encoder(img_tensor.to(device))
    finally:
        h.remove()
    B, L, D = buf[0].shape
    _, topk_idx, _, _ = sae.encode(buf[0].reshape(B * L, D))
    return set(topk_idx.reshape(-1).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--unitok_ckpt",  required=True)
    p.add_argument("--imagenet_val", required=True)
    p.add_argument("--n_images",  type=int,   default=100)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", default="sae/dense_features.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sae  = TopKSAE(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["k"]).eval().to(device)
    sae.load_state_dict(ckpt["sae"])
    print(f"SAE: k={ckpt['k']}  hidden={ckpt['hidden_dim']}")

    unitok = build_unitok(args.unitok_ckpt, device)

    dataset = ImageFolder(args.imagenet_val, transform=make_transform())
    n_classes = len(dataset.classes)

    n_images = min(args.n_images, n_classes)
    class_to_idx = {}
    for img_idx, (_, label) in enumerate(dataset.samples):
        if label not in class_to_idx:
            class_to_idx[label] = img_idx
        if len(class_to_idx) == n_images:
            break
    selected = list(class_to_idx.values())
    print(f"Probing {len(selected)} images, one per class")
    fire_count = torch.zeros(ckpt["hidden_dim"], dtype=torch.int32)

    for img_idx in tqdm(selected, desc="Scanning"):
        img_tensor, _ = dataset[img_idx]
        active = get_active_features(img_tensor.unsqueeze(0), unitok, sae, device)
        for fid in active:
            fire_count[fid] += 1

    fire_freq = fire_count.float() / len(selected) 

    threshold = args.threshold
    dense_mask     = fire_freq >= threshold
    dense_features = dense_mask.nonzero().squeeze(1)
    dense_freqs    = fire_freq[dense_features]
    order          = dense_freqs.argsort(descending=True)
    dense_features = dense_features[order]
    dense_freqs    = dense_freqs[order]

    print(f"\n── Dense features (fire in ≥{threshold*100:.0f}% of images) ──")
    print(f"  Found {len(dense_features)} / {ckpt['hidden_dim']} features")
    print(f"\n  {'Feature':>8}  {'Freq':>6}")
    for fid, freq in zip(dense_features[:50].tolist(), dense_freqs[:50].tolist()):
        print(f"  {fid:>8}  {freq*100:>5.1f}%")

    universal = (fire_count == len(selected)).nonzero().squeeze(1)
    print(f"\n  Features firing in ALL {len(selected)} images: {len(universal)}")
    if len(universal):
        print(f"  {universal.tolist()}")

    torch.save({
        "fire_count":     fire_count,
        "fire_freq":      fire_freq,
        "dense_features": dense_features,
        "dense_freqs":    dense_freqs,
        "n_images":       len(selected),
        "threshold":      threshold,
    }, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()