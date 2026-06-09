import os
import sys
import heapq
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_N_TOKENS   = 64
_PATCH_GRID = 8
_PATCH_SIZE = 32  # 256 / 8

def make_transform(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

# TODO: create a 3D mapping for all features and its top-activating images

def find_max_activating_examples(sae, unitok, imagenet_val_dir: str, feature_indices: list, k_top: int=16, img_batch: int=32, device: torch.device=None) -> dict:
    """
    Stream the entire ImageNet val set through UniTok encoder block 15 -> SAE.
    For each feature in feature_indices, track the k_top most strongly activating
    (image_idx, patch_idx) pairs.

    Uses topk_indices / topk_values from sae.encode() to avoid materializing the
    full (B, hidden_dim) sparse tensor.

    Returns:
        dict[feature_id] = [(activation_value, image_idx, patch_idx), ...]
        sorted largest activation first.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder(imagenet_val_dir, transform = make_transform())
    loader = DataLoader(dataset, batch_size=img_batch, shuffle=False, num_workers=4, pin_memory=True)

    feat_arr = np.array(feature_indices, dtype=np.int32)
    heaps: dict = {f: [] for f in feature_indices}

    act_buffer: list = []

    def _hook(module, inp, out):
        act_buffer.append(out.detach().cpu())

    handle = unitok.encoder.blocks[15].register_forward_hook(_hook)
    global_img_idx = 0
    k = sae.k

    try:
        for imgs, _ in tqdm(loader, desc="feature analysis"):
            act_buffer.clear()
            with torch.no_grad():
                imgs = imgs.to(device)
                _ = unitok.encoder(imgs)
                acts = act_buffer[0].float()  # (B, L, D)

                B, L, D = acts.shape
                acts_flat = acts.reshape(B * L, D).to(device)
                _, topk_idx, topk_vals, _ = sae.encode(acts_flat)
                # topk_idx:  (B*L, k)
                # topk_vals: (B*L, k), already ReLU'd, so non-negative

            topk_idx  = topk_idx.cpu().numpy().astype(np.int32)
            topk_vals = topk_vals.cpu().numpy().astype(np.float32)

            n_tok = B * L
            flat_feats = topk_idx.reshape(-1)    # (B*L*k,)
            flat_vals  = topk_vals.reshape(-1)   # (B*L*k,)

            # Derive image and patch indices for each flat entry
            flat_tok      = np.repeat(np.arange(n_tok, dtype=np.int32), k)
            flat_img_idx  = global_img_idx + (flat_tok // L)
            flat_patch_idx = flat_tok % L

            # Filter to tracked features with positive activation (zero = not really active)
            tracked_mask = np.isin(flat_feats, feat_arr) & (flat_vals > 0)

            if tracked_mask.any():
                tf = flat_feats[tracked_mask]
                tv = flat_vals[tracked_mask]
                ti = flat_img_idx[tracked_mask]
                tp = flat_patch_idx[tracked_mask]

                order = np.argsort(tf, kind="stable")
                tf, tv, ti, tp = tf[order], tv[order], ti[order], tp[order]

                i = 0
                while i < len(tf):
                    fid = int(tf[i])
                    j = i
                    while j < len(tf) and tf[j] == fid:
                        j += 1
                    heap = heaps[fid]
                    for idx in range(i, j):
                        val = float(tv[idx])
                        entry = (val, int(ti[idx]), int(tp[idx]))
                        if len(heap) < k_top:
                            heapq.heappush(heap, entry)
                        elif val > heap[0][0]:
                            heapq.heapreplace(heap, entry)
                    i = j

            global_img_idx += B

    finally:
        handle.remove()

    # Convert heaps to lists sorted largest activation first
    return {
        fid: sorted(heaps[fid], key=lambda x: -x[0])
        for fid in feature_indices
    }


def _load_raw_image(dataset: ImageFolder, idx: int) -> Image.Image:
    path, _ = dataset.samples[idx]
    return Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)


def _draw_patch_rect(img: Image.Image, patch_idx: int, color=(255, 0, 0), border_width: int = 3) -> Image.Image:
    row = patch_idx // _PATCH_GRID
    col = patch_idx % _PATCH_GRID
    x0  = col * _PATCH_SIZE
    y0  = row * _PATCH_SIZE
    x1  = x0 + _PATCH_SIZE - 1
    y1  = y0 + _PATCH_SIZE - 1
    draw = ImageDraw.Draw(img)
    for offset in range(border_width):
        draw.rectangle(
            [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
            outline=color,
        )
    return img


def visualize_max_activating(heap_results: dict, imagenet_val_dir: str, output_dir: str, n_features_to_show: int = 32) -> None:
    dataset_plain = ImageFolder(imagenet_val_dir)  # no transform, raw PIL images
    to_tensor     = transforms.ToTensor()

    feature_ids  = sorted(heap_results.keys())[:n_features_to_show]
    overview_imgs = []

    for fid in tqdm(feature_ids, desc="visualizing features"):
        entries = heap_results[fid]  # [(val, img_idx, patch_idx), ...]
        if not entries:
            continue

        # Group all activating patches by image, rank images by their peak activation
        img_patches: dict = {}
        for val, img_idx, patch_idx in entries:
            if img_idx not in img_patches:
                img_patches[img_idx] = {"max_val": val, "patches": []}
            img_patches[img_idx]["max_val"] = max(img_patches[img_idx]["max_val"], val)
            img_patches[img_idx]["patches"].append(patch_idx)

        top_imgs = sorted(img_patches.items(), key=lambda x: -x[1]["max_val"])[:16]

        grid_imgs = []
        for img_idx, info in top_imgs:
            pil = _load_raw_image(dataset_plain, img_idx)
            for patch_idx in info["patches"]:
                pil = _draw_patch_rect(pil, patch_idx)
            grid_imgs.append(to_tensor(pil))
            
        k = len(grid_imgs)
        grid = make_grid(torch.stack(grid_imgs), nrow=4, padding=2)
        save_image(grid, os.path.join(output_dir, f"feature_{fid:05d}_top{k}.png"))

        overview_imgs.append(grid_imgs[0])  # top-1 image per feature

    if overview_imgs:
        n_cols   = min(8, len(overview_imgs))
        overview = make_grid(torch.stack(overview_imgs), nrow=n_cols, padding=2)
        save_image(overview, os.path.join(output_dir, "feature_overview.png"))
        print(f"[features] Saved overview grid for {len(overview_imgs)} features to {output_dir}")


if __name__ == "__main__":
    from sae.model import TopKSAE
    from sae.unitok_loader import build_unitok

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--unitok_ckpt",  required=True)
    parser.add_argument("--imagenet_val", required=True)
    parser.add_argument("--output_dir",   default="sae/analysis/features_val/")
    parser.add_argument("--n_features",   type=int, default=256)
    parser.add_argument("--k_top",        type=int, default=64)
    parser.add_argument("--img_batch",    type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sae  = TopKSAE(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["k"]).eval().to(device)
    sae.load_state_dict(ckpt["sae"])

    unitok = build_unitok(args.unitok_ckpt, device)

    rng = np.random.default_rng(42)
    feature_indices = sorted(
        rng.choice(ckpt["hidden_dim"], size=args.n_features, replace=False).tolist()
    )

    results = find_max_activating_examples(
        sae=sae, unitok=unitok,
        imagenet_val_dir=args.imagenet_val,
        feature_indices=feature_indices,
        k_top=args.k_top,
        img_batch=args.img_batch,
        device=device,
    )
    visualize_max_activating(results, args.imagenet_val, args.output_dir)