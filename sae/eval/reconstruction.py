import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

from sae.eval.common import make_transform, denorm, eval_forward


def _psnr(orig: np.ndarray, recon: np.ndarray) -> float: # PSNR between two (C,H,W) float32 arrays in [0, 1]
    mse = float(np.mean((orig - recon) ** 2))
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def _ssim(orig: np.ndarray, recon: np.ndarray) -> float: # SSIM between two (C,H,W) float32 arrays in [0, 1]
    from skimage.metrics import structural_similarity
    return float(structural_similarity(orig, recon, channel_axis=0, data_range=1.0))


def _save_batch_pngs(imgs_01: torch.Tensor, out_dir: str, start_idx: int) -> None: # Save (B,3,H,W) float [0,1] tensor as zero-padded PNG files
    arr = (imgs_01.cpu().clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for i, im in enumerate(arr):
        Image.fromarray(im).save(os.path.join(out_dir, f"{start_idx + i:06d}.png"))


def run_two_pass_eval(sae, unitok, train_dir: str, val_dir: str, n_train: int, n_val: int, img_batch: int, output_dir: str, device: torch.device, text_embs_fn=None, save_images: bool = True):
    """
    Run Pass A (no SAE) and Pass B (SAE hook at block 15) on n_train + n_val images.

    text_embs_fn: optional callable(unitok, dataset, device) -> text embedding tensor
                  for zero-shot accuracy. Pass None to skip zero-shot.

    Returns:
        pass_results: dict mapping split → per-image metrics
        tmp_dirs:     dict mapping split → (orig_dir, baseline_dir, sae_dir) for rFID
    """
    splits = []
    if train_dir and n_train > 0:
        splits.append(("train", train_dir, n_train))
    if val_dir and n_val > 0:
        splits.append(("val", val_dir, n_val))

    all_results: dict = {}
    tmp_dirs:    dict = {}

    for split, data_dir, n_images in splits:
        print(f"\n[eval] 2-pass reconstruction [{split}]  n={n_images}  dir={data_dir}")

        dataset  = ImageFolder(data_dir, transform=make_transform())
        n_images = min(n_images, len(dataset))
        loader   = DataLoader(
            Subset(dataset, list(range(n_images))),
            batch_size=img_batch, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        if save_images:
            d_orig = os.path.join(output_dir, f"tmp_{split}_original")
            d_base = os.path.join(output_dir, f"tmp_{split}_baseline")
            d_sae  = os.path.join(output_dir, f"tmp_{split}_sae")
            for d in [d_orig, d_base, d_sae]:
                os.makedirs(d, exist_ok=True)
            tmp_dirs[split] = (d_orig, d_base, d_sae)

        text_embs = None
        if text_embs_fn is not None:
            print(f"[eval] Building text embeddings for zero-shot ({split})...")
            text_embs = text_embs_fn(unitok, dataset, device)

        psnr_A, ssim_A = [], []
        psnr_B, ssim_B = [], []
        top1_A = top1_B = total_zs = 0
        img_offset = 0
        grid_saved = False

        for imgs, labels in tqdm(loader, desc=f"  {split}"):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            B      = imgs.size(0)

            rec_A, clip_A = eval_forward(unitok, imgs)

            def _hook(module, inp, out):
                latent, *_ = sae.encode(out.float())
                return sae.decode(latent).to(out.dtype)

            handle = unitok.encoder.blocks[15].register_forward_hook(_hook)
            try:
                rec_B, clip_B = eval_forward(unitok, imgs)
            finally:
                handle.remove()

            orig_01  = denorm(imgs)
            rec_A_01 = (rec_A.clamp(-1, 1) + 1) / 2
            rec_B_01 = (rec_B.clamp(-1, 1) + 1) / 2

            for i in range(B):
                o = orig_01[i].cpu().float().numpy()
                a = rec_A_01[i].cpu().float().numpy()
                b = rec_B_01[i].cpu().float().numpy()
                psnr_A.append(_psnr(o, a))
                ssim_A.append(_ssim(o, a))
                psnr_B.append(_psnr(o, b))
                ssim_B.append(_ssim(o, b))

            if save_images:
                _save_batch_pngs(orig_01,  d_orig, img_offset)
                _save_batch_pngs(rec_A_01, d_base, img_offset)
                _save_batch_pngs(rec_B_01, d_sae,  img_offset)

            if text_embs is not None:
                top1_A  += (clip_A @ text_embs.T).argmax(1).eq(labels).sum().item()
                top1_B  += (clip_B @ text_embs.T).argmax(1).eq(labels).sum().item()
                total_zs += B

            if not grid_saved and split == "val":
                n_show = min(8, B)
                grid = make_grid(
                    torch.cat([orig_01[:n_show], rec_A_01[:n_show], rec_B_01[:n_show]], 0),
                    nrow=n_show, padding=2,
                )
                save_image(grid, os.path.join(output_dir, "pixel_eval_grid.png"))
                grid_saved = True

            img_offset += B

        result = {
            "n_images":      len(psnr_A),
            "baseline_psnr": float(np.mean(psnr_A)),
            "sae_psnr":      float(np.mean(psnr_B)),
            "baseline_ssim": float(np.mean(ssim_A)),
            "sae_ssim":      float(np.mean(ssim_B)),
        }
        if total_zs > 0:
            result["baseline_zeroshot_top1"] = top1_A / total_zs
            result["sae_zeroshot_top1"]      = top1_B / total_zs

        all_results[split] = result

        dp = result["sae_psnr"] - result["baseline_psnr"]
        ds = result["sae_ssim"] - result["baseline_ssim"]
        print(f"  [{split}] PSNR  baseline={result['baseline_psnr']:.2f}  SAE={result['sae_psnr']:.2f}  Δ={dp:+.2f} dB")
        print(f"  [{split}] SSIM  baseline={result['baseline_ssim']:.4f}  SAE={result['sae_ssim']:.4f}  Δ={ds:+.4f}")
        if total_zs > 0:
            dz = result["sae_zeroshot_top1"] - result["baseline_zeroshot_top1"]
            print(f"  [{split}] ZS top-1  baseline={result['baseline_zeroshot_top1']:.2%}  "
                  f"SAE={result['sae_zeroshot_top1']:.2%}  Δ={dz:+.2%}")

    return all_results, tmp_dirs
