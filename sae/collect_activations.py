import os
import sys
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Make project importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sae.config import CONFIG  # noqa: E402
from sae.unitok_loader import build_unitok  # noqa: E402

def make_transform():
    return transforms.Compose([
        transforms.Resize((CONFIG.img_size, CONFIG.img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CONFIG.unitok_ckpt)
    parser.add_argument("--block", type=int, default=CONFIG.target_block)
    parser.add_argument("--data_dir", type=str,
                        default=CONFIG.imagenet_train_dir_resolved)
    parser.add_argument("--output", type=str, default=CONFIG.activations_path)
    parser.add_argument("--batch_size", type=int, default=CONFIG.collect_batch_size)
    parser.add_argument("--num_workers", type=int, default=CONFIG.num_workers)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CONFIG.seed)

    transform = make_transform()
    print(f"[collect] Loading ImageFolder from {args.data_dir}")
    dataset = ImageFolder(args.data_dir, transform=transform)
    print(f"[collect] Found {len(dataset):,} images across {len(dataset.classes)} classes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_unitok(args.ckpt, device)

    # Hook the target block
    activation_buffer = []

    def hook(module, inp, out):
        # out shape: (B, L, embed_dim); L=64 for vitamin_large at 256px
        activation_buffer.append(out.detach().to(torch.float32).cpu())

    target_block = model.encoder.blocks[args.block]
    handle = target_block.register_forward_hook(hook)
    print(f"[collect] Hooked block {args.block}: {type(target_block).__name__}")

    expected_tokens = len(dataset) * 64  # 64 tokens per image at 256px
    print(f"[collect] Expected total tokens: {expected_tokens:,}")
    print(f"[collect] Estimated HDF5 size: {expected_tokens * CONFIG.input_dim * 4 / 1e9:.1f} GB")

    with h5py.File(args.output, "w") as f:
        dset = f.create_dataset(
            "activations",
            shape=(0, CONFIG.input_dim),
            maxshape=(None, CONFIG.input_dim),
            chunks=(8192, CONFIG.input_dim),
            dtype="float16",
            compression="gzip",
            compression_opts=4,
        )

        # Welford streaming statistics (vectorized per-batch)
        running_mean = np.zeros(CONFIG.input_dim, dtype=np.float64)
        running_M2 = np.zeros(CONFIG.input_dim, dtype=np.float64)
        n_seen = 0

        labels_dset = None

        pbar = tqdm(total=len(dataset), desc="Collecting")
        n_imgs = 0

        for batch, _labels in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model.encoder(batch)

            # The hook fires once per encoder forward (grad_ckpt=False ensures this)
            acts = activation_buffer.pop(0)
            activation_buffer.clear()

            B, L, D = acts.shape
            acts_flat = acts.reshape(-1, D).numpy().astype(np.float64)

            # store per token labels
            if labels_dset is None:
                labels_dset = f.create_dataset(
                    "labels",
                    shape=(0,), maxshape=(None,), dtype="int32",
                )
            repeated = np.repeat(_labels.numpy().astype(np.int32), L)
            cur_l = labels_dset.shape[0]
            labels_dset.resize(cur_l + len(repeated), axis=0)
            labels_dset[cur_l:] = repeated

            # Append to HDF5
            cur = dset.shape[0]
            new = cur + acts_flat.shape[0]
            dset.resize(new, axis=0)
            dset[cur:new] = acts_flat.astype(np.float32)

            # Vectorized Welford merge: treat batch as a sub-population
            batch_n = acts_flat.shape[0]
            batch_mean = acts_flat.mean(axis=0)
            batch_M2 = ((acts_flat - batch_mean) ** 2).sum(axis=0)
            delta = batch_mean - running_mean
            new_n = n_seen + batch_n
            running_mean = running_mean + delta * (batch_n / new_n)
            running_M2 = running_M2 + batch_M2 + (delta ** 2) * (n_seen * batch_n / new_n)
            n_seen = new_n

            n_imgs += B
            pbar.update(B)

        pbar.close()
        handle.remove()

        running_var = running_M2 / max(n_seen - 1, 1)
        running_std = np.sqrt(running_var)

        f.create_dataset("mean", data=running_mean.astype(np.float32))
        f.create_dataset("std", data=running_std.astype(np.float32))
        f.attrs["target_block"] = args.block
        f.attrs["num_images"] = n_imgs
        f.attrs["model_name"] = CONFIG.model_name
        f.attrs["img_size"] = CONFIG.img_size
        f.attrs["data_dir"] = args.data_dir

        print(f"\n[collect] Saved {dset.shape[0]:,} vectors to {args.output}")
        print(f"[collect] Mean |x|: {np.abs(running_mean).mean():.4f}")
        print(f"[collect] Avg std: {running_std.mean():.4f}")


if __name__ == "__main__":
    main()
