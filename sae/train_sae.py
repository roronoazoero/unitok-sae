import os
import sys
import math
import argparse
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

# Make project importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sae.config import CONFIG  # noqa: E402
from sae.model import TopKSAE  # noqa: E402


class H5ActivationDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._file = None
        with h5py.File(h5_path, "r") as f:
            self.length = f["activations"].shape[0]
            self.dim = f["activations"].shape[1]

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        return torch.from_numpy(self._file["activations"][idx]).float()


def lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one(
    activations_path: str,
    output_dir: str,
    k: int,
    expansion: int,
    steps: int,
    batch_size: int,
    lr: float,
    resume: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CONFIG.seed)

    print(f"\n{'='*30}\n[train] k={k}, expansion={expansion}\n{'='*30}")

    dataset = H5ActivationDataset(activations_path)
    print(f"[train] {len(dataset):,} vectors of dim {dataset.dim}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    sae = TopKSAE(
        input_dim=dataset.dim,
        hidden_dim=dataset.dim * expansion,
        k=k,
        dead_threshold_steps=CONFIG.dead_threshold_steps,
    ).to(device)
    print(f"[train] Params: {sum(p.numel() for p in sae.parameters()):,}")

    # Initialize pre_bias to data mean (helps optimization start)
    with h5py.File(activations_path, "r") as f:
        data_mean = torch.from_numpy(f["mean"][:]).float()
    sae.pre_bias.data.copy_(data_mean.to(device))

    optimizer = Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_schedule(s, CONFIG.warmup_steps, steps)
    )

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        sae.load_state_dict(ckpt["sae"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"[train] Resumed from step {start_step}")

    sae.train()
    data_iter = iter(loader)
    pbar = tqdm(range(start_step, steps), desc=f"k={k}")
    running = {"loss": 0.0, "recon": 0.0, "aux": 0.0, "l0": 0.0}
    log_n = 0

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = batch.to(device, non_blocking=True)

        out = sae(batch)
        aux_term = out["aux_loss"] if torch.is_tensor(out["aux_loss"]) else torch.tensor(0.0, device=device)
        loss = out["recon_loss"] + CONFIG.aux_loss_coef * aux_term

        optimizer.zero_grad()
        loss.backward()
        sae.remove_parallel_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), CONFIG.grad_clip)
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder()

        running["loss"] += loss.item()
        running["recon"] += out["recon_loss"].item()
        running["aux"] += float(aux_term.item())
        running["l0"] += (out["latent"] > 0).float().sum(-1).mean().item()
        log_n += 1

        if (step + 1) % CONFIG.log_interval == 0:
            avg = {k_: v / log_n for k_, v in running.items()}
            dead_pct = sae.get_dead_mask().float().mean().item() * 100
            pbar.set_postfix({
                "loss": f"{avg['loss']:.4f}",
                "recon": f"{avg['recon']:.4f}",
                "L0": f"{avg['l0']:.1f}",
                "dead%": f"{dead_pct:.1f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            })
            running = {k_: 0.0 for k_ in running}
            log_n = 0

        if (step + 1) % CONFIG.checkpoint_interval == 0:
            ckpt_path = os.path.join(
                output_dir,
                f"sae_block{CONFIG.target_block}_k{k}_step{step+1}.pt",
            )
            torch.save({
                "step": step + 1,
                "sae": sae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "input_dim": dataset.dim,
                "hidden_dim": dataset.dim * expansion,
                "k": k,
            }, ckpt_path)

    final_path = os.path.join(
        output_dir, f"sae_block{CONFIG.target_block}_k{k}_final.pt"
    )
    torch.save({
        "step": steps,
        "sae": sae.state_dict(),
        "input_dim": dataset.dim,
        "hidden_dim": dataset.dim * expansion,
        "k": k,
    }, final_path)
    print(f"[train] Saved {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=str, default=CONFIG.activations_path)
    parser.add_argument("--output_dir", type=str, default=CONFIG.checkpoint_dir)
    parser.add_argument("--steps", type=int, default=CONFIG.total_steps)
    parser.add_argument("--batch_size", type=int, default=CONFIG.sae_batch_size)
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--k", type=int, default=CONFIG.k)
    parser.add_argument("--expansion", type=int, default=CONFIG.expansion_factor)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help=f"Sweep k over {CONFIG.sweep_k_values}")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:
        for k_val in CONFIG.sweep_k_values:
            train_one(
                args.activations, args.output_dir, k_val, args.expansion,
                args.steps, args.batch_size, args.lr, resume=None,
            )
    else:
        train_one(
            args.activations, args.output_dir, args.k, args.expansion,
            args.steps, args.batch_size, args.lr, resume=args.resume,
        )


if __name__ == "__main__":
    main()
