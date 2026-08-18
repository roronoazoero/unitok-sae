import os
import sys
import math
import argparse
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sae.config import CONFIG
from sae.model import TopKSAE
from sae.unitok_loader import build_unitok
import wandb


def make_transform(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])


class OnTheFlyDataset(IterableDataset):
    def __init__(self, img_dir, unitok, block, device, img_size=256, img_batch=64):
        self.img_dir   = img_dir
        self.unitok    = unitok
        self.block     = block
        self.device    = device
        self.img_batch = img_batch
        self.transform = make_transform(img_size)
        self._n_imgs   = len(ImageFolder(img_dir))

    def __len__(self):
        return self._n_imgs * 256   

    def __iter__(self):
        dataset = ImageFolder(self.img_dir, transform=self.transform)
        loader  = DataLoader(
            dataset, batch_size=self.img_batch, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        for imgs, _ in loader:
            buf = []
            h = self.unitok.encoder.blocks[self.block].register_forward_hook(
                lambda m, i, o: buf.append(o.detach().float()))
            with torch.no_grad():
                self.unitok.encoder(imgs.to(self.device))
            h.remove()

            acts = buf[0].reshape(-1, buf[0].shape[-1]).cpu()  # (B*L, D)
            perm = torch.randperm(len(acts))
            yield from acts[perm]


def get_input_dim(unitok, block, device, img_size=256):
    """Run one dummy image to determine the embedding dimension at a given block."""
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    buf = []
    h = unitok.encoder.blocks[block].register_forward_hook(
        lambda m, i, o: buf.append(o.detach().float()))
    with torch.no_grad():
        unitok.encoder(dummy)
    h.remove()
    return buf[0].shape[-1]


def compute_data_mean(unitok, block, device, img_dir, n_batches=50, img_batch=64):
    """Estimate activation mean from the first n_batches image batches."""
    dataset = ImageFolder(img_dir, transform=make_transform())
    loader  = DataLoader(dataset, batch_size=img_batch, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    acc = None
    n   = 0
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        buf = []
        h = unitok.encoder.blocks[block].register_forward_hook(
            lambda m, i, o: buf.append(o.detach().float()))
        with torch.no_grad():
            unitok.encoder(imgs.to(device))
        h.remove()
        acts = buf[0].reshape(-1, buf[0].shape[-1])  # (B*L, D)
        if acc is None:
            acc = acts.mean(0)
        else:
            acc = acc + acts.mean(0)
        n += 1
    return (acc / n).cpu()


def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(sae, val_dataset, device, batch_size, n_batches=100):
    sae.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    total_mse, total_var, total_cos, total_l0, n = 0.0, 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = batch.to(device, non_blocking=True)
        out   = sae(batch)
        recon = out["reconstruction"]
        total_mse += F.mse_loss(recon, batch).item()
        total_var += batch.var().item()
        total_cos += F.cosine_similarity(recon, batch, dim=-1).mean().item()
        total_l0  += (out["latent"] > 0).float().sum(-1).mean().item()
        n += 1
    sae.train()
    mse = total_mse / max(n, 1)
    return {
        "val/recon_loss": mse,
        "val/r_squared":  1.0 - mse / max(total_var / max(n, 1), 1e-12),
        "val/cos_sim":    total_cos / max(n, 1),
        "val/L0":         total_l0  / max(n, 1),
    }


def train_one(unitok, block, train_img_dir, val_img_dir, output_dir,
              k, expansion, steps, batch_size, lr,
              resume=None, use_wandb=False, wandb_project="unitok-sae", wandb_run="",
              device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*30}\n[train] block={block}, k={k}, expansion={expansion}\n{'='*30}")
    
    input_dim  = get_input_dim(unitok, block, device)
    hidden_dim = input_dim * expansion
    print(f"[train] input_dim={input_dim}  hidden_dim={hidden_dim}")

    train_dataset = OnTheFlyDataset(train_img_dir, unitok, block, device)
    val_dataset   = OnTheFlyDataset(val_img_dir,   unitok, block, device)
    print(f"[train] ~{len(train_dataset):,} train tokens  ~{len(val_dataset):,} val tokens")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
    )

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        k=k,
        dead_threshold_steps=CONFIG.dead_threshold_steps,
    ).to(device)
    print(f"[train] SAE params: {sum(p.numel() for p in sae.parameters()):,}")

    print("[train] Estimating data mean")
    data_mean = compute_data_mean(unitok, block, device, train_img_dir)
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

    if use_wandb:
        run_name = wandb_run or f"sae_block{block}_k{k}"
        wandb.init(
            project=wandb_project, name=run_name, resume="allow",
            config={
                "block": block, "k": k, "expansion": expansion,
                "input_dim": input_dim, "hidden_dim": hidden_dim,
                "lr": lr, "batch_size": batch_size, "steps": steps,
                "warmup_steps": CONFIG.warmup_steps,
                "aux_loss_coef": CONFIG.aux_loss_coef,
                "dead_threshold_steps": CONFIG.dead_threshold_steps,
                "grad_clip": CONFIG.grad_clip,
            },
        )

    sae.train()
    data_iter = iter(train_loader)
    pbar      = tqdm(range(start_step, steps), desc=f"block{block}_k{k}")
    running   = {"loss": 0.0, "recon": 0.0, "aux": 0.0, "l0": 0.0}
    log_n     = 0

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = batch.to(device, non_blocking=True)

        out      = sae(batch)
        aux_term = out["aux_loss"] if torch.is_tensor(out["aux_loss"]) else torch.tensor(0.0, device=device)
        loss     = out["recon_loss"] + CONFIG.aux_loss_coef * aux_term

        optimizer.zero_grad()
        loss.backward()
        sae.remove_parallel_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), CONFIG.grad_clip)
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder()

        running["loss"]  += loss.item()
        running["recon"] += out["recon_loss"].item()
        running["aux"]   += float(aux_term.item())
        running["l0"]    += (out["latent"] > 0).float().sum(-1).mean().item()
        log_n += 1

        if (step + 1) % CONFIG.log_interval == 0:
            avg      = {k_: v / log_n for k_, v in running.items()}
            dead_pct = sae.get_dead_mask().float().mean().item() * 100
            cur_lr   = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss":  f"{avg['loss']:.4f}",
                "recon": f"{avg['recon']:.4f}",
                "L0":    f"{avg['l0']:.1f}",
                "dead%": f"{dead_pct:.1f}",
                "lr":    f"{cur_lr:.1e}",
            })
            if use_wandb:
                wandb.log({
                    "train/loss":       avg["loss"],
                    "train/recon_loss": avg["recon"],
                    "train/aux_loss":   avg["aux"],
                    "train/L0":         avg["l0"],
                    "train/dead_pct":   dead_pct,
                    "train/lr":         cur_lr,
                }, step=step + 1)
            running = {k_: 0.0 for k_ in running}
            log_n   = 0

        if (step + 1) % CONFIG.checkpoint_interval == 0:
            val_metrics = validate(sae, val_dataset, device, batch_size)
            print(f"\n[val  step={step+1}] " +
                  "  ".join(f"{k_}={v:.4f}" for k_, v in val_metrics.items()))
            if use_wandb:
                wandb.log(val_metrics, step=step + 1)
            ckpt_path = os.path.join(output_dir, f"sae_block{block}_k{k}_step{step+1}.pt")
            torch.save({
                "step": step + 1,
                "sae": sae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "input_dim": input_dim, "hidden_dim": hidden_dim, "k": k, "block": block,
            }, ckpt_path)

    final_path = os.path.join(output_dir, f"sae_block{block}_k{k}_final.pt")
    torch.save({
        "step": steps, "sae": sae.state_dict(),
        "input_dim": input_dim, "hidden_dim": hidden_dim, "k": k, "block": block,
    }, final_path)
    print(f"[train] Saved {final_path}")
    if use_wandb:
        wandb.finish()
    return final_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unitok_ckpt",     required=True)
    p.add_argument("--block",           type=int, default=15)
    p.add_argument("--imagenet_train",  required=True)
    p.add_argument("--imagenet_val",    required=True)
    p.add_argument("--output_dir",      default=CONFIG.checkpoint_dir)
    p.add_argument("--k",               type=int,   default=CONFIG.k)
    p.add_argument("--expansion",       type=int,   default=CONFIG.expansion_factor)
    p.add_argument("--steps",           type=int,   default=CONFIG.total_steps)
    p.add_argument("--batch_size",      type=int,   default=CONFIG.sae_batch_size)
    p.add_argument("--lr",              type=float, default=CONFIG.learning_rate)
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--wandb",           action="store_true")
    p.add_argument("--wandb_project",   type=str,   default="unitok-sae")
    p.add_argument("--wandb_run",       type=str,   default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    unitok = build_unitok(args.unitok_ckpt, device)

    train_one(
        unitok=unitok, block=args.block,
        train_img_dir=args.imagenet_train, val_img_dir=args.imagenet_val,
        output_dir=args.output_dir,
        k=args.k, expansion=args.expansion,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        resume=args.resume,
        use_wandb=args.wandb, wandb_project=args.wandb_project, wandb_run=args.wandb_run,
        device=device,
    )


if __name__ == "__main__":
    main()
