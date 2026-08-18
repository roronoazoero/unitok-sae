import argparse
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from sae.model import TopKSAE
from sae.unitok_loader import build_unitok


EDIT_KEYWORDS = {
    # "add_sun":     ["sun", "sunshine", "sunny", "sunlight"],
    "make_night":  ["night", "dark", "nighttime"],
    "add_snow":    ["snow", "snowy", "snowfall"],
    "add_rain":    ["rain", "rainy", "rainfall"],
    "make_autumn": ["autumn", "fall", "orange leaves"],
    "add_sunglasses": ["sunglass", "sunglasses", "sunnies"],
}


def get_edit_type(instr: str) -> str:
    instr_lower = instr.lower()
    for edit_type, keywords in EDIT_KEYWORDS.items():
        if any(kw in instr_lower for kw in keywords):
            return edit_type
    return "other"


def make_transform(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])


# same function as in the pgd notebook, but added a k_patches parameter that zeroes out all but the top-K patches by norm after each step
def find_delta_patchwise(
    sae, unitok, tensor_a, tensor_b, device,
    lr: float = 10.0, n_steps: int = 5000, log_every: int = 500, verbose = True, k_patches: int = None, dense_mask: torch.Tensor = None,
) -> tuple:
    with torch.no_grad():
        tgt_tokens = unitok.quant_proj(unitok.encoder(tensor_b).float()).detach()
    act_buffer = []
    def _collect(m, i, o):
        act_buffer.append(o.detach().float())
    handle = unitok.encoder.blocks[15].register_forward_hook(_collect)
    try:
        with torch.no_grad():
            unitok.encoder(tensor_a)
        src_acts = act_buffer[0]
    finally:
        handle.remove()
    B, L, D = src_acts.shape
    with torch.no_grad():
        src_latent, _, _, _ = sae.encode(src_acts.reshape(B * L, D))
        src_latent = src_latent.detach()
    delta = torch.zeros(L, sae.hidden_dim, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.5, min_lr=lr * 0.001)
    loss_steps = []
    lr_steps = []

    for _ in range(n_steps):
        optimizer.zero_grad()
        def _hook(module, inp, out):
            Bh, Lh, Dh = out.shape
            return sae.decode(src_latent + delta).reshape(Bh, Lh, Dh).to(out.dtype)
        handle = unitok.encoder.blocks[15].register_forward_hook(_hook)
        try:
            src_tokens = unitok.quant_proj(unitok.encoder(tensor_a).float())
        finally:
            handle.remove()
        loss = F.mse_loss(src_tokens, tgt_tokens)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if k_patches is not None and k_patches < L:
                patch_norms = delta.norm(dim=1)  # (L,)
                topk_idx = patch_norms.topk(k_patches).indices
                mask = torch.zeros(L, dtype=torch.bool, device=device)
                mask[topk_idx] = True
                delta[~mask] = 0.0
            if dense_mask is not None:
                delta[:, dense_mask] = 0.0
                
        scheduler.step(loss.item())
        loss_steps.append(loss.item())
        lr_steps.append(optimizer.param_groups[0]['lr'])

    return delta.detach(), loss.item()

# def find_delta_global(sae, unitok, tensor_a, tensor_b, device,
#                       lr: float = 10.0, n_steps: int = 5000, log_every: int = 500, k_c: int = 64, dense_mask: torch.Tensor = None,
#                      ) -> tuple:
#     with torch.no_grad():
#         tgt_tokens = unitok.quant_proj(unitok.encoder(tensor_b).float()).detach()
#         src_tokens_ref = unitok.quant_proj(unitok.encoder(tensor_a).float()).detach()
#         patch_diff = (tgt_tokens - src_tokens_ref).norm(dim=-1).squeeze(0) # (L,)
#         c_idx = patch_diff.topk(k_c).indices

#     buf = []
#     def _collect(m, i, o):
#         buf.append(o.detach().float())
#     handle = unitok.encoder.blocks[15].register_forward_hook(_collect)
#     with torch.no_grad():
#         unitok.encoder(tensor_a)
#     src_acts = buf[0]
#     handle.remove()

#     B, L, D = src_acts.shape
#     with torch.no_grad():
#         src_latent, _, _, _ = sae.encode(src_acts.reshape(B * L, D))
#         src_latent = src_latent.detach() # (B*L, H)

#     delta = torch.zeros(sae.hidden_dim, device=device, requires_grad=True)
#     optimizer = torch.optim.SGD([delta], lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.15, min_lr=lr * 0.001)

#     for step in range(1, n_steps+1):
#         optimizer.zero_grad()
#         def _hook(module, inp, out):
#             Bh, Lh, Dh = out.shape
#             return sae.decode(src_latent + delta.unsqueeze(0)).reshape(Bh, Lh, Dh).to(out.dtype)
#         handle = unitok.encoder.blocks[15].register_forward_hook(_hook)
#         src_tokens = unitok.quant_proj(unitok.encoder(tensor_a).float())
#         handle.remove()

#         loss = F.mse_loss(src_tokens[:, c_idx], tgt_tokens[:, c_idx])
#         loss.backward()
#         optimizer.step()
#         with torch.no_grad():
#             if dense_mask is not None:
#                 delta[dense_mask] = 0.0
#         schedule.step(loss.item())

#         if step % log_every == 0 or step == 1:
#             nnz = (delta.abs() > 1e-6).sum().item()
#             tqdm.write(f"    step {step:>5}/{n_steps}  loss={loss.item():.6f}  "
#                        f"lr={optimizer.param_groups[0]['lr']:.5f}  nnz={nnz}  "
#                        f"changed_patches={k_c}")
#     return delta.detach(), loss.item()
        
def find_delta_global_multi(sae, unitok, tensors_a, tensors_b, device, lr: float = 10.0, n_steps: int = 5000, log_every: int = 500, k_c: int = 64, dense_mask: torch.Tensor = None,
) -> tuple:
    N = len(tensors_a)
    tgt_tokens_list, src_latents = [], []
    act_buffer = []
    def _collect(m, i, o):
        act_buffer.append(o.detach().float())
    with torch.no_grad():
        for i in range(N):
            tgt_tok = unitok.quant_proj(unitok.encoder(tensors_b[i]).float()).detach()
            # src_ref = unitok.quant_proj(unitok.encoder(tensors_a[i]).float()).detach()
            # patch_diff = (tgt_tok - src_ref).norm(dim=-1).squeeze(0)   # (L,)
            tgt_tokens_list.append(tgt_tok)
            # c_idxs.append(patch_diff.topk(k_c).indices)
            handle = unitok.encoder.blocks[15].register_forward_hook(_collect)
            unitok.encoder(tensors_a[i])
            src_acts = act_buffer.pop()
            handle.remove()
            B, L, D = src_acts.shape
            latent, _, _, _ = sae.encode(src_acts.reshape(B * L, D))
            src_latents.append(latent.detach())

    delta = torch.zeros(sae.hidden_dim, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.5, min_lr=lr * 0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps,  eta_min=lr * 0.001)

    loss = None
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for i in range(N):
            def _hook(module, inp, out, idx=i):
                Bh, Lh, Dh = out.shape
                return sae.decode(src_latents[idx] + delta.unsqueeze(0)).reshape(Bh, Lh, Dh).to(out.dtype)
            handle = unitok.encoder.blocks[15].register_forward_hook(_hook)
            src_tok_i = unitok.quant_proj(unitok.encoder(tensors_a[i]).float())
            handle.remove()
            total_loss = total_loss + F.mse_loss(src_tok_i, tgt_tokens_list[i])
        loss = total_loss / N
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if dense_mask is not None:
                delta[dense_mask] = 0.0
        # scheduler.step(loss.item())
        scheduler.step()
        if step % log_every == 0 or step == 1:
            nnz = (delta.abs() > 1e-6).sum().item()
            tqdm.write(f"    step {step:>5}/{n_steps}  loss={loss.item():.6f}  "
                       f"lr={optimizer.param_groups[0]['lr']:.5f}  nnz={nnz}  "
                       f"pairs={N}")
    return delta.detach(), loss.item()

def run_edit_type(sae, unitok, ds, indices, edit_type, out_dir, device, args, dense_mask=None):
    edit_dir = Path(out_dir) / edit_type
    edit_dir.mkdir(parents=True, exist_ok=True)
    transform = make_transform()
    deltas, instructions = [], []

    for i, ds_idx in enumerate(tqdm(indices[: args.n_max], desc=edit_type)):
        pair_path = edit_dir / f"{i:04d}.pt"
        if pair_path.exists():
            saved = torch.load(pair_path, map_location="cpu")
            deltas.append(saved["delta"])
            instructions.append(saved["instruction"])
            tqdm.write(f"  [{i + 1}] resumed from disk")
            continue

        sample = ds[ds_idx]
        t_src = transform(sample["source_img"].convert("RGB")).unsqueeze(0).to(device)
        t_tgt = transform(sample["target_img"].convert("RGB")).unsqueeze(0).to(device)

        if args.global_delta:
            delta, final_loss = find_global_delta(
                sae, unitok, t_src, t_tgt, device,
                lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
                k_c=args.k_c, dense_mask=dense_mask,
            )
        else:
            delta, final_loss = find_delta_patchwise(
                sae, unitok, t_src, t_tgt, device,
                lr=args.lr, n_steps=args.n_steps, log_every=args.log_every,
                k_patches=args.k_patches,
                dense_mask=dense_mask,
            )
        torch.save({
            "delta": delta.cpu(),
            "delta_type":  "global" if args.global_delta else "patchwise",
            "instruction": sample["instruction"],
            "final_loss": final_loss,
        }, pair_path)

        deltas.append(delta.cpu())
        instructions.append(sample["instruction"])
        nnz = (delta.abs() > 1e-6).sum().item()
        tqdm.write(
            f"  [{i + 1}/{min(args.n_max, len(indices))}]"
            f"  loss={final_loss:.4f}  nnz={nnz}"
        )
    if not deltas:
        return

    stacked = torch.stack(deltas)  # (N, 256, 8192)
    out_path = Path(out_dir) / f"{edit_type}.pt"
    torch.save(
        {"deltas": stacked, "edit_type": edit_type, "instructions": instructions},
        out_path,
    )
    print(f"  → saved {len(deltas)} deltas to {out_path}")


def run_edit_type_multi(sae, unitok, ds, indices, edit_type, out_dir, device, args, dense_mask=None):
    edit_dir = Path(out_dir) / edit_type
    edit_dir.mkdir(parents=True, exist_ok=True)
    out_path = edit_dir / f"multi_global.pt"
    if out_path.exists():
        print(f"  already exists, skipping: {out_path}")
        return

    transform = make_transform()
    tensors_a, tensors_b, instructions = [], [], []
    print(f"  Loading {min(args.n_max, len(indices))} pairs")
    for ds_idx in indices[:args.n_max]:
        sample = ds[ds_idx]
        tensors_a.append(transform(sample["source_img"].convert("RGB")).unsqueeze(0).to(device))
        tensors_b.append(transform(sample["target_img"].convert("RGB")).unsqueeze(0).to(device))
        instructions.append(sample["instruction"])

    delta, final_loss = find_delta_global_multi(
        sae, unitok, tensors_a, tensors_b, device,
        lr=args.lr, n_steps=args.n_steps, log_every=args.log_every, dense_mask=dense_mask,
    )

    torch.save({
        "delta":        delta.cpu(),
        "delta_type":   "global_multi",
        "edit_type":    edit_type,
        "instructions": instructions,
        "n_pairs":      len(tensors_a),
        "final_loss":   final_loss,
    }, out_path)
    nnz = (delta.abs() > 1e-6).sum().item()
    print(f"  → loss={final_loss:.4f}  nnz={nnz}  saved to {out_path}")

def main():
    p = argparse.ArgumentParser(description="MagicBrush patchwise delta optimization")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--unitok_ckpt", required=True)
    p.add_argument("--out_dir",     default="sae/deltas")
    p.add_argument("--edit_types",  nargs="+", default=list(EDIT_KEYWORDS.keys()))
    p.add_argument("--n_max",   type=int,   default=10)
    p.add_argument("--n_steps", type=int,   default=5000)
    p.add_argument("--lr",      type=float, default=25.0)
    p.add_argument("--log_every", type=int,   default=500)
    p.add_argument("--k_patches", type=int,   default=None)
    # p.add_argument("--global_delta",   action="store_true")
    p.add_argument("--multi_global",   action="store_true")
    # p.add_argument("--k_c",          type=int,   default=64)
    p.add_argument("--dense_features", type=str, default=None)
    # p.add_argument("--eps",     type=float, default=5000.0)
    # p.add_argument("--lambda_reg", type=float, default=0.0, help="L2 regularization weight on delta (0 = unconstrained)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sae  = TopKSAE(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["k"]).eval().to(device)
    sae.load_state_dict(ckpt["sae"])
    print(f"SAE: k={ckpt['k']}  hidden={ckpt['hidden_dim']}")

    unitok = build_unitok(args.unitok_ckpt, device)

    print("Loading MagicBrush dataset...")
    from datasets import load_dataset
    ds = load_dataset("osunlp/MagicBrush", split="train")

    typed_groups: dict = defaultdict(list)
    for i, instr in enumerate(ds["instruction"]):
        typed_groups[get_edit_type(instr)].append(i)

    print("Edit type distribution:")
    for et, idxs in sorted(typed_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {et:20s}: {len(idxs)}")

    dense_mask=None
    if args.dense_features:
        d = torch.load(args.dense_features, map_location="cpu")
        dense_mask = torch.zeros(ckpt["hidden_dim"], dtype=torch.bool)
        dense_mask[d["dense_features"]] = True
        dense_mask = dense_mask.to(device)
        print(f"Dense mask: {dense_mask.sum().item()} features will be zeroed from delta each step")

    for edit_type in args.edit_types:
        indices = typed_groups.get(edit_type, [])
        if not indices:
            print(f"\nSkipping '{edit_type}': no samples found.")
            continue
        n_run = min(args.n_max, len(indices))
        print(f"\n── {edit_type} ({len(indices)} available, running {n_run}) ──")
        if args.multi_global:
            run_edit_type_multi(sae, unitok, ds, indices, edit_type, args.out_dir, device, args, dense_mask=dense_mask)
        else:
            run_edit_type(sae, unitok, ds, indices, edit_type, args.out_dir, device, args, dense_mask=dense_mask)

    print("\nAll done.")


if __name__ == "__main__":
    main()
