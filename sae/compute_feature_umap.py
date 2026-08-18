import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--output",          default="sae/analysis/umap_coords.json")
    p.add_argument("--dense_features",  default=None,
                   help="Path to dense_features.pt for coloring (optional)")
    p.add_argument("--n_neighbors",     type=int,   default=15)
    p.add_argument("--min_dist",        type=float, default=0.1)
    args = p.parse_args()

    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("Install umap-learn:  pip install umap-learn")

    print(f"Loading SAE from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    from sae.model import TopKSAE
    sae = TopKSAE(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["k"])
    sae.load_state_dict(ckpt["sae"])
    sae.eval()

    # decoder.weight: (input_dim, hidden_dim) — column i is feature i's direction
    W = sae.decoder.weight.detach().cpu().float().T.numpy()  # (hidden_dim, input_dim)
    hidden_dim, input_dim = W.shape
    print(f"Decoder directions: {hidden_dim} features × {input_dim} dims")

    print(
        f"Running UMAP  n_components=3  n_neighbors={args.n_neighbors}"
        f"  min_dist={args.min_dist}  metric=cosine ..."
    )
    reducer = umap_lib.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
        low_memory=False,
    )
    coords = reducer.fit_transform(W)  # (hidden_dim, 3)
    print(f"UMAP done.  output shape: {coords.shape}")

    fire_freq = None
    if args.dense_features:
        df_path = Path(args.dense_features)
        if df_path.exists():
            df = torch.load(df_path, map_location="cpu")
            fire_freq = df["fire_freq"].numpy().tolist()
            n_dense = int((df["fire_freq"] >= df["threshold"]).sum())
            print(f"Loaded activation frequency  ({n_dense} dense features at threshold={df['threshold']})")
        else:
            print(f"[warn] dense_features file not found: {df_path}")

    out = {
        "n_features":   hidden_dim,
        "feature_ids":  list(range(hidden_dim)),
        "x":            coords[:, 0].tolist(),
        "y":            coords[:, 1].tolist(),
        "z":            coords[:, 2].tolist(),
        "fire_freq":    fire_freq,
        "checkpoint":   args.checkpoint,
        "block":        int(ckpt.get("block", 15)),
        "k":            int(ckpt["k"]),
        "hidden_dim":   hidden_dim,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
