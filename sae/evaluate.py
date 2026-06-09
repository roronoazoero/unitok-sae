import argparse
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sae.eval.common import load_sae, build_unitok
from sae.eval.reconstruction import run_two_pass_eval
from sae.eval.rfid import compute_rfid
from sae.eval.results import save_results
from sae.eval.zeroshot import load_class_names, make_text_embs_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TopKSAE checkpoint")
    parser.add_argument("--checkpoint",           required=True)
    parser.add_argument("--unitok_ckpt",          required=True)
    parser.add_argument("--imagenet_train",        default=None)
    parser.add_argument("--imagenet_val",          default=None)
    parser.add_argument("--output_dir",            default="sae/eval_results/")
    parser.add_argument("--n_train",               type=int, default=3000)
    parser.add_argument("--n_val",                 type=int, default=2000)
    parser.add_argument("--img_batch",             type=int, default=32)
    parser.add_argument("--n_max_act_features",    type=int, default=256)
    parser.add_argument("--feature_extractor_path", default=None)
    parser.add_argument("--class_names_json",      default=None)
    parser.add_argument("--skip_rfid",             action="store_true")
    parser.add_argument("--skip_zeroshot",         action="store_true")
    parser.add_argument("--skip_features",         action="store_true")
    parser.add_argument("--device",               default=None)
    args = parser.parse_args()

    if not args.imagenet_train and not args.imagenet_val:
        parser.error("Provide at least one of --imagenet_train or --imagenet_val")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[eval] Device: {device}")

    sae, ckpt_meta = load_sae(args.checkpoint, device)
    print(f"[eval] SAE: k={ckpt_meta['k']}  hidden_dim={ckpt_meta['hidden_dim']}")

    unitok = build_unitok(args.unitok_ckpt, device)

    text_embs_fn = None
    if not args.skip_zeroshot:
        if args.class_names_json:
            class_names = load_class_names(args.class_names_json)
            text_embs_fn = make_text_embs_fn(class_names)
            print(f"[eval] Loaded {len(class_names)} class names for zero-shot")
        else:
            print("[eval] --class_names_json not provided; zero-shot eval skipped")

    metrics: dict = {"k": ckpt_meta["k"], "hidden_dim": ckpt_meta["hidden_dim"]}

    pass_results, tmp_dirs = run_two_pass_eval(
        sae          = sae,
        unitok       = unitok,
        train_dir    = args.imagenet_train,
        val_dir      = args.imagenet_val,
        n_train      = args.n_train,
        n_val        = args.n_val,
        img_batch    = args.img_batch,
        output_dir   = args.output_dir,
        device       = device,
        text_embs_fn = text_embs_fn,
        save_images  = not args.skip_rfid,
    )
    metrics["reconstruction"] = pass_results

    if not args.skip_rfid and tmp_dirs:
        metrics["rfid"] = compute_rfid(tmp_dirs, args.feature_extractor_path)

    if not args.skip_features and args.imagenet_val:
        from sae.feature_analysis import find_max_activating_examples, visualize_max_activating
        print(f"\n[eval] Feature interpretability ({args.n_max_act_features} features)")
        feat_dir = os.path.join(args.output_dir, "feature_max_activating")
        os.makedirs(feat_dir, exist_ok=True)
        rng = np.random.default_rng(42)
        feature_indices = sorted(
            rng.choice(ckpt_meta["hidden_dim"], size=args.n_max_act_features,
                       replace=False).tolist()
        )
        heap_results = find_max_activating_examples(
            sae=sae, unitok=unitok,
            imagenet_val_dir=args.imagenet_val,
            feature_indices=feature_indices,
            k_top=16,
            img_batch=args.img_batch,
            device=device,
        )
        visualize_max_activating(
            heap_results=heap_results,
            imagenet_val_dir=args.imagenet_val,
            output_dir=feat_dir,
            n_features_to_show=min(32, args.n_max_act_features),
        )

    save_results(metrics, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
