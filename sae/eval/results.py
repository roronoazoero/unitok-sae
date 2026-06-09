import datetime
import json
import os


def save_results(metrics: dict, checkpoint_path: str, output_dir: str) -> None:
    summary = {
        "checkpoint": checkpoint_path,
        "date":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        **metrics,
    }
    path = os.path.join(output_dir, "eval_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    _print_table(metrics)
    print(f"\n[eval] Summary saved → {path}")


def _print_table(metrics: dict) -> None:
    w = 64
    print("\n" + "=" * w)
    print("  EVALUATION RESULTS")
    print("=" * w)
    for split, m in metrics.get("reconstruction", {}).items():
        n  = m.get("n_images", "?")
        dp = m["sae_psnr"]  - m["baseline_psnr"]
        ds = m["sae_ssim"]  - m["baseline_ssim"]
        print(f"\n  [{split}]  n={n}")
        print(f"    PSNR   baseline={m['baseline_psnr']:6.2f} dB   SAE={m['sae_psnr']:6.2f} dB   Δ={dp:+.2f}")
        print(f"    SSIM   baseline={m['baseline_ssim']:.4f}      SAE={m['sae_ssim']:.4f}      Δ={ds:+.4f}")
        if "baseline_zeroshot_top1" in m:
            dz = m["sae_zeroshot_top1"] - m["baseline_zeroshot_top1"]
            print(f"    ZS-1   baseline={m['baseline_zeroshot_top1']:.2%}         SAE={m['sae_zeroshot_top1']:.2%}         Δ={dz:+.2%}")
    if "rfid" in metrics:
        print()
        for split, m in metrics["rfid"].items():
            df = m["sae_fid"] - m["baseline_fid"]
            print(f"  rFID [{split}]  baseline={m['baseline_fid']:.3f}  SAE={m['sae_fid']:.3f}  Δ={df:+.3f}")
    print("=" * w)
