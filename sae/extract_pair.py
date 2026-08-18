import argparse
from pathlib import Path
from datasets import load_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idx",       type=int, default=None, help="Dataset index (0-based)")
    p.add_argument("--edit_type", type=str, default=None, help="Pick first pair matching this keyword")
    p.add_argument("--out_dir",   type=str, default="sae/pairs")
    args = p.parse_args()
    print("Loading MagicBrush")
    ds = load_dataset("osunlp/MagicBrush", split="train")

    if args.idx is not None:
        idx = args.idx
    elif args.edit_type is not None:
        idx = next(
            (i for i, instr in enumerate(ds["instruction"])
             if args.edit_type.lower() in instr.lower()),
            None
        )
        if idx is None:
            print(f"No instruction containing '{args.edit_type}' found.")
            return
    else:
        idx = 0

    sample = ds[idx]
    instr  = sample["instruction"]
    src    = sample["source_img"].convert("RGB")
    tgt    = sample["target_img"].convert("RGB")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src.save(out / "source_img.png")
    tgt.save(out / "target_img.png")

    print(f"Index      : {idx}")
    print(f"Instruction: {instr}")
    print(f"Saved to   : {out}/source.png  and  target.png")

if __name__ == "__main__":
    main()