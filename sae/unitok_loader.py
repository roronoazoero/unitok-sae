import os
import sys
import torch

# Add project root to path so `models` and `utils` are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.unitok import UniTok  # noqa: E402
from utils.config import Args  # noqa: E402
from sae.config import CONFIG  # noqa: E402


def build_unitok(ckpt_path: str = None, device: torch.device = None) -> UniTok:
    ckpt_path = ckpt_path or CONFIG.unitok_ckpt
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build args matching ViTamin-Large training config
    args = Args(explicit_bool=True).parse_args(known_only=True, args=[])
    args.model = CONFIG.model_name
    args.img_size = CONFIG.img_size
    args.grad_ckpt = False
    args.device = str(device)

    model = UniTok(args)

    print(f"[unitok_loader] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "unitok" in ckpt:
            state_dict = ckpt["unitok"]
        elif (
            "trainer" in ckpt
            and isinstance(ckpt["trainer"], dict)
            and "unitok" in ckpt["trainer"]
        ):
            state_dict = ckpt["trainer"]["unitok"]
        elif "module" in ckpt:
            state_dict = ckpt["module"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip "module." prefix from DDP if present
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[unitok_loader] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model
