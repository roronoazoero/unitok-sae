import os
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sae.model import TopKSAE
from sae.unitok_loader import build_unitok  # noqa: F401 

# _MEAN = [0.485, 0.456, 0.406]
# _STD  = [0.229, 0.224, 0.225]


def make_transform(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    # ImageNet-normalized (B,3,H,W) tensor -> [0, 1]
    return ((x + 1) / 2).clamp(0, 1)


def load_sae(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae = TopKSAE(
        input_dim  = ckpt["input_dim"],
        hidden_dim = ckpt["hidden_dim"],
        k          = ckpt["k"],
    )
    sae.load_state_dict(ckpt["sae"])
    return sae.eval().to(device), ckpt


@torch.no_grad()
def eval_forward(unitok, imgs: torch.Tensor):
     # Returns: (reconstructed images in [-1,1], L2-normalized CLIP visual features)
    img_tokens = unitok.encoder(imgs).float()
    img_tokens = unitok.quant_proj(img_tokens)
    img_tokens, _, _, _ = unitok.quantizer(img_tokens)
    img_tokens = unitok.post_quant_proj(img_tokens)
    img_rec   = unitok.decoder(img_tokens).clamp(-1, 1)
    clip_feat = F.normalize(
        unitok.projection(unitok.fc_norm(img_tokens.mean(1))), dim=-1
    )
    return img_rec, clip_feat
