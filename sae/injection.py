import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image

# Make project importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sae.config import CONFIG  # noqa: E402
from sae.model import TopKSAE  # noqa: E402
from sae.unitok_loader import build_unitok  # noqa: E402


class SAEInjectionBlock(nn.Module):
    VALID_MODES = ("none", "add", "sub", "reconstruct", "steer", "clamp")

    def __init__(self, original_block: nn.Module, sae: TopKSAE):
        super().__init__()
        self.block = original_block
        self.sae = sae
        for p in self.sae.parameters():
            p.requires_grad = False
        self.mode = "none"
        self.feature_indices: list = []
        self.strength: float = 0.0
        self.clamp_value: float = 1.0

    def configure(self, mode: str, features=None, strength: float = 1.0,
                  clamp_value: float = 1.0):
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.mode = mode
        self.feature_indices = list(features) if features else []
        self.strength = strength
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.mode == "none":
            return x
        if self.mode in ("add", "sub", "steer", "clamp") and not self.feature_indices:
            return x

        B, L, D = x.shape

        if self.mode == "reconstruct":
            flat = x.reshape(-1, D)
            with torch.no_grad():
                latent, _, _ = self.sae.encode(flat)
                recon = self.sae.decode(latent)
            return recon.reshape(B, L, D)

        if self.mode in ("add", "sub"):
            idx = torch.tensor(self.feature_indices, device=x.device, dtype=torch.long)
            directions = self.sae.decoder.weight[:, idx]  # (D, n_feat)
            direction = directions.sum(dim=1)
            direction = direction / direction.norm().clamp_min(1e-8)
            sign = 1.0 if self.mode == "add" else -1.0
            return x + sign * self.strength * direction.view(1, 1, D)

        if self.mode in ("steer", "clamp"):
            flat = x.reshape(-1, D)
            with torch.no_grad():
                latent, _, _ = self.sae.encode(flat)
            latent = latent.clone()
            idx = torch.tensor(self.feature_indices, device=x.device, dtype=torch.long)
            if self.mode == "steer":
                latent[:, idx] = latent[:, idx] * (1.0 + self.strength)
            else:  # clamp
                latent[:, idx] = self.clamp_value
            recon = self.sae.decode(latent)
            return recon.reshape(B, L, D)

        return x


def install_injection(unitok, sae, target_block: int = None) -> SAEInjectionBlock:
    if target_block is None:
        target_block = CONFIG.target_block
    original = unitok.encoder.blocks[target_block]
    device = next(unitok.parameters()).device
    injector = SAEInjectionBlock(original, sae).to(device)
    unitok.encoder.blocks[target_block] = injector
    return injector


def uninstall_injection(unitok, injector: SAEInjectionBlock,
                        target_block: int = None):
    if target_block is None:
        target_block = CONFIG.target_block
    unitok.encoder.blocks[target_block] = injector.block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitok_ckpt", type=str, default=CONFIG.unitok_ckpt)
    parser.add_argument("--sae_ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--feature_idx", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="add",
                        choices=list(SAEInjectionBlock.VALID_MODES))
    parser.add_argument("--clamp_value", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="injection_demo.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unitok = build_unitok(args.unitok_ckpt, device)

    ckpt = torch.load(args.sae_ckpt, map_location=device)
    sae = TopKSAE(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"], k=ckpt["k"])
    sae.load_state_dict(ckpt["sae"])
    sae.eval().to(device)

    injector = install_injection(unitok, sae)

    transform = transforms.Compose([
        transforms.Resize((CONFIG.img_size, CONFIG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        injector.configure("none")
        baseline = unitok.img_to_reconstructed_img(img)

        injector.configure(args.mode, features=[args.feature_idx],
                          strength=args.strength, clamp_value=args.clamp_value)
        injected = unitok.img_to_reconstructed_img(img)

    out = torch.cat([baseline, injected], dim=0)
    vutils.save_image(out, args.output, nrow=2, normalize=True, value_range=(-1, 1))
    print(f"[inject] Saved {args.output} (baseline | feature {args.feature_idx} "
          f"{args.mode} @ strength {args.strength})")


if __name__ == "__main__":
    main()
