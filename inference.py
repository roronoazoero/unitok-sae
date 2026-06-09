import os
import torch
import argparse
from PIL import Image
from utils.config import Args
from models.unitok import UniTok
from sae.model import TopKSAE
from sae.injection import SAEInjectionBlock
from utils.data import normalize_01_into_pm1
from torchvision.transforms import transforms, InterpolationMode


def save_img(img: torch.Tensor, path):
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    img = Image.fromarray(img[0])
    img.save(path)

@torch.no_grad()
def extract(img: torch.Tensor, unitok: UniTok, sae: TopKSAE, target_block: int = 15) -> torch.Tensor:
    activation_buffer = []
    def hook(module, inp, out):
        activation_buffer.append(out.detach()) # out shape: (B, num_tokens, embed_dim)

    target_layer = unitok.encoder.blocks[target_block]
    handle = target_block.register_forward_hook(hook)
    try:
        unitok.encoder(img)
        activations = activation_buffer[0]
        b, n_tokens, embed_dim = activations.shape # flatten tokens
        activations_flat = activations.view(-1, embed_dim)
        sparse_features, topk_indices, topk_values = sae.encode(activations_flat)
        return sparse_features.view(b, n_tokens, -1)  # (B, num_tokens, hidden_dim)
    finally:
        handle.remove()

@torch.no_grad()
def inject_gen(img: torch.Tensor, unitok: UniTok, sae: TopKSAE, mode: str = "steer", feature_indices: list = None, strength: float = 1.0, clamp_value: str = 1.0, target_block: int = 15) -> torch.Tensor:
    if feature_indices is None:
        feature_indices = []
    
    # Save original block
    original_block = unitok.encoder.blocks[target_block]
    
    # Create injection wrapper
    injector = SAEInjectionBlock(original_block, sae)
    unitok.encoder.blocks[target_block] = injector
    
    try:
        # Configure injection
        if mode == "ablate":
            # Ablate = set features to zero (via clamping to 0)
            injector.configure("clamp", features=feature_indices, clamp_value=0.0)
        else:
            injector.configure(mode, features=feature_indices, strength=strength, clamp_value=clamp_value)
        
        # Forward pass with injection
        with torch.no_grad():
            code_idx = unitok.img_to_idx(img)
            modified_img = unitok.idx_to_img(code_idx)
        
        return modified_img
    
    finally:
        # Restore original block
        unitok.encoder.blocks[target_block] = original_block


def main(args):
    # load model
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt['trainer']['unitok'])
    unitok.to('cuda')
    unitok.eval()
    

    preprocess = transforms.Compose([
        transforms.Resize(int(unitok_cfg.img_size * unitok_cfg.resize_ratio)),
        transforms.CenterCrop(unitok_cfg.img_size),
        transforms.ToTensor(), normalize_01_into_pm1,
    ])
    img = Image.open(args.src_img).convert("RGB")
    img = preprocess(img).unsqueeze(0).to('cuda')

    with torch.no_grad():
        code_idx = unitok.img_to_idx(img)
        rec_img = unitok.idx_to_img(code_idx)

    final_img = torch.cat((img, rec_img), dim=3)
    save_img(final_img, args.rec_img)

    print('The image is saved to {}. The left one is the original image after resizing and cropping. The right one is the reconstructed image.'.format(args.rec_img))

    if args.sae_ckpt:
        print(f"\nLoading SAE from {args.sae_ckpt}")
        sae_ckpt = torch.load(args.sae_ckpt, map_location='cuda')
        sae = TopKSAE(
            input_dim=sae_ckpt["input_dim"],
            hidden_dim=sae_ckpt["hidden_dim"],
            k=sae_ckpt["k"],
        ).to('cuda')
        sae.load_state_dict(sae_ckpt["sae"])
        sae.eval()
        
        if args.output_features or args.ablate_features or args.steer_features or args.inject_mode:
            print("Extracting sparse features...")
            sparse_features = extract_sparse_features(img, unitok, sae)
            
            sparsity = (sparse_features == 0).float().mean().item()
            print(f"Sparse features shape: {sparse_features.shape}")
            print(f"Sparsity: {sparsity:.2%}")
            print(f"Mean activation: {sparse_features.abs().mean().item():.4f}")
            
            # Save sparse features
            if args.output_features:
                torch.save(sparse_features, args.output_features)
                print(f"Saved sparse features to {args.output_features}")
        
        # Inject/ablate features 
        if args.ablate_features or args.steer_features or args.inject_mode:
            print(f"\Feature injection/manipulation:")
            
            feature_list = []
            if args.ablate_features:
                feature_list = [int(x) for x in args.ablate_features.split(',')]
                mode = "ablate"
                strength = 1.0
            elif args.steer_features:
                parts = args.steer_features.split(':')
                feature_list = [int(x) for x in parts[0].split(',')]
                strength = float(parts[1]) if len(parts) > 1 else 1.0
                mode = "steer"
            elif args.inject_mode:
                mode = args.inject_mode
                if args.inject_features:
                    feature_list = [int(x) for x in args.inject_features.split(',')]
                strength = args.inject_strength
            
            modified_img = inject_and_generate(
                img,
                unitok,
                sae,
                mode=mode,
                feature_indices=feature_list,
                strength=strength,
                clamp_value=args.clamp_value,
            )
            
            print(f"Modified image generated with mode={mode}, features={feature_list}, strength={strength}")
            
            if args.output_modified:
                save_img(modified_img, args.output_modified)
                print(f"Saved modified image to {args.output_modified}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--src_img', type=str, default='')
    parser.add_argument('--rec_img', type=str, default='')
    parser.add_argument('--sae_ckpt', type=str, default='')
    parser.add_argument('--output_features', type=str, default='')
    parser.add_argument('--ablate_features', type=str, default=None, 
                        help='Comma-separated feature indices to ablate (zero out). E.g. "0,5,10"')
    parser.add_argument('--steer_features', type=str, default=None)
    parser.add_argument('--inject_mode', type=str, default=None, choices=['add', 'sub', 'steer', 'clamp', 'reconstruct'])
    parser.add_argument('--inject_features', type=str, default=None)
    parser.add_argument('--inject_strength', type=float, default=1.0,
                        help='Strength parameter for add/sub/steer modes (default: 1.0)')
    parser.add_argument('--clamp_value', type=float, default=1.0,
                        help='Value to clamp features to in clamp mode (default: 1.0)')
    parser.add_argument('--output_modified', type=str, default=None,
                        help='Path to save the modified image after feature injection')
    args = parser.parse_args()
    main(args)

