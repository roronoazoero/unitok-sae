"""Shared configuration for SAE pipeline."""
from dataclasses import dataclass

# to be changed later: replace the local paths for
@dataclass
class SAEConfig:
    unitok_ckpt: str = "${WRKDIR}/UniTok/unitok_tokenizer.pth"

    # ImageNet on Triton (set by setup_imagenet.sh into $WRKDIR/imagenet)
    imagenet_train_dir: str = "${WRKDIR}/imagenet/train"
    imagenet_val_dir: str = "${WRKDIR}/imagenet/val" # error in running this, to be fixed later

    # SAE pipeline outputs 
    activations_path: str = "sae/activations/block15.h5"
    val_activations_path: str = "sae/activations/block15_val.h5"
    checkpoint_dir: str = "sae/checkpoints"
    analysis_dir: str = "sae/analysis"

    # Model
    model_name: str = "vitamin_large"
    img_size: int = 256
    target_block: int = 15  # Middle of 31 ViT blocks 
    input_dim: int = 1024  # ViTamin-Large embed_dim

    # Activation collection
    num_images: int = 256_000  # Stratified: 256/class × 1000 classes
    collect_batch_size: int = 32
    num_workers: int = 4

    # SAE architecture 
    expansion_factor: int = 8  # hidden_dim = input_dim * expansion_factor = 8192
    k: int = 64  # Top-k sparsity (~0.8% of 8192), will experiment later
    dead_threshold_steps: int = 10_000_000

    # Training
    sae_batch_size: int = 4096
    learning_rate: float = 1e-4
    total_steps: int = 100_000
    warmup_steps: int = 1_000
    log_interval: int = 100
    checkpoint_interval: int = 10_000
    grad_clip: float = 1.0
    aux_loss_coef: float = 1.0 / 32.0  # Weight on dead-feature aux loss
    seed: int = 42

    # k-sweep
    sweep_k_values: tuple = (32, 64, 128)

    @property
    def hidden_dim(self) -> int:
        return self.input_dim * self.expansion_factor

    @property
    def imagenet_train_dir_resolved(self) -> str:
        import os
        return os.path.expandvars(self.imagenet_train_dir)

    @property
    def imagenet_val_dir_resolved(self) -> str:
        import os
        return os.path.expandvars(self.imagenet_val_dir)


CONFIG = SAEConfig()
