# UniTok-SAE: Sparse Autoencoders on UniTok's Vision Encoder

## 1. Status at handover

| Part | Status |
|---|---|
| SAE training (5 depths) | Works, stable, zero dead features |
| Reconstruction + downstream eval (rFID, zero-shot) | Works; block 15 is least disruptive |
| Max-activating images + 3-D UMAP explorer | Works (HTML files are 80–110 MB) |
| Direct feature steering | Works only for features already active in the image |
| APGD-style L0 / L1 delta search | Did not converge to useful edits |
| Plain SGD, global delta | Fails on localized edits; works for RGB → grayscale |
| Plain SGD, patchwise delta | Converges reliably (but the delta is per-image) |
| MagicBrush edit-type deltas | Multi-pair global delta doesn't converge / distorts |
| Residual (SAE-error) injection | Most promising lead, see §9.6 |

---

## 2. Repository map

The repo is a fork of upstream UniTok. Almost everything you need is under `sae/`; the rest is upstream code that `sae/` imports.

```
unitok-sae/
├── models/  utils/  open_clip/      # upstream UniTok (model defs, FID utils, CLIP text tower)
├── eval/                            # upstream LlamaGen / Liquid evals, not used by this project
├── main.py  trainer.py  launch.sh   # upstream UniTok tokenizer training, not used
├── k32.sh  k64.sh  k128.sh          # SLURM scripts for an early k-sweep (stale; use train_sae_online.py)
├── imagenet_class_index.json        # class names for zero-shot eval
└── sae/
    ├── config.py                    # SAEConfig dataclass: all default hyper-params & paths
    ├── model.py                     # TopKSAE (encoder/decoder, aux dead-feature loss)
    ├── unitok_loader.py             # build_unitok(): loads frozen UniTok checkpoint
    ├── setup_imagenet.sh            # Step 1: ImageNet subset extraction (Aalto Triton)
    ├── collect_activations.py       # Step 2 (optional/offline): dump activations to .h5
    ├── train_sae.py                 # Step 3a: train from .h5 activations (older path)
    ├── train_sae_online.py          # Step 3b: train with on-the-fly activations  ← USED FOR REPORT
    ├── evaluate.py + eval/          # Step 4: recon metrics, rFID, zero-shot
    ├── feature_analysis.py          # Step 5: top-activating images per feature (max-heaps)
    ├── find_dense_features.py       # Step 5: which features fire on ~every image
    ├── compute_feature_umap.py      # Step 5: 3-D UMAP of decoder directions
    ├── build_umap_viz.py            # Step 5: standalone interactive HTML explorer
    ├── image_umap.py                # Step 5: same, highlighting features active in ONE image
    ├── pgd.ipynb                    # Steps 6–7: steering, APGD L0/L1, SGD global/patchwise deltas
    ├── extract_pair.py              # Step 8: dump one MagicBrush pair to sae/pairs/
    ├── run_magicbrush.py            # Step 8: batch delta optimisation per edit type
    ├── magicbrush_analysis.ipynb    # Step 8: visualise / analyse MagicBrush deltas
    ├── feature_ablation.ipynb       # Step 9: residual injection, ablation, distortion features
    ├── eval_results/*/eval_summary.json   # the numbers behind Table 3 of the report
    ├── castle.jpg  castlewithsun.jpg      # test pair used in pgd.ipynb
    ├── pairs/                       # example MagicBrush source/target images
    └── UniTok                       # stray 3-line scratch script (prints ckpt keys); ignore
```

---

## 3. Environment

The work was run on Aalto Triton. Paths throughout assume `$WRKDIR` (auto-set on Triton) and that the repo lives at `$WRKDIR/UniTok`.

```bash
git clone https://github.com/roronoazoero/unitok-sae.git $WRKDIR/UniTok
cd $WRKDIR/UniTok

conda create -n unitok python=3.10 -y && conda activate unitok
pip install "torch>=2.3.1" torchvision          # match your CUDA
pip install timm h5py wandb tqdm pillow numpy matplotlib scipy \
            umap-learn datasets scikit-image jupyter ftfy regex
```

> The existing upstream-style README references `requirements.txt`, but there is no requirements.txt in the repo. The list above is reconstructed from the imports; if something is missing, it is almost certainly an upstream UniTok dependency (check upstream's requirements).

All commands below are run from the repo root (`$WRKDIR/UniTok`). Scripts add the root to `sys.path` themselves.

---

## 4. Step 1: Data

### 4.1 ImageNet (SAE training + analysis)

```bash
bash sae/setup_imagenet.sh            # train subset + val
bash sae/setup_imagenet.sh --val-only # val only
```

What it does:
- Reads Triton's shared copy at `/scratch/shareddata/dldata/imagenet/` (change `SHARED_DIR` if you are elsewhere).
- Train: extracts a deterministic-random 256 images per class (seed 42) → `$WRKDIR/imagenet/train/<synset>/` (~256 K images).
- Val: extracts all 50 K images into `$WRKDIR/imagenet/val/<synset>/`.

> The val step assumes Triton's val tar contains per-synset inner tars (`n*.tar`). On a standard ImageNet val tar (flat JPEGs) this loop does nothing; use the commented-out `valprep.sh` block in the script instead.
>
> Note: the report mistakenly says the SAEs were trained on the full ImageNet train set (1.28 M images). They were actually trained on this 256-per-class subset (~256 K images).

### 4.2 UniTok checkpoint

Download `unitok_tokenizer.pth` from Hugging Face ([FoundationVision/unitok_tokenizer](https://huggingface.co/FoundationVision/unitok_tokenizer/tree/main)) to `$WRKDIR/UniTok/unitok_tokenizer.pth`.

We use the from-scratch checkpoint, not the CLIP-initialised one (see report §3.2.3). Expect `build_unitok` to print a small number of missing/unexpected keys; that is normal with `strict=False`, but a large number means the checkpoint format changed.

### 4.3 MagicBrush (steering experiments)

Pulled automatically from Hugging Face by `run_magicbrush.py` / `extract_pair.py` via `datasets.load_dataset("osunlp/MagicBrush", split="train")`. First load downloads several GB; set `HF_HOME=$WRKDIR/.cache/huggingface` so it doesn't fill your home quota.

---

## 5. Step 2: Activations

The report's SAEs were trained with online activations (§6), so you can skip this step.

The input pipeline (identical everywhere; keep it identical if you add scripts):

```python
transforms.Resize((256, 256)) → ToTensor() → x * 2 - 1
```

Activations are captured with a forward hook on `unitok.encoder.blocks[b]`; output shape is `(B, 64, 1024)` and is flattened to `(B·64, 1024)` tokens.

Offline path (older; only if you want a fixed activation dataset):

```bash
python sae/collect_activations.py \
  --ckpt $WRKDIR/UniTok/unitok_tokenizer.pth \
  --block 15 \
  --data_dir $WRKDIR/imagenet/train \
  --output sae/activations/block15.h5
```

256 K images × 64 tokens × 1024 × fp32 ≈ 67 GB per block, which is why we moved to online training.

---

## 6. Step 3: Train the SAEs

```bash
for B in 5 10 15 20 25; do
  python sae/train_sae_online.py \
    --unitok_ckpt    $WRKDIR/UniTok/unitok_tokenizer.pth \
    --imagenet_train $WRKDIR/imagenet/train \
    --imagenet_val   $WRKDIR/imagenet/val \
    --block $B --k 64 --expansion 8 \
    --steps 100000 --batch_size 4096 --lr 1e-4 \
    --output_dir sae/checkpoints/block$B \
    --wandb --wandb_project unitok-sae --wandb_run sae_block${B}_k64
done
```

Key details (`sae/model.py`, `sae/config.py`):
- Architecture: `x − pre_bias → Linear(1024→8192) → TopK(64) + ReLU → Linear(8192→1024, no bias) + pre_bias`. Encoder/decoder are not tied.
- `pre_bias` is initialised to the activation mean estimated from 50 image batches.
- Decoder columns are unit-normalised after every step, and the gradient component parallel to each column is removed first.
- Loss = MSE + (1/32)·aux loss. The aux loss makes the top-128 "dead" latents (no fire in 10 K steps) predict the residual. In practice dead count stayed at 0.
- Schedule: Adam, 1 K-step linear warm-up then cosine decay; grad clip 1.0.
- Checkpoints every 10 K steps: `sae_block{B}_k64_step{N}.pt`, final: `sae_block{B}_k64_final.pt`. Each stores `sae`, `input_dim`, `hidden_dim`, `k`, `block`.
- `--resume path.pt` resumes from an intermediate checkpoint.

The trained checkpoints are on Hugging Face: [zo6-xzz/UniTokSAE](https://huggingface.co/datasets/zo6-xzz/UniTokSAE/tree/main)

---

## 7. Step 4: Evaluate

```bash
python sae/evaluate.py \
  --checkpoint  sae/checkpoints/block15/sae_block15_k64_final.pt \
  --unitok_ckpt $WRKDIR/UniTok/unitok_tokenizer.pth \
  --imagenet_train $WRKDIR/imagenet/train --n_train 3000 \
  --imagenet_val   $WRKDIR/imagenet/val   --n_val 50000 \
  --class_names_json imagenet_class_index.json \
  --output_dir sae/eval_results/block15_k64
# --feature_extractor_path <inception weights>   # for rFID, see utils/eval_fid.py
# --skip_rfid / --skip_zeroshot / --skip_features to run parts
```

Two-pass protocol (`sae/eval/reconstruction.py`): Pass A = vanilla UniTok; Pass B = same, but the block-b output is replaced by the SAE reconstruction. Both images are saved, rFID compares each to the originals, and zero-shot uses UniTok's CLIP head against text embeddings of the class names.

> `evaluate.py` has no `--block` flag; `run_two_pass_eval` defaults to `block=15`. To evaluate the block-5/10/20/25 SAEs you must pass `block=ckpt_meta["block"]` into `run_two_pass_eval` (and into `find_max_activating_examples`). The JSONs in `eval_results/` were produced with that edit made locally, so it is not in the repo. Please pay attention to this, because running the script as-is will silently evaluate the wrong block.

---

## 8. Step 5: Feature analysis & UMAP explorer

Run in this order for each block (shown for block 15):

```bash
CK=sae/checkpoints/block15/sae_block15_k64_final.pt
UT=$WRKDIR/UniTok/unitok_tokenizer.pth
VAL=$WRKDIR/imagenet/val
OUT=sae/analysis/block15

# (a) dense features: fire in ≥90% of 100 probe images (one per class)
python sae/find_dense_features.py --checkpoint $CK --unitok_ckpt $UT --imagenet_val $VAL \
       --n_images 100 --threshold 0.9 --out $OUT/dense_features.pt

# (b) top-64 (value, image, patch) per feature over all 50K val images.
#     --skip_images saves only heap_results.pt (small) instead of 8192 PNG grids
python sae/feature_analysis.py --checkpoint $CK --unitok_ckpt $UT --imagenet_val $VAL \
       --n_features 8192 --k_top 64 --skip_images --output_dir $OUT

# (c) 3-D UMAP of decoder columns (cosine metric, n_neighbors=15, min_dist=0.1, seed 42)
python sae/compute_feature_umap.py --checkpoint $CK \
       --dense_features $OUT/dense_features.pt --output $OUT/umap_coords.json

# (d) standalone HTML explorer (hover = top images, frequency slider, feature-ID search)
python sae/build_umap_viz.py --umap_coords $OUT/umap_coords.json \
       --heap_results $OUT/heap_results.pt --imagenet_val $VAL \
       --n_show 4 --output $OUT/feature_umap.html

# (e) optional: UMAP highlighting only features active in one query image
#     (not a focus of the project; the implementation may be buggy)
python sae/image_umap.py --image_path sae/castle.jpg --block 15 \
       --checkpoint $CK --unitok_ckpt $UT --umap_coords $OUT/umap_coords.json \
       --heap_results $OUT/heap_results.pt --imagenet_val $VAL --output $OUT/image_umap.html
```

> `find_dense_features.py` hard-codes `blocks[15]`, and `feature_analysis.py`'s CLI doesn't forward the block to `find_max_activating_examples`. For other blocks, edit both (or, preferably, add a `--block` arg).

---

## 9. Steps 6–10: Steering experiments (chronological)

These live mostly in notebooks. Open `sae/pgd.ipynb` and `sae/magicbrush_analysis.ipynb` from the repo root (they auto-detect `_root`).

> Before running any notebook: set `CHECKPOINT` and every `unitok.encoder.blocks[N]` in the hook functions to the same block. Currently the notebooks load `sae_block10_k64_final.pt` but most hooks use `blocks[15]` (and `delta_forward` uses `blocks[10]`). This is a leftover from switching between SAEs. A mismatched SAE/block silently produces garbage. Also replace hard-coded paths like `/scratch/work/vuh12/UniTok/...` with your own.

### Common building block: the intervention hook

Every steering method below is a variant of this: encode the block-b output with the SAE, modify the latent, decode, and let UniTok continue (quant_proj → quantizer → post_quant_proj → decoder).

```python
def _hook(module, inp, out):                     # out: (B, 64, 1024)
    B, L, D = out.shape
    z, *_ = sae.encode(out.float().reshape(B*L, D))
    z = z + delta                                 # delta: (8192,) global or (64, 8192) patchwise
    return sae.decode(z).reshape(B, L, D).to(out.dtype)
handle = unitok.encoder.blocks[BLOCK].register_forward_hook(_hook)
```

The optimisation loss everywhere is MSE between pre-VQ tokens: `quant_proj(encoder(A; z_A + δ))` vs `quant_proj(encoder(B))`. Pre-VQ was chosen because the quantizer is non-differentiable. The `diagnose_prevq` cell in `pgd.ipynb` checks the upper bound: adding the exact pre-VQ difference reconstructs B, so the target is reachable in principle.

### 9.1 Direct feature steering / ablation

Add `α · e_f` to feature f's latent (equivalently `α · W_dec[:, f]` to the activation) and decode; or zero features to ablate.

- Works: amplifying a feature already active in the image (feature 2549 on the purse image, α ≈ 1000–1700, report Fig. 5).
- Doesn't: injecting an inactive feature, or injecting into a blank grey image (Fig. 6). The rest of the encoder washes it out.
- Too large α → distortions.
- Dense-feature ablation (`magicbrush_analysis.ipynb`, "Dense feature ablation" cell) keeps image structure, showing dense features carry low-level info.

### 9.2 APGD-style constrained delta search

Adapted from Francesco Croce's Auto-PGD ([fra31/auto-attack](https://github.com/fra31/auto-attack); the L1 variant follows Croce & Hein, "Mind the Box"). In `pgd.ipynb`:

- L0, global delta (`find_optimal_delta` + `project_topk`): APGD step-size schedule (checkpoints at 22 % / 3 % / 6 % of steps, step halving on stagnation) with a top-k sparsity projection on δ ∈ ℝ⁸¹⁹².
- L1, global delta (`find_optimal_delta_l1` + `L1_projection`): sorting-based projection onto the L1 ball, ε up to 4·10⁴.

Outcome: neither produced a convincing edit on the castle → castle-with-sun pair. Sparsity constraints made convergence worse.

### 9.3 Plain SGD, global delta

`find_delta_prevq` (warm-up + cosine LR, optional L1 penalty, dense-feature mask) and `find_delta_global` ("supervised": loss only on the k_c patches that change most between A and B).

- Fails on localized edits: the loss plateaus well above zero (one vector can't express spatially varying changes).
- Works on RGB → grayscale (`DISTORTION = "bw"` cell; report Fig. 7). The same cell also offers `blur`, `noise`, `jpeg` distortions; these are the natural starting point for robustness work.

### 9.4 Plain SGD, patchwise delta

`find_delta_patchwise_prevq`: δ ∈ ℝ^(64×8192), SGD + ReduceLROnPlateau (patience 2000, factor 0.5). Typical: `lr=50, n_steps=10000`.

- Converges reliably on most pairs.
- Variants that made it worse: top-k patches only (`k_patches`), top-k features per patch, L1-ball projection per patch.
- Limitation: δ is fitted per image pair, so it's closer to "reconstructing B" than to finding a reusable edit direction.

### 9.5 MagicBrush

Edit types are keyword-matched from instructions (`EDIT_KEYWORDS` in `run_magicbrush.py`): `make_night`, `add_snow`, `add_rain`, `make_autumn`, `add_sunglasses`.

```bash
# one pair for the notebook (writes sae/pairs/source_img.png, target_img.png)
python sae/extract_pair.py --edit_type sunglasses

# shared global delta across N pairs of one edit type  (this path works)
python sae/run_magicbrush.py --checkpoint $CK --unitok_ckpt $UT \
       --multi_global --n_max 5 --n_steps 5000 --lr 25 \
       --dense_features $OUT/dense_features.pt \
       --out_dir sae/deltas_multi/n_5
```

Then open `magicbrush_analysis.ipynb` with `DELTAS_DIR`/`EDIT_TYPE` set, to see Source / Target / Source+δ grids and per-feature consistency across pairs.

- Result: multi-pair global delta doesn't converge and adds distortion (report Fig. 8).

> `run_magicbrush.py` hard-codes `blocks[15]`, so make sure `--checkpoint` is the block-15 SAE.

### 9.6 Residual (SAE-error) injection: the promising lead

Notebook: `sae/feature_ablation.ipynb`, function `steer_forward` (helper-functions cell), used in the "Experiment 2" and "Inject feature(s) on random images" sections. The notebook was last run with the block-25 SAE (`sae/checkpoints/block25/sae_block25_k64_final.pt`, feature IDs below refer to that SAE).

Idea: the SAE doesn't reconstruct the activation perfectly. In the hooks of §9.1 to §9.5 the block output is replaced by `decode(z)`, so every intervention also carries the SAE's reconstruction error (the rFID gap in report Table 3), and that error competes with the edit. `steer_forward` keeps the error term, so with no intervention the output is exactly the original activation and only the edited feature changes:

```python
def _hook(m, i, o):
    B, L, D = o.shape
    acts_flat = o.float().reshape(B*L, D)
    latent, _, _, _ = sae.encode(acts_flat)
    residual = acts_flat - sae.decode(latent)      # SAE reconstruction error
    latent = latent.clone()
    for fid, val in feature_interventions.items():
        if all_patches:
            latent[:, fid] = val                   # SET the feature value (not add / scale)
        else:
            latent[[b * L + L // 2 for b in range(B)], fid] = val   # centre patch only
    return (sae.decode(latent) + residual).reshape(B, L, D).to(o.dtype)
```

Two differences from the earlier hooks: the residual is added back, and the feature is set to an absolute value rather than scaled, so it also works on features that are inactive (value 0) in the image.

What came out of it:
- Injecting a feature that is not active in the image now has a visible effect, which plain replacement never achieved (§9.1). Example: feature 7196 (block 25) was picked as one of the top features of a grayscale image (Experiment 2 runs `get_top_features` on the distorted copy). Setting it to 4000 to 4750 on all patches of a colourful purse image where it is inactive (seed 10, val image #37447) progressively washes out the colours and adds texture, while the scene stays intact.
- On an image where the feature is already present (Experiment 2, building image), steering works as described in the report (amplifying a feature that is already active produces coherent visual changes).
- Required values are large (thousands, versus natural activations of roughly 100 to 500), and the useful range before distortion is narrow, so sweep in small steps. This may also be because the injected signal gets distorted on its way through the remaining encoder blocks, the quantizer and the decoder, which is worth looking into.
- Blank-image injection (feature 2549, values 1000 to 8000) still does not produce the feature's concept, only frame-like artefacts.

To reproduce: set `CHECKPOINT` to the block-25 SAE, run the config and helper cells, then the "Experiment 2" cells and the "Inject feature(s) on random images" cells (`INJECT_FEATURES = {7196: ...}`, `INJECT_VALS_SWEEP = [0, 4000, 4500, 4750]`).

> A note on how this notebook was run: I manually changed the checkpoint being loaded and the block index in the helper functions every time I ran an experiment, and I may have forgotten to change them back. I also did not keep good notes on which SAE and block each result was produced with; that is on me. As a result, the notebook as committed is not internally consistent: it loads the block-25 SAE, `steer_forward` and `ablation_forward` read `BLOCK` from the checkpoint, but `inject_feature_forward`, `multi_feature_forward`, `get_sae_latent` and the AutoPGD / delta functions at the bottom hard-code `blocks[15]`.
>
> If you continue from this notebook: make every helper use `BLOCK = ckpt["block"]` instead of a hard-coded index, put the checkpoint name into saved figure filenames, and re-run any result (bird/watermark swap, JPEG and grayscale feature injection, sea-shore cluster ablation) before building on it.

---

## 10. Suggested next steps

These are some suggestions from me, in rough priority order:

1. Clean-ups that unblock everything: add `--block` to every script / read it from `ckpt["block"]`; add a `requirements.txt`; fix `run_magicbrush.py` per-pair mode; remove hard-coded `/scratch/work/vuh12/` paths.
2. Make residual injection the default hook for all steering/optimisation and re-run §9.1–9.5 with it. Several negative results may have been caused by the SAE reconstruction error.
3. Automatic feature labelling (report §6): send top-4 images per feature to a VLM ("what do these images share?"), store labels, and add text search to the UMAP.
4. Robustness (RQ3): build on the grayscale result and the `blur/noise/jpeg` distortions in `pgd.ipynb`. Find deltas for clean → distorted and check whether they are low-rank / shared across images.
5. Cross-layer analysis: match features across the five SAEs (decoder cosine similarity) to see where concepts first appear.
6. Larger dictionaries (16 K / 32 K) or lower k for more monosemantic features.

---

## 11. References

1. Olson et al., Probing the representational power of sparse autoencoders in vision models, 2025.
2. Joseph et al., Steering CLIP's vision transformer with sparse autoencoders, 2025.
3. Ferrando et al., Language models can explain visual features via steering, 2026.
4. Ma et al., UniTok: A unified tokenizer for visual generation and understanding, 2025. Code: [FoundationVision/UniTok](https://github.com/FoundationVision/UniTok)
5. Croce & Hein, Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks (Auto-PGD), ICML 2020. Code: [fra31/auto-attack](https://github.com/fra31/auto-attack)
6. Zhang et al., MagicBrush: A manually annotated dataset for instruction-guided image editing, NeurIPS 2023. Data: `osunlp/MagicBrush` on Hugging Face.

Questions about the original work: Hien Thao Vu.
