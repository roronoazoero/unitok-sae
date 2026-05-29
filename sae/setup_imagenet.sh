#!/bin/bash
#
# What this does:
#   1. Extracts the outer train tar (1000 inner tars) into $WRKDIR/imagenet/train/
#   2. For each synset's inner tar, extracts a deterministic-random subset of
#      N images using `shuf` with a fixed seed, then deletes the inner tar.
#   3. Extracts the val tar and uses torchvision + ILSVRC2012_devkit_t12.tar.gz
#      to reorganize flat val images into per-synset subdirs.
#
# Idempotent: re-running skips already-completed steps.
# ============================================================================

set -e  # Exit on any error
VAL_ONLY=0
if [ "$1" = "--val-only" ]; then VAL_ONLY=1; fi

# Configuration 
SHARED_DIR="/scratch/shareddata/dldata/imagenet"
DEST_DIR="$WRKDIR/imagenet"
N_PER_CLASS=256          # Images per class (256 × 1000 = 256K total)
RANDOM_SEED=42           # For deterministic random subsampling

TRAIN_DEST="$DEST_DIR/train"
VAL_DEST="$DEST_DIR/val"
SHARED_TRAIN="$SHARED_DIR/ILSVRC2012_img_train.tar"
SHARED_VAL="$SHARED_DIR/ILSVRC2012_img_val.tar"
# SHARED_DEVKIT="$SHARED_DIR/ILSVRC2012_devkit_t12.tar.gz"

echo "==============================================="
echo "ImageNet partial extraction"
echo "  Source:    $SHARED_DIR"
echo "  Dest:      $DEST_DIR"
echo "  Per-class: $N_PER_CLASS images"
echo "  Seed:      $RANDOM_SEED"
echo "==============================================="

# Sanity checks 
if [ -z "$WRKDIR" ]; then
    echo "ERROR: \$WRKDIR is not set. On Aalto Triton this should be auto-set on login."
    exit 1
fi

if [ ! -f "$SHARED_TRAIN" ]; then
    echo "ERROR: Cannot find $SHARED_TRAIN"
    exit 1
fi

if [ ! -f "$SHARED_VAL" ]; then
    echo "ERROR: Cannot find $SHARED_VAL"
    exit 1
fi

# if [ ! -f "$SHARED_DEVKIT" ]; then
#     echo "ERROR: Cannot find $SHARED_DEVKIT"
#     exit 1
# fi

# Verify python + torchvision available
if ! python -c "import torchvision" 2>/dev/null; then
    echo "ERROR: torchvision not available. Activate your conda env first:"
    echo "       source activate <your_unitok_env>"
    exit 1
fi

mkdir -p "$TRAIN_DEST" "$VAL_DEST"

# Step 1: Extract outer train tar (yields 1000 inner tars)
# Detect completion: at least one inner tar OR at least one synset dir exists.
if [ "$VAL_ONLY" -eq 0 ]; then
    if ls "$TRAIN_DEST"/*.tar >/dev/null 2>&1 || ls -d "$TRAIN_DEST"/n*/ >/dev/null 2>&1; then
        echo "[1/3] Train tar already (partially) extracted, continuing..."
    else
        echo "[1/3] Extracting outer train tar (~15-25 min)..."
        cd "$TRAIN_DEST"
        tar -xf "$SHARED_TRAIN"
        n_inner=$(ls *.tar 2>/dev/null | wc -l)
        echo "  Done. $n_inner inner tars."
    fi
    
    # Step 2: Per-class partial extraction 
    echo "[2/3] Extracting $N_PER_CLASS deterministic-random images per class..."
    cd "$TRAIN_DEST"
    
    remaining_tars=$(ls *.tar 2>/dev/null | wc -l)
    if [ "$remaining_tars" -eq 0 ]; then
        echo "  No remaining inner tars. Skipping (already processed)."
    else
        total="$remaining_tars"
        processed=0
        for f in *.tar; do
            [ -f "$f" ] || continue
            synset="${f%.tar}"
            mkdir -p "$synset"
            # Deterministic random N filenames from this inner tar.
            # `yes "$RANDOM_SEED"` provides an infinite stream of the seed value to
            # `shuf --random-source`, making the shuffle reproducible.
            tar -tf "$f" \
                | shuf --random-source=<(yes "$RANDOM_SEED" 2>/dev/null) \
                | head -n "$N_PER_CLASS" > "/tmp/files_$$.txt"
            tar -xf "$f" -C "$synset" -T "/tmp/files_$$.txt"
            rm "$f" "/tmp/files_$$.txt"
            processed=$((processed + 1))
            if [ $((processed % 50)) -eq 0 ]; then
                echo "  Processed $processed / $total classes"
            fi
        done
        echo "  Done. Processed $processed classes."
    fi
fi

# Step 3: Extract + reorganize val set  
# n_val_synsets=$(find "$VAL_DEST" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
# if [ "$n_val_synsets" -ge 1000 ]; then
#     echo "[3/3] Val set already organized ($n_val_synsets synset dirs), skipping."
# else
#     echo "[3/3] Extracting val tar ..."
#     tar -xf "$SHARED_VAL" -C "$VAL_DEST"

#     echo "[3/3] Reorganizing into synset subdirs using valprep.sh ..."
#     cd "$VAL_DEST"
#     wget -qO valprep.sh \
#         https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
#     bash valprep.sh
#     rm -f valprep.sh
# fi
cd "$VAL_DEST"
for f in n*.tar; do
    synset="${f%.tar}"
    mkdir -p "$synset"
    tar -xf "$f" -C "$synset"
    rm "$f"
done
# if [ -d "$VAL_DEST/n01440764" ]; then
#     echo "[3/3] Val set already organized, skipping."
# else
#     echo "[3/3] Extracting val set using devkit (~5-10 min)..."

#     # torchvision's ImageNet class needs both tars in the same dir.
#     # Symlink rather than copy (saves ~145 GB; read-only safe).
#     ln -sf "$SHARED_VAL" "$VAL_DEST/ILSVRC2012_img_val.tar"
#     ln -sf "$SHARED_DEVKIT" "$VAL_DEST/ILSVRC2012_devkit_t12.tar.gz"

#     # Let torchvision do the parsing + reorganization
#     python -c "
# from torchvision.datasets import ImageNet
# ds = ImageNet(root='$VAL_DEST', split='val')
# print(f'Val set ready: {len(ds)} images, {len(ds.classes)} classes')
# "

#     # Cleanup symlinks (extracted JPEGs remain in $VAL_DEST organized by synset)
#     rm -f "$VAL_DEST/ILSVRC2012_img_val.tar"
#     rm -f "$VAL_DEST/ILSVRC2012_devkit_t12.tar.gz"
# fi

# Sanity check 
echo "==============================================="
echo "Verification:"
n_train_classes=$(find "$TRAIN_DEST" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_train_imgs=$(find "$TRAIN_DEST" -name '*.JPEG' | wc -l)
n_val_classes=$(find "$VAL_DEST" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_val_imgs=$(find "$VAL_DEST" -name '*.JPEG' | wc -l)
echo "  Train classes: $n_train_classes (expected 1000)"
echo "  Train images:  $n_train_imgs (expected ~$((N_PER_CLASS * 1000)))"
echo "  Val classes:   $n_val_classes (expected 1000)"
echo "  Val images:    $n_val_imgs (expected 50000)"
echo "  Total size:    $(du -sh "$DEST_DIR" 2>/dev/null | cut -f1)"
echo "==============================================="
