#!/bin/bash

# Sequential DiT iteration experiments - each gets full GPU
# Uses same train/val split as Run 11 (part1=train, part2=val)
set -e

TRAIN="saved/sessions/so101/session_so101_multiheight_part1_1345"
VAL="saved/sessions/so101/session_so101_multiheight_part2_149"

echo "=== Starting Sequential DiT Iterations (Solo GPU) ==="
echo "Train: session_so101_multiheight_part1_1345 (1345 obs)"
echo "Val: session_so101_multiheight_part2_149 (149 obs)"
echo ""
echo "PLAN:"
echo "1. Baseline (50 DDIM steps) - 0.016683 loss"
echo "2. Iter 1: 20 DDIM steps"
echo "3. Iter 2: 10 DDIM steps"
echo "4. Iter 3: Conditional training mode (PRIORITY)"
echo "5. Iter 1: Perceptual loss + baseline (PROVEN in Run 7)"
echo "6. Iter 4: FLUX VAE (16-channel latent)"
echo ""

echo "[1/6] Baseline (50 DDIM steps) - Solo GPU run"
python staged_training.py --train-session "$TRAIN" --val-session "$VAL" --config multiheight_dit_baseline.yaml --run-id multiheight_dit_iter0_solo

echo ""
echo "[2/6] Iter 1 (20 DDIM steps) - Solo GPU run"
python staged_training.py --train-session "$TRAIN" --val-session "$VAL" --config multiheight_dit_iter1_fewer_steps.yaml --run-id multiheight_dit_iter1_solo

echo ""
echo "[3/6] Iter 2 (10 DDIM steps) - Solo GPU run"
python staged_training.py --train-session "$TRAIN" --val-session "$VAL" --config multiheight_dit_iter2_10steps.yaml --run-id multiheight_dit_iter2_solo

echo ""
echo "[4/6] Iter 3: Conditional Training Mode - Solo GPU run"
python staged_training.py --train-session "$TRAIN" --val-session "$VAL" --config multiheight_dit_iter3_conditional.yaml --run-id multiheight_dit_iter3_solo

echo ""
echo "[5/6] Iter 1: Perceptual Loss - Solo GPU run"
python staged_training.py --train-session "$TRAIN" --val-session "$VAL" --config multiheight_dit_iter1_perceptual.yaml --run-id multiheight_dit_iter1_perceptual_solo

echo ""
echo "[6/6] Iter 4: FLUX VAE Backend - Solo GPU run"
python staged_training.py --train-session "$TRAIN" --val-session "$VAL" --config multiheight_dit_iter4_flux.yaml --run-id multiheight_dit_iter4_flux_solo

echo ""
echo "=== All Sequential Runs Complete ==="
