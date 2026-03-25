# DiT Iteration Experiments Status

## Overview

Running iterative experiments to improve upon Run 7 (best model) using DiT (Diffusion Transformer) architecture.

### Baselines
- **Run 7 (Best - Decoder-Only)**: Loss=0.0092, CF=19.71, Time=5.8m, Samples=18,176
- **Run 11 (Previous DiT, FAILED)**: Loss=0.0287 (3.1× worse), CF=27.19, Time=180m, Samples=183,040

### Strategy
Run iterative experiments in direct mode with fixed sessions, analyze results, and make informed adjustments.

**Sessions**:
- Train: `session_so101_multiheight_part1_1345_stage9_train_942` (942 obs)
- Val: `session_so101_multiheight_part1_1345_stage9_validate_403` (403 obs)
- Eval: `session_so101_multiheight_part1_1345` (1,343 obs, full dataset)

## Current Experiments

### Iteration 1 (Run 14): DiT with Plateau Sweeps + Reduced Inference Steps
**Status**: RUNNING (Task: bb560a6)
**Started**: 2026-03-09 ~20:00 UTC
**Expected Duration**: 2-3 hours

**Configuration**:
```yaml
Model: DiT d12 e256 with Stable Diffusion VAE (pretrained_sd)
Key Changes:
  - Enable plateau sweeps (run_id method, proven in Run 12b)
  - Reduce inference steps: 50 -> 20 (faster)
  - Enable initial LR sweep before training
  - Match Run 7's proven settings: batch_size=8, temperature=0.5
Hyperparameters:
  - Initial LR: 3e-4 (will be optimized by sweeps)
  - Batch size: 8
  - Time budget: 60 minutes
  - Loss weighting: temperature=0.5, refresh_interval=50
  - Divergence stopping: patience=30 (reduced from 50)
  - Plateau sweep: enabled, patience=25, max_sweeps=2
LR Sweep:
  - Phase A: 5 candidates, 3 min each = 15 min
  - Phase B: 2 survivors × 3 seeds × 10 min = 60 min
  - Total initial sweep: ~75 min
```

**Success Criteria**: orig_loss < 0.020 (30% improvement over Run 11)

**Expected Outcomes**:
- **Best case**: orig_loss < 0.010 (matches/beats Run 7)
- **Likely**: Loss 0.015-0.020 (significant improvement but not beating Run 7)
- **Failure**: Loss > 0.025 (indicates diffusion approach not competitive)

## Planned Iterations

### Iteration 2 (Run 15): Custom VAE (If Iteration 1 >= 0.020)
**Purpose**: Replace frozen SD VAE with custom VAE trained on robot frames
**Changes**:
- Train CanvasVAE on full robot session
- Use trained VAE in DiT instead of frozen SD VAE
- Keep other successful settings from Iteration 1

**Estimated Duration**:
- VAE training: ~30 min
- DiT training: ~2-3 hours
- Total: ~3-3.5 hours

### Iteration 3: Architectural Tweaks (If Iterations 1-2 still fail)
**Options**:
- Try conditional training mode (noise only on masked patches)
- Try fewer inference steps (10 instead of 20)
- Try different beta schedule (cosine vs linear)
- Reduce DiT depth (6 instead of 12) for efficiency
- Try sample prediction instead of epsilon prediction

## Analysis Tools

**Quick Comparison Script**: `compare_dit_runs.py`
```bash
python compare_dit_runs.py
```
Shows table comparing all runs in the experiments directory.

**Key Metrics to Track**:
1. **Primary**: `best_loss_original` (loss on full session)
2. **Secondary**: `overall_mean_divergence` (action conditioning strength)
3. **Efficiency**: samples_trained, elapsed_time_seconds
4. **Convergence**: stop_reason, divergence_triggered, sweeps_used

## Report Locations

After each iteration:
- HTML Report: `saved/staged_training_reports/session_so101_multiheight_part1_1345/multiheight_dit_run{N}/final_report_*.html`
- Metrics: `saved/staged_training_reports/session_so101_multiheight_part1_1345/multiheight_dit_run{N}/stage1_run1/metrics.json`
- Summary: `saved/staged_training_reports/session_so101_multiheight_part1_1345/multiheight_dit_run{N}/summary.json`

## Progress Tracking

### Checkpoints
All saved in: `saved/checkpoints/so101/`

Format: `best_model_auto_session_so101_multiheight_part1_1345_multiheight_dit_run{N}_*.pth`

### Next Steps
1. Wait for Iteration 1 to complete (~2-3 hours)
2. Analyze results with `compare_dit_runs.py`
3. Decide whether to continue with Iteration 2
4. If significant improvement achieved (loss < 0.010), consider fine-tuning strategies
5. Document findings and update memory

## Notes

- DiT's sample inefficiency is the main challenge (10× more samples than decoder-only)
- Frozen SD VAE (trained on natural images) may not be optimal for robot frames
- Run 11's 27.19 CF divergence suggests strong action sensitivity but poor reconstruction
- Progressive fine-tuning strategy from Run 7 won't directly apply to DiT without architectural modifications
