# DiT Systematic Parameter Exploration

## Strategy

Instead of random tuning or complex optimizations (like LR sweeps), systematically test **single parameter changes** to understand what's visually wrong with DiT and fix it.

**Sessions**:
- Training: `session_so101_multiheight_part1_1345` (1,345 observations)
- Validation: `session_so101_multiheight_part2_149` (149 observations)
- Matches Run 7's setup for fair comparison

**Comparison Baselines**:
- **Run 7** (Decoder-Only): 0.0092 loss, 5.8m, CF=19.71
- **Run 11** (DiT 50 steps): 0.0287 loss, 180m, CF=27.19

## Phase 1: Inference Step Reduction (CURRENT)

**Hypothesis**: Diffusion models average predictions over many denoising steps, causing blur. Fewer steps = sharper predictions.

**Experiments**:
- **Iter 0 Baseline**: 50 DDIM steps (original Run 11 setting)
- **Iter 1**: 20 DDIM steps (2.5× fewer)
- **Iter 2**: 10 DDIM steps (5× fewer, extreme sharpness test)

**Expected Outcome**:
- If steps=20 or 10 improves loss significantly (e.g., <0.020), blur was the issue
- If steps reduction doesn't help much, the problem is deeper (VAE quality, architecture, etc.)

**Timeline**: ~3 hours per experiment

---

## Phase 2: Architecture Adjustments (IF Phase 1 shows promise)

If inference steps help, test other parameters:

### 2.1 Conditional Training Mode
- Run 11 uses **unconditional** mode (noise added to all patches)
- Try **conditional** mode (noise only on masked patches)
- Might improve action conditioning by keeping separators clean

```yaml
dit_training_mode: "conditional"  # vs "unconditional"
```

### 2.2 Lower Dimensionality
- Test if model is over-parameterized for small dataset
- Try: d8 e256 or d12 e128 instead of d12 e256
- Reduce overfitting on limited data

### 2.3 Different VAE Backend
- Frozen SD VAE might be destroying details
- Could try: `pretrained_flux` VAE (different compression)
- Would need to measure if tradeoff is worth it

---

## Phase 3: Training Parameter Adjustment (IF still no improvement)

If architecture isn't the issue:

### 3.1 Learning Rate
- Current: 3e-4 (from Run 11)
- Try: 1e-4 or 1e-3 for stability/optimization

### 3.2 Batch Size
- Current: 8
- Try: 4 or 16 to see if gradient signal improves

### 3.3 Divergence Parameters
- Current: gap=0.002, ratio=1.5, patience=30
- Adjust to be more/less aggressive with early stopping

---

## Phase 4: Custom VAE Training (IF nothing else works)

Most invasive change:

### 4.1 Train CanvasVAE on Robot Frames
- Use `train_vae.py` to create custom VAE
- Optimized for robot arm images, not natural images
- Trade: 2-3 hour training cost, but might unlock DiT potential

### 4.2 Use Custom VAE in DiT
```yaml
vae_type: "custom"
vae_checkpoint: "saved/checkpoints/vae/custom_robot_vae.pth"
```

---

## Decision Tree

```
Phase 1 Results?
├─ Iter 2 (10 steps) < 0.015 loss?
│  └─ YES → Steps are critical! Optimize further (3-5 more iterations)
├─ Iter 1 (20 steps) < 0.020 loss?
│  └─ YES → Moderate improvement. Continue to Phase 2 (conditional mode)
├─ No improvement from step reduction?
│  └─ Problem is deeper. Try Phase 2 (conditional/architecture)
└─ Still no luck after Phase 3?
   └─ Try Phase 4 (custom VAE) or accept DiT doesn't work for this task
```

---

## Success Criteria

**Small Win**: Achieve loss < 0.020 (30% improvement over Run 11)
**Major Win**: Achieve loss < 0.015 (50% improvement over Run 11)
**Beat Baseline**: Achieve loss < 0.0092 (match Run 7)

---

## Running Experiments

All Phase 1 experiments running in parallel:

```bash
# Check progress
python compare_dit_iterations.py

# Monitor individual tasks
watch -n 60 python compare_dit_iterations.py
```

Reports will be generated in:
```
saved/staged_training_reports/session_so101_multiheight_part1_1345/
  multiheight_dit_iter0_baseline/
  multiheight_dit_iter1_fewer_steps/
  multiheight_dit_iter2_10steps/
```

---

## Notes

- **No LR sweeps**: Avoid time-consuming optimization; focus on understanding failure modes
- **Single parameter changes**: Isolate variables to understand what's actually wrong
- **Visual analysis critical**: Loss metric alone might be misleading. Check inference samples.
- **Parallel execution**: Run multiple parameter variations simultaneously (GPU permitting)
- **Document findings**: Update memory with lessons learned regardless of success/failure
