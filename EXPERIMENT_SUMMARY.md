# Experimental Summary: DiT Improvement Attempts

## Overview

Conducted iterative experiments to determine if DiT (Diffusion Transformers) could beat Run 7's decoder-only baseline. After two attempts with different strategies, conclusively determined that DiT is unsuitable for this task.

## Quick Comparison Table

| Run | Architecture | VAE | Key Settings | Loss | CF Div | Samples | Time | Result |
|-----|--------------|-----|--------------|------|--------|---------|------|--------|
| 7 | Decoder-Only d12 e256 | N/A | Fine-tuned, LR=1e-5 | **0.0092** | 19.71 | 18,176 | 5.8m | BEST |
| 11 | DiT d12 e256 | Frozen SD | From scratch, no sweeps | 0.0287 | 27.19 | 183,040 | 180m | FAIL |
| 14 | DiT d12 e256 | Frozen SD | With sweeps, LR opt | 0.2992 | 123.27 | 1,280 | 157m | CATASTROPHIC |

## Detailed Results

### Run 7: Decoder-Only (Current Best)
**Configuration**:
- Architecture: DecoderOnlyViT d12 e256
- Separator width: 32 pixels
- Perceptual loss: 0.01 weight (VGG16)
- Learning rate: 1e-5 (very low, fine-tuning)
- Batch size: 8
- Training strategy: Fine-tuned from Run 5 checkpoint

**Performance**:
- Original session loss: **0.0092** (±0.0025 std)
- Counterfactual divergence: **19.71** (strong action conditioning)
- Training efficiency: 18,176 samples in ~5.8 minutes
- Stop reason: Divergence triggered (validation diverged from training)

**Why It Works**:
- Direct prediction (no latent compression)
- Progressive fine-tuning preserves learned features
- Separator tokens visible at full resolution
- Sample-efficient for limited data

---

### Run 11: DiT No Sweeps
**Configuration**:
- Architecture: DiffusionViT d12 e256
- VAE: Pretrained Stable Diffusion (frozen, 8× compression)
- Inference steps: 50 DDIM
- Learning rate: 3e-4 (from-scratch)
- No LR sweeps enabled
- Training strategy: From scratch

**Performance**:
- Original session loss: **0.0287** (3.1× worse than Run 7)
- Counterfactual divergence: **27.19** (stronger than Run 7, but worse overall)
- Training efficiency: 183,040 samples in 180 minutes
- Samples 10× more, but still didn't converge

**Why It Failed**:
- Frozen SD VAE not optimized for robot frames
- Sample inefficiency of diffusion models
- 10× more training needed than decoder-only
- Reconstruction quality degraded by latent compression

---

### Run 14: DiT With Plateau Sweeps
**Configuration**:
- Architecture: DiffusionViT d12 e256
- VAE: Pretrained Stable Diffusion (frozen, 8× compression)
- Inference steps: 20 DDIM (reduced from 50)
- Plateau sweeps: Enabled (proven in Run 12b for decoder-only)
- Initial LR sweep: Enabled
- LR range: 1e-7 to 1e-2
- Training strategy: From scratch with optimization

**Performance**:
- Original session loss: **0.2992** (32.6× worse than Run 7!!)
- Counterfactual divergence: **123.27** (extremely high, indicates noise not conditioning)
- Training efficiency: 1,280 samples in 2.6 minutes actual training
- Total time: 157 minutes (7.4h LR sweep + 2.6m training)

**Why It Failed Catastrophically**:
- LR sweep took 7.4 hours, only 4/11 trials completed (workers timed out)
- Selected extremely small LR (1.78e-06) from incomplete data
- Only 1 minute training budget after sweep (insufficient)
- 13.6% of target training samples reached
- Optimization of wrong architecture made things worse

---

## Key Insights

### Insight 1: Frozen SD VAE Fundamentally Incompatible
- Stable Diffusion VAE trained on diverse natural images (256×256)
- Our data: Limited robot arm angles, specific camera views
- 8× compression loses fine details needed for precise action prediction
- Latent space features don't align with our prediction task

### Insight 2: Diffusion Models Need More Data
- Diffusion learns full noise distribution (many training examples needed)
- Run 11 used 10× more samples (183k) than Run 7 (18k) but still underperformed
- With only 940 training frames, decoder-only direct prediction is more efficient
- Smaller dataset → simpler models win

### Insight 3: Optimization Can't Fix Wrong Architecture
- Run 14 used sophisticated plateau sweeps + LR optimization
- Result: WORSE than Run 11 (0.2992 vs 0.0287)
- Lesson: Spending 7.4 hours optimizing the wrong approach leaves no time for actual training
- Better to use simple approach with sufficient training time

### Insight 4: High CF Divergence ≠ Good Conditioning
- Run 14: 123.27 divergence (6× higher than Run 7)
- But loss was 32× worse
- Extreme divergence indicates model predicting wildly different (and mostly wrong) outputs
- Good conditioning (Run 7): Balance between specificity and accuracy

### Insight 5: Progressive Fine-Tuning Critical for Small Data
- Run 7 built on Run 5 checkpoint (built on Run 4 checkpoint)
- Each stage refined learned features with lower learning rates
- Run 11 & 14 trained from scratch, couldn't match convergence
- With limited data, warm-starting from proven checkpoint is essential

## What We Tried

### Iteration 1 Changes (Run 14)
Attempted to improve Run 11 with:
1. ✗ Reduced inference steps: 50 → 20
2. ✗ Plateau sweeps enabled (proven for decoder-only)
3. ✗ Initial LR sweep (should optimize learning rate)
4. ✗ Matched Run 7 batch/temperature settings

**Result**: Made it 10× worse (0.0287 → 0.2992)

### Why Standard Optimizations Failed
- Plateau sweeps require time for actual training after selection
- LR sweep worked great for Run 12b decoder-only (different architecture)
- Same techniques backfired on DiT (fundamentally flawed approach)
- Time spent optimizing = less time for training

## Conclusions

### 1. Run 7 is Optimal for This Problem
- Decoder-only architecture matches task requirements
- Sample-efficient training (18k samples, 5.8 min)
- Strong action conditioning (19.71 CF divergence)
- Best reconstruction quality (0.0092 loss)

### 2. DiT Approach Must Be Abandoned
- Run 11 (no optimization): 3.1× worse
- Run 14 (with optimization): 32.6× worse
- Frozen SD VAE fundamentally incompatible with task
- Diffusion models over-engineered for this problem

### 3. Future Improvements Should Focus On:
1. **Fine-tuning from Run 7**: Progressive tuning with lower LRs on new data
2. **Larger dataset**: Collect more observations (current 1,343 obs is limiting)
3. **Alternative architectures** (if change needed):
   - Vision Transformers without diffusion
   - State-space models (Mamba, S4, etc.)
   - Ensemble of simpler models
   - NOT diffusion-based without custom VAE training
4. **Custom VAE**: Train on robot frames if pursuing diffusion
5. **Action representation**: Maybe improve action encoding beyond color separators

### 4. Key Lesson for AI Engineering
- **Sophistication ≠ Performance** on small datasets
- Theoretical elegance (diffusion) loses to practical simplicity (decoder-only)
- Optimization effectiveness depends on architecture alignment
- Test fundamental assumptions early (Run 11 should have been a stop sign)

## Files Created

1. `multiheight_dit_run14_plateau.yaml` - Iteration 1 config
2. `check_iteration_ready.py` - Monitor script (detects when training complete)
3. `compare_dit_runs.py` - Quick comparison table generator
4. `DIT_FAILURE_ANALYSIS.md` - Detailed root cause analysis
5. `ITERATION_STATUS.md` - Project documentation
6. `EXPERIMENT_SUMMARY.md` - This file

## Recommendations

**For the user**:
1. Accept Run 7 as the best baseline (0.0092 loss)
2. Abandon all DiT experiments
3. If seeking improvement:
   - Collect more training data (with current architecture)
   - Fine-tune Run 7 on new data
   - Explore non-diffusion architectures
   - Train custom VAE if diffusion becomes necessary

**For future experiments**:
1. Always test assumptions on small experiments first
2. Monitor loss trends (Run 14's losses were always terrible from start)
3. Use simple baselines as sanity checks
4. Time-box exploratory work (don't spend 7.4 hours optimizing bad architecture)
5. Trust the data: if worst run is 32× worse, abandon approach

## Status

**DiT experiments: CONCLUDED**

- Run 7 remains best model
- Run 11 & 14 definitively ruled out DiT approach
- Ready to return to Run 7-based improvements or different directions
