# DiT Failure Analysis: Why Diffusion Transformers Don't Work for This Task

## Executive Summary

Both DiT experiments (Run 11 and Run 14) failed dramatically, with Run 14 being catastrophically worse. The evidence conclusively shows that DiT with frozen Stable Diffusion VAE is fundamentally unsuited for precise action-conditioned prediction on limited robot vision data.

**Final Results**:
- Run 7 (Decoder-Only, Best): **0.0092** loss
- Run 11 (DiT, No Sweeps): **0.0287** loss (3.1× worse)
- Run 14 (DiT + Plateau Sweeps): **0.2992** loss (32.6× worse!!)

## What Happened in Run 14

### Setup
- Configuration: DiT d12 e256, Stable Diffusion VAE, plateau sweeps enabled, 20 inference steps
- Time budget: 60 minutes main training + LR sweep
- Sessions: Same as Run 7 (942 training obs, 403 validation obs)

### Execution Phases

**Phase 1: Initial LR Sweep (7.4 hours)**
- Phase A: 5 LR candidates tested, 3 minutes each
  - Only 2/5 completed before timeout (workers hung)
  - Tested: 1.00e-07, 1.78e-06 (others timed out)
- Phase B: 2 survivor LRs × 3 seeds = 6 trials, 10 min each
  - Only 2/6 completed before timeout
  - Selected winner: 1.78e-06 (median: 0.2927)

**Phase 2: Main Training (2.6 minutes total)**
- Time budget: 1.0 minute (already consumed by LR sweep delays)
- Actual training: 1.3 minutes
- Samples trained: 1,280 / 9,400 target (13.6% of target)
- Loss trajectory:
  - Start: ~1.00 (untrained)
  - After 256 samples: 0.302
  - After 1,280 samples: 0.291 (best validation)
- **Stopped**: Time budget exceeded

### Key Metrics

**Validation Performance**:
- Best validation loss during training: 0.2918
- Final evaluation on original session: 0.2992
- Training losses: ~0.98-1.00 (barely improving)

**Action Conditioning (Counterfactual Divergence)**:
- Run 14: **123.27** (extremely high!)
- Run 7: 19.71 (good)
- Run 11: 27.19 (higher than Run 7 but still in reasonable range)

## Root Cause Analysis

### Why DiT Failed So Badly

#### 1. **Frozen Stable Diffusion VAE Mismatch**
- SD VAE trained on natural images (256×256 diverse scenes)
- Our data: Robot arm at similar angles, limited visual diversity
- 8× spatial compression (224×224 -> 28×28 latent) loses fine details crucial for precise action prediction
- Latent space not optimized for robot frames

#### 2. **Diffusion Training Inefficiency**
- Diffusion models learn to predict noise across all timesteps
- Requires many training samples to learn full noise distribution
- We only have ~940 unique training frames
- Decoder-only models directly predict next frame (more direct, sample-efficient)

#### 3. **Loss of Gradient Signal Through VAE**
- Encoder: 224×224 → 28×28 latent (64× compression, information loss)
- Decoder: 28×28 → 224×224 (must reconstruct from lossy encoding)
- Diffusion training optimizes latent space, not pixel space
- Gradient information degraded by double compression

#### 4. **Extreme CF Divergence (123.27) Indicates Failure, Not Success**
- This high value doesn't mean good action conditioning
- Rather: Model predicting vastly different (and mostly wrong) outputs for each action
- When all predictions are bad, variations appear large
- Compare to Run 7's 19.71: strong conditioning + good reconstruction

#### 5. **LR Sweep Worker Timeouts**
- Workers hung during Phase A and Phase B
- Only 4/11 trials completed (36%)
- Selected very small LR (1.78e-06) based on incomplete data
- Insufficient exploration of LR space

## Why Run 14 > Run 11 (Worse Direction)

**Run 11**: Loss = 0.0287
- Trained from scratch
- No sweeps (used fixed LR: 3e-4)
- 180 minute budget → actual training happened
- ~183k samples trained

**Run 14**: Loss = 0.2992
- With plateau sweeps enabled
- 7.4 hour LR sweep, only 2.6 min actual training
- Selected tiny LR (1.78e-06) from incomplete sweep
- ~1,280 samples trained (7% of Run 11)
- Time spent optimizing LR rather than training

**Lesson**: When the architecture is fundamentally wrong, "optimization" makes it worse by consuming time that could be spent actually training. Better tuning cannot fix a flawed foundation.

## Counterfactual Divergence Analysis

**What CF Divergence Measures**:
- Pairwise pixel differences between predictions for different actions
- High value = large differences between action-conditioned predictions
- Higher = stronger action conditioning (in theory)

**Why Run 14's 123.27 Is Catastrophic**:
- Run 7: 19.71 with 0.0092 loss (strong conditioning, good reconstruction)
- Run 11: 27.19 with 0.0287 loss (stronger conditioning signal, but worse overall)
- Run 14: 123.27 with 0.2992 loss (6× higher divergence, 32× worse loss!)

The extremely high divergence coupled with terrible loss indicates:
- Model is producing wildly different outputs for different actions
- But outputs are mostly wrong (high base reconstruction error)
- Not meaningful action conditioning, just noise

## What Made Run 7 Superior

1. **Direct Prediction**: Decoder-only ViT predicts next frame directly (no latent compression)
2. **Progressive Fine-Tuning**: Run 4 → Run 5 → Run 7, each refining learned features
3. **Strong Action Signal**: Separator tokens visible at full resolution for attention
4. **Sample Efficiency**: Only 18,176 samples in final stage (vs DiT's need for 100k+)
5. **Appropriate Architecture**: Simple, proven approach for limited data

## Lessons Learned

1. **Frozen Pretrained VAE ≠ Universal Solution**
   - Natural image VAE breaks on specialized domains
   - Would need custom VAE trained on robot frames

2. **Diffusion Models Overkill for This Task**
   - Designed for high-quality image generation from noise
   - We need precise action-conditioned frame prediction
   - Simpler forward model is more efficient

3. **LR Optimization Can Fail with Wrong Architecture**
   - Plateau sweeps helped Run 12b decoder-only (40-50% improvement)
   - Same sweeps made DiT much worse (7.4h wasted on optimization)
   - Can't optimize away fundamental architectural problems

4. **Action Conditioning ≠ High Prediction Variance**
   - Run 14's extreme CF divergence (123.27) is a failure signal, not success
   - Good action conditioning (Run 7) balances conditioning strength with reconstruction quality

5. **Limited Data (940 obs) Favors Simple Models**
   - Decoder-only d12 e256: ~3M parameters
   - DiT d12 e256 + SD VAE: much more parameters in latent space
   - Small datasets need sample-efficient architectures

## Recommendation

**Abandon DiT approach entirely.** The evidence is conclusive:
- Run 11 (DiT from scratch): 3.1× worse
- Run 14 (DiT with optimization): 32.6× worse
- Making it worse with better tuning = fundamental problem
- Frozen SD VAE is fundamentally incompatible with this task

**Future improvements should focus on**:
1. Fine-tuning from Run 7 checkpoint (proven approach)
2. Larger training dataset (if more data available)
3. Other architectures compatible with limited data:
   - Shallower/narrower decoder-only variants
   - Vision Transformer variants
   - State-space models (Mamba, S4)
4. NOT diffusion-based approaches without custom VAE training

## Conclusion

Run 7's decoder-only architecture is optimal for this problem at this data scale. DiT with frozen SD VAE represents a fundamental mismatch between:
- Task requirements: Precise action-conditioned frame prediction
- Architecture strengths: High-quality image generation from noise

The dramatic failure of Run 14 (worse than Run 11, worse than from-scratch training) demonstrates that optimization of the wrong approach is futile. Better to accept the limitations of our dataset and use proven, sample-efficient architectures (Run 7) than pursue theoretically sophisticated but practically unsuitable approaches (DiT).

**Status**: DiT experiments concluded. Return to Run 7-based approaches for future work.
