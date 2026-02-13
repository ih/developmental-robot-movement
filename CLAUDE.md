# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for developmental robot movement with a canvas-based world model architecture:

1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Concrete JetBot robot interface
3. **Toroidal Dot Environment** (`toroidal_dot_env.py`, `toroidal_dot_interface.py`) - Simulated environment with white dot on black background for testing and debugging
4. **SO-101 Robot Arm Integration** (`lerobot_policy_simple_joint/`, `convert_lerobot_to_explorer.py`) - LeRobot custom policy and dataset converter for SO-101 follower arm
5. **Autoencoder Concat Predictor World Model** (`autoencoder_concat_predictor_world_model.py`) - Canvas-based world model using targeted masked autoencoder with frame concatenation
6. **Action Selectors** (`toroidal_action_selectors.py`, `recorded_policy.py`) - Pluggable action selection strategies
7. **Concat World Model Explorer** (`concat_world_model_explorer/`) - Modular web-based interface for exploring and visualizing the concat world model
8. **Staged Training** (`staged_training.py`, `staged_training_config.py`, `create_staged_splits.py`) - Automated progressive training pipeline with HTML reports
9. **Learning Rate Sweep** (`lr_sweep.py`) - Time-budgeted learning rate optimization with two-phase search (broad exploration + deep validation)

## Environment
- Python virtual environment: `C:\Projects\pythonenv-deeprl`
- Activate before running Python commands: `C:\Projects\pythonenv-deeprl\Scripts\activate`

## Architecture

### RobotInterface Abstraction
- **Abstract base class**: Defines standard interface for any robot type
- **Key methods**: `get_observation()`, `execute_action()`, `action_space`, `cleanup()`
- **Extensible design**: Easy to implement new robot types by inheriting from RobotInterface

### JetBot Implementation
- **RemoteJetBot class**: Handles RPyC connections, camera capture, and motor control
- **JetBotInterface**: Wrapper implementing RobotInterface for JetBot robots
- **Live video feed**: Real-time display using OpenCV windows
- **Connection**: Connects to JetBot at configurable IP on port 18861

### Toroidal Dot Environment
- **ToroidalDotEnvironment**: Simple 224x224 simulated environment with a white dot on black background
- **Toroidal wrapping**: Horizontal movement wraps around at image edges (x-axis is toroidal)
- **Binary action space**: Action 0 (stay), Action 1 (move right by configurable pixels)
- **Flexible initialization**: Supports both random and fixed (x, y) positions via `initial_x` and `initial_y` parameters
- **ToroidalDotRobot class**: Implements RobotInterface for simulated environment
- **Fast iteration**: No hardware needed, perfect for debugging and testing world model
- **Configurable parameters**: Dot radius, movement speed, image size via ToroidalDotConfig
- **Default parameters**: DOT_RADIUS=5, DOT_MOVE_PIXELS=27, IMG_SIZE=224, DOT_ACTION_DELAY=0.0

### SO-101 Robot Arm Integration
- **LeRobot custom policy**: `lerobot_policy_simple_joint` package for single-joint control with 3 discrete actions
- **Action space**: Action 0 (stay), Action 1 (move positive by position_delta), Action 2 (move negative by position_delta)
- **Action parameters**: `action_duration` controls how long each discrete action lasts (default 0.5s, can be auto-calibrated), `position_delta` controls movement magnitude (default 0.1 radians)
- **Servo auto-calibration**: `run_lerobot_record.py` can automatically measure servo settling time to determine optimal `action_duration`, ensuring the recorded timing matches actual robot movement
  - Sends test movement, polls `Present_Position` at 10ms intervals until stable (range < 0.5 units for 50ms)
  - Applies 1.2x safety margin to measured settling time
  - Visual verification: runs test sequence [MOVE+, STAY, MOVE-, STAY] with canvas preview showing colored separators
  - Skip with `--skip-calibration` (uses default 0.5s) or `--skip-verification` (calibrates but no visual check)
- **Configurable joint**: Control any SO-101 joint (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **Policy modes**: Random exploration (infinite), fixed action sequences (execute once, no wrapping), or deterministic (always stay)
- **Discrete action logging**: Automatically records exact discrete actions during `lerobot-record` sessions to JSONL logs in `meta/discrete_action_logs/` directory (included in Hub uploads)
  - Log header contains all recording parameters (joint_name, action_duration, position_delta, etc.)
  - Each log entry contains timestamp and discrete action chosen
  - Enables 100% accurate action reconstruction without velocity-based approximation
- **Dataset converter**: `convert_lerobot_to_explorer.py` converts LeRobot v3.0 datasets to concat_world_model_explorer format
  - **Hub download support**: Converter can download datasets directly via HuggingFace repo_id (e.g., `username/dataset-name`)
  - **Automatic parameter extraction**: Reads action_duration and position_delta from log header when available
  - **Parquet-anchored mapping**: Uses per-frame action/state data to find exact video frame of first MOVE action, anchoring the proportional mapping to ground truth
  - **Timestamp-based matching**: Uses precise frame-to-action matching via log timestamps
  - **Backward compatible**: Falls back to even distribution for datasets without timestamps
- **Dual-camera support**: Stacks base_0_rgb (224×224) and left_wrist_0_rgb (224×224) vertically for 448×224 combined frames
- **Compatible with concat_world_model_explorer**: Converted sessions can be loaded and visualized in the world model explorer

### Autoencoder Concat Predictor World Model
- **Canvas-based approach**: Concatenates history frames horizontally with action-colored separators between them
- **Targeted masking**: Uses masked autoencoder to inpaint masked next-frame slot for prediction with full masking (MASK_RATIO = 1.0)
- **Main loop architecture**: Get frame → compute prediction error → single-step train on canvas → select action → predict next frame → execute action
- **MAE-native training**: Optimizes masked patches only (not full canvas reconstruction) for efficient inpainting learning
- **Action encoding**: Actions encoded as thin colored separators (e.g., red for stay, green for move)
- **Non-square canvases**: Supports non-square image dimensions (e.g., 224x688 for 3 frames + 2 separators)
- **Pluggable action selectors**: Custom action selection via `action_selector` parameter (returns action only, no metadata)
- **Visualization state**: Tracks training canvas with mask overlay, inpainting output, composite reconstruction, and predicted frame
- **Training callback**: Optional callback for periodic UI updates during training
- **Quality gating**: Single training step per iteration with inpainting threshold of 0.0001

### Action Selectors
- **Pluggable architecture**: Action selectors are functions that take observations and return actions
- **Toroidal action selectors** (`toroidal_action_selectors.py`):
  - `create_constant_action_selector(action)`: Always returns the same action
  - `create_sequence_action_selector(sequence)`: Cycles through a sequence of actions
  - `create_random_duration_action_selector(min_duration, max_duration, seed)`: Randomly chooses actions and maintains them for random consecutive steps
  - Pre-defined sequences: `SEQUENCE_ALWAYS_MOVE`, `SEQUENCE_ALWAYS_STAY`, `SEQUENCE_ALTERNATE`, `SEQUENCE_DOUBLE_MOVE`, `SEQUENCE_TRIPLE_MOVE`
- **Recorded action selector** (`recorded_policy.py`):
  - `create_recorded_action_selector(reader)`: Replays actions from recorded sessions
  - Optional action filtering for selective replay
- **Usage**: Pass to `AutoencoderConcatPredictorWorldModel` via `action_selector` parameter

## Key Components

### Concat World Model Explorer
- **concat_world_model_explorer/**: Modular web-based Gradio interface for running AutoencoderConcatPredictorWorldModel on recorded toroidal dot sessions
  - Refactored from monolithic script into organized package structure
  - `app.py`: Main Gradio application with UI layout and event handlers
  - `state.py`: Centralized application state management
  - `session_manager.py`: Session loading and frame navigation
  - `canvas_ops.py`: Canvas building and preprocessing operations
  - `inference.py`: Single-frame inference without training
  - `evaluation.py`: Full-session evaluation with comprehensive statistics
  - `training.py`: Batch training with 4-phase performance optimizations
  - `checkpoint_manager.py`: Model checkpoint save/load with metadata
  - `attention.py`: Decoder attention visualization with multiple modes
  - `visualization.py`: Plotting and display utilities
  - `utils.py`: Shared helper functions
- **Canvas-based approach**: Uses targeted masked autoencoder with full masking (MASK_RATIO = 1.0) for next-frame inpainting
- **Session replay**: Load and replay recorded sessions with authentic single-step world model training
- **Live progress tracking**: Real-time display of prediction error and iteration timing during execution
- **Comprehensive visualizations**: Shows 4 training views - original canvas, masked canvas overlay, full inpainting output, and composite reconstruction
- **Prediction display**: Current frame, predicted next frame, and prediction error visualization
- **Metric graphs**: Plots showing prediction error and iteration time over all iterations
- **Authentic training**: Uses actual `AutoencoderConcatPredictorWorldModel.train_autoencoder()` method with periodic UI updates
- **Model checkpoint management**: Save and load trained model weights with full optimizer and scheduler state
  - Save weights with custom names to `saved/checkpoints/toroidal_dot/`
  - Load previously saved checkpoints to resume training
  - Checkpoint metadata includes timestamp, config, and training metrics
  - Preserves optimizer and scheduler state for seamless training continuation
- **Decoder attention visualization**: Interactive visualization of decoder attention patterns from masked patches to unmasked patches
  - Quantile-based filtering: Show top N% of attention connections (e.g., 95% = strongest 5%)
  - Layer selection: Toggle individual decoder layers (0-4) on/off
  - Head selection: Toggle individual attention heads (0-3) on/off for fine-grained analysis
  - Aggregation methods: Mean, max, or sum across selected heads
  - Visualization types: Patch-to-patch lines (color-coded by layer, thickness by magnitude) or heatmap matrix
  - Real-time statistics: Connection counts, attention weights, and per-layer/per-head metrics
- **Batch training performance optimizations** (4-phase optimization achieving 6.92x absolute speedup):
  - **Phase 1 - Canvas pre-building**: Pre-load and build all training canvases at session load time
    - Eliminates 60-100ms per-batch PIL resize overhead
    - Caches ~6400 canvases in memory (~3GB) for instant access
    - Implemented in `session_explorer_lib.py:prebuild_all_canvases()`
  - **Phase 2 - DataLoader integration**: PyTorch DataLoader with pinned memory for async GPU transfers
    - New `models/canvas_dataset.py` with `PrebuiltCanvasDataset` and `CanvasCollateFn`
    - Pinned memory enables non-blocking CPU→GPU transfers
    - Windows: single-process (num_workers=0), Linux: multi-process (num_workers=4)
  - **Phase 3 - GPU mask generation**: Vectorized GPU-based patch mask computation
    - `compute_randomized_patch_mask_for_last_slot_gpu()` in `autoencoder_concat_predictor.py`
    - Replaces Python random operations with torch.rand()/torch.randperm() on GPU
    - 10-20x faster than CPU version
  - **Phase 4 - CUDA stream pipelining**: Overlap GPU training with CPU→GPU transfers
    - Separate CUDA stream for async data transfers
    - Double-buffering: prefetch batch N+1 while training batch N
    - Maximizes GPU utilization, eliminates idle time
  - **Performance results** (BS=64): 29.44s → 18.72s (1.57x speedup, 36.4% faster)
  - **Absolute speedup** (BS=1 → BS=64): 129.50s → 18.72s (6.92x faster, 342 samples/sec)
- **Comprehensive training diagnostics**: Multi-tier logging system for debugging and analysis
  - **TrainingLogger**: Automatic per-batch metrics logging with rotated JSONL files
  - **Interval summaries**: Compact aggregated statistics every N batches for LLM-friendly analysis
  - **Batch composition tracking**: Monitor which samples appear in each batch to detect averaging issues
  - **Gradient flow monitoring**: Track gradient norms across all model components
  - **Real-time analysis**: `analyze_training_logs.py` tool for ongoing training diagnostics
  - **Automatic diagnostics**: Detects plateau, gradient explosion, batch averaging, and other common training issues
  - **File structure**: `saved/training_logs/{session}/run_{timestamp}/` with config, summaries, and raw metrics

### Neural Vision System
- **MaskedAutoencoderViT**: Vision Transformer-based autoencoder with powerful transformer encoder and decoder
- **Symmetric architecture**: Decoder uses transformer blocks with same depth and attention heads as encoder by default (configurable via `decoder_depth` and `decoder_num_heads` parameters)
- **Dynamic masking**: Randomized mask ratios for better generalization (shared config: MASK_RATIO_MIN and MASK_RATIO_MAX)
- **Targeted masking for prediction**: Full masking (MASK_RATIO = 1.0) of next-frame slot for inpainting-based prediction
- **MAE-native optimization**: Trains only on masked patches for efficient learning
- **Quality gating**: Single training step per iteration with reconstruction threshold
- **Non-square image support**: Handles concatenated canvases with non-square dimensions

### Action Space
- **Duration-based actions**: Motor commands with automatic stopping after specified duration
- **Simplified action space**: Single motor control with motor_right values {0, 0.12} = 2 total actions (motor_left always 0)
- **Forward-only movement**: Gentler gearbox operation with stop and forward-only commands
- **Format**: `{'motor_left': 0, 'motor_right': value, 'duration': 0.2}`
- **Smooth ramping**: Gradual acceleration/deceleration for gearbox protection
- **Automatic stopping**: Motors automatically ramp down to 0 after duration expires
- **Motor configuration**: Uses right motor for movement control, left motor remains at 0

### Configuration
- **config.py**: Contains image transforms, constants, and world model parameters
- **AutoencoderConcatPredictorWorldModelConfig class**: Configuration for canvas-based world model (frame size, separator width, canvas history size, reconstruction threshold, optimizer parameters)
- **Robot-specific directories**:
  - `JETBOT_CHECKPOINT_DIR = saved/checkpoints/jetbot/` - JetBot model checkpoints
  - `TOROIDAL_DOT_CHECKPOINT_DIR = saved/checkpoints/toroidal_dot/` - Toroidal dot model checkpoints
  - `SO101_CHECKPOINT_DIR = saved/checkpoints/so101/` - SO-101 model checkpoints
  - `JETBOT_RECORDING_DIR = saved/sessions/jetbot/` - JetBot session recordings
  - `TOROIDAL_DOT_RECORDING_DIR = saved/sessions/toroidal_dot/` - Toroidal dot session recordings
  - `SO101_RECORDING_DIR = saved/sessions/so101/` - SO-101 session recordings
- **ToroidalDotConfig class**: Configuration for simulated environment (IMG_SIZE, DOT_RADIUS, DOT_MOVE_PIXELS, ACTION_CHANNELS_DOT, ACTION_RANGES_DOT)
- **SO101Config class**: Configuration for SO-101 follower arm (JOINT_NAMES, DEFAULT_MOVE_DURATION, DEFAULT_MOVE_SPEED, ACTION_SPACE, FRAME_SIZE with dual-camera stacking)
- **Recording configuration**: `RECORDING_MODE` boolean controls recording vs online mode, `RECORDING_MAX_DISK_GB` limits total disk usage
- **IP addresses**: JetBot connection IPs specified in integration example (modify as needed)

## Running the Code

### JetBot Live Feed Only
```bash
python jetbot_remote_client.py
```
- Displays live camera feed from JetBot
- Requires JetBot running RPyC server
- Press 'q' or Ctrl+C to stop

### SO-101 Robot Arm with LeRobot

#### Install the Policy Package
```bash
cd lerobot_policy_simple_joint
pip install -e .
```

#### Record with lerobot-record

**Recommended**: Use `run_lerobot_record.py` wrapper for action sequences (auto-calculates episode time):
```bash
python run_lerobot_record.py \
    --robot.type=so101_follower \
    --robot.port=COM8 \
    --robot.id=my_so101_follower \
    --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --policy.type=simple_joint \
    --policy.joint_name=wrist_roll.pos \
    --policy.action_duration=0.5 \
    --policy.position_delta=10 \
    --policy.action_sequence="[1, 0, 2, 0, 1]" \
    --dataset.repo_id=${HF_USER}/so101-test \
    --dataset.num_episodes=1 \
    --dataset.single_task="Single joint movement"
```
- Automatically calculates `episode_time_s` = (num_actions × action_duration) + 5s buffer
- Buffer accounts for camera warmup, calibration, and final action completion
- Action sequence executes once and stops (no wrapping)
- Policy outputs action 0 (stay) after sequence completes
- Start action_sequence with 1 or 2 (move) instead of 0 (stay) for immediate visible movement

**Alternative**: Direct lerobot-record command (manual episode time):
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_so101 \
    --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
    --policy.type=simple_joint \
    --policy.joint_name=shoulder_pan.pos \
    --policy.action_duration=0.5 \
    --policy.position_delta=0.1 \
    --policy.use_random_policy=true \
    --dataset.repo_id=${HF_USER}/so101-single-joint \
    --dataset.num_episodes=10 \
    --dataset.single_task="Single joint movement"
```

#### Convert LeRobot Dataset to Explorer Format

**Recommended** (downloads from Hub, reads parameters from log):
```bash
python convert_lerobot_to_explorer.py \
    --lerobot-path ${HF_USER}/so101-single-joint \
    --output-dir saved/sessions/so101 \
    --cameras base_0_rgb left_wrist_0_rgb \
    --stack-cameras vertical
```
- Converter automatically downloads dataset from HuggingFace Hub
- Reads `action_duration` and `position_delta` from discrete action log header
- No need to manually specify recording parameters
- Uses timestamp-based matching for frame-to-action pairing

**Alternative** (local dataset with manual parameters):
```bash
python convert_lerobot_to_explorer.py \
    --lerobot-path ~/.cache/huggingface/lerobot/${HF_USER}/so101-single-joint \
    --output-dir saved/sessions/so101 \
    --cameras base_0_rgb left_wrist_0_rgb \
    --stack-cameras vertical \
    --joint-name shoulder_pan.pos \
    --action-duration 0.5
```
- Specify local path to dataset instead of Hub repo_id
- Manually provide `--action-duration` and `--position-delta` if logs are unavailable

### Concat World Model Explorer

**Usage:**
```bash
# Auto-find available port starting from 7861
python -m concat_world_model_explorer

# Use specific port (for running multiple instances)
python -m concat_world_model_explorer --port 7862

# Create public Gradio link
python -m concat_world_model_explorer --share
```

**Features:**
- **Canvas-based world model**: Interactive web UI for exploring AutoencoderConcatPredictorWorldModel on recorded sessions
- **Session selection**: Choose from recorded sessions in `saved/sessions/toroidal_dot/` or `saved/sessions/so101/`
- **Frame navigation**: Browse session frames with slider and text input
- **World model execution**: Run world model for specified number of iterations with single-step training per iteration
- **Full masking approach**: Uses MASK_RATIO = 1.0 for complete next-frame inpainting
- **Live progress tracking**: Real-time display of prediction error, learning rate, and iteration timing during execution
- **Comprehensive visualizations**:
  - Four training canvas views (original canvas, masked overlay, full inpainting output, composite reconstruction)
  - Loss graphs (full history + rolling window)
  - Learning rate schedule graph (log scale for warmup + cosine decay visualization)
  - Training observation samples (random frames + current frame)
- **Post-run visualizations**: Current frame, predicted frame, and prediction error
- **Metric graphs**: Plots tracking loss and learning rate over all iterations
- **Authentic training**: Uses actual `AutoencoderConcatPredictorWorldModel.train_autoencoder()` method with MAE-native masked patch optimization
- **Model checkpoint management**:
  - **Save weights**: Enter checkpoint name and click "💾 Save Weights" to save model, optimizer, and scheduler state
  - **Load weights**: Select checkpoint from dropdown and click "📂 Load Weights" to restore previous training state
  - **Checkpoint location**: All checkpoints saved to robot-specific directories (e.g., `saved/checkpoints/toroidal_dot/`)
  - **Metadata tracking**: Each checkpoint includes timestamp, config parameters, and training metrics
  - **Instance-aware naming**: Multiple instances running on different ports save checkpoints with unique names to prevent collisions
- **Inference-only evaluation**:
  - **Single canvas inference**: Run inference without training on selected frame to see predictions
  - **Full session evaluation**: Calculate loss statistics over all observations for objective model comparison
  - **Comprehensive metrics**: Mean, median, std dev, percentiles, loss over time plots, and distribution histograms
- **Decoder attention visualization**: Interactive exploration of decoder attention patterns
  - **Patch selection**: Automatic dot detection based on brightness threshold or manual patch selection
  - **Frame-based analysis**: Visualize attention from selected frame (not just training canvas)
  - **Multiple visualization types**: Patch-to-patch connection lines, attention matrix heatmap, or heatmap overlay on canvas
  - **Configurable aggregation**: Choose how to aggregate attention across heads (mean/max/sum) and selected patches
  - **Layer and head filtering**: Toggle individual decoder layers (0-4) and attention heads (0-3) for detailed analysis
  - **Attention direction**: Shows attention FROM selected patches (e.g., dot patches) TO all other patches in the canvas

**Parallel Training:**
- Multiple instances can run simultaneously on different ports
- Each instance automatically gets a unique checkpoint prefix (e.g., `p7861`, `p7862`)
- Checkpoints are named: `best_model_auto_{session}_{instance_id}_{samples}_{type}_{loss}.pth`
- Requires sufficient GPU memory for each instance (~1-4GB per instance depending on batch size)

### Staged Training

Automated training pipeline that progressively trains on increasing data sizes with HTML reports.

**Create Staged Splits:**
```bash
# Create progressive train/validation splits from a session
python create_staged_splits.py --session-path saved/sessions/so101/my_session

# Customize initial size and train ratio
python create_staged_splits.py --session-path saved/sessions/so101/my_session --initial-size 20 --train-ratio 0.8
```
- Creates progressively larger training/validation splits (10, 20, 40, 80, ... observations)
- Default 70/30 train/validate split at each stage
- Output: `{session}_stage{N}_train_{size}` and `{session}_stage{N}_validate_{size}`

**Run Staged Training:**
```bash
# Basic usage (plateau-triggered sweeps enabled by default)
python staged_training.py --root-session saved/sessions/so101/my_session

# Multiple runs per stage for robustness
python staged_training.py --root-session saved/sessions/so101/my_session --runs-per-stage 3

# With time budget per stage (main training only; sweeps use lr_sweep phase budgets)
python staged_training.py --root-session saved/sessions/so101/my_session --stage-time-budget-min 10

# Single stage only (run just stage 2)
python staged_training.py --root-session saved/sessions/so101/my_session --stage 2

# Direct session mode (no staged splits needed)
python staged_training.py --train-session saved/sessions/so101/train --val-session saved/sessions/so101/val

# Direct mode with explicit original session for evaluation
python staged_training.py --train-session .../train --val-session .../val --original-session saved/sessions/so101/full_session

# Resume from a checkpoint
python staged_training.py --train-session .../train --val-session .../val --checkpoint saved/checkpoints/model.pth

# Custom LR sweep configuration (applies to both plateau and upfront sweeps)
python staged_training.py --root-session saved/sessions/so101/my_session \
    --lr-sweep-phase-a-candidates 5 \
    --lr-sweep-phase-a-budget-min 2.0 \
    --lr-sweep-phase-b-seeds 3 \
    --lr-sweep-phase-b-budget-min 5.0

# Use upfront sweeps instead of plateau-triggered sweeps (legacy mode)
python staged_training.py --root-session saved/sessions/so101/my_session --disable-plateau-sweep

# Disable baseline comparison
python staged_training.py --root-session saved/sessions/so101/my_session --disable-baseline

# Custom configuration file
python staged_training.py --root-session saved/sessions/so101/my_session --config my_config.yaml
```

**Features:**
- **Progressive training**: Trains on each stage's data until stopping criteria are met, then moves to next stage
- **Dynamic sample budget**: `stage_samples_multiplier` scales training samples based on stage size (e.g., 61 frames × 1000 = 61,000 samples) to prevent overfitting on small stages
- **Divergence-based early stopping**: Automatically stops when validation loss diverges from training
- **Plateau-triggered LR sweeps** (default mode): When validation loss plateaus, triggers a multi-phase LR sweep
  - Uses constant LR scheduler (no decay) - LR adaptation happens through sweep-triggered restarts
  - Uses current weights for sweep, continues with winning LR and weights
  - Configurable: `plateau_patience=25` updates, `improvement_threshold=0.5%`, `cooldown=5` updates
  - Maximum sweeps per stage (default 2) prevents infinite optimization loops
  - Early stop on sweep failure: stops training if sweep produces no improvement (`min_sweep_improvement`)
  - Sweeps use full Phase A → Phase B multi-phase structure
  - GPU memory management: frees model before sweep, reloads from winning checkpoint after
- **Loss-weighted sampling**: Focuses on high-loss samples for efficient learning
- **Upfront LR sweep** (legacy mode): Automatic LR optimization before each stage (when `plateau_sweep.enabled=False`)
  - **Phase A**: Broad exploration with many LR candidates, short time budgets
  - **Phase B**: Deep validation with top survivors, multiple seeds
  - Ranking by median/mean/min best validation loss across seeds
- **Baseline comparison**: Optionally run parallel baseline training (fresh weights each stage) to compare against staged training (weight carryover)
- **Serial runs** (`--serial-runs`, default): Runs `runs_per_stage` sequentially to reduce peak GPU memory; parallel mode still available
- **Initial LR sweep** (`initial_sweep_enabled=True`): Runs an upfront LR sweep before each stage regardless of whether plateau sweeps are enabled; disable with `--disable-initial-sweep`
- **Time budget control**: Optional per-stage time budget for main training
- **Interrupt/crash recovery**: Catches `KeyboardInterrupt` and exceptions, recovers the interrupted stage from auto-saved checkpoints on disk, and generates a partial report with all completed stages
- **HTML reports**: Generates comprehensive reports with training progress, inference visualizations, staged vs baseline comparison, LR sweep results (including plateau sweep history), config diff vs last commit, full training loss timeline, multi-run statistics, and evaluation metrics
- **Progressive reporting**: Final report is updated after each stage for real-time progress visibility
- **Best checkpoint selection**: Selects best checkpoint based on hybrid loss over original (full) session
- **W&B integration**: Optional Weights & Biases logging with run_id in run names and baseline config tracking
- **Concurrent execution**: Multiple instances can run with isolated checkpoints using `--run-id`

**Configuration** (`staged_training_config.py`):
- All parameters match Gradio app defaults
- Key parameters: `stage_samples_multiplier`, `divergence_patience`, `loss_weight_temperature`
- **Sweep mode**: Controlled by `plateau_sweep.enabled` (default True = plateau-triggered sweeps, False = upfront sweeps before each stage)
- **Plateau Sweep config**: `plateau_sweep.plateau_patience` (25 updates), `plateau_sweep.plateau_improvement_threshold` (0.5%), `plateau_sweep.cooldown_updates` (5), `plateau_sweep.max_sweeps_per_stage` (2), `plateau_sweep.min_sweep_improvement` (0.0)
- **LR Sweep config** (shared by both modes): `lr_sweep.lr_min`, `lr_sweep.lr_max`, `lr_sweep.phase_a_num_candidates`, `lr_sweep.phase_a_time_budget_min`, `lr_sweep.phase_b_seeds`, `lr_sweep.phase_b_time_budget_min`
- Baseline config: `enable_baseline` (default False), `baseline_runs_per_stage` (default 1)
- Stage time budget: `stage_time_budget_min` (0 = unlimited)
- `serial_runs` (default True): run multiple runs per stage serially instead of in parallel
- `initial_sweep_enabled` (default True): run upfront LR sweep before each stage (orthogonal to `plateau_sweep.enabled`)
- Supports YAML config files for reproducible experiments
- **Deprecated fields**: `val_plateau_patience`, `val_plateau_min_delta`, `plateau_factor`, `plateau_patience` (replaced by plateau_sweep when enabled)

**Reports:**
- Per-stage reports: `saved/staged_training_reports/{session}/stage{N}_run{M}/report.html`
- Baseline reports: `saved/staged_training_reports/{session}/stage{N}_baseline_run{M}/report.html`
- Final summary: `saved/staged_training_reports/{session}/final_report_{date}.html` (dated, e.g., `final_report_2026_feb_07.html`)
- Also copied to: `docs/final_report_{date}.html` for easy access
- Includes: training progress graphs, hybrid loss over session graphs, full training loss timeline across all stages, config diff vs last commit, multi-run statistics (when runs_per_stage > 1), staged vs baseline comparison (winner, per-stage metrics), inference visualizations, evaluation statistics

### Recording Sessions
To create new sessions for exploration:
1. Set `RECORDING_MODE = True` in `config.py`
2. Run a robot interface with recording enabled (e.g., ToroidalDotRobot with action selectors)
3. Sessions are automatically saved to robot-specific directories
4. **Robot-specific directories**: JetBot and toroidal dot sessions stored separately for organization
5. **Disk space management**: Automatic cleanup of oldest sessions when total recordings exceed configurable disk limit (default 10 GB per robot type)

## Dependencies

Install dependencies with:
```bash
pip install -r requirements.txt
```

Required Python packages:
- rpyc (robot communication)
- opencv-python (computer vision)
- numpy, matplotlib (data processing and visualization)
- torch, torchvision, timm (neural networks and vision transformers)
- PIL (image processing)
- ipywidgets, IPython (notebook compatibility)
- tqdm (progress bars)
- gradio (web-based interface)
- nest_asyncio (async support for Gradio)

## File Structure

### Core World Model
- `autoencoder_concat_predictor_world_model.py`: Canvas-based world model using targeted masked autoencoder with frame concatenation
- `concat_world_model_explorer/`: Modular web-based Gradio interface for AutoencoderConcatPredictorWorldModel exploration
  - `app.py`: Main Gradio application and UI layout
  - `state.py`: Application state management
  - `session_manager.py`: Session loading and frame handling
  - `canvas_ops.py`: Canvas building and preprocessing
  - `inference.py`: Single-frame inference operations
  - `evaluation.py`: Full-session evaluation and statistics
  - `training.py`: Batch training with performance optimizations
  - `checkpoint_manager.py`: Model checkpoint save/load operations
  - `attention.py`: Decoder attention visualization
  - `visualization.py`: Plotting and display utilities
  - `utils.py`: Shared helper functions
  - `training_logger.py`: Multi-tier training diagnostics logger with automatic summarization
- `config.py`: Shared configuration, image transforms, robot-specific directories, and world model parameters
- `world_model_utils.py`: Utility functions for training and tensor operations
- `analyze_training_logs.py`: Standalone tool for analyzing ongoing or completed training runs with batch composition, loss trajectory, and gradient diagnostics

### Staged Training
- `staged_training.py`: Automated staged training pipeline with HTML report generation
  - Progressive training on increasing data sizes
  - Divergence-based early stopping
  - Best checkpoint selection based on original session loss
  - Baseline comparison training (fresh weights each stage) for comparing against staged (weight carryover)
  - Comprehensive HTML reports with training progress, staged vs baseline comparison, and inference visualizations
  - Integrated LR sweep before each stage with parallel execution support
  - Direct session mode: train on specific train/val sessions without staged splits
  - Single stage mode: run only a specific stage from staged splits
  - Starting checkpoint support for resuming training
- `staged_training_config.py`: Dataclass configuration for staged training runs
  - All parameters match Gradio app defaults
  - `PlateauSweepConfig`: Plateau-triggered LR sweep configuration (enabled by default)
  - `LRSweepConfig`: Legacy upfront LR sweep configuration
  - Baseline config: `enable_baseline`, `baseline_runs_per_stage`
  - YAML serialization support for reproducible experiments
- `lr_sweep.py`: Time-budgeted learning rate optimization module
  - Two-phase search: Phase A (broad exploration) and Phase B (deep validation)
  - Parallel trial execution with multiprocessing
  - `run_plateau_triggered_sweep()`: Mid-training sweep for plateau detection mode (returns lr, checkpoint, best_val)
  - Fallback logic when Phase B trials fail (falls back to Phase A survivor or config default)
  - GPU memory cleanup between sweep phases and worker processes
  - Worker GPU assignment by index (not PID) for reliable multi-GPU distribution
  - Data structures: `LRTrialResult`, `LRAggregatedResult`, `LRSweepPhaseResult`, `LRSweepStageResult`, `StageTiming`
  - Checkpoint path tracking for winning trials
  - Resume support for interrupted sweeps
- `create_staged_splits.py`: Utility to create progressive train/validation splits from a session
  - Doubling data size at each stage (10, 20, 40, 80, ...)
  - Configurable train/validation ratio (default 70/30)

### Robot Interfaces
- `robot_interface.py`: Abstract base class defining robot interaction contract
- `jetbot_interface.py`: JetBot implementation of RobotInterface with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client with live feed capability
- `toroidal_dot_env.py`: Simulated 224x224 toroidal environment with white dot
- `toroidal_dot_interface.py`: ToroidalDotRobot implementation of RobotInterface
- `lerobot_policy_simple_joint/`: LeRobot custom policy package for SO-101 single-joint control
  - `lerobot_policy_simple_joint/__init__.py`: Package exports and version
  - `lerobot_policy_simple_joint/configuration_simple_joint.py`: SimpleJointConfig class with configurable joint and movement parameters
  - `lerobot_policy_simple_joint/modeling_simple_joint.py`: SimpleJointPolicy class implementing 3 discrete actions with duration-based movement
  - `lerobot_policy_simple_joint/processor_simple_joint.py`: Identity pre/post processors for LeRobot plugin system
  - `pyproject.toml`: Package metadata and dependencies
  - `README.md`: Usage documentation for the policy
- `run_lerobot_record.py`: Wrapper script for lerobot-record with Windows camera patches, auto-calculated episode timing, and servo auto-calibration
  - **Windows compatibility**: DSHOW camera backend and synchronous read patches
  - **Auto episode timing**: Calculates episode_time_s from action_sequence length + 5s buffer
  - **Servo auto-calibration**: Measures actual servo settling time to set optimal `action_duration` before recording
  - **Visual verification**: Runs test sequence with canvas preview for manual timing verification
  - **Discrete action logging**: Injects log directory into policy config via CLI args
  - **Debug output**: Prints full lerobot-record command with all parameters before execution
  - **CLI flags**: `--skip-calibration`, `--skip-verification` to bypass calibration steps
- `convert_lerobot_to_explorer.py`: Converter script for LeRobot v3.0 datasets to concat_world_model_explorer format with dual-camera stacking

### Models
- `models/__init__.py`: Module exports for clean imports
- `models/base_autoencoder.py`: Base class for autoencoder implementations
- `models/vit_autoencoder.py`: MaskedAutoencoderViT with powerful transformer encoder and decoder
- `models/autoencoder_concat_predictor.py`: Canvas building utilities, TargetedMAEWrapper, and GPU-accelerated mask generation
- `models/canvas_dataset.py`: PyTorch Dataset and DataLoader utilities for high-performance batch training with pinned memory and CUDA streams

### Action Selection and Recording
- `toroidal_action_selectors.py`: Action selector factories for toroidal dot environment (constant, sequence, and random duration selectors)
- `recorded_policy.py`: Action selector factory for recorded action playback
- `record_toroidal_random_duration_sessions.py`: Script for recording training and validation sessions with random duration action selector
- `filter_validation_overlap.py`: Utility to filter validation sessions and remove (position, last_action) pairs seen during training
- `recording_writer.py`: Recording system with automatic disk space management
- `recording_reader.py`: Reads recorded sessions with smart observation/action sequencing
- `recording_robot.py`: Robot wrapper for recording sessions
- `replay_robot.py`: Robot interface replacement for replaying recorded sessions

### Visualization and Analysis
- `attention_viz.py`: Decoder attention visualization utilities with multiple visualization modes
  - Patch-to-patch connection lines showing attention FROM selected patches TO all patches
  - Attention matrix heatmaps for detailed numerical analysis
  - Heatmap overlay on canvas for spatial attention visualization
  - Automatic dot detection and manual patch selection
  - Quantile-based filtering for focusing on strongest connections
  - Layer and head selection for fine-grained analysis
- `session_explorer_lib.py`: Shared library of utilities for session management, frame processing, model operations, and canvas pre-building for batch training optimization

### Testing and Development
- `test_concat_world_model.py`: Test script for AutoencoderConcatPredictorWorldModel
- `test_jetbot_actions.ipynb`: Jupyter notebook for interactive JetBot action space testing
- `test_toroidal_dot_actions.ipynb`: Jupyter notebook for interactive toroidal dot environment testing

### Configuration
- `requirements.txt`: Python package dependencies
- `.gitignore`: Excludes `.claude/` directory, `CLAUDE.md`, wandb logs, checkpoints, and common Python artifacts

## Implementation Notes

### Adding New Robot Types
To add support for a new robot:

1. Create a new class inheriting from `RobotInterface`
2. Implement the required methods: `get_observation()`, `execute_action()`, `action_space`, `cleanup()`
3. Define your robot's action format (must include any parameters needed)
4. Use the recording system to capture sessions for exploration in the concat world model explorer

### Action Format Requirements
- Actions must be dictionaries
- Include any parameters your robot needs (motor speeds, duration, etc.)
- Actions are encoded as colored separators in the canvas-based world model

### Canvas-Based Architecture
- **Frame concatenation**: History frames are concatenated horizontally with thin colored separators encoding actions
- **Targeted masking**: Next-frame slot is fully masked (MASK_RATIO = 1.0) for inpainting-based prediction
- **MAE-native optimization**: Only masked patches are used in loss calculation for efficient training
- **Non-square support**: Architecture handles concatenated canvases with non-square dimensions (e.g., 224x688)
- **Single-step training**: One training step per world model iteration for real-time learning

### Decoder Attention Analysis
- **Attention visualization**: The concat world model explorer provides interactive decoder attention visualization
- **Patch selection modes**:
  - **Automatic dot detection**: Automatically identifies bright patches based on configurable brightness threshold
  - **Manual selection**: Specify patch indices manually (comma-separated, ranges, or mixed: "0,5,10-15")
- **Frame-based visualization**: Analyze attention from selected frame (builds canvas from current frame context)
- **Attention direction**: Shows attention FROM selected patches TO all other patches in canvas
- **Three visualization types**:
  - **Patch-to-patch lines**: Connection lines color-coded by layer with thickness indicating attention strength
  - **Matrix heatmap**: Numerical heatmap showing selected patches to all patches
  - **Overlay heatmap**: Spatial attention heatmap overlaid on canvas image
- **Quantile filtering**: Focus on strongest attention connections by filtering to top percentiles (for line visualization)
- **Layer/head selection**: Toggle individual decoder layers (0-4) and attention heads (0-3) for detailed analysis
- **Aggregation methods**:
  - Head aggregation: Choose mean, max, or sum across attention heads
  - Selected patch aggregation: Choose how to combine attention from multiple selected patches
- **Canvas-aware coordinates**: Automatically adjusts patch coordinates from frame space to canvas space
- **Real-time statistics**: View selected patches, connection counts, attention weights, and per-layer/head metrics
