# Developmental Robot Movement

Research code for developmental robot movement with canvas-based world model architecture using masked autoencoder inpainting for next-frame prediction.

## Project Overview

This repository contains research code for a canvas-based world model that learns to predict future observations by inpainting masked next-frame slots in horizontally concatenated frame sequences.

**Key Components:**
1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Physical robot interface
3. **Toroidal Dot Environment** (`toroidal_dot_env.py`, `toroidal_dot_interface.py`) - Simulated environment for testing
4. **SO-101 Robot Arm** (`lerobot_policy_simple_joint/`, `convert_lerobot_to_explorer.py`) - LeRobot integration for SO-101 follower arm
5. **Autoencoder Concat Predictor World Model** (`autoencoder_concat_predictor_world_model.py`) - Canvas-based world model
6. **Concat World Model Explorer** (`concat_world_model_explorer/`) - Interactive web-based interface for exploring sessions
7. **Staged Training** (`staged_training.py`, `staged_training_config.py`, `create_staged_splits.py`) - Automated progressive training pipeline with HTML reports
8. **Learning Rate Sweep** (`lr_sweep.py`) - Time-budgeted LR optimization with two-phase search (broad exploration + deep validation)

## Architecture

### Canvas-Based World Model

The concat world model uses a unique approach to visual prediction:

- **Frame concatenation**: History frames are concatenated horizontally with colored action separators
- **Targeted masking**: Next-frame slot is fully masked (MASK_RATIO = 1.0) for inpainting-based prediction
- **MAE-native training**: Optimizes only masked patches using the MaskedAutoencoderViT architecture
- **Hybrid loss**: Combines plain MSE and focal MSE for edge-aware training
- **VGG perceptual loss**: Optional VGG16 feature-space loss for sharper predictions (`PERCEPTUAL_LOSS_WEIGHT` in config, 0.0 = disabled)
- **Action encoding**: Actions encoded as thin colored separators between frames (e.g., red for stay, green for move)
- **Non-square canvases**: Handles non-square concatenated images (e.g., 224x688 for 3 frames + 2 separators)
- **Multi-resolution support**: Automatic frame size detection from loaded sessions (e.g., 448x224 for SO-101 dual-camera stacking)
- **Single-step training**: One training step per world model iteration for real-time learning

### RobotInterface Abstraction

- **Abstract base class**: Defines standard interface for any robot type
- **Key methods**: `get_observation()`, `execute_action()`, `action_space`, `cleanup()`
- **Extensible design**: Easy to implement new robot types by inheriting from RobotInterface

### JetBot Implementation

- **RemoteJetBot class**: Handles RPyC connections, camera capture, and motor control
- **JetBotInterface**: Wrapper implementing RobotInterface for JetBot robots
- **Live video feed**: Real-time display using OpenCV windows
- **Connection**: Connects to JetBot at configurable IP on port 18861
- **Duration-based actions**: Motor commands with automatic stopping after specified duration
- **Simplified action space**: motor_right values {0, 0.12} = 2 actions (motor_left always 0)
- **Forward-only movement**: Gentler gearbox operation with stop and forward-only commands

### Toroidal Dot Environment

- **Simulated environment**: 224x224 black image with white dot for fast testing
- **Toroidal wrapping**: Horizontal movement wraps around at edges (x-axis is toroidal)
- **Binary action space**: 0 (stay) and 1 (move right)
- **Random initialization**: Each session starts with dot at random (x, y) position
- **No hardware needed**: Perfect for debugging and development
- **Configurable**: Dot size, movement speed, image size via ToroidalDotConfig

### SO-101 Robot Arm

- **LeRobot integration**: Custom policy package for SO-101 follower arm control with two policy types:
  - **SimpleJoint** (`simple_joint`): Controls a single joint with 3 discrete actions (stay, move+, move-) per episode
  - **MultiSecondaryJoint** (`multi_secondary_joint`): Controls a primary joint within each episode while a secondary joint randomly changes position between episodes (for multi-height recordings)
- **Shared base class**: `BaseJointPolicy` provides common functionality (action timing, discrete action logging, sequence/random modes) for both policy types
- **Configurable joints**: Control any SO-101 joint (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **Servo auto-calibration**: Two-phase settle detection (departure + stabilization) with 4-movement test and motor calibration loading
  - Runs move+, return, move-, return sequence; uses worst-case settle time with 1.2x safety margin
  - Full pipeline verification (record → convert → build canvases) for visual timing approval
  - Always runs calibration regardless of explicit `--policy.action_duration`
  - Skip with `--skip-calibration` or `--skip-verification` flags
- **MultiSecondaryJoint reset phase**: `run_lerobot_record.py` patches LeRobot's `record_loop` to handle the reset phase between episodes — physically moves the secondary joint servo to the new target and waits `reset_time_s` for settling; auto-sets `reset_time_s = 3 × action_duration` for secondary joint policies
- **Dual-camera support**: Stacks base_0_rgb and left_wrist_0_rgb cameras vertically for 448x224 combined frames
- **Discrete action logging**: Automatic JSONL logs with frame index for exact frame-to-action correspondence; MultiSecondaryJoint logs also include `height_target` for the secondary joint position
- **Dataset converter**: Convert LeRobot v3.0 datasets to concat_world_model_explorer format
  - Frame-index-based mapping: uses logged frame indices for exact frame-to-action correspondence
  - `--combine-episodes`: Combines all episodes into a single session (designed for multi-height recordings where each episode is at a different secondary joint position)

### Action Selectors

- **Pluggable architecture**: Action selectors are functions that take observations and return actions
- **Toroidal action selectors** (`toroidal_action_selectors.py`):
  - `create_constant_action_selector(action)`: Always returns the same action
  - `create_sequence_action_selector(sequence)`: Cycles through a sequence of actions
  - `create_random_duration_action_selector(min_duration, max_duration, seed)`: Random actions with random durations
  - Pre-defined sequences: `SEQUENCE_ALWAYS_MOVE`, `SEQUENCE_ALWAYS_STAY`, `SEQUENCE_ALTERNATE`, `SEQUENCE_DOUBLE_MOVE`, `SEQUENCE_TRIPLE_MOVE`
- **Recorded action selector** (`recorded_policy.py`):
  - `create_recorded_action_selector(reader)`: Replays actions from recorded sessions
  - Optional action filtering for selective replay

## Running the Code

### Concat World Model Explorer (Main Tool)

```bash
python -m concat_world_model_explorer
```

**The primary way to interact with the concat world model.** This modular web-based interface provides:

- **Session selection**: Choose from recorded sessions in `saved/sessions/toroidal_dot/` or `saved/sessions/so101/`
- **Frame navigation**: Browse session frames with slider and text input
- **World model execution**: Run world model for specified number of iterations
- **Full masking approach**: Uses MASK_RATIO = 1.0 for complete next-frame inpainting
- **Live progress tracking**: Real-time prediction error and iteration timing
- **Comprehensive visualizations**:
  - Original canvas
  - Masked canvas overlay
  - Full inpainting output
  - Composite reconstruction
  - Current frame
  - Predicted next frame
  - Prediction error
- **Metric graphs**: Plots tracking prediction error and iteration time
- **Batch training with configurable modes**:
  - **Random sampling**: Sample batches with replacement
  - **Epoch-based sampling**: Shuffle and see each sample once per epoch
  - **Resume training**: Continue from checkpoint with adaptive learning rate warmup
  - **Train until divergence**: Run indefinitely until validation loss diverges from training loss
    - Requires a validation session
    - Uses ReduceLROnPlateau scheduler for unknown training length
    - Configurable divergence detection (gap and ratio thresholds)
    - Automatic chunked epoch regeneration for memory efficiency
- **Model checkpoint management**:
  - Save/load model weights, optimizer, and scheduler state
  - Metadata tracking (timestamp, config, training metrics)
  - Checkpoint browser for easy model comparison
- **Inference-only evaluation**:
  - Single canvas inference on selected frame (no training)
  - Full session evaluation with comprehensive statistics
  - Metrics: mean, median, std dev, percentiles, loss plots, distributions
- **Decoder attention visualization**: Interactive exploration of attention patterns
  - **Patch selection**: Automatic dot detection (brightness-based) or manual selection (indices/ranges)
  - **Frame-based analysis**: Visualize attention from selected frame (not just training canvas)
  - **Attention direction**: Shows attention FROM selected patches TO all patches in canvas
  - **Three visualization types**:
    - Patch-to-patch connection lines (color-coded by layer, thickness by attention strength)
    - Attention matrix heatmap (numerical view of selected → all patches)
    - Heatmap overlay on canvas (spatial attention visualization)
  - **Quantile filtering**: Show top N% strongest connections (for line visualization)
  - **Layer selection**: Toggle decoder layers 0-4 on/off
  - **Head selection**: Toggle attention heads 0-3 for fine-grained analysis
  - **Aggregation methods**:
    - Head aggregation: mean, max, or sum across selected heads
    - Selected patch aggregation: mean, max, or sum across multiple selected patches
  - **Real-time statistics**: Selected patches, connection counts, attention weights, per-layer/head metrics

Access the interface at http://localhost:7860 after starting the server.

### JetBot Live Feed

```bash
python jetbot_remote_client.py
```
- Displays live camera feed from JetBot
- Requires JetBot running RPyC server
- Press 'q' or Ctrl+C to stop

### Recording Sessions

To create new sessions for exploration, you need to implement a script that:
1. Sets `RECORDING_MODE = True` in `config.py`
2. Instantiates a robot interface (JetBot or ToroidalDotRobot)
3. Wraps it with `RecordingRobot` from `recording_robot.py`
4. Runs actions using action selectors or custom logic

Sessions are automatically saved to robot-specific directories:
- JetBot: `saved/sessions/jetbot/`
- Toroidal dot: `saved/sessions/toroidal_dot/`
- SO-101: `saved/sessions/so101/`

**Disk space management**: Automatic cleanup of oldest sessions when total recordings exceed configurable disk limit (default 10 GB per robot type).

### SO-101 Robot Arm with LeRobot

#### Install the Policy Package
```bash
cd lerobot_policy_simple_joint
pip install -e .
```

#### Record with lerobot-record

Use `run_lerobot_record.py` wrapper for action sequences (auto-calibrates and calculates episode time):
```bash
python run_lerobot_record.py \
    --robot.type=so101_follower \
    --robot.port=COM8 \
    --robot.id=my_so101_follower \
    --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --policy.type=simple_joint \
    --policy.joint_name=wrist_roll.pos \
    --policy.position_delta=10 \
    --policy.action_sequence="[1, 0, 2, 0, 1]" \
    --dataset.repo_id=${HF_USER}/so101-test \
    --dataset.num_episodes=1 \
    --dataset.single_task="Single joint movement"
```

The wrapper always auto-calibrates by measuring servo settling time (overrides any explicit `--policy.action_duration`).
Use `--skip-calibration` to skip calibration entirely, or `--skip-verification` to calibrate without the visual preview.

#### Convert LeRobot Dataset to Explorer Format
```bash
# Convert each episode as a separate session (default)
python convert_lerobot_to_explorer.py \
    --lerobot-path ${HF_USER}/so101-single-joint \
    --output-dir saved/sessions/so101 \
    --cameras base_0_rgb left_wrist_0_rgb \
    --stack-cameras vertical

# Combine all episodes into one session (for multi-height recordings)
python convert_lerobot_to_explorer.py \
    --lerobot-path ${HF_USER}/so101-multi-height \
    --output-dir saved/sessions/so101 \
    --cameras base_0_rgb left_wrist_0_rgb \
    --stack-cameras vertical \
    --combine-episodes
```

The converter downloads from HuggingFace Hub and reads action parameters from discrete action logs automatically. Use `--combine-episodes` to concatenate all episodes into a single session with stay-action transitions between them.

### Interactive Testing

```bash
# JetBot action space testing
jupyter notebook test_jetbot_actions.ipynb

# Toroidal dot environment testing
jupyter notebook test_toroidal_dot_actions.ipynb
```

These notebooks provide interactive environments for testing robot interfaces and action spaces.

### Staged Training

Automated training pipeline that progressively trains on increasing data sizes with comprehensive HTML reports.

**1. Create Staged Splits:**
```bash
# Create progressive train/validation splits from a session
python create_staged_splits.py --session-path saved/sessions/so101/my_session

# Customize initial size and train ratio
python create_staged_splits.py --session-path saved/sessions/so101/my_session --initial-size 20 --train-ratio 0.8
```
- Creates progressively larger training/validation splits (10, 20, 40, 80, ... observations)
- Default 70/30 train/validate split at each stage
- Output: `{session}_stage{N}_train_{size}` and `{session}_stage{N}_validate_{size}`

**2. Run Staged Training:**
```bash
# Basic usage (plateau-triggered sweeps enabled by default)
python staged_training.py --root-session saved/sessions/so101/my_session

# Multiple runs per stage for robustness
python staged_training.py --root-session saved/sessions/so101/my_session --runs-per-stage 3

# With time budget per stage (main training only; sweeps use lr_sweep phase budgets)
python staged_training.py --root-session saved/sessions/so101/my_session --stage-time-budget-min 10

# Custom LR sweep configuration (applies to both plateau and upfront sweeps)
python staged_training.py --root-session saved/sessions/so101/my_session \
    --lr-sweep-phase-a-candidates 5 \
    --lr-sweep-phase-a-budget-min 2.0 \
    --lr-sweep-phase-b-seeds 3 \
    --lr-sweep-phase-b-budget-min 5.0

# Use upfront sweeps instead of plateau-triggered sweeps (legacy mode)
python staged_training.py --root-session saved/sessions/so101/my_session --disable-plateau-sweep

# Disable the initial LR sweep that runs before each stage
python staged_training.py --root-session saved/sessions/so101/my_session --disable-initial-sweep

# Disable baseline comparison
python staged_training.py --root-session saved/sessions/so101/my_session --disable-baseline

# Custom configuration via YAML
python staged_training.py --root-session saved/sessions/so101/my_session --config my_config.yaml
```

**Features:**
- **Progressive training**: Trains on each stage's data until divergence, then moves to next stage
- **Plateau-triggered LR sweeps** (default mode): LR optimization triggered when validation loss plateaus
  - Uses current weights for sweep, continues with winning LR and weights
  - Maximum sweeps per stage (default 3) prevents infinite optimization loops
- **Upfront LR sweeps** (legacy mode): Automatic LR optimization before each stage (use `--disable-plateau-sweep`)
- **Two-phase sweep structure** (both modes):
  - **Phase A**: Broad exploration with many LR candidates, short time budgets
  - **Phase B**: Deep validation with top survivors, multiple seeds for robust selection
  - Ranking by median/mean/min best validation loss across seeds
- **Divergence-based early stopping**: Automatically stops when validation loss diverges from training
- **EMA-smoothed divergence detection**: Uses exponential moving average of training loss for robust detection
- **Loss-weighted sampling**: Focuses on high-loss samples for efficient learning
- **Serial runs** (`--serial-runs`, default): Runs `runs_per_stage` sequentially to reduce peak GPU memory; parallel mode still available
- **Initial LR sweep** (`initial_sweep_enabled=True`): Runs an upfront LR sweep before each stage regardless of whether plateau sweeps are enabled; disable with `--disable-initial-sweep`
- **Time budget control**: Optional per-stage time budget for main training
- **Interrupt/crash recovery**: Catches `KeyboardInterrupt` and exceptions, recovers the interrupted stage from auto-saved checkpoints, and generates a partial report with all completed stages
- **Baseline comparison**: Optionally run parallel baseline training (fresh weights each stage) to compare against staged training (weight carryover)
- **Progressive reporting**: Final report updated after each stage for real-time progress visibility
- **HTML reports**: Comprehensive reports with training progress, hybrid loss graphs, config diff vs last commit, full training loss timeline, multi-run statistics, LR sweep results (including plateau sweep history), staged vs baseline comparison, and inference visualizations
- **Best checkpoint selection**: Selects best checkpoint based on hybrid loss over original (full) session
- **W&B integration**: Optional Weights & Biases logging with run_id in run names and baseline config tracking

**Configuration** (`staged_training_config.py`):
- All parameters match Gradio app defaults
- **Sweep mode**: `plateau_sweep.enabled` (default True = plateau-triggered sweeps, False = upfront sweeps)
- Key parameters: `batch_size`, `divergence_patience`, `loss_weight_temperature`
- Plateau Sweep config: `plateau_sweep.plateau_patience` (25 updates), `plateau_sweep.plateau_improvement_threshold`, `plateau_sweep.cooldown_updates`, `plateau_sweep.max_sweeps_per_stage` (2)
- LR Sweep config (shared by both modes): `lr_sweep.lr_min`, `lr_sweep.lr_max`, `lr_sweep.phase_a_num_candidates`, `lr_sweep.phase_a_time_budget_min`, `lr_sweep.phase_b_seeds`, `lr_sweep.phase_b_time_budget_min`
- Baseline config: `enable_baseline` (default False), `baseline_runs_per_stage` (default 1)
- `serial_runs` (default True): run multiple runs per stage serially instead of in parallel
- `initial_sweep_enabled` (default True): upfront LR sweep before each stage (orthogonal to `plateau_sweep.enabled`)
- Stage time budget: `stage_time_budget_min` (0 = unlimited)
- Supports YAML config files for reproducible experiments

**Reports:**
- Per-stage reports: `saved/staged_training_reports/{session}/stage{N}_run{M}/report.html`
- Baseline reports: `saved/staged_training_reports/{session}/stage{N}_baseline_run{M}/report.html`
- Final summary: `saved/staged_training_reports/{session}/final_report_{run_id}_{date}.html` (e.g., `final_report_shoulder_session_multiheight_2026_feb_19.html`)
- Also copied to: `docs/final_report_{run_id}_{date}.html` for easy access
- Includes: training progress graphs, hybrid loss over session graphs, full training loss timeline across all stages, config diff vs last commit, multi-run statistics (when `runs_per_stage > 1`), staged vs baseline comparison (winner, per-stage metrics), inference visualizations, evaluation statistics

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
- `autoencoder_concat_predictor_world_model.py`: Canvas-based world model implementation
- `concat_world_model_explorer/`: Modular web-based Gradio interface for exploration
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
- `config.py`: Configuration for world model, robots, and recording
- `world_model_utils.py`: Utility functions for training and tensor operations

### Staged Training
- `staged_training.py`: Automated staged training pipeline with HTML report generation
  - Progressive training on increasing data sizes
  - Divergence-based early stopping with EMA-smoothed training loss
  - Best checkpoint selection based on original session loss
  - Baseline comparison training (fresh weights each stage) for comparing against staged (weight carryover)
  - Serial and parallel run modes for `runs_per_stage > 1` (serial is default)
  - Initial LR sweep before each stage (orthogonal to plateau-triggered sweeps)
  - Interrupt/crash recovery with partial report generation
  - Comprehensive HTML reports with config diff, full loss timeline, multi-run stats, staged vs baseline comparison, and inference visualizations
- `staged_training_config.py`: Dataclass configuration for staged training runs
  - All parameters match Gradio app defaults
  - Baseline config: `enable_baseline`, `baseline_runs_per_stage`
  - LRSweepConfig nested config for LR sweep parameters
  - YAML serialization support for reproducible experiments
- `lr_sweep.py`: Time-budgeted learning rate optimization module
  - Two-phase search: Phase A (broad exploration) and Phase B (deep validation)
  - Parallel trial execution with multiprocessing
  - Data structures: `LRTrialResult`, `LRAggregatedResult`, `LRSweepPhaseResult`, `LRSweepStageResult`, `StageTiming`
  - Resume support for interrupted sweeps
- `create_staged_splits.py`: Utility to create progressive train/validation splits from a session
  - Doubling data size at each stage (10, 20, 40, 80, ...)
  - Configurable train/validation ratio (default 70/30)

### Robot Interfaces
- `robot_interface.py`: Abstract base class for robot interaction
- `jetbot_interface.py`: JetBot implementation with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client
- `toroidal_dot_env.py`: Simulated toroidal environment
- `toroidal_dot_interface.py`: ToroidalDotRobot implementation
- `lerobot_policy_simple_joint/`: LeRobot custom policy package for SO-101 joint control
  - `base_joint_policy.py`: Shared base class (`BaseJointPolicy`) for discrete joint control policies
  - `configuration_simple_joint.py` / `modeling_simple_joint.py`: `SimpleJointPolicy` — single joint, 3 discrete actions
  - `configuration_multi_secondary_joint.py` / `modeling_multi_secondary_joint.py`: `MultiSecondaryJointPolicy` — primary joint within episodes, secondary joint changes between episodes
  - `processor_simple_joint.py` / `processor_multi_secondary_joint.py`: Identity pre/post processors
- `run_lerobot_record.py`: Wrapper for lerobot-record with servo auto-calibration, visual verification, auto-calculated episode timing, and `record_loop` patch for MultiSecondaryJoint reset phase handling
- `convert_lerobot_to_explorer.py`: Dataset converter for LeRobot v3.0 to explorer format with `--combine-episodes` support

### Models
- `models/__init__.py`: Module exports
- `models/base_autoencoder.py`: Base class for autoencoders
- `models/vit_autoencoder.py`: MaskedAutoencoderViT with powerful transformer encoder/decoder
- `models/autoencoder_concat_predictor.py`: Canvas building and TargetedMAEWrapper for masked inpainting
- `models/canvas_dataset.py`: PyTorch Dataset and DataLoader for high-performance batch training
- `models/perceptual_loss.py`: VGG16 perceptual loss module for sharper predictions (optional, controlled by `PERCEPTUAL_LOSS_WEIGHT`)

### Action Selection and Recording
- `toroidal_action_selectors.py`: Action selector factories (constant and sequence selectors)
- `recorded_policy.py`: Recorded action replay selector
- `recording_writer.py`: Recording system with disk space management
- `recording_reader.py`: Reads recorded sessions
- `recording_robot.py`: Robot wrapper for recording
- `replay_robot.py`: Robot interface for replaying sessions

### Visualization and Analysis
- `attention_viz.py`: Decoder attention visualization with multiple visualization modes
  - Patch-to-patch connection lines (FROM selected patches TO all patches)
  - Attention matrix heatmaps for numerical analysis
  - Heatmap overlay on canvas for spatial visualization
  - Automatic dot detection and manual patch selection
  - Quantile-based filtering and layer/head selection
- `session_explorer_lib.py`: Session management, frame processing, and model operations

### Testing and Development
- `test_concat_world_model.py`: Test script for concat world model
- `test_jetbot_actions.ipynb`: Interactive JetBot testing notebook
- `test_toroidal_dot_actions.ipynb`: Interactive toroidal dot testing notebook

### Configuration
- `requirements.txt`: Python package dependencies
- `.gitignore`: Excludes logs, checkpoints, and artifacts

## Implementation Notes

### Adding New Robot Types

To add support for a new robot:

1. Create a new class inheriting from `RobotInterface`
2. Implement the required methods: `get_observation()`, `execute_action()`, `action_space`, `cleanup()`
3. Define your robot's action format (dictionary with parameters)
4. Use the recording system to capture sessions for exploration

### Action Format Requirements

- Actions must be dictionaries
- Include any parameters your robot needs (motor speeds, duration, etc.)
- Actions are encoded as colored separators in the canvas-based world model

### Canvas-Based Architecture Details

**Frame Concatenation:**
- History frames concatenated horizontally
- Thin colored separators encode actions between frames
- Example: 3 frames (224x224 each) + 2 separators (224x8 each) = 224x688 canvas

**Targeted Masking:**
- Next-frame slot is fully masked (MASK_RATIO = 1.0)
- Autoencoder learns to inpaint the missing frame
- Only masked patches contribute to training loss (MAE-native optimization)

**Training Strategy:**
- Single training step per world model iteration
- Inpainting threshold of 0.0001 for quality gating
- AdamW optimizer with cosine decay learning rate schedule

**Batch Training Performance (4-phase optimization):**
- **Phase 1**: Canvas pre-building at session load (~6400 canvases cached in memory)
- **Phase 2**: PyTorch DataLoader with pinned memory for async GPU transfers
- **Phase 3**: GPU-accelerated mask generation using vectorized torch operations
- **Phase 4**: CUDA stream pipelining for overlapping GPU training with CPU→GPU transfers
- **Results**: 6.92x speedup (129.5s → 18.7s for 6400 samples at batch size 64)

### Decoder Attention Analysis

The concat world model explorer provides powerful attention visualization:

- **Patch selection modes**:
  - **Automatic dot detection**: Identifies bright patches using configurable brightness threshold (0-1)
  - **Manual selection**: Specify indices manually (supports formats: "0,5,10" or "0-10" or "0,5,10-15")
- **Frame-based analysis**: Visualize attention from selected frame (builds canvas from current frame context)
- **Attention direction**: Shows attention FROM selected patches (e.g., dot patches) TO all other patches in canvas
- **Three visualization types**:
  - **Patch-to-patch lines**: Connection lines color-coded by layer, thickness by attention strength
  - **Matrix heatmap**: Numerical heatmap showing selected patches to all patches
  - **Overlay heatmap**: Spatial attention heatmap overlaid on full canvas
- **Quantile filtering**: Focus on strongest connections by filtering to top percentiles (for line visualization)
- **Layer/head selection**: Toggle individual decoder layers (0-4) and attention heads (0-3)
- **Aggregation methods**:
  - Head aggregation: Mean, max, or sum across selected heads
  - Selected patch aggregation: Mean, max, or sum across multiple selected patches
- **Canvas-aware coordinates**: Automatically adjusts patch coordinates from frame space to canvas space
- **Real-time statistics**: Selected patches, connection counts, attention weights, per-layer/head metrics

Use this to understand how specific patches (like those containing the dot) attend to different parts of the canvas when the decoder is reconstructing the next frame.

### Configuration

The `config.py` file contains:

- **AutoencoderConcatPredictorWorldModelConfig**: Canvas size, separator width, history size, training thresholds
- **ToroidalDotConfig**: Simulated environment parameters (dot size, movement speed, image dimensions)
- **SO101Config**: Configuration for SO-101 follower arm (joint names, action parameters, dual-camera frame size)
- **Robot-specific directories**: Separate checkpoint and recording directories for JetBot, toroidal dot, and SO-101
- **Recording configuration**: `RECORDING_MODE` boolean and `RECORDING_MAX_DISK_GB` for disk management

## Next Steps

1. **Generate sessions**: Use recording system to capture robot observations and actions
2. **Explore sessions**: Launch concat world model explorer to visualize and analyze sessions
3. **Analyze attention**: Use decoder attention visualization to understand prediction mechanism
4. **Iterate**: Adjust config parameters, generate new sessions, and explore results
