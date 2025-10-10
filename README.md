# Developmental Robot Movement

Research code for developmental robot movement with hierarchical world models and uncertainty-based exploration.

Currently a work in progress with [notes here](https://docs.google.com/document/d/e/2PACX-1vTdVqbwuou38bDDVrR4LonrjLcE2SXu6SYXUpeU9nmfKAc9raojYJW40eqHxlj8fqNR1FU9o24JCzPX/pub):

## Project Overview

This repository contains research code for developmental robot movement with a modular architecture:

1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Concrete JetBot robot interface
3. **Toroidal Dot Environment** (`toroidal_dot_env.py`, `toroidal_dot_interface.py`) - Simulated environment for testing
4. **Adaptive World Model** (`adaptive_world_model.py`) - Hierarchical world model with uncertainty-based action selection
5. **Robot Runner** (`robot_runner.py`) - Lightweight action execution without learning components
6. **Action Selectors** (`toroidal_action_selectors.py`, `recorded_policy.py`) - Pluggable action selection strategies
7. **Integration Examples** (`jetbot_world_model_example.py`, `toroidal_dot_world_model_example.py`, `toroidal_dot_runner_example.py`) - Integration with different robots/environments

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
- **Simulated environment**: 224x224 black image with white dot for fast testing
- **Toroidal wrapping**: Horizontal movement wraps around at edges
- **Binary actions**: 0 (stay) and 1 (move right)
- **No hardware needed**: Perfect for debugging and development
- **Configurable**: Dot size, movement speed via ToroidalDotConfig
- **Isolated checkpoints**: Uses separate directory from JetBot due to different action space

### Adaptive World Model
- **Dependency injection**: Takes RobotInterface in constructor for modularity
- **Main loop architecture**: Continuous cycle of perception → prediction → action selection → execution
- **Neural vision system**: MaskedAutoencoderViT for visual encoding/decoding with joint training
- **Quality gating**: Robot only acts when visual reconstruction quality meets threshold
- **Hierarchical predictors**: Multiple levels of action abstraction with automatic level creation
- **Uncertainty-based exploration**: Selects actions with highest prediction uncertainty (entropy)
- **Pluggable action selectors**: Custom action selection via `action_selector` parameter (defaults to uncertainty-based)
- **Real-time training visualization**: Shows current vs reconstructed frames during autoencoder training
- **Interactive visualization**: Real-time display of current frame, decoded frame, and predictions for all motor combinations

### Robot Runner
- **RobotRunner class**: Lightweight runner for executing actions without learning or world model components
- **No neural networks**: Pure action execution without training, checkpoints, or predictions
- **Action selector support**: Takes any action selector function for deterministic or custom policies
- **Visual display**: Optional matplotlib-based observation display with action information
- **Statistics tracking**: Automatic tracking of action distribution, timing, and execution counts
- **Interactive mode**: Optional user confirmation/override before each action
- **Recording compatible**: Works with RecordingRobot wrapper for data collection

### Action Selectors
- **Pluggable architecture**: Action selectors are functions that take observations and return (action, metadata) tuples
- **Toroidal action selectors** (`toroidal_action_selectors.py`):
  - `create_constant_action_selector(action)`: Always returns the same action
  - `create_sequence_action_selector(sequence)`: Cycles through a sequence of actions
  - Pre-defined sequences: `SEQUENCE_ALWAYS_MOVE`, `SEQUENCE_ALWAYS_STAY`, `SEQUENCE_ALTERNATE`, `SEQUENCE_DOUBLE_MOVE`, `SEQUENCE_TRIPLE_MOVE`
- **Recorded action selector** (`recorded_policy.py`):
  - `create_recorded_action_selector(reader)`: Replays actions from recorded sessions
  - Optional action filtering for selective replay
- **Usage**: Pass to `AdaptiveWorldModel` or `RobotRunner` via `action_selector` parameter

## Key Components

### Session Explorer Notebook
- **session_explorer.ipynb**: Interactive Jupyter notebook for exploring recorded sessions and training models. Provides frame playback with action callouts, lets you run autoencoder/predictor checkpoints to compare predictions against ground truth, and includes training sections to improve models on specific frames or sequences.
- **Multi-robot support**: Automatically detects robot type from session metadata and loads appropriate checkpoints
- **Robot-agnostic**: Works seamlessly with JetBot or toroidal dot sessions
- **Attention introspection**: Visualize transformer attention patterns with heatmaps, breakdown charts, and quantitative metrics (APA, ALF, TTAR, RI@16, entropy)
- **Action space sweep**: Predictions across full robot action space instead of +/-10% variants for comprehensive action effect analysis

### Neural Vision System
- **MaskedAutoencoderViT**: Vision Transformer-based autoencoder with powerful encoder and lightweight MLP decoder
- **Dynamic masking**: Randomized mask ratios (30%-85%) during autoencoder training for improved generalization
- **TransformerActionConditionedPredictor**: Causal transformer that interleaves visual features with action tokens
- **Fresh prediction training**: Predictors trained using fresh predictions with consistent loss calculation
- **Dual loss training**: Combined patch-space reconstruction + latent-space prediction losses for encoder optimization (current weights: 1.0 patch, 0.2 latent)
- **Action reconstruction loss**: Optional action classification from predicted features (currently disabled with weight 0)
- **Joint training architecture**: Autoencoder reconstruction loss + action predictor gradient flow with shared encoder
- **Sequence length management**: Handles long histories (4096 token capacity with clipping for GPU safety)
- **Quality gating**: Robot stops acting when reconstruction loss > threshold, focuses on vision training
- **Real-time visualization**: Training progress display showing current vs reconstructed frames
- **Comprehensive experiment tracking**: Weights & Biases integration for reconstruction, predictor training, and timing metrics
- **Learning progress persistence**: Automatic save/load of model weights, training progress, and history buffers
- **Attention introspection**: TransformerActionConditionedPredictor returns per-layer attention maps when `return_attn=True` for analysis

### Attention Analysis Infrastructure
- **EncoderLayerWithAttn**: Custom transformer encoder layer that optionally captures and returns per-head attention weights
- **Attention metrics**: APA (Attention to Previous Action), ALF (Attention to Last Frame), TTAR (Token-Type Attention Ratio), RI@16 (Recency Index), entropy
- **Action sensitivity testing**: Build action variants across full action space to measure prediction diversity
- **Counterfactual analysis**: Action shuffle/zero tests to validate that predictions meaningfully depend on action inputs
- **Gradient flow tracking**: Monitor gradient magnitudes flowing through action-related parameters vs total network
- **Token indexing**: Automatic derivation of frame/action/future token positions from sequence for robust metric calculation

### Action Space
- **Duration-based actions**: Motor commands with automatic stopping after specified duration
- **Simplified action space**: Single motor control with motor_right values {0, 0.12} = 2 total actions (motor_left always 0)
- **Forward-only movement**: Gentler gearbox operation with stop and forward-only commands
- **Format**: `{'motor_left': 0, 'motor_right': value, 'duration': 0.2}`
- **Smooth ramping**: Gradual acceleration/deceleration for gearbox protection
- **Automatic stopping**: Motors automatically ramp down to 0 after duration expires

### Configuration
- **config.py**: Contains image transforms, constants, and adaptive world model parameters
- **AdaptiveWorldModelConfig class**: Centralized configuration for all model parameters
- **Robot-specific directories**:
  - `JETBOT_CHECKPOINT_DIR = saved/checkpoints/jetbot/`
  - `TOROIDAL_DOT_CHECKPOINT_DIR = saved/checkpoints/toroidal_dot/`
  - `JETBOT_RECORDING_DIR = saved/sessions/jetbot/`
  - `TOROIDAL_DOT_RECORDING_DIR = saved/sessions/toroidal_dot/`
- **ToroidalDotConfig class**: Settings for simulated environment (dot size, movement speed, action space)
- **Recording configuration**: `RECORDING_MODE` boolean controls recording vs online mode, `RECORDING_MAX_DISK_GB` limits total disk usage per robot type
- **Configurable intervals**: Logging frequency, visualization uploads, checkpoint saving, and display intervals
- **Action timing**: Configurable delay between actions via `ACTION_DELAY` parameter (default 0 seconds)
- **IP addresses**: JetBot connection IPs specified in integration example (modify as needed)

## Running the Code

### JetBot Live Feed Only
```bash
python jetbot_remote_client.py
```
- Displays live camera feed from JetBot
- Requires JetBot running RPyC server
- Press 'q' or Ctrl+C to stop

### World Model with Real JetBot
```bash
python jetbot_world_model_example.py
```
- Integrates AdaptiveWorldModel with actual JetBot hardware
- Requires JetBot running RPyC server
- Interactive mode with real robot control
- Press Enter to continue with proposed action, type custom action dict, or 'stop' to exit
- **Automatic checkpoint saving**: Learning progress saved every 10 predictor training steps
- **Resume learning**: Automatically loads previous training progress on restart
- **Connection error handling**: Automatically terminates when JetBot becomes unreachable (e.g., battery runs out, network disconnection)

### World Model Testing (No Robot)
```bash
python adaptive_world_model.py
```
- Runs world model with stub robot for testing
- Shows visualization of predictions for all actions
- All ML components are stubs - only tests main loop logic
- No physical robot required

### Toroidal Dot Environment with World Model
```bash
python toroidal_dot_world_model_example.py
```
- Simulated environment with white dot on black background
- Fast iteration without hardware requirements
- Separate checkpoints: `saved/checkpoints/toroidal_dot/`
- Separate recordings: `saved/sessions/toroidal_dot/`
- Uses `SEQUENCE_ALWAYS_MOVE` action selector by default
- Interactive testing notebook: `test_toroidal_dot_actions.ipynb`

### Toroidal Dot Environment without Learning
```bash
python toroidal_dot_runner_example.py
```
- Runs ToroidalDotRobot with RobotRunner (no learning or neural networks)
- Executes deterministic action sequences using action selectors
- Useful for testing action selectors and collecting data with known policies
- Supports recording mode for data collection
- Uses `SEQUENCE_ALWAYS_MOVE` action selector by default
- Lightweight alternative to world model for pure action execution

### Recording and Replay System
```bash
# Record a JetBot session (set RECORDING_MODE = True in config.py)
python jetbot_world_model_example.py

# Record a toroidal dot session (set RECORDING_MODE = True in config.py)
python toroidal_dot_world_model_example.py

# Replay all recorded sessions
python replay_session_example.py

# Replay only specific actions (e.g., only forward movement)
python replay_session_example.py --filter-action motor_right=0.12

# Continuous replay with plateau detection
python continuous_replay.py
```
- **Recording mode**: Captures robot observations and actions with automatic disk space management
- **Robot-specific directories**: JetBot (`saved/sessions/jetbot/`) and toroidal dot (`saved/sessions/toroidal_dot/`) stored separately
- **Replay mode**: Replays all recorded sessions using the exact same main loop, enabling GPU utilization for predictor training
- **Action filtering**: Optional command-line filtering to replay only specific actions
- **Robot-agnostic replay**: Can replay any robot's recorded sessions regardless of robot type
- **Isolated checkpoints**: Each robot type maintains separate model checkpoints due to different action spaces
- **Disk space management**: Automatic cleanup of oldest sessions per robot type (default 10 GB each)

### Continuous Replay Training
```bash
# Run until plateau with default settings
python continuous_replay.py

# Run with custom plateau detection parameters
python continuous_replay.py --patience 10 --min-delta 0.0001

# Run with action filtering
python continuous_replay.py --filter-action action=1

# Run with max epochs limit
python continuous_replay.py --max-epochs 50
```
- **Automatic training**: Runs replay sessions repeatedly until predictor loss plateaus
- **Plateau detection**: Configurable patience, minimum delta, and minimum epochs before stopping
- **Loss tracking**: Extracts predictor loss from wandb or checkpoint state after each epoch
- **Action filtering**: Optional filtering to train on specific actions only
- **Max epochs**: Configurable maximum number of training epochs (default: 100)
- **Statistics**: Detailed loss history and training summary on completion

### Session Explorer and Training
```bash
jupyter notebook session_explorer.ipynb
```
- **Interactive session exploration**: Load and browse recorded robot sessions with frame-by-frame playback
- **Model inference**: Run autoencoder and predictor checkpoints on selected frames to compare predictions with ground truth
- **Adaptive training**: Train models on specific frames or sequences using the exact same algorithms as the live system
- **Threshold-based training**: Train until reconstruction/prediction loss drops below specified thresholds
- **Step-based training**: Train for a specific number of iterations with real-time progress monitoring
- **AdaptiveWorldModel integration**: Uses actual `AdaptiveWorldModel.train_autoencoder()` and `train_predictor()` methods for authentic training experience
- **Model saving capabilities**: Save trained autoencoder and predictor models with configurable paths, compatible with AdaptiveWorldModel checkpoint format
- **Interactive training controls**: Pause and resume buttons for both autoencoder and predictor training with responsive UI updates and proper state management

### Interactive JetBot Testing
```bash
jupyter notebook test_jetbot_actions.ipynb
```
- Interactive testing of JetBot action space in Jupyter notebook
- Test individual actions with visual feedback
- Capture before/after frames and analyze differences
- Useful for validating robot behavior and camera setup

## Dependencies

Install the required dependencies:
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
- wandb (experiment tracking and logging)

## File Structure

- `robot_interface.py`: Abstract base class defining robot interaction contract
- `jetbot_interface.py`: JetBot implementation of RobotInterface with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client with live feed capability
- `toroidal_dot_env.py`: Simulated 224x224 toroidal environment with white dot
- `toroidal_dot_interface.py`: ToroidalDotRobot implementation of RobotInterface
- `models/`: Neural network architectures directory
  - `models/__init__.py`: Module exports for clean imports
  - `models/base_autoencoder.py`: Base class for autoencoder implementations
  - `models/base_predictor.py`: Base class for predictor implementations
  - `models/vit_autoencoder.py`: MaskedAutoencoderViT implementation with fixed positional embeddings
  - `models/cnn_autoencoder.py`: CNN-based autoencoder implementation
  - `models/transformer_predictor.py`: TransformerActionConditionedPredictor with sequence length management
  - `models/lstm_predictor.py`: LSTM-based predictor implementation
  - `models/action_classifier.py`: Action classification module for action reconstruction loss
  - `models/encoder_layer_with_attn.py`: Custom transformer encoder layer with attention capture
- `adaptive_world_model.py`: Main world model implementation with comprehensive training and logging
- `robot_runner.py`: Lightweight runner for executing actions without learning components
- `jetbot_world_model_example.py`: Integration example connecting JetBot with world model
- `toroidal_dot_world_model_example.py`: Integration example connecting toroidal dot environment with world model
- `toroidal_dot_runner_example.py`: Integration example using RobotRunner with toroidal dot (no learning)
- `toroidal_action_selectors.py`: Action selector factories for toroidal dot environment (constant and sequence selectors)
- `recording_writer.py`: Recording system with automatic disk space management
- `recording_reader.py`: Reads recorded sessions with smart observation/action sequencing
- `replay_robot.py`: Robot interface replacement for replaying recorded sessions
- `recorded_policy.py`: Action selector factory for recorded action playback with optional filtering
- `replay_session_example.py`: Robot-agnostic replay script with command-line action filtering
- `continuous_replay.py`: Continuous replay training with plateau detection for automatic model improvement
- `session_explorer.ipynb`: Multi-robot session exploration and training notebook with automatic robot type detection
- `session_explorer.py`: Python script version of session explorer notebook
- `test_jetbot_actions.ipynb`: Interactive Jupyter notebook for JetBot action space testing
- `test_toroidal_dot_actions.ipynb`: Interactive Jupyter notebook for toroidal dot environment testing
- `config.py`: Shared configuration, image transforms, robot-specific directories, and adaptive world model parameters
- `requirements.txt`: Python package dependencies
- `.gitignore`: Excludes `.claude/` directory, `CLAUDE.md`, wandb logs, checkpoints, and common Python artifacts

## Implementation Notes

### Adding New Robot Types
To add support for a new robot:

1. Create a new class inheriting from `RobotInterface`
2. Implement the required methods: `get_observation()`, `execute_action()`, `action_space`, `cleanup()`
3. Define your robot's action format (must include any parameters needed)
4. Create an integration example similar to `jetbot_world_model_example.py`

### Action Format Requirements
- Actions must be dictionaries
- Include any parameters your robot needs (motor speeds, duration, etc.)
- The world model will learn about all parameters in the action space

## Experiment Tracking with Weights & Biases

The system includes optional integration with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization.

### Enabling wandb Logging

To enable wandb logging, provide a project name when creating the AdaptiveWorldModel:

```python
# With wandb logging and checkpoints enabled
model = AdaptiveWorldModel(robot_interface, wandb_project="my-robot-experiment", checkpoint_dir="my_checkpoints")

# Without wandb logging but with checkpoints (default)
model = AdaptiveWorldModel(robot_interface, checkpoint_dir="checkpoints")

# Minimal setup (no wandb, default checkpoint directory)
model = AdaptiveWorldModel(robot_interface)
```

### Logging and Visualization Features

**Efficient Logging:**
- **Throttled logging**: All metrics logged every 100 steps (LOG_INTERVAL) to reduce computational overhead
- **Optimized calculations**: Detailed gradient norms and update-to-weight ratios calculated only when logging occurs
- **Configurable intervals**: Adjust `LOG_INTERVAL` in config.py to control logging frequency

**Visual Monitoring:**
- **Prediction visualizations**: Complete visualization grids uploaded to wandb periodically
- **Remote monitoring**: View current frame, decoded frame, and all action predictions via wandb interface
- **Upload frequency**: Visualizations uploaded every 10 steps (configurable via `VISUALIZATION_UPLOAD_INTERVAL`)

### Logged Metrics

The following metrics are automatically logged when wandb is enabled:

**Vision System:**
- **`reconstruction_loss`**: Quality of visual reconstruction (lower is better, logged every 100 steps)
- **`autoencoder_training_loss`**: Training loss during autoencoder updates
- **`autoencoder_training_step`**: Dedicated counter for autoencoder training iterations
- **`mask_ratio`**: Dynamic mask ratio used during autoencoder training (0.3-0.85 range)

**Prediction System:**
- **`predictor_training_loss`**: Combined patch + latent loss for predictor training (action loss currently disabled)
- **`predictor_patch_loss`**: Patch-space reconstruction loss (anchors visual quality, weight=1.0)
- **`predictor_latent_loss`**: Latent-space prediction loss (trains encoder for predictability, weight=0.2)
- **`predictor_action_loss`**: Action classification loss (currently disabled, weight=0)
- **`predictor_explained_variance`**: R² in patch space - scale-invariant predictor quality metric (target: >0.2 indicates meaningful learning)
- **`predictor_training_step`**: Dedicated counter for predictor training iterations
- **`predictor_grad_norm`**: Global gradient norm for predictor parameters
- **`predictor_lr`**: Current learning rate for predictor optimizer
- **`predictor_uwr_95th`**: 95th percentile update-to-weight ratio across all predictor parameters

**Detailed Transformer Metrics:**
- **Per-layer gradient norms**: `grad_norms/transformer/transformer_layer_X_sublayer` (median values)
- **Per-layer UWR**: `uwr/transformer/transformer_layer_X_sublayer` (median values)
- **Sublayer tracking**: Separate metrics for self_attn, linear1, linear2, etc. components

**Dual Loss Training Approach:**
- **Patch loss (weight=1.0)**: Maintains visual reconstruction quality and prevents representation collapse
- **Latent loss (weight=0.2)**: Encourages encoder to learn prediction-friendly representations
- **Action loss (weight=0.0)**: Action classification from predicted features (currently disabled)
- **Combined approach**: Encoder gradients flow from both predictor and reconstruction tasks

**Interpreting Explained Variance (R²):**
- **≈0.0**: Predictor no better than predicting mean patch values (common early in training)
- **0.2-0.6**: Materially learning useful structure; should trend upward over training steps
- **<0**: Worse than baseline (indicates potential data/scale mismatches or unstable updates)
- **→1.0**: Excellent alignment in patch space (theoretical maximum)

**Training Progress Metrics:**
- **`consecutive_autoencoder_iterations`**: Number of consecutive autoencoder training steps before proceeding to prediction training
- **`predictor_training_iterations`**: Number of predictor training iterations needed before meeting threshold

**Action Timing Statistics:**
- **`action_timing/mean_interval`**: Average time between actions (logged every 100 actions)
- **`action_timing/median_interval`**: Median time between actions
- **`action_timing/min_interval`**: Minimum time between actions
- **`action_timing/max_interval`**: Maximum time between actions
- **`action_timing/std_interval`**: Standard deviation of action intervals
- **`action_count`**: Total number of actions taken

**Fresh Prediction Quality:**
- **`prediction_errors/level_X`**: Prediction error for each predictor level (guides training decisions)

**Visual Data:**
- **`predictions_visualization`**: Complete visualization showing current frame, decoded frame, and predictions for all actions

### Setup

1. Install wandb: `pip install wandb` (included in requirements.txt)
2. Login to wandb: `wandb login`
3. Run with project name parameter to enable logging

## Learning Progress Persistence

The system automatically saves and loads learning progress to enable continuous training across sessions with a rolling backup system for maximum reliability.

### Features

- **Rolling backup system**: Automatic backup creation before each save for safety
- **Automatic checkpointing**: Model weights, optimizers, and training progress saved periodically
- **Predictable behavior**: Every run picks up exactly where the last run finished
- **Automatic safety net**: One-deep rollback point if new checkpoint gets corrupted
- **Resume training**: Automatically loads existing checkpoints on startup
- **Configurable directory**: Set checkpoint location via `checkpoint_dir` parameter
- **Periodic saves**: Checkpoints saved every 500 predictor training steps (configurable)

### File Naming Convention

- **Primary files**: `autoencoder.pth`, `predictor_0.pth`, `state.pkl`
- **Backup files**: `autoencoder_backup.pth`, `predictor_0_backup.pth`, `state_backup.pkl`

### Saved Components

- Neural network weights (autoencoder, predictors)
- Optimizer states for continued training
- Training step counters and learning progress
- Recent history buffers (last 100 entries)

### Usage

```python
# Custom checkpoint directory
model = AdaptiveWorldModel(robot_interface, checkpoint_dir="jetbot_checkpoints")

# Override learning rates (useful for experimentation)
model = AdaptiveWorldModel(robot_interface, autoencoder_lr=1e-4, predictor_lr=5e-4)

# Default directory ("checkpoints") with config learning rates
model = AdaptiveWorldModel(robot_interface)
```

### Learning Rate Management

The system uses a hierarchy for determining learning rates:

1. **Explicit parameters** (e.g., `predictor_lr=1e-3`) - Override everything
2. **Saved optimizer state** - Used when resuming from checkpoint and no explicit override
3. **Config defaults** - Used when starting fresh and no explicit override

This allows for flexible experimentation while respecting existing training state.

Checkpoints are excluded from git via `.gitignore` to avoid committing large model files.