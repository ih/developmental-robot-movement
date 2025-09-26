# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for developmental robot movement with a modular architecture:

1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Concrete JetBot robot interface
3. **Adaptive World Model** (`adaptive_world_model.py`) - Hierarchical world model with uncertainty-based action selection
4. **Integration Example** (`jetbot_world_model_example.py`) - Shows how to connect world model with JetBot

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

### Adaptive World Model
- **Dependency injection**: Takes RobotInterface in constructor for modularity
- **Main loop architecture**: Continuous cycle of perception → prediction → action selection → execution
- **Neural vision system**: MaskedAutoencoderViT for visual encoding/decoding with joint training
- **Quality gating**: Robot only acts when visual reconstruction quality meets threshold
- **Hierarchical predictors**: Multiple levels of action abstraction with automatic level creation
- **Uncertainty-based exploration**: Selects actions with highest prediction uncertainty (entropy)
- **Real-time training visualization**: Shows current vs reconstructed frames during autoencoder training
- **Interactive visualization**: Real-time display of current frame, decoded frame, and predictions for all motor combinations

## Key Components

### Session Explorer Notebook
- **session_explorer.ipynb**: Interactive Jupyter notebook for exploring recorded sessions and training models. Browse recorded sessions with frame-by-frame playback, run autoencoder/predictor checkpoints against stored frames, and train models on specific frames or sequences using authentic AdaptiveWorldModel methods.
- **Comprehensive weight visualization**: Real-time visualization of both autoencoder and predictor network weights during inference and training, including weight distribution histograms, layer norms, and detailed change tracking for transformer components (action embeddings, position embeddings, self-attention, MLP layers)

### Neural Vision System
- **MaskedAutoencoderViT**: Vision Transformer-based autoencoder with powerful encoder and lightweight MLP decoder
- **Dynamic masking**: Randomized mask ratios (30%-85%) during autoencoder training for better generalization
- **TransformerActionConditionedPredictor**: Causal transformer that interleaves visual features with action tokens
- **Fresh prediction training**: Predictors trained using fresh predictions with consistent loss calculation
- **Dual loss training**: Combined patch-space reconstruction + latent-space prediction losses for encoder optimization
- **Prediction-friendly representations**: Encoder learns features that are both visually meaningful and easy to predict
- **Joint training architecture**: Autoencoder reconstruction loss + dual predictor losses with shared encoder gradients
- **Sequence length management**: Handles long histories (4096 token capacity with clipping for GPU safety)
- **Quality gating**: Robot stops acting when reconstruction loss > threshold, focuses on vision training
- **Real-time visualization**: Training progress display showing current vs reconstructed frames
- **Comprehensive experiment tracking**: Weights & Biases integration for reconstruction, predictor training, and timing metrics
- **Learning progress persistence**: Automatic save/load of model weights, training progress, and history buffers

### Action Space
- **Duration-based actions**: Motor commands with automatic stopping after specified duration
- **Simplified action space**: Single motor control with motor_right values {0, 0.12} = 2 total actions (motor_left always 0)
- **Forward-only movement**: Gentler gearbox operation with stop and forward-only commands
- **Format**: `{'motor_left': 0, 'motor_right': value, 'duration': 0.2}`
- **Smooth ramping**: Gradual acceleration/deceleration for gearbox protection
- **Automatic stopping**: Motors automatically ramp down to 0 after duration expires
- **Motor configuration**: Uses right motor for movement control, left motor remains at 0

### Configuration
- **config.py**: Contains image transforms, constants, and adaptive world model parameters
- **AdaptiveWorldModelConfig class**: Centralized configuration for all model parameters including thresholds, learning rates, and training intervals
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

### World Model with Real JetBot
```bash
python jetbot_world_model_example.py
```
- Integrates AdaptiveWorldModel with actual JetBot hardware
- Requires JetBot running RPyC server
- Interactive mode with real robot control
- Press Enter to continue with proposed action, type custom action dict, or 'stop' to exit
- **Automatic checkpoint saving**: Learning progress saved every 500 predictor training steps (configurable)
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

### Recording and Replay System
```bash
# Record a session (set RECORDING_MODE = True in config.py)
python jetbot_world_model_example.py

# Replay all recorded sessions
python replay_session_example.py
```
- **Recording mode**: Captures robot observations and actions with automatic disk space management
- **Replay mode**: Replays all recorded sessions using the exact same main loop, enabling GPU utilization for predictor training
- **Robot-agnostic replay**: Can replay any robot's recorded sessions regardless of robot type
- **Checkpoint sharing**: All modes (online, record, replay) share the same checkpoint directory for continuous learning
- **Disk space management**: Automatic cleanup of oldest sessions when total recordings exceed configurable disk limit (default 10 GB)

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
- wandb (experiment tracking and logging)

## File Structure

- `robot_interface.py`: Abstract base class defining robot interaction contract
- `jetbot_interface.py`: JetBot implementation of RobotInterface with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client with live feed capability
- `models/`: Neural network architectures directory
  - `models/__init__.py`: Module exports for clean imports
  - `models/autoencoder.py`: MaskedAutoencoderViT implementation with fixed positional embeddings
  - `models/predictor.py`: TransformerActionConditionedPredictor with sequence length management
- `adaptive_world_model.py`: Main world model implementation with comprehensive training and logging
- `jetbot_world_model_example.py`: Integration example connecting JetBot with world model
- `recording_writer.py`: Recording system with automatic disk space management
- `recording_reader.py`: Reads recorded sessions with smart observation/action sequencing
- `replay_robot.py`: Robot interface replacement for replaying recorded sessions
- `recorded_policy.py`: Action selector factory for recorded action playback
- `replay_session_example.py`: Robot-agnostic replay script for recorded sessions
- `test_jetbot_actions.ipynb`: Jupyter notebook for interactive JetBot action space testing
- `config.py`: Shared configuration, image transforms, and adaptive world model parameters
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

### Experiment Tracking
- **Weights & Biases integration**: Pass `wandb_project` parameter to `AdaptiveWorldModel` constructor to enable logging
- **Core metrics**: `reconstruction_loss`, `autoencoder_training_loss`, `predictor_training_loss`, `step`, `training_step`
- **Dual loss components**: `predictor_patch_loss` (visual quality anchor), `predictor_latent_loss` (prediction-friendly encoder training)
- **Predictor quality**: `predictor_explained_variance` (R² in patch space - scale-invariant metric showing predictor improvement over baseline)
- **Training metrics**: `predictor_grad_norm`, `predictor_lr`, `predictor_uwr_95th`, `mask_ratio`
- **Action timing statistics**: `action_timing/mean_interval`, `action_timing/median_interval`, `action_timing/min_interval`, `action_timing/max_interval`, `action_timing/std_interval`
- **Fresh prediction quality**: `prediction_errors/level_X` for each predictor level (guides training decisions)
- **Detailed transformer metrics**: Per-layer gradient norms and UWR for transformer layers (e.g., `grad_norms/transformer/transformer_layer_0_self_attn`, `uwr/transformer/transformer_layer_1_linear1`)
- **New metrics**: `consecutive_autoencoder_iterations` and `predictor_training_iterations` for training phase analysis
- **Usage**: `AdaptiveWorldModel(robot_interface, wandb_project="my-experiment")` or `None` to disable

### Learning Progress Persistence
- **Rolling backup system**: Automatic backup creation before each save for safety
- **Automatic checkpointing**: Model weights, optimizers, and training progress saved periodically
- **Predictable behavior**: Every run picks up exactly where the last run finished
- **Automatic safety net**: One-deep rollback point if new checkpoint gets corrupted
- **Checkpoint directory**: Configurable via `checkpoint_dir` parameter (defaults to "checkpoints")
- **Resume training**: Automatically loads existing checkpoints on startup (primary files first, then backup files)
- **Saved components**:
  - Neural network weights (autoencoder, predictors)
  - Optimizer states for continued training
  - Training step counters and learning progress
  - Recent history buffers (last 100 entries)
- **File naming**:
  - Primary files: `autoencoder.pth`, `predictor_0.pth`, `state.pkl`
  - Backup files: `autoencoder_backup.pth`, `predictor_0_backup.pth`, `state_backup.pkl`
- **Usage**: `AdaptiveWorldModel(robot_interface, checkpoint_dir="jetbot_checkpoints")`

### Configuration Management
- **Centralized parameters**: All adaptive world model parameters moved to `config.py` in `AdaptiveWorldModelConfig` class
- **Recording configuration**: `RECORDING_MODE` boolean and `RECORDING_MAX_DISK_GB` for disk space management
- **Configurable intervals**: Logging, visualization upload, checkpoint saving, and training display frequencies
- **Training parameters**:
  - Mask ratio bounds (MASK_RATIO_MIN=0.3, MASK_RATIO_MAX=0.85)
  - Learning rates (AUTOENCODER_LR=1e-4, PREDICTOR_LR=1e-4)
  - Quality thresholds (RECONSTRUCTION_THRESHOLD=0.0005, PREDICTION_THRESHOLD=0.0005)
- **Action timing**: Configurable delay between actions via `ACTION_DELAY` parameter for controlled execution timing
- **Easy tuning**: Modify parameters in one location instead of scattered throughout the code
- **Default values**: Reasonable defaults for all parameters (LOG_INTERVAL=100, ACTION_DELAY=0, etc.)
- **Recent changes**: Tightened quality thresholds and unified learning rates for more consistent training

### Logging and Monitoring
- **Throttled logging frequency**: All metrics logged periodically (LOG_INTERVAL=100 steps) to reduce computational overhead
- **Periodic visualization uploads**: Complete prediction visualizations uploaded to wandb every N steps
- **Comprehensive metrics**: Reconstruction loss, predictor training loss, timing metrics, action counts, and mask ratios
- **Scale-invariant quality tracking**: Explained variance (R²) in patch space provides intuitive predictor performance assessment
- **Visual monitoring**: Remote access to current frame, decoded frame, and all action predictions via wandb
- **Training insights**: Track dynamic mask ratios and reconstruction-based predictor training progress
- **Detailed transformer metrics**: Per-layer gradient norms and update-to-weight ratios for individual transformer layers and sublayers
- **Optimized metric calculation**: Gradient norms and UWR calculations throttled to reduce performance impact during training
- **Hierarchical metric organization**: Transformer layers tracked separately with detailed sublayer breakdowns (e.g., `grad_norms/transformer/transformer_layer_0_self_attn`)
- **Statistical focus**: Only median values tracked for gradient norms and per-layer UWR, plus 95th percentile for global UWR
- **Learning rate flexibility**: Override learning rates at runtime while respecting saved optimizer states

### Enhanced Visualization System
- **Real-time prediction comparison**: Visual display comparing last predicted frame with actual current frame
- **Integrated error metrics**: Reconstruction loss displayed above decoded frame, prediction error above last prediction
- **Performance optimization**: Display updates on configurable intervals (DISPLAY_INTERVAL=10) instead of every iteration
- **Smart interactive mode**: Always shows current visualization in interactive mode for user decision-making
- **Comprehensive layout**: Current frame → Decoded frame (with reconstruction loss) → Last prediction (with error) → Action predictions
- **Visual error analysis**: Direct pixel-by-pixel comparison between predictions and reality
- **Efficient rendering**: Reduced CPU/GPU usage during long training runs while maintaining visual feedback