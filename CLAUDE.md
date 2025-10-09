# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for developmental robot movement with a modular architecture:

1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Concrete JetBot robot interface
3. **Toroidal Dot Environment** (`toroidal_dot_env.py`, `toroidal_dot_interface.py`) - Simulated environment with white dot on black background for testing and debugging
4. **Adaptive World Model** (`adaptive_world_model.py`) - Hierarchical world model with uncertainty-based action selection
5. **Robot Runner** (`robot_runner.py`) - Lightweight action execution without learning components
6. **Action Selectors** (`toroidal_action_selectors.py`, `recorded_policy.py`) - Pluggable action selection strategies
7. **Integration Examples** (`jetbot_world_model_example.py`, `toroidal_dot_world_model_example.py`, `toroidal_dot_runner_example.py`) - Shows how to connect world model and runner with different robots/environments

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
- **Random initialization**: Each session starts with dot at random (x, y) position
- **ToroidalDotRobot class**: Implements RobotInterface for simulated environment
- **Fast iteration**: No hardware needed, perfect for debugging and testing world model
- **Configurable parameters**: Dot radius, movement speed, image size via ToroidalDotConfig

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
- **session_explorer.ipynb**: Interactive Jupyter notebook for exploring recorded sessions and training models. Browse recorded sessions with frame-by-frame playback, run autoencoder/predictor checkpoints against stored frames, and train models on specific frames or sequences using authentic AdaptiveWorldModel methods.
- **Multi-robot support**: Automatically detects robot type from session metadata and loads appropriate checkpoints (JetBot or toroidal dot)
- **Robot-agnostic exploration**: Works seamlessly with sessions from any robot type, dynamically adapting action space and model paths
- **Action distribution analysis**: Automatic table showing action usage statistics when loading sessions, comparing current session vs all sessions with counts and percentages for each action in the action space
- **Comprehensive weight visualization**: Real-time visualization of both autoencoder and predictor network weights during inference and training, including weight distribution histograms, layer norms, and detailed change tracking for transformer components (action embeddings, position embeddings, self-attention, MLP layers)
- **Model saving capabilities**: Save trained autoencoder and predictor models with configurable paths, compatible with AdaptiveWorldModel checkpoint format
- **Interactive training controls**: Pause and resume buttons for both autoencoder and predictor training with responsive UI updates and proper state management
- **Attention introspection**: Visualize transformer attention patterns with heatmaps, breakdown charts, and quantitative metrics (APA, ALF, TTAR, RI@16, entropy)
- **Action space sweep**: Predictions across full robot action space with MSE for each action, clearly labeled with action values and recorded action indicator

### Neural Vision System
- **MaskedAutoencoderViT**: Vision Transformer-based autoencoder with powerful encoder and lightweight MLP decoder
- **Dynamic masking**: Randomized mask ratios (30%-85%) during autoencoder training for better generalization
- **TransformerActionConditionedPredictor**: Causal transformer that interleaves visual features with action tokens
- **FiLM conditioning**: Feature-wise Linear Modulation integrates action information into transformer layers via learned affine transformations (gamma, beta)
- **Action normalization**: All action channels mapped to [-1, 1] range before embedding for consistent FiLM conditioning
- **ActionEmbedding module**: Learned MLP embedding of normalized actions for semantic action representation
- **Delta latent prediction**: Predictors learn residual/delta features rather than absolute features, improving training stability
- **Action reconstruction loss**: Predictors must classify which discrete action was taken from pixel-level differences between predicted and previous frames, learning action-aware visual representations through convolutional classification of image changes
- **Fresh prediction training**: Predictors trained using fresh predictions with consistent loss calculation (single training pass per error threshold)
- **Triple loss training**: Combined patch-space reconstruction + latent-space prediction + action classification losses for encoder optimization
- **Prediction-friendly representations**: Encoder learns features that are both visually meaningful and easy to predict
- **Joint training architecture**: Autoencoder reconstruction loss + triple predictor losses (patch, latent, action) with shared encoder gradients
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
- **Gradient flow tracking**: L2 norm-based monitoring of gradients flowing through action-related parameters (action_embed, film_layers) vs total network
- **Token indexing**: Automatic derivation of frame/action/future token positions from sequence for robust metric calculation

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
- **Robot-specific directories**:
  - `JETBOT_CHECKPOINT_DIR = saved/checkpoints/jetbot/` - JetBot model checkpoints
  - `TOROIDAL_DOT_CHECKPOINT_DIR = saved/checkpoints/toroidal_dot/` - Toroidal dot model checkpoints
  - `JETBOT_RECORDING_DIR = saved/sessions/jetbot/` - JetBot session recordings
  - `TOROIDAL_DOT_RECORDING_DIR = saved/sessions/toroidal_dot/` - Toroidal dot session recordings
- **Action normalization config**: `ACTION_CHANNELS`, `ACTION_RANGES` define action space and normalization ranges for FiLM conditioning
- **ToroidalDotConfig class**: Configuration for simulated environment (IMG_SIZE, DOT_RADIUS, DOT_MOVE_PIXELS, ACTION_CHANNELS_DOT, ACTION_RANGES_DOT)
- **FiLM parameters**: `ACTION_EMBED_DIM` (64), `FILM_HIDDEN_DIM` (128), `FILM_BLOCK_IDS` ([0, 2]) control action conditioning architecture
- **Delta latent flag**: `DELTA_LATENT` (True) enables residual prediction mode for improved training stability
- **Prediction loss weights**: `PRED_PATCH_W` (0.2), `PRED_LATENT_W` (0.8), `PRED_ACTION_W` (1.0) balance visual quality, latent prediction, and action classification
- **Prediction history size**: `PREDICTION_HISTORY_SIZE` (3) controls number of recent frames used for context in predictions
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

### Toroidal Dot Environment with World Model
```bash
python toroidal_dot_world_model_example.py
```
- Integrates AdaptiveWorldModel with simulated toroidal dot environment
- Perfect for debugging and fast iteration without hardware
- Uses separate checkpoint directory (`saved/checkpoints/toroidal_dot/`)
- Records to separate session directory (`saved/sessions/toroidal_dot/`)
- Binary action space: 0 (stay), 1 (move right)
- Configurable via `ToroidalDotConfig` in `config.py`
- Uses `SEQUENCE_ALWAYS_MOVE` action selector by default
- **Interactive notebook**: `test_toroidal_dot_actions.ipynb` for testing and visualization

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
```
- **Recording mode**: Captures robot observations and actions with automatic disk space management
- **Robot-specific directories**: JetBot and toroidal dot sessions stored separately for organization
- **Replay mode**: Replays all recorded sessions using the exact same main loop, enabling GPU utilization for predictor training
- **Robot-agnostic replay**: Can replay any robot's recorded sessions regardless of robot type
- **Isolated checkpoints**: Each robot type maintains separate model checkpoints due to different action space dimensionality
- **Disk space management**: Automatic cleanup of oldest sessions when total recordings exceed configurable disk limit (default 10 GB per robot type)

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
- `toroidal_dot_env.py`: Simulated 224x224 toroidal environment with white dot
- `toroidal_dot_interface.py`: ToroidalDotRobot implementation of RobotInterface
- `models/`: Neural network architectures directory
  - `models/__init__.py`: Module exports for clean imports
  - `models/autoencoder.py`: MaskedAutoencoderViT implementation with fixed positional embeddings
  - `models/predictor.py`: TransformerActionConditionedPredictor with FiLM conditioning, delta latent prediction, image-based action classification, and attention introspection
  - `models/encoder_layer_with_attn.py`: Custom transformer encoder layer that optionally returns attention maps
- `adaptive_world_model.py`: Main world model implementation with comprehensive training and logging
- `robot_runner.py`: Lightweight runner for executing actions without learning components
- `jetbot_world_model_example.py`: Integration example connecting JetBot with world model
- `toroidal_dot_world_model_example.py`: Integration example connecting toroidal dot environment with world model (uses SEQUENCE_ALWAYS_MOVE)
- `toroidal_dot_runner_example.py`: Integration example using RobotRunner with toroidal dot (no learning)
- `toroidal_action_selectors.py`: Action selector factories for toroidal dot environment (constant and sequence selectors)
- `recording_writer.py`: Recording system with automatic disk space management
- `recording_reader.py`: Reads recorded sessions with smart observation/action sequencing
- `replay_robot.py`: Robot interface replacement for replaying recorded sessions
- `recorded_policy.py`: Action selector factory for recorded action playback
- `replay_session_example.py`: Robot-agnostic replay script for recorded sessions
- `session_explorer.ipynb`: Interactive Jupyter notebook for multi-robot session exploration with automatic robot type detection
- `session_explorer.py`: Python script version of session explorer notebook with identical functionality
- `test_jetbot_actions.ipynb`: Jupyter notebook for interactive JetBot action space testing
- `test_toroidal_dot_actions.ipynb`: Jupyter notebook for interactive toroidal dot environment testing
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

### Experiment Tracking
- **Weights & Biases integration**: Pass `wandb_project` parameter to `AdaptiveWorldModel` constructor to enable logging
- **Core metrics**: `reconstruction_loss`, `autoencoder_training_loss`, `predictor_training_loss`, `step`, `training_step`
- **Triple loss components**: `predictor_patch_loss` (visual quality anchor), `predictor_latent_loss` (prediction-friendly encoder training), `predictor_action_loss` (action classification from predicted features)
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
- **Action normalization parameters**: `ACTION_CHANNELS`, `ACTION_RANGES` define action space mapping to [-1, 1] for FiLM conditioning
- **FiLM architecture parameters**: `ACTION_EMBED_DIM`, `FILM_HIDDEN_DIM`, `FILM_BLOCK_IDS` control action conditioning layer configuration
- **Delta latent mode**: `DELTA_LATENT` flag enables residual prediction architecture for improved gradient flow
- **Action timing**: Configurable delay between actions via `ACTION_DELAY` parameter for controlled execution timing
- **Easy tuning**: Modify parameters in one location instead of scattered throughout the code
- **Default values**: Reasonable defaults for all parameters (LOG_INTERVAL=100, ACTION_DELAY=0, etc.)
- **Recent changes**: Image-based action classification from pixel differences, increased action loss weight (PRED_ACTION_W=1.0), reduced prediction history (PREDICTION_HISTORY_SIZE=3) for stronger action-aware learning

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