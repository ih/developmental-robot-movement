# Developmental Robot Movement

Research code for developmental robot movement with hierarchical world models and uncertainty-based exploration.

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

### Neural Vision System
- **MaskedAutoencoderViT**: Vision Transformer-based autoencoder with powerful encoder and lightweight MLP decoder
- **Dynamic masking**: Randomized mask ratios (30%-85%) during autoencoder training for improved generalization
- **TransformerActionConditionedPredictor**: Causal transformer that interleaves visual features with action tokens
- **Fresh prediction training**: Predictors trained using fresh predictions with consistent loss calculation
- **Dual loss training**: Combined patch-space reconstruction + latent-space prediction losses for encoder optimization
- **Joint training architecture**: Autoencoder reconstruction loss + action predictor gradient flow with shared encoder
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

### Configuration
- **config.py**: Contains image transforms, constants, and adaptive world model parameters
- **AdaptiveWorldModelConfig class**: Centralized configuration for all model parameters
- **Configurable intervals**: Logging frequency, visualization uploads, checkpoint saving, and display intervals
- **Action timing**: Configurable delay between actions via `ACTION_DELAY` parameter (default 0.1 seconds)
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
- `models/`: Neural network architectures directory
  - `models/__init__.py`: Module exports for clean imports
  - `models/autoencoder.py`: MaskedAutoencoderViT implementation with fixed positional embeddings
  - `models/predictor.py`: TransformerActionConditionedPredictor with sequence length management
- `adaptive_world_model.py`: Main world model implementation with comprehensive training and logging
- `jetbot_world_model_example.py`: Integration example connecting JetBot with world model
- `test_jetbot_actions.ipynb`: Interactive Jupyter notebook for JetBot action space testing
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
- **`reconstruction_loss`**: Quality of visual reconstruction (lower is better, logged periodically)
- **`autoencoder_training_loss`**: Training loss during autoencoder updates
- **`mask_ratio`**: Dynamic mask ratio used during autoencoder training (0.3-0.85 range)

**Prediction System:**
- **`predictor_training_loss`**: Combined patch + latent loss for predictor training
- **`predictor_patch_loss`**: Patch-space reconstruction loss (anchors visual quality)
- **`predictor_latent_loss`**: Latent-space prediction loss (trains encoder for predictability)
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
- **Latent loss (weight=0.1)**: Encourages encoder to learn prediction-friendly representations
- **Combined approach**: Encoder gradients flow from both predictor and reconstruction tasks

**Interpreting Explained Variance (R²):**
- **≈0.0**: Predictor no better than predicting mean patch values (common early in training)
- **0.2-0.6**: Materially learning useful structure; should trend upward over training steps
- **<0**: Worse than baseline (indicates potential data/scale mismatches or unstable updates)
- **→1.0**: Excellent alignment in patch space (theoretical maximum)

**Action Timing Statistics:**
- **`action_timing/mean_interval`**: Average time between actions (logged every 100 actions)
- **`action_timing/median_interval`**: Median time between actions
- **`action_timing/min_interval`**: Minimum time between actions
- **`action_timing/max_interval`**: Maximum time between actions
- **`action_timing/std_interval`**: Standard deviation of action intervals
- **`action_count`**: Total number of actions taken
- **`step`** and **`training_step`**: Overall timestep counters for tracking progress

**Fresh Prediction Quality:**
- **`prediction_errors/level_X`**: Prediction error for each predictor level (guides training decisions)

**Visual Data:**
- **`predictions_visualization`**: Complete visualization showing current frame, decoded frame, and predictions for all actions

### Setup

1. Install wandb: `pip install wandb` (included in requirements.txt)
2. Login to wandb: `wandb login`
3. Run with project name parameter to enable logging

## Learning Progress Persistence

The system automatically saves and loads learning progress to enable continuous training across sessions with a dual save system for maximum reliability.

### Features

- **Dual save system**: Automatic periodic saves during training + manual saves on program exit
- **Automatic checkpointing**: Model weights, optimizers, and training progress saved periodically with standard filenames
- **Manual saves**: On program exit (Ctrl+C), saves to files with `_manual` suffix for safe restart
- **Smart loading**: On startup, prioritizes manual save files over automatic saves for best recovery
- **Resume training**: Automatically loads existing checkpoints on startup (manual saves first, then auto saves)
- **Configurable directory**: Set checkpoint location via `checkpoint_dir` parameter
- **Periodic saves**: Checkpoints saved every 500 predictor training steps (configurable)

### File Naming Convention

- **Auto saves**: `autoencoder.pth`, `predictor_0.pth`, `state.pkl`
- **Manual saves**: `autoencoder_manual.pth`, `predictor_0_manual.pth`, `state_manual.pkl`

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