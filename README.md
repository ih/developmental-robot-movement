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
- **TransformerActionConditionedPredictor**: Causal transformer that interleaves visual features with action tokens
- **Joint training architecture**: Autoencoder reconstruction loss + action predictor gradient flow with shared encoder
- **Sequence length management**: Handles long histories (4096 token capacity with clipping for GPU safety)
- **Quality gating**: Robot stops acting when reconstruction loss > threshold, focuses on vision training
- **Real-time visualization**: Training progress display showing current vs reconstructed frames
- **Comprehensive experiment tracking**: Weights & Biases integration for reconstruction, predictor training, and timing metrics
- **Learning progress persistence**: Automatic save/load of model weights, training progress, and history buffers

### Action Space
- **Duration-based actions**: Motor commands with automatic stopping after specified duration
- **Simplified action space**: Single motor control with motor_left values {-0.2, 0, 0.2} = 3 total actions (motor_right always 0)
- **Format**: `{'motor_left': value, 'motor_right': 0, 'duration': 0.1}`
- **Automatic stopping**: Motors automatically set to 0 after duration expires

### Configuration
- **config.py**: Contains image transforms, constants, and adaptive world model parameters
- **AdaptiveWorldModelConfig class**: Centralized configuration for all model parameters
- **Configurable intervals**: Logging frequency, visualization uploads, checkpoint saving, and display intervals
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
- **Periodic logging**: Metrics logged every 10 steps (configurable) instead of every step to reduce noise
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

**Prediction System:**
- **`predictor_training_loss`**: MSE loss between predicted and actual features
- **`predictor_level`**: Which predictor level is being trained
- **`predictor_training_step`**: Dedicated counter for predictor training iterations

**Performance Metrics:**
- **`time_between_actions`**: Duration in seconds between consecutive actions (logged periodically)
- **`action_count`**: Total number of actions taken
- **`step`** and **`training_step`**: Overall timestep counters for tracking progress

**Visual Data:**
- **`predictions_visualization`**: Complete visualization showing current frame, decoded frame, and predictions for all actions

### Setup

1. Install wandb: `pip install wandb` (included in requirements.txt)
2. Login to wandb: `wandb login`
3. Run with project name parameter to enable logging

## Learning Progress Persistence

The system automatically saves and loads learning progress to enable continuous training across sessions.

### Features

- **Automatic checkpointing**: Model weights, optimizers, and training progress saved periodically
- **Resume training**: Automatically loads existing checkpoints on startup
- **Configurable directory**: Set checkpoint location via `checkpoint_dir` parameter
- **Periodic saves**: Checkpoints saved every 10 predictor training steps
- **Final save**: Checkpoint automatically saved when program exits

### Saved Components

- Neural network weights (autoencoder, predictors)
- Optimizer states for continued training
- Training step counters and learning progress
- Recent history buffers (last 100 entries)

### Usage

```python
# Custom checkpoint directory
model = AdaptiveWorldModel(robot_interface, checkpoint_dir="jetbot_checkpoints")

# Default directory ("checkpoints")
model = AdaptiveWorldModel(robot_interface)
```

Checkpoints are excluded from git via `.gitignore` to avoid committing large model files.