# Developmental Robot Movement

Research code for developmental robot movement with canvas-based world model architecture using masked autoencoder inpainting for next-frame prediction.

## Project Overview

This repository contains research code for a canvas-based world model that learns to predict future observations by inpainting masked next-frame slots in horizontally concatenated frame sequences.

**Key Components:**
1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Physical robot interface
3. **Toroidal Dot Environment** (`toroidal_dot_env.py`, `toroidal_dot_interface.py`) - Simulated environment for testing
4. **Autoencoder Concat Predictor World Model** (`autoencoder_concat_predictor_world_model.py`) - Canvas-based world model
5. **Concat World Model Explorer** (`concat_world_model_explorer/`) - Interactive web-based interface for exploring sessions

## Architecture

### Canvas-Based World Model

The concat world model uses a unique approach to visual prediction:

- **Frame concatenation**: History frames are concatenated horizontally with colored action separators
- **Targeted masking**: Next-frame slot is fully masked (MASK_RATIO = 1.0) for inpainting-based prediction
- **MAE-native training**: Optimizes only masked patches using the MaskedAutoencoderViT architecture
- **Action encoding**: Actions encoded as thin colored separators between frames (e.g., red for stay, green for move)
- **Non-square canvases**: Handles non-square concatenated images (e.g., 224x688 for 3 frames + 2 separators)
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

### Action Selectors

- **Pluggable architecture**: Action selectors are functions that take observations and return actions
- **Toroidal action selectors** (`toroidal_action_selectors.py`):
  - `create_constant_action_selector(action)`: Always returns the same action
  - `create_sequence_action_selector(sequence)`: Cycles through a sequence of actions
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

- **Session selection**: Choose from recorded sessions in `saved/sessions/toroidal_dot/`
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

**Disk space management**: Automatic cleanup of oldest sessions when total recordings exceed configurable disk limit (default 10 GB per robot type).

### Interactive Testing

```bash
# JetBot action space testing
jupyter notebook test_jetbot_actions.ipynb

# Toroidal dot environment testing
jupyter notebook test_toroidal_dot_actions.ipynb
```

These notebooks provide interactive environments for testing robot interfaces and action spaces.

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

### Robot Interfaces
- `robot_interface.py`: Abstract base class for robot interaction
- `jetbot_interface.py`: JetBot implementation with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client
- `toroidal_dot_env.py`: Simulated toroidal environment
- `toroidal_dot_interface.py`: ToroidalDotRobot implementation

### Models
- `models/__init__.py`: Module exports
- `models/base_autoencoder.py`: Base class for autoencoders
- `models/vit_autoencoder.py`: MaskedAutoencoderViT with powerful transformer encoder/decoder
- `models/autoencoder_concat_predictor.py`: Canvas building and TargetedMAEWrapper for masked inpainting

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
- **Robot-specific directories**: Separate checkpoint and recording directories for JetBot and toroidal dot
- **Recording configuration**: `RECORDING_MODE` boolean and `RECORDING_MAX_DISK_GB` for disk management

## Next Steps

1. **Generate sessions**: Use recording system to capture robot observations and actions
2. **Explore sessions**: Launch concat world model explorer to visualize and analyze sessions
3. **Analyze attention**: Use decoder attention visualization to understand prediction mechanism
4. **Iterate**: Adjust config parameters, generate new sessions, and explore results
