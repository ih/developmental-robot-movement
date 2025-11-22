# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for developmental robot movement with a canvas-based world model architecture:

1. **RobotInterface** (`robot_interface.py`) - Abstract base class defining robot interaction contract
2. **JetBot Implementation** (`jetbot_interface.py`, `jetbot_remote_client.py`) - Concrete JetBot robot interface
3. **Toroidal Dot Environment** (`toroidal_dot_env.py`, `toroidal_dot_interface.py`) - Simulated environment with white dot on black background for testing and debugging
4. **Autoencoder Concat Predictor World Model** (`autoencoder_concat_predictor_world_model.py`) - Canvas-based world model using targeted masked autoencoder with frame concatenation
5. **Action Selectors** (`toroidal_action_selectors.py`, `recorded_policy.py`) - Pluggable action selection strategies
6. **Concat World Model Explorer** (`concat_world_model_explorer_gradio.py`) - Interactive web-based interface for exploring and visualizing the concat world model

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
  - Pre-defined sequences: `SEQUENCE_ALWAYS_MOVE`, `SEQUENCE_ALWAYS_STAY`, `SEQUENCE_ALTERNATE`, `SEQUENCE_DOUBLE_MOVE`, `SEQUENCE_TRIPLE_MOVE`
- **Recorded action selector** (`recorded_policy.py`):
  - `create_recorded_action_selector(reader)`: Replays actions from recorded sessions
  - Optional action filtering for selective replay
- **Usage**: Pass to `AutoencoderConcatPredictorWorldModel` via `action_selector` parameter

## Key Components

### Concat World Model Explorer
- **concat_world_model_explorer_gradio.py**: Web-based Gradio interface for running AutoencoderConcatPredictorWorldModel on recorded toroidal dot sessions
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
  - `JETBOT_RECORDING_DIR = saved/sessions/jetbot/` - JetBot session recordings
  - `TOROIDAL_DOT_RECORDING_DIR = saved/sessions/toroidal_dot/` - Toroidal dot session recordings
- **ToroidalDotConfig class**: Configuration for simulated environment (IMG_SIZE, DOT_RADIUS, DOT_MOVE_PIXELS, ACTION_CHANNELS_DOT, ACTION_RANGES_DOT)
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

### Concat World Model Explorer
```bash
python concat_world_model_explorer_gradio.py
```
- **Canvas-based world model**: Interactive web UI for exploring AutoencoderConcatPredictorWorldModel on toroidal dot sessions
- **Session selection**: Choose from recorded sessions in `saved/sessions/toroidal_dot/`
- **Frame navigation**: Browse session frames with slider and text input
- **World model execution**: Run world model for specified number of iterations with single-step training per iteration
- **Full masking approach**: Uses MASK_RATIO = 1.0 for complete next-frame inpainting
- **Live progress tracking**: Real-time display of prediction error and iteration timing during execution
- **Comprehensive visualizations**: Four training views (original canvas, masked overlay, full inpainting output, composite) plus prediction display
- **Post-run visualizations**: Current frame, predicted frame, and prediction error
- **Metric graphs**: Plots tracking prediction error and iteration time over all iterations
- **Authentic training**: Uses actual `AutoencoderConcatPredictorWorldModel.train_autoencoder()` method with MAE-native masked patch optimization
- **Model checkpoint management**:
  - **Save weights**: Enter checkpoint name and click "💾 Save Weights" to save model, optimizer, and scheduler state
  - **Load weights**: Select checkpoint from dropdown and click "📂 Load Weights" to restore previous training state
  - **Checkpoint location**: All checkpoints saved to `saved/checkpoints/toroidal_dot/`
  - **Metadata tracking**: Each checkpoint includes timestamp, config parameters, and training metrics
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
- `concat_world_model_explorer_gradio.py`: Web-based Gradio interface for AutoencoderConcatPredictorWorldModel exploration
- `config.py`: Shared configuration, image transforms, robot-specific directories, and world model parameters
- `world_model_utils.py`: Utility functions for training and tensor operations

### Robot Interfaces
- `robot_interface.py`: Abstract base class defining robot interaction contract
- `jetbot_interface.py`: JetBot implementation of RobotInterface with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client with live feed capability
- `toroidal_dot_env.py`: Simulated 224x224 toroidal environment with white dot
- `toroidal_dot_interface.py`: ToroidalDotRobot implementation of RobotInterface

### Models
- `models/__init__.py`: Module exports for clean imports
- `models/base_autoencoder.py`: Base class for autoencoder implementations
- `models/vit_autoencoder.py`: MaskedAutoencoderViT with powerful transformer encoder and decoder
- `models/autoencoder_concat_predictor.py`: Canvas building utilities and TargetedMAEWrapper for targeted masked autoencoder inpainting

### Action Selection and Recording
- `toroidal_action_selectors.py`: Action selector factories for toroidal dot environment (constant and sequence selectors)
- `recorded_policy.py`: Action selector factory for recorded action playback
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
- `session_explorer_lib.py`: Shared library of utilities for session management, frame processing, and model operations

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
