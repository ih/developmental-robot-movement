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
- **Joint training architecture**: Autoencoder reconstruction loss + action predictor gradient flow
- **Quality gating**: Robot stops acting when reconstruction loss > threshold, focuses on vision training
- **Real-time visualization**: Training progress display showing current vs reconstructed frames

### Action Space
- **Duration-based actions**: Motor commands with automatic stopping after specified duration
- **Motor combinations**: Cross product of motor_left and motor_right values {-0.15, 0, 0.15} = 9 total motor combinations
- **Format**: `{'motor_left': value, 'motor_right': value, 'duration': 0.1}`
- **Automatic stopping**: Motors automatically set to 0 after duration expires

### Configuration
- **config.py**: Contains image transforms and constants
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

### World Model Testing (No Robot)
```bash
python adaptive_world_model.py
```
- Runs world model with stub robot for testing
- Shows visualization of predictions for all actions
- All ML components are stubs - only tests main loop logic
- No physical robot required

## Dependencies

Required Python packages:
- rpyc (robot communication)
- opencv-python (computer vision)
- numpy, matplotlib (data processing and visualization)
- torch, torchvision, timm (neural networks and vision transformers)
- PIL (image processing)
- ipywidgets, IPython (notebook compatibility)
- tqdm (progress bars)

## File Structure

- `robot_interface.py`: Abstract base class defining robot interaction contract
- `jetbot_interface.py`: JetBot implementation of RobotInterface with duration-based actions
- `jetbot_remote_client.py`: Low-level JetBot RPyC client with live feed capability
- `models.py`: Neural network architectures including MaskedAutoencoderViT
- `adaptive_world_model.py`: Main world model implementation with neural vision system
- `jetbot_world_model_example.py`: Integration example connecting JetBot with world model
- `config.py`: Shared configuration and image transforms
- `.gitignore`: Excludes `.claude/` directory, `CLAUDE.md`, and common Python artifacts

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