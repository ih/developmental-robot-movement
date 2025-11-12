# Concat World Model Explorer - Gradio App

A web-based interface for exploring recorded toroidal dot sessions using the canvas-based `AutoencoderConcatPredictorWorldModel`. This interactive tool lets you visualize how the world model learns to predict future frames through masked autoencoder inpainting.

## Features

### 1. Session Management
- **Session Selection**: Browse and select from recorded toroidal dot sessions
- **Automatic Detection**: Scans `saved/sessions/toroidal_dot/` directory for available sessions
- **Session Info**: Displays session metadata including total frames, actions, and timestamps

### 2. Frame Navigation
- **Slider Control**: Scrub through frames using an interactive slider
- **Direct Jump**: Enter a frame number and jump directly to that frame
- **Frame Display**: View current frame with metadata (step number, timestamp, action)

### 3. World Model Execution
- **Iteration Control**: Run world model for a specified number of iterations (1-500)
- **Single-step Training**: Each iteration performs one training step on the canvas
- **Live Progress Tracking**: Real-time updates of prediction error and iteration timing
- **Authentic Training**: Uses actual `AutoencoderConcatPredictorWorldModel.train_autoencoder()` method

### 4. Comprehensive Visualizations

**Training Views** (4 visualizations):
- **Original Canvas**: Horizontally concatenated frames with action separators
- **Masked Canvas Overlay**: Shows which patches are masked (next-frame slot fully masked)
- **Full Inpainting Output**: Complete reconstruction including masked next-frame
- **Composite Reconstruction**: Overlay showing reconstruction quality

**Prediction Views** (3 visualizations):
- **Current Frame**: The actual current observation
- **Predicted Next Frame**: Model's inpainting of the masked next-frame slot
- **Prediction Error**: Pixel-wise difference between prediction and actual next frame

### 5. Metric Graphs
- **Prediction Error Plot**: Tracks MSE prediction error over iterations
- **Iteration Time Plot**: Shows training time per iteration

### 6. Decoder Attention Visualization

Interactive exploration of decoder attention patterns:

**Filtering Controls:**
- **Quantile-based Filtering**: Show only top N% of attention connections (e.g., 95% = strongest 5%)
- **Layer Selection**: Toggle individual decoder layers (0-4) on/off
- **Head Selection**: Toggle individual attention heads (0-3) for fine-grained analysis
- **Aggregation Method**: Choose mean, max, or sum aggregation across selected heads

**Visualization Modes:**
- **Patch-to-Patch Lines**: Connection lines showing attention flow
  - Color-coded by decoder layer
  - Thickness proportional to attention magnitude
  - Visual overlay on canvas image
- **Heatmap Matrix**: Full attention matrix as color-coded heatmap
  - Rows: decoder patches (masked next-frame slot)
  - Columns: encoder patches (visible history and separators)
  - Color intensity: attention weight magnitude

**Real-time Statistics:**
- Total connection count
- Sum of attention weights
- Per-layer connection counts
- Per-head connection counts
- Attention weight distributions

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- gradio (web interface)
- torch, torchvision (neural networks)
- matplotlib, numpy (visualization)
- PIL (image processing)
- nest_asyncio (async support)

## Usage

### Starting the Web App

Run the Gradio app:

```bash
python concat_world_model_explorer_gradio.py
```

The app will start a local web server (default: http://localhost:7860) and automatically open in your browser.

### Basic Workflow

1. **Select a Session**
   - Click "🔄 Refresh Sessions" to scan for available sessions
   - Select a session from the dropdown
   - Click "Load Session" to load the session data

2. **Navigate Frames**
   - Use the slider to browse through frames
   - Or enter a frame number and click "Jump to Frame"
   - View current frame and metadata

3. **Run World Model**
   - Set the number of iterations (1-500)
   - Click "Run World Model" to execute
   - Watch live progress updates during execution
   - View training and prediction visualizations after completion

4. **View Metric Graphs**
   - Scroll down to see prediction error and iteration time plots
   - Analyze how prediction quality changes over iterations

5. **Explore Decoder Attention** (after running world model)
   - Adjust quantile threshold to filter attention connections
   - Toggle decoder layers on/off
   - Toggle attention heads on/off
   - Switch between aggregation methods (mean/max/sum)
   - Choose visualization mode (lines or heatmap)
   - View real-time statistics about attention patterns

### Understanding the Visualizations

**Training Canvas Visualizations:**
- The canvas concatenates history frames horizontally
- Action separators are thin colored bands between frames (e.g., red=stay, green=move)
- The next-frame slot is fully masked (MASK_RATIO = 1.0) for inpainting
- The autoencoder learns to predict what the next frame should be

**Prediction Quality:**
- Lower prediction error (MSE) indicates better prediction accuracy
- Prediction error typically decreases as training progresses
- Iteration time shows computational cost per training step

**Decoder Attention:**
- Strong attention to recent frames indicates temporal modeling
- Attention to action separators indicates action-conditional prediction
- Layer-specific patterns reveal hierarchical processing
- Head-specific patterns show specialized attention mechanisms

## Configuration

### Server Settings

By default, the app runs on:
- Host: `0.0.0.0` (accessible from network)
- Port: `7860`

To change these settings, edit the `demo.launch()` call at the bottom of `concat_world_model_explorer_gradio.py`:

```python
demo.launch(share=False, server_name="127.0.0.1", server_port=8080)
```

### Share Publicly

To create a public shareable link (via Gradio's tunneling service):

```python
demo.launch(share=True)
```

### World Model Configuration

The world model uses configuration from `config.py`:
- **Frame size**: 224x224 (toroidal dot environment)
- **Separator width**: 8 pixels
- **Canvas history**: 3 frames + 2 separators = 224x688 canvas
- **Mask ratio**: 1.0 (full masking of next-frame slot)
- **Reconstruction threshold**: 0.0001
- **Optimizer**: AdamW with cosine decay learning rate

To modify these parameters, edit `AutoencoderConcatPredictorWorldModelConfig` in `config.py`.

## Troubleshooting

### Port Already in Use

If port 7860 is already in use, change the port:

```python
demo.launch(server_port=7861)
```

### CUDA Out of Memory

If execution fails with CUDA OOM:
- Reduce the number of iterations
- Close other GPU applications
- Reduce canvas history size in config

### Session Not Loading

- Ensure the session directory contains `session_meta.json` and event shards
- Check that frame files exist in the session directory
- Verify the session is in `saved/sessions/toroidal_dot/`

### Attention Visualization Not Appearing

- Run the world model at least once to generate attention data
- Ensure at least one decoder layer is selected
- Try increasing the quantile threshold to show more connections

### Slow Performance

- Reduce the number of iterations per run
- Disable attention visualization during long runs
- Use a machine with GPU acceleration
- Reduce the number of visible attention connections (lower quantile)

## Advanced Usage

### Creating New Sessions for Exploration

To generate new sessions:
1. Set `RECORDING_MODE = True` in `config.py`
2. Implement a script that:
   - Creates a `ToroidalDotRobot` instance
   - Wraps it with `RecordingRobot`
   - Runs actions using action selectors
   - Sessions automatically saved to `saved/sessions/toroidal_dot/`

Example action selectors available in `toroidal_action_selectors.py`:
- `SEQUENCE_ALWAYS_MOVE`: Always move right
- `SEQUENCE_ALTERNATE`: Alternate between stay and move
- `SEQUENCE_DOUBLE_MOVE`: Two moves, then stay

### Model Checkpoints

The world model automatically loads/saves checkpoints from:
- `saved/checkpoints/toroidal_dot/autoencoder.pth`

Checkpoints are automatically created/updated during world model execution.

### Understanding Canvas Structure

For a 3-frame history with 2 action separators:
```
[Frame t-2][Action separator][Frame t-1][Action separator][MASKED FRAME t]
  224x224       224x8            224x224       224x8          224x224
|<---------------------- Total Canvas: 224x688 ------------------------>|
```

The model learns to inpaint the masked frame based on history and action separators.

### Attention Pattern Interpretation

**Common patterns to look for:**
- **Recency bias**: Stronger attention to recent frames (frame t-1) vs older frames
- **Action conditioning**: Attention to colored action separators
- **Spatial correspondence**: Attention to similar spatial regions across frames
- **Layer specialization**: Different layers attending to different temporal scales
- **Head specialization**: Different heads attending to different features

## Development

### Adding New Features

The Gradio app is structured with clear function definitions:
- `load_session()`: Session loading logic
- `run_world_model()`: World model execution with progress tracking
- `update_attention_viz()`: Attention visualization generation

To add new features:
1. Define a new function with inputs/outputs
2. Add UI components in the `with gr.Blocks()` section
3. Connect function to UI with `.click()` or `.change()` event handlers

### Code Organization

Key files:
- `concat_world_model_explorer_gradio.py`: Main Gradio interface
- `autoencoder_concat_predictor_world_model.py`: World model implementation
- `attention_viz.py`: Attention visualization utilities
- `models/autoencoder_concat_predictor.py`: Canvas building and targeted MAE wrapper
- `session_explorer_lib.py`: Shared utilities for session management

## Technical Details

### Canvas-Based Architecture

The concat world model uses a unique approach:
1. Concatenate history frames horizontally with action separators
2. Fully mask the next-frame slot (rightmost 224x224 pixels)
3. Use MaskedAutoencoderViT to inpaint the masked region
4. Train only on masked patches (MAE-native optimization)
5. Predicted frame = inpainted next-frame slot

### Training Strategy

- **Single-step training**: One forward/backward pass per iteration
- **Quality gating**: Training continues until inpainting threshold is met
- **Gradient-based optimization**: AdamW with cosine decay learning rate
- **Non-square support**: Custom positional embeddings for 224x688 canvases

### Attention Mechanism

The decoder attention shows how the model combines information:
- **Encoder output**: All visible patches (history frames + separators)
- **Decoder queries**: Masked patches (next-frame slot)
- **Attention weights**: How much each decoder patch attends to each encoder patch
- **Multi-head, multi-layer**: 4 heads per layer, 5 decoder layers

## License

Same license as the main project.
