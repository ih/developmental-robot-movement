# Session Explorer Gradio App

A web-based interface for exploring recorded robot sessions, running inference, and training models. This is a Gradio port of the `session_explorer.ipynb` notebook with the same functionality in a modern web UI.

## Features

The Gradio app replicates all functionality from the Jupyter notebook:

### 1. Session Management
- **Session Selection**: Browse and select from recorded robot sessions
- **Auto-detection**: Automatically detects robot type (JetBot or Toroidal Dot) from session metadata
- **Session Info**: Displays session metadata including total events, observations, actions, and timestamps

### 2. Frame Navigation
- **Slider Control**: Scrub through frames using an interactive slider
- **Direct Jump**: Enter a frame number and jump directly to that frame
- **Frame Display**: View current frame with observation metadata (step number, timestamp)

### 3. Model Checkpoint Loading
- **Autoencoder Loading**: Load pre-trained autoencoder models
- **Predictor Loading**: Load pre-trained predictor models
- **Auto-path Detection**: Paths automatically updated based on robot type

### 4. Autoencoder Inference
- **Single Frame**: Run autoencoder on currently selected frame
- **Reconstruction Comparison**: Side-by-side view of input and reconstructed frames
- **Loss Display**: Show reconstruction MSE loss

### 5. Predictor Inference
- **History-based Prediction**: Use configurable history length (2-8 frames)
- **Prediction Comparison**: Compare predicted frame vs actual frame
- **Action Sweep**: Visualize counterfactual predictions for all possible actions
- **Error Metrics**: Display prediction MSE for each action variant

### 6. Autoencoder Training
- **Threshold Training**: Train until reconstruction loss reaches target threshold
- **Fixed-step Training**: Train for a specified number of steps
- **Progress Tracking**: Real-time progress bars and loss displays
- **Training Plots**: Visualize loss curves during training
- **Status Updates**: Live status and loss updates

### 7. Predictor Training
- **Threshold Training**: Train until prediction loss reaches target threshold
- **Fixed-step Training**: Train for a specified number of steps
- **Joint Training**: Uses AdaptiveWorldModel's joint autoencoder-predictor training
- **Progress Tracking**: Real-time progress bars and loss displays
- **Training Plots**: Visualize loss curves during training

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

The Gradio dependency has been added to `requirements.txt`.

## Usage

### Starting the Web App

Run the Gradio app:

```bash
python session_explorer_gradio.py
```

The app will start a local web server (default: http://localhost:7860) and automatically open in your browser.

### Basic Workflow

1. **Select a Session**
   - Click "🔄 Refresh" to scan for available sessions
   - Select a session from the dropdown
   - Click "Load Session"

2. **Navigate Frames**
   - Use the slider to browse through frames
   - Or enter a frame number and click "Jump"

3. **Load Models** (Optional)
   - The paths are auto-populated based on robot type
   - Click "Load Models" to load both autoencoder and predictor

4. **Run Inference**
   - **Autoencoder**: Click "Run Autoencoder" to see reconstruction
   - **Predictor**: Adjust history length, then click "Run Predictor" to see predictions

5. **Train Models** (Optional)
   - **Autoencoder**: Set threshold/steps and click "Train to Threshold" or "Train N Steps"
   - **Predictor**: Set threshold/steps and click "Train to Threshold" or "Train N Steps"

## Comparison with Jupyter Notebook

### Advantages of Gradio App
- **No Jupyter Required**: Runs as standalone web app
- **Cleaner UI**: Better organized interface with clear sections
- **Better for Demos**: Easier to share and demo to others
- **Progress Bars**: Built-in progress tracking for training
- **Responsive**: Modern web UI that works on any device

### Notebook Advantages
- **Advanced Features**: Full attention diagnostics and weight visualization
- **Pause/Resume**: Interactive training with pause/resume buttons
- **Custom Code**: Easy to add custom analysis cells
- **Rich Output**: Inline matplotlib plots with more interactive features

### Missing Features (compared to notebook)
The Gradio app currently does not include:
- Attention introspection (APA, ALF, TTAR, RI@16, entropy metrics)
- Attention heatmaps and breakdown charts
- Weight visualization (pre/post training weight statistics)
- Pause/Resume functionality during training
- History preview (showing multiple frames before current)

These features can be added in future versions if needed.

## Configuration

### Server Settings
By default, the app runs on:
- Host: `0.0.0.0` (accessible from network)
- Port: `7860`

To change these settings, edit the `demo.launch()` call at the bottom of `session_explorer_gradio.py`:

```python
demo.launch(share=False, server_name="127.0.0.1", server_port=8080)
```

### Share Publicly
To create a public shareable link (via Gradio's tunneling service):

```python
demo.launch(share=True)
```

## Troubleshooting

### Port Already in Use
If port 7860 is already in use, change the port:

```python
demo.launch(server_port=7861)
```

### CUDA Out of Memory
If training fails with CUDA OOM:
- Reduce the number of training steps
- Close other GPU applications
- Use smaller batch sizes (modify AdaptiveWorldModel config)

### Session Not Loading
- Ensure the session directory contains `session_meta.json` and event shards
- Check that frame files exist in the session directory
- Verify robot type is correctly detected

## Advanced Usage

### Custom Action Selectors
The app uses the same `AdaptiveWorldModel` backend as the notebook, so all action selectors and configurations work identically.

### Model Checkpoints
The app automatically determines checkpoint directories based on robot type:
- JetBot: `saved/checkpoints/jetbot/`
- Toroidal Dot: `saved/checkpoints/toroidal_dot/`

### Wandb Integration
To enable Weights & Biases logging, set `WANDB_PROJECT` at the top of `session_explorer_gradio.py`:

```python
WANDB_PROJECT = "my-project-name"
```

## Development

### Adding New Features
The Gradio app is structured with clear function definitions for each action:
- `load_session()`: Session loading logic
- `run_autoencoder()`: Autoencoder inference
- `run_predictor()`: Predictor inference
- `train_autoencoder_threshold()`: Autoencoder training
- `train_predictor_threshold()`: Predictor training

To add new features:
1. Define a new function with inputs/outputs
2. Add UI components in the `with gr.Blocks()` section
3. Connect function to UI with `.click()` or `.change()` event handlers

### Testing
Test the app with different robot types and session configurations to ensure compatibility.

## License

Same license as the main project.
