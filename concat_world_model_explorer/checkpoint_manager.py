"""
Model checkpoint management for saving and loading trained models.
"""

import os
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime

import config
from . import state
from .utils import format_loss


def list_available_checkpoints():
    """List all available checkpoint files in the checkpoint directory"""
    # Get checkpoint directory based on current session's robot type
    if not state.session_state.get("session_dir"):
        return []

    checkpoint_dir = Path(state.get_checkpoint_dir_for_session(state.session_state["session_dir"]))
    if not checkpoint_dir.exists():
        return []

    # Find all .pth files
    checkpoint_files = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.name for f in checkpoint_files]


def refresh_checkpoints():
    """Refresh checkpoint dropdown list"""
    checkpoints = list_available_checkpoints()
    choices = checkpoints if checkpoints else ["No checkpoints available"]
    return gr.Dropdown(choices=choices, value=choices[0] if checkpoints else None)


def save_model_weights(checkpoint_name):
    """Save current model weights to a checkpoint file"""
    if state.world_model is None:
        return "Error: No model loaded. Please load a session first.", gr.Dropdown()

    if not checkpoint_name or checkpoint_name.strip() == "":
        return "Error: Please provide a checkpoint name.", gr.Dropdown()

    checkpoint_name = checkpoint_name.strip()

    # Add .pth extension if not present
    if not checkpoint_name.endswith('.pth'):
        checkpoint_name += '.pth'

    # Use dynamic checkpoint directory based on session's robot type
    checkpoint_dir = state.get_checkpoint_dir_for_session(state.session_state["session_dir"])
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    try:
        # Save model state dict, optimizer state, scheduler state, and metadata
        checkpoint = {
            'model_state_dict': state.world_model.autoencoder.state_dict(),
            'optimizer_state_dict': state.world_model.ae_optimizer.state_dict(),
            'scheduler_state_dict': state.world_model.ae_scheduler.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'frame_size': config.AutoencoderConcatPredictorWorldModelConfig.FRAME_SIZE,
                'separator_width': config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                'canvas_history_size': config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
            }
        }

        # Add training metrics if available
        if hasattr(state.world_model, '_metrics_history') and state.world_model._metrics_history['iteration']:
            checkpoint['metrics'] = {
                'total_iterations': len(state.world_model._metrics_history['iteration']),
                'final_training_loss': state.world_model._metrics_history['training_loss'][-1] if state.world_model._metrics_history['training_loss'] else None,
                'final_prediction_error': state.world_model._metrics_history['prediction_error'][-1] if state.world_model._metrics_history['prediction_error'] else None,
            }

        torch.save(checkpoint, checkpoint_path)
        state.current_checkpoint_name = checkpoint_name

        status_msg = f"✅ **Model weights saved successfully!**\n\n"
        status_msg += f"**Checkpoint:** {checkpoint_name}\n"
        status_msg += f"**Location:** {checkpoint_path}\n"
        status_msg += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        if 'metrics' in checkpoint:
            status_msg += f"\n**Training Metrics:**\n"
            status_msg += f"- Total iterations: {checkpoint['metrics']['total_iterations']}\n"
            if checkpoint['metrics']['final_training_loss'] is not None:
                status_msg += f"- Final training loss: {format_loss(checkpoint['metrics']['final_training_loss'])}\n"
            if checkpoint['metrics']['final_prediction_error'] is not None:
                status_msg += f"- Final prediction error: {format_loss(checkpoint['metrics']['final_prediction_error'])}\n"

        return status_msg, refresh_checkpoints()

    except Exception as e:
        import traceback
        error_msg = f"❌ **Error saving checkpoint:**\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, gr.Dropdown()


def load_model_weights(checkpoint_name):
    """Load model weights from a checkpoint file"""
    if state.world_model is None:
        return "Error: No model loaded. Please load a session first."

    if not checkpoint_name or checkpoint_name == "No checkpoints available":
        return "Error: Please select a valid checkpoint."

    # Use dynamic checkpoint directory based on session's robot type
    checkpoint_dir = state.get_checkpoint_dir_for_session(state.session_state["session_dir"])
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        return f"Error: Checkpoint file not found: {checkpoint_path}"

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)

        # Load model state
        if 'model_state_dict' in checkpoint:
            state.world_model.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback: assume entire checkpoint is the state dict
            state.world_model.autoencoder.load_state_dict(checkpoint)

        # Track which components were loaded
        optimizer_loaded = False
        scheduler_loaded = False
        warnings = []

        # Try to load optimizer state if available (may fail if parameter groups differ)
        if 'optimizer_state_dict' in checkpoint:
            try:
                state.world_model.ae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_loaded = True
            except (ValueError, KeyError) as e:
                warnings.append(f"⚠️ Optimizer state not loaded (parameter group mismatch): {str(e)}")
                print(f"[LOAD WARNING] Skipping optimizer state: {str(e)}")

        # Try to load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            try:
                state.world_model.ae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler_loaded = True
            except (ValueError, KeyError) as e:
                warnings.append(f"⚠️ Scheduler state not loaded: {str(e)}")
                print(f"[LOAD WARNING] Skipping scheduler state: {str(e)}")

        state.current_checkpoint_name = checkpoint_name

        status_msg = f"✅ **Model weights loaded successfully!**\n\n"
        status_msg += f"**Checkpoint:** {checkpoint_name}\n"
        status_msg += f"**Location:** {checkpoint_path}\n"

        # Show what was loaded
        status_msg += f"\n**Loaded Components:**\n"
        status_msg += f"- Model weights: ✅\n"
        status_msg += f"- Optimizer state: {'✅' if optimizer_loaded else '❌ (skipped)'}\n"
        status_msg += f"- Scheduler state: {'✅' if scheduler_loaded else '❌ (skipped)'}\n"

        # Add warnings if any
        if warnings:
            status_msg += f"\n**Warnings:**\n"
            for warning in warnings:
                status_msg += f"{warning}\n"
            status_msg += f"\n*Note: Model weights loaded successfully. Optimizer/scheduler will be reinitialized on next training run.*\n"

        if 'timestamp' in checkpoint:
            status_msg += f"\n**Saved at:** {checkpoint['timestamp']}\n"

        # Show loss and samples_seen for auto-saved best models
        if 'loss' in checkpoint:
            status_msg += f"**Loss at save:** {checkpoint['loss']:.6f}\n"
        if 'samples_seen' in checkpoint:
            status_msg += f"**Samples seen:** {checkpoint['samples_seen']:,}\n"

        if 'config' in checkpoint:
            status_msg += f"\n**Model Configuration:**\n"
            status_msg += f"- Frame size: {checkpoint['config'].get('frame_size')}\n"
            status_msg += f"- Separator width: {checkpoint['config'].get('separator_width')}\n"
            status_msg += f"- Canvas history size: {checkpoint['config'].get('canvas_history_size')}\n"

        if 'metrics' in checkpoint:
            status_msg += f"\n**Training Metrics (at save time):**\n"
            status_msg += f"- Total iterations: {checkpoint['metrics']['total_iterations']}\n"
            if checkpoint['metrics']['final_training_loss'] is not None:
                status_msg += f"- Final training loss: {format_loss(checkpoint['metrics']['final_training_loss'])}\n"
            if checkpoint['metrics']['final_prediction_error'] is not None:
                status_msg += f"- Final prediction error: {format_loss(checkpoint['metrics']['final_prediction_error'])}\n"

        return status_msg

    except Exception as e:
        import traceback
        error_msg = f"❌ **Error loading checkpoint:**\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg


def get_checkpoint_info():
    """Get current checkpoint status information"""
    if state.current_checkpoint_name:
        return f"**Current checkpoint:** {state.current_checkpoint_name}"
    else:
        return "**Current checkpoint:** None (using fresh model)"
