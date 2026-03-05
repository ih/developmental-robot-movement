"""
Model checkpoint management for saving and loading trained models.
"""

import os
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime

import config
import world_model_utils
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


FRESH_WEIGHTS_OPTION = "🆕 Fresh (untrained) weights"


def refresh_checkpoints():
    """Refresh checkpoint dropdown list"""
    checkpoints = list_available_checkpoints()
    # Always include the fresh weights option at the top
    choices = [FRESH_WEIGHTS_OPTION] + checkpoints if checkpoints else [FRESH_WEIGHTS_OPTION]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


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

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Save model state dict, optimizer state, scheduler state, and metadata
        scheduler_type = type(state.world_model.ae_scheduler).__name__
        scheduler = state.world_model.ae_scheduler

        # Extract scheduler-specific parameters for recreation on load
        scheduler_params = {}
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler_params = {
                'patience': scheduler.patience,
                'factor': scheduler.factor,
                'min_lrs': scheduler.min_lrs,  # List of min_lr per param group
                'mode': scheduler.mode,
            }

        checkpoint = {
            'model_state_dict': state.world_model.autoencoder.state_dict(),
            'optimizer_state_dict': state.world_model.ae_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_type': scheduler_type,  # Save scheduler type for compatibility check
            'scheduler_params': scheduler_params,  # Save params for scheduler recreation
            'timestamp': datetime.now().isoformat(),
            # Preserve original peak LR for global schedule calculation when resuming
            'original_peak_lr': state.loaded_checkpoint_metadata.get('original_peak_lr')
                                or state.world_model.ae_optimizer.param_groups[0]['lr'],
            'model_type': config.AutoencoderConcatPredictorWorldModelConfig.MODEL_TYPE,
            'config': {
                'frame_size': state.world_model.frame_size,  # Use actual frame_size from model, not global config
                'separator_width': config.AutoencoderConcatPredictorWorldModelConfig.SEPARATOR_WIDTH,
                'canvas_history_size': config.AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE,
                'embed_dim': state.world_model.autoencoder.embed_dim,
                'depth': len(state.world_model.autoencoder.blocks),
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
    """Load model weights from a checkpoint file, or reset to fresh untrained weights"""
    if state.world_model is None:
        return "Error: No model loaded. Please load a session first."

    if not checkpoint_name or checkpoint_name == "No checkpoints available":
        return "Error: Please select a valid checkpoint."

    # Handle fresh weights option - reinitialize model
    if checkpoint_name == FRESH_WEIGHTS_OPTION:
        try:
            from models import TargetedMAEWrapper, TargetedDecoderOnlyWrapper, LatentDiffusionWrapper

            # Get current autoencoder dimensions
            old_ae = state.world_model.autoencoder
            img_height, img_width = old_ae.image_size
            patch_size = old_ae.patch_size
            embed_dim = old_ae.embed_dim

            # Create fresh autoencoder with same architecture as current model
            if isinstance(old_ae, LatentDiffusionWrapper):
                # DiT: recreate DiT with fresh weights, keep VAE frozen
                from models.vit_dit import DiffusionViT
                from models.noise_scheduler import NoiseScheduler
                old_dit = old_ae.dit
                Config = config.AutoencoderConcatPredictorWorldModelConfig
                fresh_dit = DiffusionViT(
                    img_height=old_dit.image_size[0],
                    img_width=old_dit.image_size[1],
                    in_channels=old_dit.in_channels,
                    patch_size=old_dit.patch_size,
                    embed_dim=old_dit.embed_dim,
                    depth=len(old_dit.blocks),
                    num_heads=old_dit.blocks[0].attn.num_heads,
                    prediction_type=old_dit.prediction_type,
                ).to(state.device)
                fresh_scheduler = NoiseScheduler(
                    num_train_timesteps=Config.DIT_NUM_TRAIN_TIMESTEPS,
                    beta_start=Config.DIT_BETA_START,
                    beta_end=Config.DIT_BETA_END,
                    beta_schedule=Config.DIT_BETA_SCHEDULE,
                    prediction_type=Config.DIT_PREDICTION_TYPE,
                )
                state.world_model.autoencoder = LatentDiffusionWrapper(
                    vae=old_ae.vae,  # Reuse frozen VAE
                    dit=fresh_dit,
                    noise_scheduler=fresh_scheduler,
                    num_inference_steps=Config.DIT_NUM_INFERENCE_STEPS,
                    training_mode=Config.DIT_TRAINING_MODE,
                ).to(state.device)
            elif isinstance(old_ae, TargetedDecoderOnlyWrapper):
                depth = len(old_ae.blocks)
                num_heads = old_ae.blocks[0].attn.num_heads
                state.world_model.autoencoder = TargetedDecoderOnlyWrapper(
                    img_height=img_height,
                    img_width=img_width,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                ).to(state.device)
            else:
                decoder_embed_dim = old_ae.decoder_embed.out_features
                depth = len(old_ae.blocks)
                num_heads = old_ae.blocks[0].attn.num_heads
                decoder_depth = len(old_ae.decoder_blocks)
                decoder_num_heads = old_ae.decoder_blocks[0].attn.num_heads
                state.world_model.autoencoder = TargetedMAEWrapper(
                    img_height=img_height,
                    img_width=img_width,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    decoder_embed_dim=decoder_embed_dim,
                    decoder_depth=decoder_depth,
                    decoder_num_heads=decoder_num_heads,
                ).to(state.device)

            # Recreate optimizer with fresh state
            param_groups = world_model_utils.create_param_groups(
                state.world_model.autoencoder,
                config.AutoencoderConcatPredictorWorldModelConfig.WEIGHT_DECAY
            )
            state.world_model.ae_optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR
            )

            # Create fresh scheduler (ReduceLROnPlateau for flexibility)
            state.world_model.ae_scheduler = world_model_utils.create_reduce_on_plateau_scheduler(
                state.world_model.ae_optimizer,
                patience=6,
                factor=0.1,
                min_lr=1e-7
            )

            # Clear loaded checkpoint metadata
            state.loaded_checkpoint_metadata = {}

            return "✅ **Reset to fresh untrained weights**\n\nModel, optimizer, and scheduler have been reinitialized."
        except Exception as e:
            import traceback
            return f"❌ **Error resetting to fresh weights:**\n\n{str(e)}\n\n{traceback.format_exc()}"

    # Use dynamic checkpoint directory based on session's robot type
    checkpoint_dir = state.get_checkpoint_dir_for_session(state.session_state["session_dir"])
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        return f"Error: Checkpoint file not found: {checkpoint_path}"

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=state.device, weights_only=False)

        # Load model state (supports depth growth: loading shallower checkpoint into deeper model)
        model_sd = checkpoint.get('model_state_dict', checkpoint)
        growth_info = world_model_utils.load_state_dict_with_depth_growth(
            state.world_model.autoencoder, model_sd
        )

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
            # Check scheduler type compatibility
            current_scheduler_type = type(state.world_model.ae_scheduler).__name__
            saved_scheduler_type = checkpoint.get('scheduler_type', 'Unknown')
            scheduler_params = checkpoint.get('scheduler_params', {})
            scheduler_state = checkpoint['scheduler_state_dict']

            # If scheduler type is unknown, try to detect from state_dict keys
            if saved_scheduler_type == 'Unknown':
                # ReduceLROnPlateau has 'factor', 'patience', 'min_lrs' etc.
                # LambdaLR has 'base_lrs' but not 'factor' or 'patience'
                if 'factor' in scheduler_state and 'patience' in scheduler_state:
                    saved_scheduler_type = 'ReduceLROnPlateau'
                    print(f"[LOAD] Detected scheduler type from state_dict: {saved_scheduler_type}")

            if saved_scheduler_type != 'Unknown' and saved_scheduler_type != current_scheduler_type:
                # Scheduler types don't match - try to recreate the correct scheduler type
                if saved_scheduler_type == 'ReduceLROnPlateau':
                    # Recreate ReduceLROnPlateau scheduler
                    # Try to get params from checkpoint, fallback to state_dict, then defaults
                    try:
                        # Get min_lr: prefer scheduler_params, fallback to state_dict, then default
                        if scheduler_params and 'min_lrs' in scheduler_params:
                            min_lr = scheduler_params['min_lrs'][0]
                        elif 'min_lrs' in scheduler_state:
                            min_lr = scheduler_state['min_lrs'][0]
                        else:
                            min_lr = 1e-7

                        # Get patience: prefer scheduler_params, fallback to state_dict, then default
                        if scheduler_params and 'patience' in scheduler_params:
                            patience = scheduler_params['patience']
                        elif 'patience' in scheduler_state:
                            patience = scheduler_state['patience']
                        else:
                            patience = 5

                        # Get factor: prefer scheduler_params, fallback to state_dict, then default
                        if scheduler_params and 'factor' in scheduler_params:
                            factor = scheduler_params['factor']
                        elif 'factor' in scheduler_state:
                            factor = scheduler_state['factor']
                        else:
                            factor = 0.5

                        state.world_model.ae_scheduler = world_model_utils.create_reduce_on_plateau_scheduler(
                            state.world_model.ae_optimizer,
                            patience=patience,
                            factor=factor,
                            min_lr=min_lr,
                        )
                        state.world_model.ae_scheduler.load_state_dict(scheduler_state)
                        scheduler_loaded = True
                        print(f"[LOAD] Recreated ReduceLROnPlateau scheduler (patience={patience}, factor={factor}, min_lr={min_lr}) and loaded state")
                    except Exception as e:
                        warnings.append(f"⚠️ Scheduler recreation failed: {str(e)}")
                        print(f"[LOAD WARNING] Failed to recreate scheduler: {str(e)}")
                else:
                    # Can't recreate - skip loading
                    warnings.append(f"⚠️ Scheduler state not loaded: type mismatch (saved: {saved_scheduler_type}, current: {current_scheduler_type})")
                    print(f"[LOAD WARNING] Skipping scheduler state: type mismatch ({saved_scheduler_type} vs {current_scheduler_type})")
            else:
                try:
                    state.world_model.ae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    scheduler_loaded = True
                except (ValueError, KeyError) as e:
                    warnings.append(f"⚠️ Scheduler state not loaded: {str(e)}")
                    print(f"[LOAD WARNING] Skipping scheduler state: {str(e)}")

        state.current_checkpoint_name = checkpoint_name

        # Populate checkpoint metadata for resume functionality
        current_lr = state.world_model.ae_optimizer.param_groups[0]['lr'] if optimizer_loaded else None
        state.loaded_checkpoint_metadata = {
            'samples_seen': checkpoint.get('samples_seen', 0),
            'loss': checkpoint.get('loss', None),
            'learning_rate': current_lr,
            # Original peak LR for global schedule calculation (fall back to current LR if not saved)
            'original_peak_lr': checkpoint.get('original_peak_lr', current_lr),
            'scheduler_step': state.world_model.ae_scheduler.last_epoch if scheduler_loaded else 0,
            'timestamp': checkpoint.get('timestamp', None),
            'checkpoint_name': checkpoint_name,
            'optimizer_loaded': optimizer_loaded,
            'scheduler_loaded': scheduler_loaded,
        }
        print(f"[CHECKPOINT] Loaded metadata: samples_seen={state.loaded_checkpoint_metadata['samples_seen']}, "
              f"loss={state.loaded_checkpoint_metadata['loss']}, lr={state.loaded_checkpoint_metadata['learning_rate']}")
        print(f"[CHECKPOINT] Loaded original_peak_lr: {state.loaded_checkpoint_metadata['original_peak_lr']}")

        status_msg = f"✅ **Model weights loaded successfully!**\n\n"
        status_msg += f"**Checkpoint:** {checkpoint_name}\n"
        status_msg += f"**Location:** {checkpoint_path}\n"

        # Show what was loaded
        status_msg += f"\n**Loaded Components:**\n"
        status_msg += f"- Model weights: ✅\n"
        status_msg += f"- Optimizer state: {'✅' if optimizer_loaded else '❌ (skipped)'}\n"
        status_msg += f"- Scheduler state: {'✅' if scheduler_loaded else '❌ (skipped)'}\n"

        # Show depth growth info if applicable
        if growth_info['depth_changed']:
            status_msg += f"\n**Depth Growth:**\n"
            status_msg += f"- Blocks: {growth_info['blocks_saved']} -> {growth_info['blocks_current']}\n"
            if growth_info['decoder_blocks_saved'] > 0 or growth_info['decoder_blocks_current'] > 0:
                status_msg += f"- Decoder blocks: {growth_info['decoder_blocks_saved']} -> {growth_info['decoder_blocks_current']}\n"
            status_msg += f"- New blocks initialized with zero-init residual (identity pass-through)\n"

        # Add warnings if any
        if warnings:
            status_msg += f"\n**Warnings:**\n"
            for warning in warnings:
                status_msg += f"{warning}\n"
            status_msg += f"\n*Note: Model weights loaded successfully. A fresh scheduler will be created when training starts, using the checkpoint's learning rate as the starting point.*\n"

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

        # Resume info
        status_msg += f"\n**Resume Info:**\n"
        status_msg += f"- To continue training, enable 'Resume Mode' in batch training\n"
        if state.loaded_checkpoint_metadata['samples_seen'] > 0:
            status_msg += f"- Starting samples: {state.loaded_checkpoint_metadata['samples_seen']:,}\n"
        if state.loaded_checkpoint_metadata['learning_rate'] is not None:
            status_msg += f"- Current LR: {state.loaded_checkpoint_metadata['learning_rate']:.2e}\n"

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
