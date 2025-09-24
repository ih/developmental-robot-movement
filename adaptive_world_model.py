# Adaptive World Model with Hierarchical Action Learning
# Based on Day 6 outline with future generalizations from the document

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import wandb
import os
import pickle
import random
import signal
from models import MaskedAutoencoderViT, TransformerActionConditionedPredictor
from config import AdaptiveWorldModelConfig

class AdaptiveWorldModel:
    def __init__(self, robot_interface, interactive=False, wandb_project=None, checkpoint_dir="checkpoints", autoencoder_lr=None, predictor_lr=None, action_selector=None):
        # Store the robot interface
        self.robot = robot_interface

        # Store action selector (defaults to internal select_action_by_uncertainty)
        self.action_selector = action_selector if action_selector is not None else self.select_action_by_uncertainty

        # Checkpoint management
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = AdaptiveWorldModelConfig.CHECKPOINT_SAVE_INTERVAL

        # Store explicit learning rates for potential override
        self.explicit_autoencoder_lr = autoencoder_lr
        self.explicit_predictor_lr = predictor_lr
        
        # Device setup - use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize wandb if project name is provided
        if wandb_project:
            wandb.init(project=wandb_project, config={
                "device": str(self.device),
                "interactive": interactive,
                "lookahead": AdaptiveWorldModelConfig.LOOKAHEAD,
                "max_lookahead_margin": AdaptiveWorldModelConfig.MAX_LOOKAHEAD_MARGIN,
                "prediction_history_size": AdaptiveWorldModelConfig.PREDICTION_HISTORY_SIZE,
                "uncertainty_threshold": AdaptiveWorldModelConfig.UNCERTAINTY_THRESHOLD,
                "reconstruction_threshold": AdaptiveWorldModelConfig.RECONSTRUCTION_THRESHOLD,
                "prediction_threshold": AdaptiveWorldModelConfig.PREDICTION_THRESHOLD,
                "autoencoder_lr": AdaptiveWorldModelConfig.AUTOENCODER_LR,
                "predictor_lr": AdaptiveWorldModelConfig.PREDICTOR_LR,
                "mask_ratio_min": AdaptiveWorldModelConfig.MASK_RATIO_MIN,
                "mask_ratio_max": AdaptiveWorldModelConfig.MASK_RATIO_MAX,
                "pred_patch_w": AdaptiveWorldModelConfig.PRED_PATCH_W,
                "pred_latent_w": AdaptiveWorldModelConfig.PRED_LATENT_W,
                "max_history_size": AdaptiveWorldModelConfig.MAX_HISTORY_SIZE,
                "checkpoint_history_limit": AdaptiveWorldModelConfig.CHECKPOINT_HISTORY_LIMIT,
            })
            self.wandb_enabled = True
        else:
            self.wandb_enabled = False
        
        # Core components
        self.autoencoder = MaskedAutoencoderViT().to(self.device)
        self.predictors = [TransformerActionConditionedPredictor().to(self.device)]  # Start with one predictor
        self.action_encoders = []  # For future hierarchical actions
        self.action_decoders = []
        
        # Parameters from config
        self.lookahead = AdaptiveWorldModelConfig.LOOKAHEAD
        self.max_lookahead_margin = AdaptiveWorldModelConfig.MAX_LOOKAHEAD_MARGIN
        self.prediction_history_size = AdaptiveWorldModelConfig.PREDICTION_HISTORY_SIZE
        self.uncertainty_threshold = AdaptiveWorldModelConfig.UNCERTAINTY_THRESHOLD
        self.interactive = interactive
        
        # Matplotlib figure for persistent display
        self.fig = None
        self.axes = None
        
        # The action space is now retrieved from the robot interface
        self.base_actions = self.robot.action_space
        
        # History buffers
        self.frame_features_history = []
        self.action_history = []
        self.prediction_buffer = []
        
        # Training visualization counter
        self.autoencoder_training_step = 0
        self.predictor_training_step = 0
        
        # Action timing tracking
        self.last_action_time = None
        self.action_count = 0
        self.action_time_intervals = []  # Store action time intervals since last log
        
        # Display counter for interval-based display updates
        self.display_counter = 0

        # Counter for consecutive autoencoder training iterations before reaching step 3
        self.consecutive_autoencoder_iterations = 0
        
        # Load checkpoint if it exists
        self.load_checkpoint()

    def _get_autoencoder_lr(self):
        """Get autoencoder learning rate using hierarchy: explicit > saved optimizer > config"""
        if self.explicit_autoencoder_lr is not None:
            return self.explicit_autoencoder_lr
        elif hasattr(self, 'autoencoder_optimizer') and self.autoencoder_optimizer is not None:
            return self.autoencoder_optimizer.param_groups[0]['lr']
        else:
            return AdaptiveWorldModelConfig.AUTOENCODER_LR

    def _get_predictor_lr(self):
        """Get predictor learning rate using hierarchy: explicit > saved optimizer > config"""
        if self.explicit_predictor_lr is not None:
            return self.explicit_predictor_lr
        else:
            return AdaptiveWorldModelConfig.PREDICTOR_LR
    
    def save_checkpoint(self):
        """Save current learning progress to disk with rolling backup"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create backup of existing checkpoint files before saving new ones
        self._backup_existing_checkpoints()

        # Save autoencoder model and optimizer
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': getattr(self, 'autoencoder_optimizer', {}).state_dict() if hasattr(self, 'autoencoder_optimizer') else None,
        }, os.path.join(self.checkpoint_dir, 'autoencoder.pth'))

        # Save predictors
        for i, predictor in enumerate(self.predictors):
            torch.save({
                'model_state_dict': predictor.state_dict() if hasattr(predictor, 'state_dict') else {},
                'optimizer_state_dict': getattr(predictor, 'optimizer', {}).state_dict() if hasattr(predictor, 'optimizer') else None,
                'level': predictor.level,
            }, os.path.join(self.checkpoint_dir, f'predictor_{i}.pth'))

        # Save learning progress and history (keep only recent history to avoid huge files)
        history_limit = AdaptiveWorldModelConfig.CHECKPOINT_HISTORY_LIMIT
        state = {
            'autoencoder_training_step': self.autoencoder_training_step,
            'predictor_training_step': self.predictor_training_step,
            'action_count': self.action_count,
            'display_counter': self.display_counter,
            'lookahead': self.lookahead,
            'frame_features_history': self.frame_features_history[-history_limit:] if self.frame_features_history else [],
            'action_history': self.action_history[-history_limit:] if self.action_history else [],
            'prediction_buffer': self.prediction_buffer,
            'last_action_time': self.last_action_time,
        }

        with open(os.path.join(self.checkpoint_dir, 'state.pkl'), 'wb') as f:
            pickle.dump(state, f)

        print(f"Checkpoint saved at predictor step {self.predictor_training_step}")

    def _backup_existing_checkpoints(self):
        """Create backup copies of existing checkpoint files before overwriting"""
        checkpoint_files = [
            'autoencoder.pth',
            'state.pkl'
        ]

        # Add predictor files that exist
        for i in range(len(self.predictors)):
            predictor_file = f'predictor_{i}.pth'
            if os.path.exists(os.path.join(self.checkpoint_dir, predictor_file)):
                checkpoint_files.append(predictor_file)

        # Create backups by renaming existing files
        for filename in checkpoint_files:
            src_path = os.path.join(self.checkpoint_dir, filename)
            if os.path.exists(src_path):
                backup_filename = filename.replace('.pth', '_backup.pth').replace('.pkl', '_backup.pkl')
                backup_path = os.path.join(self.checkpoint_dir, backup_filename)
                # Remove old backup if it exists
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                # Rename current file to backup
                os.rename(src_path, backup_path)

    def load_checkpoint(self):
        """Load learning progress from disk if checkpoint exists"""
        if not os.path.exists(self.checkpoint_dir):
            print("No checkpoint directory found, starting fresh")
            return

        # Try primary checkpoint files first, then backup files
        autoencoder_path = os.path.join(self.checkpoint_dir, 'autoencoder.pth')
        backup_autoencoder_path = os.path.join(self.checkpoint_dir, 'autoencoder_backup.pth')

        if os.path.exists(autoencoder_path):
            use_backup = False
            print("Loading from primary checkpoint files...")
        elif os.path.exists(backup_autoencoder_path):
            use_backup = True
            autoencoder_path = backup_autoencoder_path
            print("Primary checkpoint missing, loading from backup files...")
        else:
            print("No autoencoder checkpoint found")
            return
        
        # Load autoencoder
        if os.path.exists(autoencoder_path):
            checkpoint = torch.load(autoencoder_path, map_location=self.device)
            self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer (respecting explicit learning rate overrides)
            if checkpoint['optimizer_state_dict'] is not None:
                if not hasattr(self, 'autoencoder_optimizer'):
                    lr = self._get_autoencoder_lr()
                    self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)

                if self.explicit_autoencoder_lr is None:
                    # Use saved optimizer state if no explicit override
                    self.autoencoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    # Override learning rate but keep other optimizer state
                    print(f"Overriding autoencoder learning rate to {self.explicit_autoencoder_lr}")
                    for param_group in self.autoencoder_optimizer.param_groups:
                        param_group['lr'] = self.explicit_autoencoder_lr
            print("Autoencoder checkpoint loaded")
        
        # Load predictors (using backup if needed)
        suffix = '_backup' if use_backup else ''
        predictor_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('predictor_') and f.endswith(f'{suffix}.pth')]
        for predictor_file in sorted(predictor_files):
            predictor_path = os.path.join(self.checkpoint_dir, predictor_file)
            checkpoint = torch.load(predictor_path, map_location=self.device)

            # Find or create corresponding predictor
            # Extract index from filename like 'predictor_0_backup.pth' or 'predictor_0.pth'
            file_parts = predictor_file.replace(suffix, '').split('_')
            predictor_idx = int(file_parts[1].split('.')[0])
            if predictor_idx < len(self.predictors):
                predictor = self.predictors[predictor_idx]
                if hasattr(predictor, 'load_state_dict') and checkpoint['model_state_dict']:
                    predictor.load_state_dict(checkpoint['model_state_dict'])

                # Restore optimizer (respecting explicit learning rate overrides)
                if checkpoint['optimizer_state_dict'] is not None:
                    if not hasattr(predictor, 'optimizer'):
                        lr = self._get_predictor_lr()
                        predictor.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

                    if self.explicit_predictor_lr is None:
                        # Use saved optimizer state if no explicit override
                        predictor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    else:
                        # Override learning rate but keep other optimizer state
                        print(f"Overriding predictor learning rate to {self.explicit_predictor_lr}")
                        for param_group in predictor.optimizer.param_groups:
                            param_group['lr'] = self.explicit_predictor_lr

        # Load learning progress and history (using backup if needed)
        state_path = os.path.join(self.checkpoint_dir, f'state{suffix}.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            self.autoencoder_training_step = state.get('autoencoder_training_step', state.get('training_step', 0))
            self.predictor_training_step = state.get('predictor_training_step', 0)
            self.action_count = state.get('action_count', 0)
            self.display_counter = state.get('display_counter', 0)
            self.lookahead = state.get('lookahead', 1)
            self.frame_features_history = state.get('frame_features_history', [])
            self.action_history = state.get('action_history', [])
            self.prediction_buffer = state.get('prediction_buffer', [])
            self.last_action_time = state.get('last_action_time', None)
            
            print(f"Learning progress loaded: {self.autoencoder_training_step} autoencoder steps, {self.predictor_training_step} predictor steps, {self.action_count} actions")
    
    def to_model_tensor(self, frame_np):
        """Convert frame to properly scaled tensor for model input"""
        # frame_np is HxWx3 RGB, uint8 or float
        if frame_np.dtype == np.uint8:
            img = frame_np.astype(np.float32) / 255.0
        else:
            img = np.clip(frame_np.astype(np.float32), 0.0, 1.0)
        return torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(self.device)  # [1,3,H,W]
        
    def main_loop(self):
        while True:
            # Step 1: Capture and encode current frame using the interface
            current_frame = self.robot.get_observation()
            if current_frame is None:
                continue  # Skip this iteration if observation failed
            
            # Convert to properly scaled tensor [1, 3, H, W]
            frame_tensor = self.to_model_tensor(current_frame)
            current_features = self.autoencoder.encode(frame_tensor)
            
            # Step 2: Validate image encoding quality using proper reconstruction
            # Use the same path as training (forward + unpatchify) for consistent metrics
            with torch.no_grad():  # No gradients needed for display decoding
                decoded_tensor = self.autoencoder.reconstruct(frame_tensor)
                decoded_frame = decoded_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                decoded_frame = np.clip(decoded_frame, 0, 1)  # Clip to valid range for display
            
            # Calculate reconstruction loss in patch space (same as training)
            with torch.no_grad():
                pred_patches, _ = self.autoencoder(frame_tensor, mask_ratio=0.0)
                target_patches = self.patchify(frame_tensor)
                reconstruction_loss = torch.nn.functional.mse_loss(pred_patches, target_patches).item()
            
            # Log reconstruction loss to wandb (periodically)
            if self.wandb_enabled and self.autoencoder_training_step % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
                wandb.log({
                    "reconstruction_loss": reconstruction_loss
                })
            
            reconstruction_threshold = AdaptiveWorldModelConfig.RECONSTRUCTION_THRESHOLD
            if reconstruction_loss > reconstruction_threshold:
                # Train autoencoder if reconstruction is poor
                train_loss = self.train_autoencoder(current_frame)

                # Show current and reconstructed frames while training (periodically)
                # if self.autoencoder_training_step % AdaptiveWorldModelConfig.DISPLAY_TRAINING_INTERVAL == 0:
                #     self.display_reconstruction_training(current_frame, decoded_frame, reconstruction_loss)
                self.autoencoder_training_step += 1
                self.consecutive_autoencoder_iterations += 1


                continue  # Skip action execution until reconstruction improves
            
            # Step 3: Train predictors until fresh predictions meet threshold
            # Log consecutive autoencoder training iterations if any occurred
            if self.consecutive_autoencoder_iterations > 0:
                if self.wandb_enabled:
                    wandb.log({"consecutive_autoencoder_iterations": self.consecutive_autoencoder_iterations})
                self.consecutive_autoencoder_iterations = 0  # Reset counter

            prediction_errors = []
            if self.prediction_buffer:
                # Get prediction context for fresh predictions
                history_features, history_actions = self.get_prediction_context()

                predictor_training_iterations = 0
                while True:  # Keep training until all fresh prediction errors are below threshold
                    predictor_training_iterations += 1
                    # Make fresh predictions with current context
                    fresh_predictions = []
                    for predictor in self.predictors:
                        with torch.no_grad():
                            prediction = predictor.forward(history_features, history_actions)
                            fresh_predictions.append(prediction)

                    # Evaluate fresh predictions using same loss as training
                    prediction_errors = self.evaluate_fresh_predictions(fresh_predictions, frame_tensor)

                    # Log prediction errors to wandb (periodically)
                    if self.wandb_enabled and self.predictor_training_step % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
                        log_dict = {}

                        # Log individual predictor errors with grouping
                        for level, error in enumerate(prediction_errors):
                            log_dict[f"prediction_errors/level_{level}"] = error

                        wandb.log(log_dict)

                    prediction_threshold = AdaptiveWorldModelConfig.PREDICTION_THRESHOLD
                    if all(error < prediction_threshold for error in prediction_errors):
                        # All fresh predictions accurate - ready to proceed
                        if self.wandb_enabled:
                            wandb.log({"predictor_training_iterations": predictor_training_iterations})
                        break
                    else:
                        # Train predictors that had errors using the same predictions we evaluated
                        for level, error in enumerate(prediction_errors):
                            if error > prediction_threshold:
                                self.train_predictor(
                                    level,
                                    frame_tensor,
                                    fresh_predictions[level],
                                    history_features,
                                    history_actions
                                )

                # Adjust lookahead based on accuracy horizon
                self.lookahead = self.find_accurate_prediction_horizon()
            
            # Step 4: Select action based on uncertainty maximization (or recorded action in replay mode)
            best_action, all_action_predictions = self.action_selector(current_features)

            # Display frames for visual feedback (on interval, or always in interactive mode)
            self.display_counter += 1
            should_display = (self.display_counter % AdaptiveWorldModelConfig.DISPLAY_INTERVAL == 0) or self.interactive
            if should_display:
                try:
                    self.display_frames(current_frame, decoded_frame, all_action_predictions, prediction_errors, reconstruction_loss)
                except Exception as e:
                    print(f"Warning: Could not display frames: {e}", flush=True)
            
            # Interactive mode: show information and get user input
            if self.interactive:
                action_to_execute = self.interactive_prompt(
                    current_frame, decoded_frame, reconstruction_loss, 
                    prediction_errors, best_action, all_action_predictions
                )
            else:
                action_to_execute = best_action
            
            # Step 5: Take action and make predictions using the interface (only if reconstruction is good)
            current_time = time.time()
            
            # Calculate time between actions and accumulate statistics
            if self.last_action_time is not None:
                time_between_actions = current_time - self.last_action_time
                self.action_time_intervals.append(time_between_actions)

                # Log action timing statistics periodically
                if self.wandb_enabled and self.action_count % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
                    if self.action_time_intervals:
                        wandb.log({
                            "action_timing/mean_interval": np.mean(self.action_time_intervals),
                            "action_timing/median_interval": np.median(self.action_time_intervals),
                            "action_timing/min_interval": np.min(self.action_time_intervals),
                            "action_timing/max_interval": np.max(self.action_time_intervals),
                            "action_timing/std_interval": np.std(self.action_time_intervals),
                            "action_count": self.action_count,
                        })
                        # Clear intervals after logging
                        self.action_time_intervals = []
            
            self.robot.execute_action(action_to_execute)
            self.last_action_time = current_time
            self.action_count += 1
            self.make_predictions(current_features, action_to_execute)
            
            # Add delay between actions
            time.sleep(AdaptiveWorldModelConfig.ACTION_DELAY)
            
            # Step 6: Upload visualizations periodically
            if self.wandb_enabled and self.action_count % AdaptiveWorldModelConfig.VISUALIZATION_UPLOAD_INTERVAL == 0:
                self.upload_visualizations_to_wandb(current_frame, decoded_frame, all_action_predictions, prediction_errors, reconstruction_loss)
            
            # Step 7: Update history buffers
            self.frame_features_history.append(current_features.detach())
            self.action_history.append(action_to_execute)
            self.maintain_history_window()
    
    def interactive_prompt(self, current_frame, decoded_frame, reconstruction_loss, 
                          prediction_errors, best_action, all_action_predictions):
        """Interactive mode: display info and get user input"""
        print("\n" + "="*60, flush=True)
        print("INTERACTIVE MODE", flush=True)
        print("="*60, flush=True)
        
        # Print metrics first
        print(f"Reconstruction Loss: {reconstruction_loss:.4f}", flush=True)
        if prediction_errors:
            print(f"Prediction Errors: {[f'{e:.4f}' for e in prediction_errors]}", flush=True)
        else:
            print("Prediction Errors: None (no previous predictions)", flush=True)
        print(f"Proposed Best Action: {best_action}", flush=True)
        print(f"Current Lookahead: {self.lookahead}", flush=True)
        print(f"History Size: {len(self.frame_features_history)}", flush=True)
        
        
        # Get user input
        while True:
            user_input = input("\nOptions:\n"
                             "1. Continue with proposed action (press Enter)\n"
                             "2. Replace action (type new action as dict, e.g., {'motor_1': 1})\n"
                             "3. Stop (type 'stop')\n"
                             "Choice: ").strip()
            
            if user_input == "":
                return best_action
            elif user_input.lower() == "stop":
                raise KeyboardInterrupt("Stopped by user")
            else:
                try:
                    # Try to parse as action dict
                    new_action = eval(user_input)
                    if isinstance(new_action, dict):
                        print(f"Using custom action: {new_action}")
                        return new_action
                    else:
                        print("Invalid action format. Please use dict format like {'motor_1': 1}")
                except:
                    print("Invalid input. Please try again.")
    
    def display_frames(self, current_frame, decoded_frame, all_action_predictions, prediction_errors=None, reconstruction_loss=None):
        """Display current frame, decoded frame, last prediction, and all action predictions"""
        # Calculate grid layout: current + decoded + last predicted (if available) + all action predictions
        total_predictions = sum(len(pred_data['predictions']) for pred_data in all_action_predictions)
        last_pred_cols = 1 if hasattr(self, 'last_predicted_frame') and self.last_predicted_frame is not None else 0
        num_cols = 2 + last_pred_cols + total_predictions  # current + decoded + last pred + predictions
        
        # Create figure on first call or if layout changed
        if self.fig is None or len(self.axes) != num_cols:
            if self.fig is not None:
                plt.close(self.fig)
            
            self.fig, self.axes, _ = self._create_prediction_visualization_figure(current_frame, decoded_frame, all_action_predictions, prediction_errors, reconstruction_loss)
            
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
        else:
            # Update existing figure using helper
            self._populate_prediction_visualization_axes(self.axes, current_frame, decoded_frame, all_action_predictions, prediction_errors, reconstruction_loss)
            plt.tight_layout()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to ensure rendering
    
    def _populate_prediction_visualization_axes(self, axes, current_frame, decoded_frame, all_action_predictions, prediction_errors=None, reconstruction_loss=None):
        """Helper function to populate axes with prediction visualization content"""
        # Clear axes
        for ax in axes:
            ax.clear()
            ax.axis('off')
        
        col_idx = 0
        
        # Current frame
        axes[col_idx].imshow(current_frame)
        axes[col_idx].set_title("Current Frame", fontsize=10)
        col_idx += 1
        
        # Decoded frame with reconstruction loss above it
        axes[col_idx].imshow(decoded_frame)
        if reconstruction_loss is not None:
            title = f"Decoded Frame\nReconstruction Loss: {reconstruction_loss:.4f}"
        else:
            title = "Decoded Frame"
        axes[col_idx].set_title(title, fontsize=10)
        col_idx += 1
        
        # Show last predicted frame with prediction error above it if available
        if hasattr(self, 'last_predicted_frame') and self.last_predicted_frame is not None and col_idx < len(axes):
            axes[col_idx].imshow(self.last_predicted_frame)
            action_str = str(getattr(self, 'last_action', 'Unknown'))
            
            # Calculate prediction error for display
            if prediction_errors and len(prediction_errors) > 0:
                avg_error = np.mean(prediction_errors)
                title = f"Last Prediction\n{action_str}\nPrediction Error: {avg_error:.4f}"
            else:
                title = f"Last Prediction\n{action_str}"
            
            axes[col_idx].set_title(title, fontsize=10)
            col_idx += 1
        
        # Predictions for each action
        for action_data in all_action_predictions:
            action = action_data['action']
            uncertainty = action_data['uncertainty']
            predictions = action_data['predictions']
            
            for level, pred_data in enumerate(predictions):
                if col_idx < len(axes):
                    pred_frame = pred_data['frame']
                    predictor_level = pred_data['level']
                    
                    axes[col_idx].imshow(pred_frame)
                    
                    # Multi-line title with action, level, and uncertainty
                    title = f"Action: {action}\nLevel {predictor_level}\nUncertainty: {uncertainty:.3f}"
                    axes[col_idx].set_title(title, fontsize=8)
                    
                    # Add border color based on uncertainty (higher = more red)
                    border_color = plt.cm.Reds(min(uncertainty, 1.0))
                    for spine in axes[col_idx].spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2)
                    
                    col_idx += 1
    
    def _create_prediction_visualization_figure(self, current_frame, decoded_frame, all_action_predictions, prediction_errors=None, reconstruction_loss=None):
        """Helper function to create a figure with current frame, decoded frame, and predictions"""
        # Calculate grid layout: current + decoded + last predicted (if available) + all action predictions
        total_predictions = sum(len(pred_data['predictions']) for pred_data in all_action_predictions)
        last_pred_cols = 1 if hasattr(self, 'last_predicted_frame') and self.last_predicted_frame is not None else 0
        num_cols = 2 + last_pred_cols + total_predictions  # current + decoded + last pred + predictions
        
        # Create figure
        fig, axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 4))
        
        if num_cols == 1:
            axes = [axes]
        
        # Populate axes using helper
        self._populate_prediction_visualization_axes(axes, current_frame, decoded_frame, all_action_predictions, prediction_errors, reconstruction_loss)
        
        plt.tight_layout()
        return fig, axes, num_cols
    
    def upload_visualizations_to_wandb(self, current_frame, decoded_frame, all_action_predictions, prediction_errors=None, reconstruction_loss=None):
        """Upload current visualizations to wandb for remote monitoring"""
        if not self.wandb_enabled:
            return
        
        try:
            # Create visualization figure using helper
            fig, axes, num_cols = self._create_prediction_visualization_figure(current_frame, decoded_frame, all_action_predictions, prediction_errors, reconstruction_loss)
            
            # Upload to wandb
            wandb.log({
                "predictions_visualization": wandb.Image(fig),
                "action_count": self.action_count,
            })
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not upload visualization to wandb: {e}")
    
    def display_reconstruction_training(self, current_frame, decoded_frame, reconstruction_loss):
        """Display current and reconstructed frames during autoencoder training"""
        # Create training figure if it doesn't exist
        if not hasattr(self, 'training_fig'):
            self.training_fig, self.training_axes = plt.subplots(1, 2, figsize=(8, 4))
            self.training_axes[0].set_title("Current Frame")
            self.training_axes[1].set_title("Reconstructed Frame")
            for ax in self.training_axes:
                ax.axis('off')
            plt.ion()
            plt.show(block=False)
        
        # Clear and update axes
        for ax in self.training_axes:
            ax.clear()
            ax.axis('off')
        
        # Display frames
        self.training_axes[0].imshow(current_frame)
        self.training_axes[0].set_title(f"Current Frame")
        
        self.training_axes[1].imshow(decoded_frame)
        self.training_axes[1].set_title(f"Reconstructed (Loss: {reconstruction_loss:.4f})")
        
        plt.tight_layout()
        self.training_fig.canvas.draw()
        self.training_fig.canvas.flush_events()
        plt.pause(0.01)
    
    def select_action_by_uncertainty(self, current_features):
        """Choose action with most uncertain outcome (highest entropy)"""
        max_uncertainty = -float('inf')
        best_action = None
        all_action_predictions = []
        
        # Get relevant history for prediction
        history_features, history_actions = self.get_prediction_context()
        
        with torch.no_grad():  # No gradients needed for action scoring and frame previews
            for action in self.get_available_actions():
                # Predict next states for this action
                predictions = []
                prediction_frames = []
                
                for predictor in self.predictors:
                    next_state = predictor.forward(
                        history_features, 
                        history_actions + [action]
                    )
                    predictions.append(next_state)
                    
                    # Generate frame for this prediction
                    # Convert features to tensor and use forward_decoder
                    if isinstance(next_state, np.ndarray):
                        features_tensor = torch.from_numpy(next_state).unsqueeze(0).float().to(self.device)
                    else:
                        features_tensor = next_state.unsqueeze(0).to(self.device) if len(next_state.shape) == 1 else next_state.to(self.device)
                    
                    # Create identity ids_restore for unmasked features
                    B, seq_len, _ = features_tensor.shape
                    L = seq_len - 1  # Remove CLS token count
                    ids_restore = torch.arange(L, device=features_tensor.device).unsqueeze(0).repeat(B, 1)
                    
                    # Use forward_decoder to get patches then unpatchify
                    pred_patches = self.autoencoder.forward_decoder(features_tensor, ids_restore)
                    decoded_tensor = self.autoencoder.unpatchify(pred_patches)
                    pred_frame = decoded_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    pred_frame = np.clip(pred_frame, 0, 1)  # Clip to valid range for display
                    prediction_frames.append({
                        'frame': pred_frame,
                        'level': predictor.level,
                        'features': next_state
                    })
                
                # Calculate uncertainty (entropy) over predictions
                uncertainty = calculate_entropy(predictions)
                
                # Store all prediction data for this action
                all_action_predictions.append({
                    'action': action,
                    'uncertainty': uncertainty,
                    'predictions': prediction_frames
                })
                
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    best_action = action
        
        return best_action, all_action_predictions

    def _calculate_predictor_metrics(self, predictor, pred_patches, target_patches, weights_before=None):
        """Calculate detailed training metrics for predictor including UWR"""
        metrics = {}

        # Calculate explained variance (R²) in patch space - scale-invariant predictor quality metric
        with torch.no_grad():
            resid = pred_patches - target_patches
            var_y = target_patches.var(unbiased=False)
            var_resid = resid.var(unbiased=False)
            explained_variance = 1.0 - (var_resid / (var_y + 1e-12))  # R^2 in [−∞, 1]

        # Detailed per-layer gradient norms for transformer layers
        transformer_layer_grad_stats = {}
        other_layer_grad_stats = {}

        for name, param in predictor.named_parameters():
            if param.grad is not None:
                grad_norms = param.grad.data.abs().flatten()

                # Parse layer name for more detailed tracking
                if 'transformer_layers' in name:
                    # Extract layer index for transformer layers (e.g., 'transformer_layers.0.self_attn.in_proj_weight')
                    parts = name.split('.')
                    if len(parts) >= 3:
                        layer_idx = parts[1]  # Get layer index
                        sublayer = '.'.join(parts[2:4])  # Get sublayer type (e.g., 'self_attn.in_proj_weight')
                        layer_key = f"transformer_layer_{layer_idx}_{sublayer}"
                    else:
                        layer_key = f"transformer_layer_{parts[1] if len(parts) > 1 else 'unknown'}"

                    if layer_key not in transformer_layer_grad_stats:
                        transformer_layer_grad_stats[layer_key] = []
                    transformer_layer_grad_stats[layer_key].extend(grad_norms.tolist())
                else:
                    # Group other layers by base name
                    layer_name = name.split('.')[0]
                    if layer_name not in other_layer_grad_stats:
                        other_layer_grad_stats[layer_name] = []
                    other_layer_grad_stats[layer_name].extend(grad_norms.tolist())

        # Process transformer layers
        for layer_key, grad_values in transformer_layer_grad_stats.items():
            if grad_values:
                grad_tensor = torch.tensor(grad_values)
                metrics[f"grad_norms/transformer/{layer_key}"] = grad_tensor.median().item()

        # Process other layers
        for layer_name, grad_values in other_layer_grad_stats.items():
            if grad_values:
                grad_tensor = torch.tensor(grad_values)
                metrics[f"grad_norms/{layer_name}"] = grad_tensor.median().item()

        # Calculate UWR metrics if weights_before provided
        uwr_95th = 0.0
        if weights_before:
            transformer_layer_uwrs = {}
            other_layer_uwrs = {}
            all_param_uwrs = []

            for name, param in predictor.named_parameters():
                if name in weights_before:
                    weight_before = weights_before[name]
                    actual_update = torch.abs(param.data - weight_before)
                    weight_magnitude = torch.abs(weight_before) + 1e-8
                    param_uwr = actual_update / weight_magnitude
                    param_uwr_values = param_uwr.flatten().tolist()
                    all_param_uwrs.extend(param_uwr_values)

                    # Group by layer type for detailed tracking
                    if 'transformer_layers' in name:
                        parts = name.split('.')
                        if len(parts) >= 3:
                            layer_idx = parts[1]
                            sublayer = '.'.join(parts[2:4])
                            layer_key = f"transformer_layer_{layer_idx}_{sublayer}"
                        else:
                            layer_key = f"transformer_layer_{parts[1] if len(parts) > 1 else 'unknown'}"

                        if layer_key not in transformer_layer_uwrs:
                            transformer_layer_uwrs[layer_key] = []
                        transformer_layer_uwrs[layer_key].extend(param_uwr_values)
                    else:
                        layer_name = name.split('.')[0]
                        if layer_name not in other_layer_uwrs:
                            other_layer_uwrs[layer_name] = []
                        other_layer_uwrs[layer_name].extend(param_uwr_values)

            # Process transformer layer UWRs
            for layer_key, uwr_values in transformer_layer_uwrs.items():
                if uwr_values:
                    uwr_tensor = torch.tensor(uwr_values)
                    metrics[f"uwr/transformer/{layer_key}"] = uwr_tensor.median().item()

            # Process other layer UWRs
            for layer_name, uwr_values in other_layer_uwrs.items():
                if uwr_values:
                    uwr_tensor = torch.tensor(uwr_values)
                    metrics[f"uwr/{layer_name}"] = uwr_tensor.median().item()

            # Calculate global UWR 95th percentile
            if all_param_uwrs:
                param_uwrs_tensor = torch.tensor(all_param_uwrs)
                uwr_95th = torch.quantile(param_uwrs_tensor, 0.95).item()

        # Store key metrics for later use
        metrics['explained_variance'] = explained_variance
        metrics['uwr_95th'] = uwr_95th

        return metrics

    def train_predictor(self, level, current_frame_tensor, predicted_features=None, history_features=None, history_actions=None):
        """Train neural predictor at specified level (gradients flow to autoencoder automatically)."""
        predictor = self.predictors[level]

        # Initialize predictor optimizer if not exists
        if not hasattr(predictor, 'optimizer'):
            predictor.optimizer = torch.optim.Adam(predictor.parameters(), lr=self._get_predictor_lr())

        # Initialize autoencoder optimizer for joint training
        if not hasattr(self, 'autoencoder_optimizer'):
            self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self._get_autoencoder_lr())

        # Ensure frame tensor lives on the correct device
        current_frame_tensor = current_frame_tensor.to(self.device)

        # Determine prediction context and recompute predictions with gradients
        if predicted_features is not None:
            if history_features is None or history_actions is None:
                raise ValueError("history_features and history_actions are required when providing predicted_features")
            history_features_for_forward = history_features
            history_actions_for_forward = history_actions
        else:
            history_features_for_forward, history_actions_for_forward = self.get_prediction_context()

        history_features_for_forward = [feat.detach().to(self.device) for feat in (history_features_for_forward or [])]
        history_actions_for_forward = list(history_actions_for_forward or [])

        predicted_features = predictor.forward(history_features_for_forward, history_actions_for_forward)

        # Initialize target patches from actual frame (same as evaluation)
        target_patches = self.patchify(current_frame_tensor)

        # Zero gradients for both predictor and autoencoder
        predictor.optimizer.zero_grad()
        self.autoencoder_optimizer.zero_grad()

        # Convert predicted features back to patches using decoder (same as evaluation)
        B, seq_len, _ = predicted_features.shape
        L = seq_len - 1  # Remove CLS token count
        ids_restore = torch.arange(L, device=predicted_features.device).unsqueeze(0).repeat(B, 1)

        # Use decoder to convert features to patches
        pred_patches = self.autoencoder.forward_decoder(predicted_features, ids_restore)

        # Calculate patch-space loss
        loss_patch = torch.nn.functional.mse_loss(pred_patches, target_patches)

        # Calculate latent-space loss (grad flows into encoder to make it prediction-friendly)
        target_latent = self.autoencoder.encode(current_frame_tensor)  # No stop-grad!
        loss_latent = torch.nn.functional.mse_loss(
            predicted_features[:, 1:, :],  # Exclude CLS token
            target_latent[:, 1:, :]        # Exclude CLS token
        )

        # Combine losses with weights
        w_patch = AdaptiveWorldModelConfig.PRED_PATCH_W
        w_latent = AdaptiveWorldModelConfig.PRED_LATENT_W
        prediction_loss = w_patch * loss_patch + w_latent * loss_latent

        # Backward pass - this will train both predictor and autoencoder!
        prediction_loss.backward()

        # Log gradient norms (global and per-layer) - throttled
        grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), float('inf'))

        # Capture weights before step for UWR calculation (throttled)
        weights_before = {}
        if self.predictor_training_step % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
            for name, param in predictor.named_parameters():
                if param.grad is not None:
                    weights_before[name] = param.data.clone()

        predictor.optimizer.step()
        self.autoencoder_optimizer.step()
        # Calculate detailed metrics periodically to reduce overhead
        if self.predictor_training_step % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
            # Calculate all metrics together
            detailed_metrics = self._calculate_predictor_metrics(predictor, pred_patches, target_patches, weights_before)
            # Log predictor training metrics to wandb
            if self.wandb_enabled:
                log_dict = {
                    "predictor_training_loss": prediction_loss.item(),
                    "predictor_patch_loss": loss_patch.item(),
                    "predictor_latent_loss": loss_latent.item(),
                    "predictor_explained_variance": detailed_metrics['explained_variance'].item(),
                    "predictor_training_step": self.predictor_training_step,
                    "predictor_grad_norm": grad_norm,
                    "predictor_lr": predictor.optimizer.param_groups[0]['lr'],
                    "predictor_uwr_95th": detailed_metrics['uwr_95th'],
                }
                # Add per-layer gradient and UWR statistics
                for key, value in detailed_metrics.items():
                    if key not in ['explained_variance', 'uwr_95th']:
                        log_dict[key] = value

                wandb.log(log_dict)
            
        # Increment predictor training step counter
        self.predictor_training_step += 1
        
        # Save checkpoint periodically based on predictor training steps
        if self.predictor_training_step % self.save_interval == 0:
            self.save_checkpoint()

        return prediction_loss.item()


    def train_autoencoder(self, ground_truth_frame):
        """Train only the autoencoder with masked reconstruction"""
        # Convert to properly scaled tensor
        frame_tensor = self.to_model_tensor(ground_truth_frame)
        
        # Initialize optimizer if not exists
        if not hasattr(self, 'autoencoder_optimizer'):
            self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self._get_autoencoder_lr())
        
        # Zero gradients
        self.autoencoder_optimizer.zero_grad()
        
        # Forward pass with masking for reconstruction loss (randomized mask ratio)
        mask_ratio = random.uniform(AdaptiveWorldModelConfig.MASK_RATIO_MIN, AdaptiveWorldModelConfig.MASK_RATIO_MAX)
        pred_patches, latent = self.autoencoder(frame_tensor, mask_ratio=mask_ratio)
        
        # Calculate reconstruction loss
        target_patches = self.patchify(frame_tensor)
        reconstruction_loss = torch.nn.functional.mse_loss(pred_patches, target_patches)
        
        # Backward pass and optimization
        reconstruction_loss.backward()
        self.autoencoder_optimizer.step()
        
        train_loss_value = reconstruction_loss.item()
        
        # Log training loss to wandb
        if self.wandb_enabled:
            wandb.log({
                "autoencoder_training_loss": train_loss_value,
                "mask_ratio": mask_ratio,
                "autoencoder_training_step": self.autoencoder_training_step
            })
        
        return train_loss_value

    def patchify(self, imgs):
        """Convert images to patches (helper for training)"""
        patch_size = int(self.autoencoder.patch_embed.patch_size[0])
        B, C, H, W = imgs.shape
        h = H // patch_size
        w = W // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, patch_size**2 * C)
        return x
    
    def create_new_hierarchy_level(self):
        """Create new predictor for higher-level actions"""
        # Create action encoder for current action sequences
        action_encoder = ActionEncoder(
            sequence_length=self.lookahead,
            input_actions=self.get_current_level_actions()
        )
        action_decoder = ActionDecoder()
        
        # Train on recent action sequences
        recent_sequences = self.extract_action_sequences(self.lookahead)
        action_encoder.train(recent_sequences)
        
        # Create new predictor for encoded actions
        new_predictor = ActionConditionedPredictor(
            level=len(self.predictors),
            action_space=action_encoder.latent_space
        )
        
        # Add to hierarchy
        self.predictors.append(new_predictor)
        self.action_encoders.append(action_encoder)
        self.action_decoders.append(action_decoder)
        
        print(f"Created hierarchy level {len(self.predictors)}")
    
    def get_available_actions(self):
        """Get actions at appropriate abstraction level"""
        if not self.action_encoders:
            return self.base_actions
        
        # Could return mix of low and high level actions
        # or just highest level based on current lookahead
        current_level = min(self.lookahead // self.max_lookahead_margin, 
                          len(self.action_encoders))
        
        if current_level == 0:
            return self.base_actions
        else:
            return self.action_encoders[current_level-1].get_abstract_actions()
    
    def get_prediction_context(self):
        """Get appropriate history window for prediction"""
        # Need enough history for lookahead + context
        context_size = self.prediction_history_size + self.lookahead
        
        # Get last context_size frames and actions
        start_idx = max(0, len(self.frame_features_history) - context_size)
        
        features = self.frame_features_history[start_idx:]
        actions = self.action_history[start_idx:]
        
        return features, actions
    
    def find_accurate_prediction_horizon(self):
        """Find how far ahead predictions remain accurate"""
        threshold = 0.5
        for horizon in range(1, self.lookahead + 1):
            if horizon-1 < len(self.prediction_buffer) and self.prediction_buffer[horizon-1].get('error', 0) > threshold:
                return max(1, horizon - 1)
        return self.lookahead
    
    def maintain_history_window(self):
        """Keep history buffers at reasonable size"""
        max_history = AdaptiveWorldModelConfig.MAX_HISTORY_SIZE
        
        if len(self.frame_features_history) > max_history:
            self.frame_features_history = self.frame_features_history[-max_history:]
            self.action_history = self.action_history[-max_history:]
    
    def make_predictions(self, current_features, action):
        """Generate predictions for future timesteps"""
        self.prediction_buffer = []
        
        history_features, history_actions = self.get_prediction_context()
        history_actions.append(action)
        
        # Store the predicted frame for the first predictor (level 0) for visualization
        self.last_predicted_frame = None
        self.last_action = action
        
        for level, predictor in enumerate(self.predictors):
            prediction = predictor.forward(
                history_features,
                history_actions
            ).detach()
            
            # Generate predicted frame for visualization (store the first predictor's prediction)
            if level == 0:
                with torch.no_grad():
                    # Convert features to tensor and use forward_decoder
                    if isinstance(prediction, np.ndarray):
                        features_tensor = torch.from_numpy(prediction).unsqueeze(0).float().to(self.device)
                    else:
                        features_tensor = prediction.unsqueeze(0).to(self.device) if len(prediction.shape) == 1 else prediction.to(self.device)
                    
                    # Create identity ids_restore for unmasked features
                    B, seq_len, _ = features_tensor.shape
                    L = seq_len - 1  # Remove CLS token count
                    ids_restore = torch.arange(L, device=features_tensor.device).unsqueeze(0).repeat(B, 1)
                    
                    # Use forward_decoder to get patches then unpatchify
                    pred_patches = self.autoencoder.forward_decoder(features_tensor, ids_restore)
                    decoded_tensor = self.autoencoder.unpatchify(pred_patches)
                    pred_frame = decoded_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    self.last_predicted_frame = np.clip(pred_frame, 0, 1)  # Clip to valid range for display
            
            self.prediction_buffer.append({
                'prediction': prediction,
                'level': predictor.level,
                'timestamp': current_time()
            })
    
    def _compute_prediction_loss(self, predicted_features, actual_frame_tensor):
        """Compute prediction loss using same method as training (patch + latent loss)"""
        with torch.no_grad():
            # Calculate target patches and latent features
            target_patches = self.patchify(actual_frame_tensor)
            target_latent = self.autoencoder.encode(actual_frame_tensor)

            # Convert predicted features to patches using decoder
            B, seq_len, _ = predicted_features.shape
            L = seq_len - 1  # Remove CLS token count
            ids_restore = torch.arange(L, device=predicted_features.device).unsqueeze(0).repeat(B, 1)
            pred_patches = self.autoencoder.forward_decoder(predicted_features, ids_restore)

            # Calculate patch-space loss
            loss_patch = torch.nn.functional.mse_loss(pred_patches, target_patches)

            # Calculate latent-space loss
            loss_latent = torch.nn.functional.mse_loss(
                predicted_features[:, 1:, :],  # Exclude CLS token
                target_latent[:, 1:, :]        # Exclude CLS token
            )

            # Combine losses with same weights as training
            w_patch = AdaptiveWorldModelConfig.PRED_PATCH_W
            w_latent = AdaptiveWorldModelConfig.PRED_LATENT_W
            total_loss = w_patch * loss_patch + w_latent * loss_latent

            return total_loss.item()

    def evaluate_predictions(self, actual_frame_tensor):
        """Compare past predictions with actual outcome using same loss as training"""
        errors = []

        for pred_data in self.prediction_buffer:
            predicted_features = pred_data['prediction']
            error = self._compute_prediction_loss(predicted_features, actual_frame_tensor)
            errors.append(error)

        return errors

    def evaluate_fresh_predictions(self, fresh_predictions, actual_frame_tensor):
        """Evaluate freshly made predictions using same loss as training"""
        errors = []

        for predicted_features in fresh_predictions:
            error = self._compute_prediction_loss(predicted_features, actual_frame_tensor)
            errors.append(error)

        return errors
    
    def get_max_predictor_lookahead(self):
        """Get maximum lookahead capability of current predictors"""
        return max([p.max_lookahead for p in self.predictors])
    
    def get_current_level_actions(self):
        """Get actions from current abstraction level"""
        return self.base_actions
    
    def extract_action_sequences(self, length):
        """Extract action sequences of given length from history"""
        sequences = []
        for i in range(len(self.action_history) - length + 1):
            sequences.append(self.action_history[i:i+length])
        return sequences


# Supporting functions and classes (stub implementations)


class ActionConditionedPredictor:
    """Predicts future states given past states and actions"""
    def __init__(self, level, action_space=None): 
        self.level = level
        self.action_space = action_space
        self.max_lookahead = 10  # Default max lookahead
    
    def forward(self, feature_history, action_history):
        # Stub forward method for neural network interface
        return torch.randn(256).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Match autoencoder embed_dim
    
    def parameters(self):
        # Stub parameters method for optimizer
        return []

class ActionEncoder:
    """Encodes sequences of actions into abstract actions"""
    def __init__(self, sequence_length, input_actions): 
        self.sequence_length = sequence_length
        self.input_actions = input_actions
        self.latent_space = ['abstract_action_1', 'abstract_action_2']
    
    def train(self, action_sequences): 
        # Stub training
        pass
    
    def encode(self, action_sequence): 
        # Return dummy encoded action
        return 'encoded_action'
    
    def get_abstract_actions(self): 
        return self.latent_space

class ActionDecoder:
    """Decodes abstract actions back to primitive sequences"""
    def decode(self, abstract_action): 
        # Return dummy action sequence
        return [{'motor_1': 0}, {'motor_2': 0}]

def calculate_entropy(predictions):
    """Calculate uncertainty/entropy over predicted distributions"""
    if not predictions:
        return 0.0
    
    # Stack predictions and calculate variance on GPU
    if predictions and hasattr(predictions[0], 'device'):
        # PyTorch tensors - keep on GPU
        predictions_tensor = torch.stack([p.flatten() for p in predictions])
        return torch.var(predictions_tensor).item()
    else:
        # Fallback for numpy arrays
        predictions_array = np.array([p.flatten() if hasattr(p, 'flatten') else [p] for p in predictions])
        return np.var(predictions_array)

def calculate_loss(predicted, actual):
    """Compute prediction error"""
    import numpy as np, torch
    if isinstance(predicted, torch.Tensor): 
        predicted = predicted.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor): 
        actual = actual.detach().cpu().numpy()
    if isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray):
        return float(np.mean((predicted - actual) ** 2))
    return 0.0  # Return 0 for non-compatible types rather than random

# Robot interface functions removed - now handled by RobotInterface

def create_masked_versions(image, num_masks=5):
    """Create multiple masked versions for training"""
    masked_versions = []
    for _ in range(num_masks):
        # Create a copy and mask random patches
        masked = image.copy()
        h, w = masked.shape[:2]
        mask_h, mask_w = h//4, w//4
        start_h, start_w = np.random.randint(0, h-mask_h), np.random.randint(0, w-mask_w)
        masked[start_h:start_h+mask_h, start_w:start_w+mask_w] = 0
        masked_versions.append(masked)
    return masked_versions

def current_time():
    """Get current timestamp"""
    return time.time()

# Example usage with stub robot for testing
class StubRobot:
    """Stub robot implementation for testing the world model"""
    def __init__(self):
        motor_values = [-0.15, 0, 0.15]
        duration = 0.1  # Fixed duration in seconds
        self.action_space = []
        for left in motor_values:
            for right in motor_values:
                self.action_space.append({
                    'motor_left': left, 
                    'motor_right': right,
                    'duration': duration
                })
    
    def get_observation(self):
        # Return dummy 224x224x3 image with some pattern
        image = np.random.rand(224, 224, 3)
        # Add some visual pattern for better visualization
        image[50:150, 50:150] = [0.8, 0.2, 0.2]  # Red square
        return image
    
    def execute_action(self, action):
        print(f"Executing action: {action}")
        time.sleep(0.1)  # Simulate action duration
        return True
    
    def cleanup(self):
        pass

if __name__ == "__main__":
    # Interactive mode with stub robot
    stub_robot = StubRobot()
    # Enable wandb logging by providing a project name, or set to None to disable
    model = AdaptiveWorldModel(stub_robot, interactive=True, wandb_project="adaptive-world-model-test")
    try:
        model.main_loop()
    except KeyboardInterrupt:
        print("\nStopped by user")
        stub_robot.cleanup()
        # Clean up wandb run
        if model.wandb_enabled:
            wandb.finish()