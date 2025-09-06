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
from models import MaskedAutoencoderViT, TransformerActionConditionedPredictor
from config import AdaptiveWorldModelConfig

class AdaptiveWorldModel:
    def __init__(self, robot_interface, interactive=False, wandb_project=None, checkpoint_dir="checkpoints"):
        # Store the robot interface
        self.robot = robot_interface
        
        # Checkpoint management
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = AdaptiveWorldModelConfig.CHECKPOINT_SAVE_INTERVAL
        
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
                "log_interval": AdaptiveWorldModelConfig.LOG_INTERVAL,
                "visualization_upload_interval": AdaptiveWorldModelConfig.VISUALIZATION_UPLOAD_INTERVAL
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
        self.training_step = 0
        
        # Action timing tracking
        self.last_action_time = None
        self.action_count = 0
        
        # Load checkpoint if it exists
        self.load_checkpoint()
    
    def save_checkpoint(self):
        """Save current learning progress to disk"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
            'training_step': self.training_step,
            'predictor_training_step': getattr(self, 'predictor_training_step', 0),
            'action_count': self.action_count,
            'lookahead': self.lookahead,
            'frame_features_history': self.frame_features_history[-history_limit:] if self.frame_features_history else [],
            'action_history': self.action_history[-history_limit:] if self.action_history else [],
            'prediction_buffer': self.prediction_buffer,
            'last_action_time': self.last_action_time,
        }
        
        with open(os.path.join(self.checkpoint_dir, 'state.pkl'), 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Checkpoint saved at step {self.training_step}")
    
    def load_checkpoint(self):
        """Load learning progress from disk if checkpoint exists"""
        if not os.path.exists(self.checkpoint_dir):
            print("No checkpoint directory found, starting fresh")
            return
        
        # Load autoencoder
        autoencoder_path = os.path.join(self.checkpoint_dir, 'autoencoder.pth')
        if os.path.exists(autoencoder_path):
            checkpoint = torch.load(autoencoder_path, map_location=self.device)
            self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer if it was saved
            if checkpoint['optimizer_state_dict'] is not None:
                if not hasattr(self, 'autoencoder_optimizer'):
                    self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-4)
                self.autoencoder_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Autoencoder checkpoint loaded")
        
        # Load predictors
        predictor_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('predictor_') and f.endswith('.pth')]
        for predictor_file in sorted(predictor_files):
            predictor_path = os.path.join(self.checkpoint_dir, predictor_file)
            checkpoint = torch.load(predictor_path, map_location=self.device)
            
            # Find or create corresponding predictor
            predictor_idx = int(predictor_file.split('_')[1].split('.')[0])
            if predictor_idx < len(self.predictors):
                predictor = self.predictors[predictor_idx]
                if hasattr(predictor, 'load_state_dict') and checkpoint['model_state_dict']:
                    predictor.load_state_dict(checkpoint['model_state_dict'])
                
                # Restore optimizer if it was saved
                if checkpoint['optimizer_state_dict'] is not None:
                    if not hasattr(predictor, 'optimizer'):
                        predictor.optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
                    predictor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load learning progress and history
        state_path = os.path.join(self.checkpoint_dir, 'state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            self.training_step = state.get('training_step', 0)
            self.predictor_training_step = state.get('predictor_training_step', 0)
            self.action_count = state.get('action_count', 0)
            self.lookahead = state.get('lookahead', 1)
            self.frame_features_history = state.get('frame_features_history', [])
            self.action_history = state.get('action_history', [])
            self.prediction_buffer = state.get('prediction_buffer', [])
            self.last_action_time = state.get('last_action_time', None)
            
            print(f"Learning progress loaded: {self.training_step} training steps, {self.action_count} actions")
    
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
            if self.wandb_enabled and self.training_step % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
                wandb.log({
                    "reconstruction_loss": reconstruction_loss,
                    "step": self.training_step
                })
            
            threshold = AdaptiveWorldModelConfig.RECONSTRUCTION_THRESHOLD
            if reconstruction_loss > threshold:
                # Train autoencoder if reconstruction is poor
                train_loss = self.train_autoencoder(current_frame)
                
                # Show current and reconstructed frames while training (periodically)
                if self.training_step % AdaptiveWorldModelConfig.DISPLAY_TRAINING_INTERVAL == 0:
                    self.display_reconstruction_training(current_frame, decoded_frame, reconstruction_loss)
                self.training_step += 1
                
                
                continue  # Skip action execution until reconstruction improves
            
            # Step 3: Check past predictions accuracy (if any)
            prediction_errors = []
            if self.prediction_buffer:
                prediction_errors = self.evaluate_predictions(frame_tensor)
                
                if all(error < threshold for error in prediction_errors):
                    # All predictions accurate - increase lookahead
                    ## comment out for now
                    print("Low prediction error, Ready to create higher level abstraction")
                    # self.lookahead += 1
                    
                    # # Check if we need a new hierarchical level
                    # if self.lookahead > self.get_max_predictor_lookahead() + self.max_lookahead_margin:
                    #     self.create_new_hierarchy_level()
                else:
                    # Train predictors that had errors
                    for level, error in enumerate(prediction_errors):
                        if error > threshold:
                            self.train_predictor(level, frame_tensor)
                    
                    # Adjust lookahead based on accuracy horizon
                    self.lookahead = self.find_accurate_prediction_horizon()
            
            # Step 4: Select action based on uncertainty maximization
            best_action, all_action_predictions = self.select_action_by_uncertainty(current_features)
            
            # Display frames for visual feedback (always)
            try:
                self.display_frames(current_frame, decoded_frame, all_action_predictions)
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
            
            # Calculate time between actions and log to wandb (periodically)
            if self.last_action_time is not None:
                time_between_actions = current_time - self.last_action_time
                if self.wandb_enabled and self.action_count % AdaptiveWorldModelConfig.LOG_INTERVAL == 0:
                    wandb.log({
                        "time_between_actions": time_between_actions,
                        "action_count": self.action_count,
                        "step": self.training_step
                    })
            
            self.robot.execute_action(action_to_execute)
            self.last_action_time = current_time
            self.action_count += 1
            self.make_predictions(current_features, action_to_execute)
            
            # Step 6: Upload visualizations periodically
            if self.wandb_enabled and self.action_count % AdaptiveWorldModelConfig.VISUALIZATION_UPLOAD_INTERVAL == 0:
                self.upload_visualizations_to_wandb(current_frame, decoded_frame, all_action_predictions)
            
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
    
    def display_frames(self, current_frame, decoded_frame, all_action_predictions):
        """Display current frame, decoded frame, and predictions for all actions"""
        # Calculate grid layout: current + decoded + all action predictions
        total_predictions = sum(len(pred_data['predictions']) for pred_data in all_action_predictions)
        num_cols = 2 + total_predictions  # current + decoded + all predictions
        
        # Create figure on first call or if layout changed
        if self.fig is None or len(self.axes) != num_cols:
            if self.fig is not None:
                plt.close(self.fig)
            
            self.fig, self.axes, _ = self._create_prediction_visualization_figure(current_frame, decoded_frame, all_action_predictions)
            
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
        else:
            # Update existing figure using helper
            self._populate_prediction_visualization_axes(self.axes, current_frame, decoded_frame, all_action_predictions)
            plt.tight_layout()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to ensure rendering
    
    def _populate_prediction_visualization_axes(self, axes, current_frame, decoded_frame, all_action_predictions):
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
        
        # Decoded frame
        axes[col_idx].imshow(decoded_frame)
        axes[col_idx].set_title("Decoded Frame", fontsize=10)
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
    
    def _create_prediction_visualization_figure(self, current_frame, decoded_frame, all_action_predictions):
        """Helper function to create a figure with current frame, decoded frame, and predictions"""
        # Calculate grid layout: current + decoded + all action predictions
        total_predictions = sum(len(pred_data['predictions']) for pred_data in all_action_predictions)
        num_cols = 2 + total_predictions  # current + decoded + all predictions
        
        # Create figure
        fig, axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 4))
        
        if num_cols == 1:
            axes = [axes]
        
        # Populate axes using helper
        self._populate_prediction_visualization_axes(axes, current_frame, decoded_frame, all_action_predictions)
        
        plt.tight_layout()
        return fig, axes, num_cols
    
    def upload_visualizations_to_wandb(self, current_frame, decoded_frame, all_action_predictions):
        """Upload current visualizations to wandb for remote monitoring"""
        if not self.wandb_enabled:
            return
        
        try:
            # Create visualization figure using helper
            fig, axes, num_cols = self._create_prediction_visualization_figure(current_frame, decoded_frame, all_action_predictions)
            
            # Upload to wandb
            wandb.log({
                "predictions_visualization": wandb.Image(fig),
                "action_count": self.action_count,
                "step": self.training_step
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
    
    def train_predictor(self, level, current_frame_tensor):
        """Train neural predictor at specified level (gradients flow to autoencoder automatically)"""
        predictor = self.predictors[level]
        
        # Initialize predictor optimizer if not exists
        if not hasattr(predictor, 'optimizer'):
            predictor.optimizer = torch.optim.Adam(predictor.parameters(), lr=AdaptiveWorldModelConfig.PREDICTOR_LR)
        
        # Initialize autoencoder optimizer for joint training
        if not hasattr(self, 'autoencoder_optimizer'):
            self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=AdaptiveWorldModelConfig.AUTOENCODER_LR)
        
        # Get appropriate history window
        history_features, history_actions = self.get_prediction_context()
        history_features = [f.detach() for f in history_features]  # Detach history to prevent backprop through old encoder graphs
        
        # Get fresh target features from current frame (this creates a new computational graph)
        actual_tensor, _ = self.autoencoder.forward_encoder(current_frame_tensor, mask_ratio=0.0)
        
        # Zero gradients for both predictor and autoencoder
        predictor.optimizer.zero_grad()
        self.autoencoder_optimizer.zero_grad()
        
        # Forward pass through predictor
        predicted_tensor = predictor.forward(history_features, history_actions)
        
        # Calculate prediction loss
        prediction_loss = torch.nn.functional.mse_loss(predicted_tensor, actual_tensor)
        
        # Backward pass - this will train both predictor and autoencoder!
        prediction_loss.backward()
        predictor.optimizer.step()
        self.autoencoder_optimizer.step()
                
        # Log predictor training metrics to wandb
        if self.wandb_enabled:
            wandb.log({
                "predictor_training_loss": prediction_loss.item(),
                "predictor_level": level,
                "predictor_training_step": getattr(self, 'predictor_training_step', 0),
                "step": self.training_step
            })
            
        # Increment predictor training step counter
        self.predictor_training_step = getattr(self, 'predictor_training_step', 0) + 1
        
        # Save checkpoint periodically based on predictor training steps
        if self.predictor_training_step % self.save_interval == 0:
            self.save_checkpoint()
    
    def train_autoencoder(self, ground_truth_frame):
        """Train only the autoencoder with masked reconstruction"""
        # Convert to properly scaled tensor
        frame_tensor = self.to_model_tensor(ground_truth_frame)
        
        # Initialize optimizer if not exists
        if not hasattr(self, 'autoencoder_optimizer'):
            self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=AdaptiveWorldModelConfig.AUTOENCODER_LR)
        
        # Zero gradients
        self.autoencoder_optimizer.zero_grad()
        
        # Forward pass with masking for reconstruction loss
        pred_patches, latent = self.autoencoder(frame_tensor, mask_ratio=AdaptiveWorldModelConfig.MASK_RATIO)
        
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
                "training_step": self.training_step
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
        
        for predictor in self.predictors:
            prediction = predictor.forward(
                history_features,
                history_actions
            ).detach()
            self.prediction_buffer.append({
                'prediction': prediction,
                'level': predictor.level,
                'timestamp': current_time()
            })
    
    def evaluate_predictions(self, actual_frame_tensor):
        """Compare past predictions with actual outcome using patch space reconstruction loss"""
        errors = []
        
        # Calculate target patches from actual frame (same as main loop step 2)
        target_patches = self.patchify(actual_frame_tensor)
        
        for pred_data in self.prediction_buffer:
            predicted_features = pred_data['prediction']
            
            # Convert predicted features back to patches using decoder
            with torch.no_grad():
                # Create identity ids_restore since we're working with unmasked features
                B, seq_len, _ = predicted_features.shape
                L = seq_len - 1  # Remove CLS token count
                ids_restore = torch.arange(L, device=predicted_features.device).unsqueeze(0).repeat(B, 1)
                
                # Use decoder to convert features to patches
                pred_patches = self.autoencoder.forward_decoder(predicted_features, ids_restore)
                
                # Calculate reconstruction loss in patch space (same as training)
                reconstruction_loss = torch.nn.functional.mse_loss(pred_patches, target_patches).item()
                errors.append(reconstruction_loss)
            
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