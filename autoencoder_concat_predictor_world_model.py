"""
autoencoder_concat_predictor_world_model.py

World model using targeted masked autoencoder with canvas-based frame concatenation.
Builds canvases by concatenating history frames with action-colored separators between them.
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Callable, Tuple, Any

from robot_interface import RobotInterface
from models.autoencoder_concat_predictor import (
    TargetedMAEWrapper,
    TargetedDecoderOnlyWrapper,
    build_canvas,
    canvas_to_tensor,
    compute_patch_mask_for_last_slot,
    compute_randomized_patch_mask_for_last_slot,
)
from config import AutoencoderConcatPredictorWorldModelConfig as Config
import world_model_utils


class AutoencoderConcatPredictorWorldModel:
    """
    World model that uses targeted masked autoencoder with canvas-based prediction.

    Main loop:
    1. Get current frame and add to interleaved history
    2. Output reconstruction loss between last prediction and current frame
    3. Create training canvas from history ending in current frame
    4. Train autoencoder on canvas (one step)
    5. Select action and add to history
    6. Create prediction canvas with selected action and masked next frame
    7. Run reconstruction and store predicted next frame
    8. Execute action on robot
    9. Trim history to maintain size
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        action_selector: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        training_callback: Optional[Callable] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the autoencoder concat predictor world model.

        Args:
            robot_interface: RobotInterface implementation for robot interaction
            action_selector: Function that takes (observation, action_space) and returns (action, metadata)
                           If None, uses default random action selection
            device: torch.device for model computations (defaults to cuda if available)
            training_callback: Optional callback(iteration, loss) called after each training step
            frame_size: Optional (H, W) tuple for frame dimensions. If None, uses config default.
        """
        self.robot = robot_interface
        self.action_selector = action_selector or self._default_action_selector
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_callback = training_callback

        # Use provided frame_size or fall back to config
        if frame_size is not None:
            frame_height, frame_width = frame_size
        else:
            frame_height = Config.FRAME_SIZE[0]
            frame_width = Config.FRAME_SIZE[1]

        # Store the frame size used for this model instance
        self.frame_size = (frame_height, frame_width)

        # Initialize autoencoder model for canvas dimensions
        # Canvas: height=frame_height, width = CANVAS_HISTORY_SIZE * frame_width + (CANVAS_HISTORY_SIZE - 1) * SEPARATOR_WIDTH
        canvas_height = frame_height
        canvas_width = Config.CANVAS_HISTORY_SIZE * frame_width + (Config.CANVAS_HISTORY_SIZE - 1) * Config.SEPARATOR_WIDTH

        if Config.MODEL_TYPE == "decoder_only":
            self.autoencoder = TargetedDecoderOnlyWrapper(
                img_height=canvas_height,
                img_width=canvas_width,
                patch_size=Config.PATCH_SIZE,
                embed_dim=Config.EMBED_DIM,
                depth=Config.DECODER_ONLY_DEPTH,
                num_heads=Config.NUM_HEADS,
            ).to(self.device)
        else:
            self.autoencoder = TargetedMAEWrapper(
                img_height=canvas_height,
                img_width=canvas_width,
                patch_size=Config.PATCH_SIZE,
                embed_dim=Config.EMBED_DIM,
                depth=Config.ENCODER_DEPTH,
                num_heads=Config.NUM_HEADS,
                decoder_embed_dim=Config.DECODER_EMBED_DIM,
                decoder_depth=Config.DECODER_DEPTH,
                decoder_num_heads=Config.DECODER_NUM_HEADS,
            ).to(self.device)

        # Create optimizer with parameter groups for weight decay
        param_groups = world_model_utils.create_param_groups(
            self.autoencoder,
            Config.WEIGHT_DECAY
        )
        self.ae_optimizer = optim.AdamW(param_groups, lr=Config.AUTOENCODER_LR)

        # Create learning rate scheduler
        self.ae_scheduler = world_model_utils.create_warmup_cosine_scheduler(
            self.ae_optimizer,
            warmup_steps=Config.WARMUP_STEPS,
            lr_min_ratio=Config.LR_MIN_RATIO,
        )

        # Interleaved history: [frame0, action0, frame1, action1, frame2, ...]
        # Ensures frames always followed by actions
        self.interleaved_history = []

        # Store last prediction for comparison with current frame
        self.last_prediction = None

        # Visualization state tracking
        self.last_training_canvas = None
        self.last_training_mask = None  # Patch mask used during last training iteration
        self.last_training_loss = None
        self.last_training_iterations = 0
        self.last_training_loss_history = []  # List of losses during last training
        self.last_prediction_canvas = None
        self.last_grad_diagnostics = None  # Gradient diagnostics from last training step

        print(f"AutoencoderConcatPredictorWorldModel initialized on {self.device}")
        print(f"Canvas history size: {Config.CANVAS_HISTORY_SIZE}")
        print(f"Frame size: {self.frame_size}")
        print(f"Separator width: {Config.SEPARATOR_WIDTH}")

    def _default_action_selector(self, observation: np.ndarray, action_space: list) -> dict:
        """Default random action selection."""
        import random
        action = random.choice(action_space)
        return action

    def train_autoencoder(self, training_canvas: np.ndarray) -> float:
        """
        Train the autoencoder on a canvas using targeted randomized masking on the last frame.

        Args:
            training_canvas: HxWx3 canvas numpy array (uint8)

        Returns:
            loss_value: float, the loss value for this training step
        """
        # Convert canvas to tensor
        canvas_tensor = canvas_to_tensor(training_canvas).to(self.device)

        # Get canvas dimensions
        canvas_height, canvas_width = canvas_tensor.shape[-2:]

        # Compute number of frames in the canvas
        num_frames = Config.CANVAS_HISTORY_SIZE

        # Compute randomized patch mask for the last frame slot
        import config
        patch_mask = compute_randomized_patch_mask_for_last_slot(
            img_size=(canvas_height, canvas_width),
            patch_size=16,
            num_frame_slots=num_frames,
            sep_width=Config.SEPARATOR_WIDTH,
            mask_ratio_min=config.TRAIN_MASK_RATIO_MIN,
            mask_ratio_max=config.TRAIN_MASK_RATIO_MAX,
        ).to(self.device)

        # Store the mask for visualization (no extra computation cost)
        self.last_training_mask = patch_mask

        # Train with targeted masking on last frame
        self.autoencoder.train()
        loss_value, grad_diagnostics = self.autoencoder.train_on_canvas(canvas_tensor, patch_mask, self.ae_optimizer)
        self.last_grad_diagnostics = grad_diagnostics  # Store for visualization
        self.ae_scheduler.step()
        return loss_value

    def reset_state(self):
        """Reset the world model state for a new session."""
        self.interleaved_history = []
        self.last_prediction = None
        self.last_training_canvas = None
        self.last_training_mask = None
        self.last_training_loss = None
        self.last_training_iterations = 0
        self.last_training_loss_history = []
        self.last_prediction_canvas = None
        self.last_grad_diagnostics = None

    def get_canvas_with_mask_overlay(self, canvas: np.ndarray, patch_mask: torch.Tensor) -> np.ndarray:
        """
        Visualize which patches are masked by overlaying a semi-transparent color.

        Shows the original canvas with masked patches highlighted in red overlay.

        Args:
            canvas: HxWx3 canvas numpy array (uint8)
            patch_mask: [1, num_patches] boolean tensor; True = masked patches

        Returns:
            Canvas with red overlay on masked patches as HxWx3 uint8 array
        """
        # Start with original canvas
        canvas_overlay = canvas.copy().astype(np.float32)

        # Get canvas dimensions and patch size
        canvas_height, canvas_width = canvas.shape[:2]
        patch_size = int(getattr(self.autoencoder, "patch_size", 16))
        num_patches_height = canvas_height // patch_size
        num_patches_width = canvas_width // patch_size

        # Reshape patch mask to 2D grid: [1, num_patches] -> [num_patches_height, num_patches_width]
        mask_grid = patch_mask.cpu().view(num_patches_height, num_patches_width).numpy()

        # Create red overlay for masked patches
        for patch_row in range(num_patches_height):
            for patch_col in range(num_patches_width):
                if mask_grid[patch_row, patch_col]:
                    # Calculate pixel boundaries for this patch
                    y_start = patch_row * patch_size
                    y_end = y_start + patch_size
                    x_start = patch_col * patch_size
                    x_end = x_start + patch_size

                    # Apply semi-transparent red overlay (50% blend)
                    canvas_overlay[y_start:y_end, x_start:x_end, 0] = canvas_overlay[y_start:y_end, x_start:x_end, 0] * 0.5 + 255 * 0.5  # Red
                    canvas_overlay[y_start:y_end, x_start:x_end, 1] = canvas_overlay[y_start:y_end, x_start:x_end, 1] * 0.5  # Green
                    canvas_overlay[y_start:y_end, x_start:x_end, 2] = canvas_overlay[y_start:y_end, x_start:x_end, 2] * 0.5  # Blue

        return canvas_overlay.astype(np.uint8)

    def get_canvas_inpainting_full_output(self, canvas: np.ndarray, patch_mask: torch.Tensor) -> np.ndarray:
        """
        Get full model output when inpainting a canvas with specified patches masked.

        Returns what the model reconstructs for ALL patches (both masked and unmasked).
        Useful for debugging: shows if the model does anything weird with unmasked regions.

        Args:
            canvas: HxWx3 canvas numpy array (uint8)
            patch_mask: [1, num_patches] boolean tensor; True = masked patches to inpaint

        Returns:
            Full model output as HxWx3 uint8 array
        """
        # Convert canvas to tensor
        canvas_tensor = canvas_to_tensor(canvas).to(self.device)

        # Run forward pass with masking - model predicts all patches
        self.autoencoder.eval()
        with torch.no_grad():
            pred_patches, _ = self.autoencoder.forward_with_patch_mask(canvas_tensor, patch_mask)
            img_pred = self.autoencoder.unpatchify(pred_patches)  # [1, 3, H, W]
            full_output_np = (
                img_pred.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            ).astype(np.uint8)

        return full_output_np

    def get_canvas_inpainting_composite(self, canvas: np.ndarray, patch_mask: torch.Tensor) -> np.ndarray:
        """
        Get composite inpainting result: original pixels + model inpainted regions.

        Returns a composite canvas showing:
        - Original canvas pixels where patches are NOT masked (visible to encoder)
        - Model predictions where patches ARE masked (inpainted by model)

        This clearly shows only what the model must generate from scratch.

        Args:
            canvas: HxWx3 canvas numpy array (uint8)
            patch_mask: [1, num_patches] boolean tensor; True = masked patches to inpaint

        Returns:
            Composite canvas as HxWx3 uint8 array (original + inpainted regions)
        """
        # Convert canvas to tensor
        canvas_tensor = canvas_to_tensor(canvas).to(self.device)

        # Run forward pass with masking to get predictions for all patches
        self.autoencoder.eval()
        with torch.no_grad():
            pred_patches, _ = self.autoencoder.forward_with_patch_mask(canvas_tensor, patch_mask)
            img_pred = self.autoencoder.unpatchify(pred_patches)  # [1, 3, H, W]

        # Create composite: original pixels where mask=False, predictions where mask=True
        # Get canvas dimensions and patch size
        _, _, canvas_height, canvas_width = canvas_tensor.shape
        patch_size = int(getattr(self.autoencoder, "patch_size", 16))
        num_patches_height = canvas_height // patch_size
        num_patches_width = canvas_width // patch_size

        # Start with original canvas
        composite_tensor = canvas_tensor.clone()

        # Reshape patch mask to 2D grid: [1, num_patches] -> [num_patches_height, num_patches_width]
        mask_grid = patch_mask.view(num_patches_height, num_patches_width)

        # Replace masked patch regions with model predictions
        for patch_row in range(num_patches_height):
            for patch_col in range(num_patches_width):
                # If this patch is masked, copy the prediction to the composite
                if mask_grid[patch_row, patch_col]:
                    # Calculate pixel boundaries for this patch
                    y_start = patch_row * patch_size
                    y_end = y_start + patch_size
                    x_start = patch_col * patch_size
                    x_end = x_start + patch_size

                    # Copy predicted pixels for this masked patch
                    composite_tensor[:, :, y_start:y_end, x_start:x_end] = img_pred[:, :, y_start:y_end, x_start:x_end]

        # Convert composite tensor to numpy
        composite_np = (
            composite_tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)

        return composite_np

    def predict_next_frame(self, prediction_canvas_history: list) -> np.ndarray:
        """
        Predict next frame by creating canvas with blank slot and inpainting.

        Args:
            prediction_canvas_history: Interleaved history to use for prediction canvas

        Returns:
            Predicted next frame as HxWx3 uint8 array
        """
        H, W = Config.FRAME_SIZE

        # Add blank frame at end for prediction slot
        blank_frame = np.zeros((H, W, 3), dtype=np.uint8)
        prediction_canvas = build_canvas(
            prediction_canvas_history + [blank_frame],
            frame_size=Config.FRAME_SIZE,
            sep_width=Config.SEPARATOR_WIDTH,
        )

        # Store for visualization
        self.last_prediction_canvas = prediction_canvas.copy()

        # Convert to tensor
        canvas_tensor = canvas_to_tensor(prediction_canvas).to(self.device)

        # Get patch size from the autoencoder model (don't hard-code)
        patch_size = int(getattr(self.autoencoder, "patch_size", 16))

        # Compute patch mask for last slot
        num_frames = (len(prediction_canvas_history) + 1) // 2 + 1  # +1 for blank frame
        Ht, Wt = canvas_tensor.shape[-2:]
        patch_mask = compute_patch_mask_for_last_slot(
            img_size=(Ht, Wt),
            patch_size=patch_size,
            K=num_frames,
            sep_width=Config.SEPARATOR_WIDTH,
        ).to(self.device)

        # Run inference
        self.autoencoder.eval()
        with torch.no_grad():
            pred_patches, _ = self.autoencoder.forward_with_patch_mask(canvas_tensor, patch_mask)
            img_pred = self.autoencoder.unpatchify(pred_patches)
            img_np = (img_pred.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        # Extract predicted next frame from last slot
        W0 = (Wt - (num_frames - 1) * Config.SEPARATOR_WIDTH) // num_frames
        x0 = (num_frames - 1) * (W0 + Config.SEPARATOR_WIDTH)
        x1 = x0 + W0
        next_frame = img_np[:, x0:x1, :]

        return next_frame

    def run(self, max_iterations: Optional[int] = None):
        """
        Run the main loop.

        Args:
            max_iterations: Maximum number of iterations (None for infinite)
        """
        iteration = 0

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                print(f"\n=== Iteration {iteration} ===")

                # Step 1: Get current frame and add to history
                current_frame = self.robot.get_observation()
                if current_frame is None:
                    print("Failed to get observation, skipping iteration")
                    continue

                self.interleaved_history.append(current_frame)
                print(f"History length after adding frame: {len(self.interleaved_history)}")

                # Step 2: Output reconstruction loss between prediction and current frame
                if self.last_prediction is not None:
                    pred_tensor = world_model_utils.to_model_tensor(self.last_prediction, self.device)
                    curr_tensor = world_model_utils.to_model_tensor(current_frame, self.device)
                    prediction_error = torch.nn.functional.mse_loss(pred_tensor, curr_tensor).item()
                    print(f"Prediction error: {prediction_error:.6f}")
                else:
                    print("No previous prediction available")

                # Step 3-4: Create training canvas and train if history is exact size
                if len(self.interleaved_history) == 2 * Config.CANVAS_HISTORY_SIZE - 1:
                    # Build training canvas from current history
                    training_canvas = build_canvas(
                        self.interleaved_history,
                        frame_size=Config.FRAME_SIZE,
                        sep_width=Config.SEPARATOR_WIDTH,
                    )

                    # Store for visualization
                    self.last_training_canvas = training_canvas.copy()

                    # Train autoencoder on canvas (one step)
                    print("Training autoencoder on canvas (one step)...")
                    loss = self.train_autoencoder(training_canvas)
                    self.last_training_loss_history = [loss]

                    # Call training callback if provided
                    if self.training_callback:
                        self.training_callback(1, loss)

                    self.last_training_loss = loss
                    self.last_training_iterations = 1
                    print(f"Training step complete, loss: {loss:.6f}")
                else:
                    print(f"Skipping training (need exactly {2 * Config.CANVAS_HISTORY_SIZE - 1} elements, have {len(self.interleaved_history)})")

                # Step 5: Select action and add to history
                action = self.action_selector(current_frame, self.robot.action_space)
                print(f"Selected action: {action}")
                self.interleaved_history.append(action)
                print(f"History length after adding action: {len(self.interleaved_history)}")

                # Step 6-7: Create prediction canvas and predict next frame if history is exact size
                if len(self.interleaved_history) == 2 * Config.CANVAS_HISTORY_SIZE:
                    # Remove first frame+action pair to maintain same canvas size as training
                    prediction_canvas_history = self.interleaved_history[2:]

                    print("Predicting next frame...")
                    self.last_prediction = self.predict_next_frame(prediction_canvas_history)
                    print("Prediction complete")
                else:
                    print(f"Skipping prediction (need exactly {2 * Config.CANVAS_HISTORY_SIZE} elements, have {len(self.interleaved_history)})")
                    self.last_prediction = None
                    self.last_prediction_canvas = None

                # Step 8: Execute action on robot
                success = self.robot.execute_action(action)
                if not success:
                    print("Action execution failed")

                # Step 9: Trim history to maintain size
                self.interleaved_history = world_model_utils.maintain_history_window(
                    self.interleaved_history,
                    max_size=2 * (Config.CANVAS_HISTORY_SIZE - 1)
                )
                print(f"History length after trim: {len(self.interleaved_history)}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.robot.cleanup()
            print("Robot cleanup complete")


if __name__ == "__main__":
    print("This module requires a RobotInterface implementation to run")
    print("See toroidal_dot_world_model_example.py or jetbot_world_model_example.py for usage")
