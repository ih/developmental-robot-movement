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
    build_canvas,
    canvas_to_tensor,
    compute_patch_mask_for_last_slot,
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
    4. Train autoencoder on canvas until reconstruction loss below threshold
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
    ):
        """
        Initialize the autoencoder concat predictor world model.

        Args:
            robot_interface: RobotInterface implementation for robot interaction
            action_selector: Function that takes (observation, action_space) and returns (action, metadata)
                           If None, uses default random action selection
            device: torch.device for model computations (defaults to cuda if available)
        """
        self.robot = robot_interface
        self.action_selector = action_selector or self._default_action_selector
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize autoencoder model
        self.autoencoder = TargetedMAEWrapper(
            image_size=Config.FRAME_SIZE[0],
            patch_size=16,
            embed_dim=256,
            decoder_embed_dim=128,
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

        print(f"AutoencoderConcatPredictorWorldModel initialized on {self.device}")
        print(f"Canvas history size: {Config.CANVAS_HISTORY_SIZE}")
        print(f"Frame size: {Config.FRAME_SIZE}")
        print(f"Separator width: {Config.SEPARATOR_WIDTH}")

    def _default_action_selector(self, observation: np.ndarray, action_space: list) -> Tuple[dict, dict]:
        """Default random action selection."""
        import random
        action = random.choice(action_space)
        metadata = {"selection_method": "random"}
        return action, metadata

    def train_canvas_reconstruction(self, training_canvas: np.ndarray, num_frames: int) -> float:
        """
        Train autoencoder to reconstruct canvas until loss below threshold.

        Args:
            training_canvas: HxWx3 canvas with history frames
            num_frames: Number of frames in canvas

        Returns:
            Final reconstruction loss
        """
        # Convert canvas to tensor
        canvas_tensor = canvas_to_tensor(training_canvas).to(self.device)

        # Compute patch mask for last slot
        H, W = canvas_tensor.shape[-2:]
        patch_mask = compute_patch_mask_for_last_slot(
            img_size=(H, W),
            patch_size=16,
            K=num_frames,
            sep_width=Config.SEPARATOR_WIDTH,
        ).to(self.device)

        # Train until reconstruction loss below threshold
        self.autoencoder.train()
        loss = float('inf')

        while loss > Config.CANVAS_RECONSTRUCTION_THRESHOLD:
            loss = self.autoencoder.train_on_canvas(
                canvas_tensor,
                patch_mask,
                self.ae_optimizer
            )
            self.ae_scheduler.step()

        return loss

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

        # Convert to tensor
        canvas_tensor = canvas_to_tensor(prediction_canvas).to(self.device)

        # Compute patch mask for last slot
        num_frames = (len(prediction_canvas_history) + 1) // 2 + 1  # +1 for blank frame
        Ht, Wt = canvas_tensor.shape[-2:]
        patch_mask = compute_patch_mask_for_last_slot(
            img_size=(Ht, Wt),
            patch_size=16,
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
                    num_frames = (len(self.interleaved_history) + 1) // 2

                    # Train autoencoder
                    print("Training autoencoder on canvas...")
                    final_loss = self.train_canvas_reconstruction(training_canvas, num_frames)
                    print(f"Training complete, final loss: {final_loss:.6f}")
                else:
                    print(f"Skipping training (need exactly {2 * Config.CANVAS_HISTORY_SIZE - 1} elements, have {len(self.interleaved_history)})")

                # Step 5: Select action and add to history
                action, metadata = self.action_selector(current_frame, self.robot.action_space)
                print(f"Selected action: {action} ({metadata.get('selection_method', 'unknown')})")
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
