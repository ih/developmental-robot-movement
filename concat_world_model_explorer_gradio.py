"""
Concat World Model Explorer Gradio App

A web-based UI for running AutoencoderConcatPredictorWorldModel on recorded robot sessions.
Allows running the world model for N iterations and visualizing training and prediction results.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr

import config
from autoencoder_concat_predictor_world_model import AutoencoderConcatPredictorWorldModel
from recording_reader import RecordingReader
from replay_robot import ReplayRobot
from session_explorer_lib import (
    list_session_dirs,
    load_session_metadata,
    load_session_events,
    extract_observations,
    extract_actions,
    load_frame_image,
    format_timestamp,
)
from recorded_policy import create_recorded_action_selector

# Session directories
TOROIDAL_DOT_SESSIONS_DIR = config.TOROIDAL_DOT_RECORDING_DIR
TOROIDAL_DOT_CHECKPOINT_DIR = config.TOROIDAL_DOT_CHECKPOINT_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global state
session_state = {}
world_model = None
replay_robot = None

def format_loss(loss_value):
    """Format loss value for display"""
    if loss_value is None:
        return "--"
    if loss_value < 0.001:
        return f"{loss_value:.2e}"
    else:
        return f"{loss_value:.6f}"

def refresh_sessions():
    """Refresh session list"""
    sessions = list_session_dirs()
    # Filter for toroidal dot sessions only
    toroidal_sessions = [s for s in sessions if "toroidal" in s.lower()]
    choices = [os.path.basename(s) + " - " + s for s in toroidal_sessions]
    return gr.Dropdown(choices=choices, value=choices[-1] if choices else None)

def load_session(session_choice):
    """Load a session from dropdown selection"""
    global session_state, world_model, replay_robot

    if not session_choice:
        return "No session selected", None, "", 0, 0

    # Extract session_dir from choice
    session_dir = session_choice.split(" - ")[-1]

    metadata = load_session_metadata(session_dir)
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)

    session_state.update({
        "session_name": os.path.basename(session_dir),
        "session_dir": session_dir,
        "metadata": metadata,
        "events": events,
        "observations": observations,
        "actions": actions,
    })

    # Build session info
    if not observations:
        info = f"**{session_state['session_name']}** has no observation frames."
        return info, None, "", 0, 0

    details = [
        f"**Session:** {session_state['session_name']}",
        f"**Total events:** {len(events)}",
        f"**Observations:** {len(observations)}",
        f"**Actions:** {len(actions)}",
    ]
    if metadata:
        start_time = metadata.get("start_time")
        if start_time:
            details.append(f"**Start:** {start_time}")
        robot_type_display = metadata.get("robot_type")
        if robot_type_display:
            details.append(f"**Robot:** {robot_type_display}")

    info = "\n\n".join(details)

    # Load first frame
    first_frame = load_frame_image(observations[0]["full_path"])
    frame_info = f"**Observation 1 / {len(observations)}**\n\nStep: {observations[0]['step']}\n\nTimestamp: {format_timestamp(observations[0]['timestamp'])}"

    max_frames = len(observations) - 1

    # Initialize world model and replay robot
    action_space = metadata.get("action_space", [])

    # Create recording reader
    reader = RecordingReader(session_dir)

    # Create replay robot
    replay_robot = ReplayRobot(reader, action_space)

    # TODO: Change action selectors throughout the codebase to only return an action
    # Currently recorded_action_selector returns (action, all_action_predictions) but
    # AutoencoderConcatPredictorWorldModel expects just action to be returned

    # Create world model with recorded action selector
    recorded_selector = create_recorded_action_selector(reader)

    def action_selector_adapter(observation, action_space):
        action, _ = recorded_selector()
        return action

    # Note: training_callback will be set dynamically in run_world_model when progress is available
    world_model = AutoencoderConcatPredictorWorldModel(
        replay_robot,
        action_selector=action_selector_adapter,
        device=device,
        training_callback=None  # Will be set in run_world_model
    )

    info += "\n\n**World model initialized and ready to run**"

    return info, first_frame, frame_info, 0, max_frames

def update_frame(frame_idx):
    """Update frame display"""
    if not session_state.get("observations"):
        return None, ""

    observations = session_state["observations"]
    if frame_idx >= len(observations):
        frame_idx = len(observations) - 1

    obs = observations[frame_idx]
    frame = load_frame_image(obs["full_path"])
    frame_info = f"**Observation {frame_idx + 1} / {len(observations)}**\n\nStep: {obs['step']}\n\nTimestamp: {format_timestamp(obs['timestamp'])}"

    return frame, frame_info

def run_world_model(num_iterations, progress=gr.Progress()):
    """Run the world model for N iterations with live metrics tracking"""
    global world_model, replay_robot

    if world_model is None:
        return "Please load a session first", "", None, None, None, None, None, None, None, None, None, "", "--"

    if num_iterations <= 0:
        return "Number of iterations must be greater than 0", "", None, None, None, None, None, None, None, None, None, "", "--"

    # Initialize or retrieve metrics history
    if not hasattr(world_model, '_metrics_history'):
        world_model._metrics_history = {
            'iteration': [],
            'training_loss': [],
            'training_iterations': [],
            'prediction_error': [],
            'iteration_time': [],
        }

    try:
        import time
        loop_count = 0  # Track how many times we loop back to the beginning

        # Run world model with progress updates and metric tracking
        for i in range(num_iterations):
            start_time = time.time()

            # Set up training callback to update progress bar during training
            iter_num = len(world_model._metrics_history['iteration']) + 1

            def training_progress_callback(train_iter, train_loss):
                desc = f"Iter {iter_num}: Training step {train_iter}, loss={format_loss(train_loss)}"
                progress((i + 0.5) / num_iterations, desc=desc)

            world_model.training_callback = training_progress_callback

            # Run one iteration, catching StopIteration to loop the session
            iteration_successful = False
            while not iteration_successful:
                try:
                    world_model.run(max_iterations=1)
                    iteration_successful = True
                except StopIteration:
                    # Session ended, reset reader to loop back to beginning
                    # Model training progress is preserved (weights, optimizer state)
                    # but we need to clear the frame/action history buffer
                    loop_count += 1
                    print(f"Session ended at iteration {i+1}/{num_iterations}, looping back to beginning (loop #{loop_count})...")
                    replay_robot.reader.reset()
                    # Clear the interleaved history to avoid corrupted frame/action structure
                    world_model.interleaved_history = []
                    # Clear prediction state for clean restart
                    world_model.last_prediction = None
                    world_model.last_prediction_canvas = None
                    # Retry the iteration from the beginning of the session

            iteration_time = time.time() - start_time

            # Get current metrics from world model's stored state
            # Don't call get_observation() again as it would skip a frame
            last_prediction_np = world_model.last_prediction
            prediction_error = None

            # Get current frame from history if available
            if len(world_model.interleaved_history) > 0 and last_prediction_np is not None:
                # Find the last frame in history (skip actions)
                for idx in range(len(world_model.interleaved_history) - 1, -1, -1):
                    if idx % 2 == 0:  # Even indices are frames
                        import world_model_utils
                        current_frame_np = world_model.interleaved_history[idx]
                        pred_tensor = world_model_utils.to_model_tensor(last_prediction_np, device)
                        curr_tensor = world_model_utils.to_model_tensor(current_frame_np, device)
                        prediction_error = torch.nn.functional.mse_loss(pred_tensor, curr_tensor).item()
                        break

            # Record metrics
            iter_num = len(world_model._metrics_history['iteration']) + 1
            world_model._metrics_history['iteration'].append(iter_num)
            world_model._metrics_history['training_loss'].append(world_model.last_training_loss if world_model.last_training_loss else None)
            world_model._metrics_history['training_iterations'].append(world_model.last_training_iterations)
            world_model._metrics_history['prediction_error'].append(prediction_error)
            world_model._metrics_history['iteration_time'].append(iteration_time)

            # Update progress with final iteration metrics
            desc = f"Iter {iter_num} complete: "
            if world_model.last_training_loss:
                desc += f"Final Loss={format_loss(world_model.last_training_loss)} ({world_model.last_training_iterations} steps) "
            if prediction_error:
                desc += f"Pred Err={format_loss(prediction_error)} "
            desc += f"Time={iteration_time:.2f}s"

            progress((i + 1) / num_iterations, desc=desc)

        # Get final state from world model's stored state
        last_prediction_np = world_model.last_prediction
        current_frame_np = None
        prediction_error = None

        # Get current frame from history
        if len(world_model.interleaved_history) > 0:
            for idx in range(len(world_model.interleaved_history) - 1, -1, -1):
                if idx % 2 == 0:  # Even indices are frames
                    current_frame_np = world_model.interleaved_history[idx]
                    break

        if current_frame_np is not None and last_prediction_np is not None:
            import world_model_utils
            pred_tensor = world_model_utils.to_model_tensor(last_prediction_np, device)
            curr_tensor = world_model_utils.to_model_tensor(current_frame_np, device)
            prediction_error = torch.nn.functional.mse_loss(pred_tensor, curr_tensor).item()

        # Create current metrics display
        current_metrics = f"**Latest Iteration Metrics:**\n\n"
        current_metrics += f"- Training Loss: {format_loss(world_model.last_training_loss)}\n"
        current_metrics += f"- Training Iterations: {world_model.last_training_iterations}\n"
        current_metrics += f"- Prediction Error: {format_loss(prediction_error)}\n"
        current_metrics += f"- Iteration Time: {world_model._metrics_history['iteration_time'][-1]:.2f}s\n"

        # Create metrics history plots
        fig_metrics, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Training Loss over iterations
        valid_train_loss = [(i, loss) for i, loss in zip(world_model._metrics_history['iteration'],
                                                          world_model._metrics_history['training_loss']) if loss is not None]
        if valid_train_loss:
            iters, losses = zip(*valid_train_loss)
            axes[0, 0].plot(iters, losses, 'b-o', linewidth=2, markersize=4)
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Training Loss')
            axes[0, 0].set_title('Training Loss Over Time')
            axes[0, 0].grid(True, alpha=0.3)

        # Training Iterations over iterations
        valid_train_iters = [(i, ti) for i, ti in zip(world_model._metrics_history['iteration'],
                                                        world_model._metrics_history['training_iterations']) if ti > 0]
        if valid_train_iters:
            iters, train_iters = zip(*valid_train_iters)
            axes[0, 1].plot(iters, train_iters, 'g-o', linewidth=2, markersize=4)
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Training Iterations')
            axes[0, 1].set_title('Training Iterations Per World Model Iteration')
            axes[0, 1].grid(True, alpha=0.3)

        # Prediction Error over iterations
        valid_pred_error = [(i, err) for i, err in zip(world_model._metrics_history['iteration'],
                                                        world_model._metrics_history['prediction_error']) if err is not None]
        if valid_pred_error:
            iters, errors = zip(*valid_pred_error)
            axes[1, 0].plot(iters, errors, 'r-o', linewidth=2, markersize=4)
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Prediction Error (MSE)')
            axes[1, 0].set_title('Prediction Error Over Time')
            axes[1, 0].grid(True, alpha=0.3)

        # Iteration Time over iterations
        axes[1, 1].plot(world_model._metrics_history['iteration'],
                       world_model._metrics_history['iteration_time'], 'purple', linewidth=2, marker='o', markersize=4)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Iteration Time Over Time')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Create visualizations
        status_msg = f"**Completed {num_iterations} iterations** (Total: {len(world_model._metrics_history['iteration'])} iterations)"
        if loop_count > 0:
            status_msg += f"\n\n*Session looped {loop_count} time{'s' if loop_count > 1 else ''} to complete all iterations*"

        # Current frame and prediction
        fig_frames = None
        if current_frame_np is not None:
            if last_prediction_np is not None:
                fig_frames, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(current_frame_np)
                axes[0].set_title("Current Frame")
                axes[0].axis("off")
                axes[1].imshow(last_prediction_np)
                axes[1].set_title(f"Last Predicted Frame\nError: {format_loss(prediction_error)}")
                axes[1].axis("off")
                plt.tight_layout()
            else:
                fig_frames, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(current_frame_np)
                ax.set_title("Current Frame (no prediction)")
                ax.axis("off")
                plt.tight_layout()

        # Training canvas visualizations
        fig_training_canvas = None
        fig_training_canvas_masked = None
        fig_training_inpainting_full = None
        fig_training_inpainting_composite = None
        fig_training_loss_plot = None
        training_info = ""

        if world_model.last_training_canvas is not None:
            # 1. Original training canvas
            fig_training_canvas, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(world_model.last_training_canvas)
            ax.set_title(f"Training Canvas (Original)")
            ax.axis("off")
            plt.tight_layout()

            training_info = f"**Training Loss:** {format_loss(world_model.last_training_loss)}\n\n"
            training_info += f"**Training Iterations:** {world_model.last_training_iterations}"

            # Plot training loss history
            if world_model.last_training_loss_history:
                fig_training_loss_plot, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.plot(range(1, len(world_model.last_training_loss_history) + 1),
                       world_model.last_training_loss_history, 'b-', linewidth=2)
                ax.set_xlabel('Training Iteration')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Progress (Current Iteration)')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')  # Log scale to see progress better
                plt.tight_layout()

            # Generate additional visualizations if mask is available
            if world_model.last_training_mask is not None:
                # 2. Canvas with mask overlay (shows which patches are masked)
                canvas_with_mask = world_model.get_canvas_with_mask_overlay(
                    world_model.last_training_canvas,
                    world_model.last_training_mask
                )
                fig_training_canvas_masked, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.imshow(canvas_with_mask)
                ax.set_title("Training Canvas with Mask (Red = Masked Patches)")
                ax.axis("off")
                plt.tight_layout()

                # 3. Full model output (what model reconstructs for everything)
                inpainting_full = world_model.get_canvas_inpainting_full_output(
                    world_model.last_training_canvas,
                    world_model.last_training_mask
                )
                fig_training_inpainting_full, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.imshow(inpainting_full)
                ax.set_title("Training Inpainting - Full Model Output (All Patches)")
                ax.axis("off")
                plt.tight_layout()

                # 4. Composite (original + inpainted masked regions only)
                inpainting_composite = world_model.get_canvas_inpainting_composite(
                    world_model.last_training_canvas,
                    world_model.last_training_mask
                )
                fig_training_inpainting_composite, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.imshow(inpainting_composite)
                ax.set_title("Training Inpainting - Composite (Original + Inpainted Masked Regions)")
                ax.axis("off")
                plt.tight_layout()

        # Prediction canvas and predicted frame
        fig_prediction_canvas = None
        fig_predicted_frame = None

        if world_model.last_prediction_canvas is not None:
            fig_prediction_canvas, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.imshow(world_model.last_prediction_canvas)
            ax.set_title("Prediction Canvas (with blank slot)")
            ax.axis("off")
            plt.tight_layout()

        if last_prediction_np is not None:
            fig_predicted_frame, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(last_prediction_np)
            ax.set_title("Predicted Next Frame")
            ax.axis("off")
            plt.tight_layout()

        return status_msg, current_metrics, fig_metrics, fig_frames, fig_training_canvas, fig_training_canvas_masked, fig_training_inpainting_full, fig_training_inpainting_composite, fig_training_loss_plot, fig_prediction_canvas, fig_predicted_frame, training_info, format_loss(prediction_error)

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", None, None, None, None, None, None, None, None, None, "", "--"

# Build Gradio interface
with gr.Blocks(title="Concat World Model Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Concat World Model Explorer")
    gr.Markdown("Run AutoencoderConcatPredictorWorldModel on recorded toroidal dot sessions.")

    # Session Selection
    with gr.Row():
        session_dropdown = gr.Dropdown(label="Session", choices=[], interactive=True)
        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        load_session_btn = gr.Button("Load Session", variant="primary")

    session_info = gr.Markdown("No session loaded.")

    # Frame Viewer
    gr.Markdown("## Current Frame")
    with gr.Row():
        with gr.Column(scale=2):
            frame_image = gr.Image(label="Current Frame", type="pil", interactive=False)
        with gr.Column(scale=1):
            frame_info = gr.Markdown("Load a session to view frames.")

    # Frame Navigation
    with gr.Row():
        frame_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Frame", interactive=True)

    with gr.Row():
        frame_number_input = gr.Number(value=0, label="Jump to Frame", precision=0)
        jump_btn = gr.Button("Jump", size="sm")

    gr.Markdown("---")

    # World Model Runner
    gr.Markdown("## Run World Model")
    gr.Markdown("Execute the world model for a specified number of iterations.")

    with gr.Row():
        num_iterations_input = gr.Number(value=10, label="Number of Iterations", precision=0, minimum=1)
        run_btn = gr.Button("Run World Model", variant="primary")

    run_status = gr.Markdown("")

    gr.Markdown("---")

    # Metrics Display
    gr.Markdown("## Iteration Metrics")
    current_metrics_display = gr.Markdown("")
    metrics_history_plot = gr.Plot(label="Metrics History")

    gr.Markdown("---")

    # Visualizations
    gr.Markdown("## Visualizations")

    gr.Markdown("### Current State")
    frames_plot = gr.Plot(label="Current Frame & Last Prediction")
    prediction_error_display = gr.Textbox(label="Prediction Error", value="--", interactive=False)

    gr.Markdown("### Training Results")
    training_info_display = gr.Markdown("")
    training_canvas_plot = gr.Plot(label="1. Training Canvas (Original)")
    training_canvas_masked_plot = gr.Plot(label="2. Training Canvas with Mask Overlay")
    training_inpainting_full_plot = gr.Plot(label="3. Training Inpainting - Full Model Output")
    training_inpainting_composite_plot = gr.Plot(label="4. Training Inpainting - Composite")
    training_loss_history_plot = gr.Plot(label="Training Loss Progress")

    gr.Markdown("### Prediction Results")
    prediction_canvas_plot = gr.Plot(label="Prediction Canvas")
    predicted_frame_plot = gr.Plot(label="Predicted Next Frame")

    # Event handlers
    refresh_btn.click(
        fn=refresh_sessions,
        inputs=[],
        outputs=[session_dropdown]
    )

    load_session_btn.click(
        fn=load_session,
        inputs=[session_dropdown],
        outputs=[session_info, frame_image, frame_info, frame_slider, frame_slider]
    )

    frame_slider.change(
        fn=update_frame,
        inputs=[frame_slider],
        outputs=[frame_image, frame_info]
    )

    def jump_to_frame(frame_num):
        frame_num = int(frame_num)
        img, info = update_frame(frame_num)
        return img, info, frame_num

    jump_btn.click(
        fn=jump_to_frame,
        inputs=[frame_number_input],
        outputs=[frame_image, frame_info, frame_slider]
    )

    run_btn.click(
        fn=run_world_model,
        inputs=[num_iterations_input],
        outputs=[
            run_status,
            current_metrics_display,
            metrics_history_plot,
            frames_plot,
            training_canvas_plot,
            training_canvas_masked_plot,
            training_inpainting_full_plot,
            training_inpainting_composite_plot,
            training_loss_history_plot,
            prediction_canvas_plot,
            predicted_frame_plot,
            training_info_display,
            prediction_error_display,
        ]
    )

    # Initialize session dropdown on load
    demo.load(
        fn=refresh_sessions,
        inputs=[],
        outputs=[session_dropdown]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
