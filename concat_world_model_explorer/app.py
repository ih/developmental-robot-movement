"""
Concat World Model Explorer Gradio App

A web-based UI for running AutoencoderConcatPredictorWorldModel on recorded robot sessions.
Allows batch training with periodic progress updates and final full-session evaluation.
"""

import gradio as gr
import config

# Import all modules from the modular package
from . import (
    state,
    session_manager,
    canvas_ops,
    inference,
    evaluation,
    training,
    visualization,
    checkpoint_manager,
    attention,
)


# Build Gradio interface
with gr.Blocks(title="Concat World Model Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Concat World Model Explorer")
    gr.Markdown("Run AutoencoderConcatPredictorWorldModel on recorded robot sessions.")

    # Session Selection
    gr.Markdown("### Session Selection")
    with gr.Row():
        robot_type_dropdown = gr.Dropdown(
            label="Robot Type",
            choices=session_manager.get_robot_types(),
            value="so101",
            interactive=True
        )
        session_dropdown = gr.Dropdown(label="Session", choices=[], interactive=True)
        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        load_session_btn = gr.Button("Load Session", variant="primary")

    session_info = gr.Markdown("No session loaded.")

    gr.Markdown("---")

    # Model Checkpoint Management
    gr.Markdown("## Model Checkpoint Management")
    gr.Markdown("Save and load trained model weights.")

    checkpoint_status = gr.Markdown("**Current checkpoint:** None (using fresh model)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Save Model Weights")
            save_checkpoint_name = gr.Textbox(
                label="Checkpoint Name",
                placeholder="e.g., my_model_checkpoint",
                info="Extension .pth will be added automatically"
            )
            save_checkpoint_btn = gr.Button("💾 Save Weights", variant="primary")
            save_checkpoint_status = gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### Load Model Weights")
            checkpoint_dropdown = gr.Dropdown(
                label="Select Checkpoint",
                choices=[],
                interactive=True
            )
            with gr.Row():
                refresh_checkpoints_btn = gr.Button("🔄 Refresh", size="sm")
                load_checkpoint_btn = gr.Button("📂 Load Weights", variant="primary", scale=2)
            load_checkpoint_status = gr.Markdown("")

    gr.Markdown("---")

    # Frame Viewer
    gr.Markdown("## Current Frame & Canvas")
    with gr.Row():
        with gr.Column(scale=2):
            frame_image = gr.Image(label="Current Frame", type="pil", interactive=False)
        with gr.Column(scale=1):
            frame_info = gr.Markdown("Load a session to view frames.")

    # Canvas ending at current frame
    canvas_image = gr.Image(label="Canvas (ending at current frame)", type="numpy", interactive=False)

    # Frame Navigation
    with gr.Row():
        frame_number_input = gr.Number(value=3, label="Jump to Frame", precision=0)
        jump_btn = gr.Button("Jump", size="sm")

    gr.Markdown("---")

    # Counterfactual Testing
    gr.Markdown("## Counterfactual Testing")
    gr.Markdown("Test what the model predicts if the last action before the selected frame was different.")

    with gr.Row():
        # Choices are updated dynamically when a session is loaded
        counterfactual_action_radio = gr.Radio(
            choices=[
                ("Stay (action=0, RED)", 0),
                ("Move Positive (action=1, GREEN)", 1),
                ("Move Negative (action=2, BLUE)", 2),
            ],
            value=1,
            label="Counterfactual Last Action",
            info="What action should we pretend was taken?"
        )
        run_counterfactual_btn = gr.Button("🔀 Run Counterfactual Inference", variant="secondary")

    counterfactual_status = gr.Markdown("")

    with gr.Accordion("Counterfactual Results", open=True):
        # Row 1: True vs Counterfactual canvases side-by-side
        with gr.Row():
            counterfactual_true_canvas = gr.Plot(label="True Canvas (Actual Action)")
            counterfactual_cf_canvas = gr.Plot(label="Counterfactual Canvas (Modified Action)")

        # Row 2: Inference composites side-by-side
        with gr.Row():
            counterfactual_true_inference = gr.Plot(label="True Inference (Composite)")
            counterfactual_cf_inference = gr.Plot(label="Counterfactual Inference (Composite)")

        # Row 3: Difference heatmap and statistics
        with gr.Row():
            counterfactual_diff_heatmap = gr.Plot(label="Prediction Difference Heatmap")
            counterfactual_stats = gr.Markdown("")

    gr.Markdown("---")

    # Full Session Evaluation
    gr.Markdown("## Full Session Evaluation")
    gr.Markdown("Evaluate model performance across all observations in the session to get objective metrics for model comparison.")

    with gr.Row():
        evaluate_session_btn = gr.Button("📊 Evaluate Model on Full Session", variant="primary")

    eval_status = gr.Markdown("")

    # Full Session Evaluation Results (collapsible)
    with gr.Accordion("Full Session Evaluation Results", open=False):
        eval_statistics = gr.Markdown("")
        eval_loss_over_time = gr.Plot(label="Loss Over Observations")
        eval_distribution = gr.Plot(label="Loss Distribution")

    gr.Markdown("---")

    # Single Canvas Training
    gr.Markdown("## Train on Single Canvas")
    gr.Markdown("Train the autoencoder on a canvas built from the selected frame and its history.")

    with gr.Row():
        single_canvas_training_steps = gr.Number(value=10, label="Training Steps", precision=0, minimum=1)
        single_canvas_train_btn = gr.Button("Train on Selected Frame", variant="primary")

    single_canvas_status = gr.Markdown("")

    # Single Canvas Training Visualizations (collapsible)
    with gr.Accordion("Single Canvas Training Results", open=False):
        single_canvas_training_info = gr.Markdown("")
        single_canvas_grad_diag = gr.Markdown("")
        single_canvas_loss_history = gr.Plot(label="Training Loss History")
        single_canvas_training_canvas = gr.Plot(label="1. Training Canvas")
        single_canvas_training_masked = gr.Plot(label="2. Canvas with Mask Overlay")
        single_canvas_inpainting_full = gr.Plot(label="3. Full Inpainting Output")
        single_canvas_inpainting_composite = gr.Plot(label="4. Composite Reconstruction")

    gr.Markdown("---")

    # ========== BATCH TRAINING SECTION ==========
    gr.Markdown("---")
    gr.Markdown("## Run World Model (Batch Training)")
    gr.Markdown("Train the model on batches of observations with periodic full-session evaluation.")

    with gr.Row():
        total_samples_input = gr.Number(
            value=10000000,
            label="Total Samples",
            precision=0,
            minimum=1,
            interactive=True,
            info="Number of training samples (loops through session if needed)"
        )
        batch_size_input = gr.Number(
            value=1,
            label="Batch Size",
            precision=0,
            minimum=1,
            interactive=True,
            info="Samples per batch"
        )
        sampling_mode_dropdown = gr.Dropdown(
            choices=["Random (with replacement)", "Epoch-based (shuffle each epoch)", "Loss-weighted"],
            value="Epoch-based (shuffle each epoch)",
            label="Sampling Mode",
            info="Random: repeat samples. Epoch: see each once. Loss-weighted: prioritize high-loss samples."
        )

    # Loss-weighted sampling parameters
    gr.Markdown("### Loss-Weighted Sampling Parameters")
    gr.Markdown("Controls for loss-weighted sampling mode (only used when 'Loss-weighted' is selected above).")
    with gr.Row():
        loss_weight_temperature_input = gr.Slider(
            value=0.5,
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            label="Temperature",
            info="Lower = focus more on high-loss samples, Higher = more uniform sampling"
        )
        loss_weight_refresh_input = gr.Number(
            value=50,
            label="Weight Refresh Interval",
            precision=0,
            minimum=10,
            maximum=1000,
            interactive=True,
            info="Batches between weight updates (lower = more responsive)"
        )

    with gr.Row():
        update_interval_input = gr.Number(
            value=100,
            label="Update Interval",
            precision=0,
            minimum=10,
            interactive=True,
            info="Evaluate every N samples (lower = more frequent updates)"
        )
        window_size_input = gr.Number(
            value=50,
            label="Rolling Window Size",
            precision=0,
            minimum=1,
            interactive=True,
            info="Number of recent checkpoints to show in rolling window graph"
        )
        num_random_obs_input = gr.Number(
            value=2,
            label="Random Observations to Visualize",
            precision=0,
            minimum=1,
            maximum=20,
            interactive=True,
            info="Number of random observations to sample and visualize on each update"
        )
        num_best_models_input = gr.Number(
            value=1,
            label="Best Models to Keep",
            precision=0,
            minimum=1,
            maximum=10,
            interactive=True,
            info="Maximum number of best model checkpoints to keep (auto-deletes worse models)"
        )

    # Validation Set Controls
    gr.Markdown("### Validation Set (Optional)")
    gr.Markdown("Select a different session to use as validation set for monitoring generalization during training.")
    with gr.Row():
        validation_session_dropdown = gr.Dropdown(
            label="Validation Session",
            choices=[],
            interactive=True,
            info="Select a session to use as validation set"
        )
        clear_validation_btn = gr.Button("🗑️ Clear", size="sm")

    validation_status = gr.Markdown("No validation session selected")

    # Divergence-based Early Stopping Controls
    gr.Markdown("### Early Stopping on Divergence")
    gr.Markdown("Stop training when validation loss diverges from training loss. Requires a validation session.")
    with gr.Row():
        stop_on_divergence_checkbox = gr.Checkbox(
            label="Stop on Divergence",
            value=True,
            info="Stop training when validation loss exceeds training loss by threshold"
        )
        divergence_patience_input = gr.Number(
            value=25,
            label="Patience",
            precision=0,
            minimum=1,
            maximum=100,
            interactive=True,
            info="Number of consecutive divergence checks before stopping"
        )
        divergence_min_updates_input = gr.Number(
            value=5,
            label="Min Updates",
            precision=0,
            minimum=1,
            interactive=True,
            info="Minimum update intervals before checking divergence"
        )

    with gr.Row():
        divergence_gap_input = gr.Number(
            value=0.1,
            label="Divergence Gap",
            minimum=0,
            interactive=True,
            info="Stop if (val_loss - train_loss) >= this value"
        )
        divergence_ratio_input = gr.Number(
            value=2.5,
            label="Divergence Ratio",
            minimum=1.0,
            interactive=True,
            info="Stop if (val_loss / train_loss) >= this ratio"
        )

    with gr.Row():
        val_spike_threshold_input = gr.Number(
            value=2.0,
            label="Val Spike Threshold",
            minimum=1.0,
            interactive=True,
            info="Spike if val_loss > best_val * this multiplier (2.0 = 100% above best)"
        )
        val_spike_window_input = gr.Number(
            value=25,
            label="Spike Window",
            precision=0,
            minimum=3,
            interactive=True,
            info="Number of recent checks to track for spike frequency"
        )
        val_spike_frequency_input = gr.Number(
            value=0.75,
            label="Spike Frequency",
            minimum=0.1,
            maximum=1.0,
            interactive=True,
            info="Trigger if this fraction of window are spikes (0.75 = 75%)"
        )

    # Resume Mode Controls
    gr.Markdown("### Resume Training from Checkpoint")
    with gr.Row():
        resume_mode_checkbox = gr.Checkbox(
            label="Resume Mode",
            value=False,
            info="Continue training from loaded checkpoint (preserves samples count, plots, W&B steps)"
        )
        samples_mode_radio = gr.Radio(
            choices=["Train additional samples", "Train to total samples"],
            value="Train additional samples",
            label="Samples Mode",
            info="How to interpret 'Total Samples' when resuming"
        )
        starting_samples_input = gr.Number(
            value=0,
            label="Starting Samples Seen",
            precision=0,
            minimum=0,
            interactive=True,
            info="Prefilled from checkpoint; adjust if needed"
        )

    with gr.Row():
        preserve_optimizer_checkbox = gr.Checkbox(
            label="Keep Optimizer State",
            value=True,
            info="Preserve momentum/velocity from checkpoint (recommended for resuming)"
        )
        preserve_scheduler_checkbox = gr.Checkbox(
            label="Keep Scheduler State",
            value=True,
            info="Continue LR schedule from checkpoint step"
        )

    # Learning Rate Controls
    gr.Markdown("### Learning Rate Settings")
    with gr.Row():
        custom_lr_input = gr.Number(
            value=0,
            label="Custom Base LR (0=default)",
            minimum=0,
            interactive=True,
            info=f"Override base LR (default: {config.AutoencoderConcatPredictorWorldModelConfig.AUTOENCODER_LR}). Set 0 to use config default."
        )
        disable_lr_scaling_checkbox = gr.Checkbox(
            label="Disable LR Scaling",
            value=False,
            info="Use exact LR instead of scaling by batch size"
        )
        custom_warmup_input = gr.Number(
            value=-1,
            label="Warmup Steps (-1=default)",
            precision=0,
            minimum=-1,
            interactive=True,
            info="-1 uses scaled default, 0 disables warmup, >0 sets exact steps"
        )
        lr_min_ratio_input = gr.Number(
            value=0.001,
            label="LR Min Ratio",
            minimum=0,
            maximum=1,
            interactive=True,
            info="Minimum LR as ratio of base LR for cosine decay"
        )
        resume_warmup_ratio_input = gr.Number(
            value=0.01,
            label="Resume Warmup Ratio",
            minimum=0,
            maximum=1,
            interactive=True,
            info="Warmup steps as ratio of session steps when resuming (0=instant jump, 0.01=1%)"
        )
        plateau_factor_input = gr.Number(
            value=0.1,
            label="Plateau Factor",
            minimum=0.01,
            maximum=1.0,
            interactive=True,
            info="ReduceLROnPlateau: multiply LR by this factor when loss plateaus"
        )

    # Weights & Biases logging controls
    gr.Markdown("### Logging")
    with gr.Row():
        enable_wandb_input = gr.Checkbox(
            label="Enable Weights & Biases Logging",
            value=False,
            info="Log metrics and visualizations to wandb during training"
        )
        wandb_run_name_input = gr.Textbox(
            label="W&B Run Name (optional)",
            placeholder="Auto-generated if empty",
            info="Custom name for this wandb run"
        )

    # Training info display (dynamically updated based on total_samples and batch_size)
    training_info_display = gr.Markdown("")

    # Pre-flight summary (shows configuration before training)
    preflight_summary = gr.Markdown("")

    with gr.Row():
        run_batch_btn = gr.Button("🚀 Run Batch Training", variant="primary")
        stop_training_btn = gr.Button("🛑 Stop Training", variant="stop")
        show_preflight_btn = gr.Button("📋 Show Config", variant="secondary")

    batch_training_status = gr.Markdown("")

    gr.Markdown("---")

    # Training Progress
    gr.Markdown("## Training Progress")

    with gr.Row():
        with gr.Column(scale=1):
            loss_vs_samples_plot = gr.Plot(label="Loss vs Samples Seen (Full History)")
        with gr.Column(scale=1):
            loss_vs_recent_plot = gr.Plot(label="Loss vs Recent Checkpoints (Rolling Window)")

    lr_vs_samples_plot = gr.Plot(label="Learning Rate vs Samples Seen")

    sample_weights_plot = gr.Plot(label="Sample Weights Distribution (Loss-Weighted Only)")

    gr.Markdown("---")

    # Training Observation Samples
    gr.Markdown("---")

    gr.Markdown("## Training Observation Samples")
    gr.Markdown("Random observations + current frame, re-sampled on each training update.")

    observation_samples_status = gr.Markdown("")
    observation_samples_plot = gr.Plot(label="Observation Samples (Original + Composite Grid)")

    gr.Markdown("---")

    # Latest Evaluation Results
    gr.Markdown("## Latest Evaluation Results")
    gr.Markdown("Most recent full-session evaluation from training.")

    latest_eval_loss_plot = gr.Plot(label="Loss Over Session")
    latest_eval_dist_plot = gr.Plot(label="Loss Distribution")

    gr.Markdown("---")

    # Attention Visualization
    gr.Markdown("## Decoder Attention Visualization")
    gr.Markdown("Visualize decoder attention FROM selected patches TO all other patches.")
    gr.Markdown("*Note: Uses the currently selected frame to build a canvas and run inference to get attention.*")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")

            # Patch selection mode
            attn_selection_mode = gr.Radio(
                choices=["Automatic Dot Detection", "Manual Selection"],
                value="Automatic Dot Detection",
                label="Patch Selection Mode"
            )

            # Brightness threshold for automatic detection
            attn_brightness_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Brightness Threshold (for automatic detection)",
                info="Minimum avg brightness (0-1) to detect dot patches"
            )

            # Manual patch selection
            attn_manual_patches = gr.Textbox(
                label="Manual Patch Indices (for manual selection)",
                placeholder="e.g., 0,5,10 or 0-10",
                info="Comma-separated indices or ranges (e.g., 0,5,10-15)"
            )

            # Visualization type
            attn_viz_type = gr.Radio(
                choices=["Patch-to-Patch Lines", "Heatmap Matrix", "Heatmap Overlay on Frame"],
                value="Patch-to-Patch Lines",
                label="Visualization Type"
            )

            # Aggregation method
            attn_aggregation = gr.Radio(
                choices=["mean", "max", "sum"],
                value="mean",
                label="Head Aggregation Method"
            )

            # Selected patch aggregation method
            attn_selected_aggregation = gr.Radio(
                choices=["mean", "max", "sum"],
                value="mean",
                label="Selected Patch Aggregation Method",
                info="How to aggregate attention from multiple selected patches"
            )

            # Quantile slider
            attn_quantile = gr.Slider(
                minimum=0.0,
                maximum=100.0,
                value=95.0,
                step=1.0,
                label="Attention Quantile (%)",
                info="Show top N% of connections (e.g., 95 = show strongest 5%)"
            )

            # Layer toggles
            gr.Markdown("**Select Layers to Display:**")
            attn_layer0 = gr.Checkbox(label="Layer 0", value=True)
            attn_layer1 = gr.Checkbox(label="Layer 1", value=True)
            attn_layer2 = gr.Checkbox(label="Layer 2", value=True)
            attn_layer3 = gr.Checkbox(label="Layer 3", value=True)
            attn_layer4 = gr.Checkbox(label="Layer 4", value=True)

            # Head toggles
            gr.Markdown("**Select Attention Heads to Display:**")
            attn_head0 = gr.Checkbox(label="Head 0", value=True)
            attn_head1 = gr.Checkbox(label="Head 1", value=True)
            attn_head2 = gr.Checkbox(label="Head 2", value=True)
            attn_head3 = gr.Checkbox(label="Head 3", value=True)

            # Generate button
            generate_attn_btn = gr.Button("Generate Attention Visualization", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Visualization")
            attn_status = gr.Markdown("")
            attn_plot = gr.Plot(label="Attention Visualization")
            attn_stats = gr.Markdown("")

    gr.Markdown("---")

    # Batch Size Comparison Testing
    gr.Markdown("## Batch Size Comparison")
    gr.Markdown(
        "Compare training efficiency across different batch sizes. "
        "Each test trains over the same total number of samples for fair comparison."
    )

    with gr.Row():
        with gr.Column(scale=1):
            batch_sizes_input = gr.Textbox(
                value="1,2,4,8,16",
                label="Batch Sizes to Test",
                info="Comma-separated (e.g., 1,2,4,8,16)"
            )
            comparison_total_samples_input = gr.Number(
                value=1000,
                label="Total Samples Per Test",
                precision=0,
                minimum=100,
                info="Same for all batch sizes (fair comparison)"
            )
            run_comparison_btn = gr.Button("🔬 Run Batch Comparison", variant="primary")

        with gr.Column(scale=1):
            comparison_status = gr.Markdown("")

    with gr.Accordion("Batch Comparison Results", open=False):
        comparison_summary = gr.Markdown("")
        comparison_time_plot = gr.Plot(label="Total Training Time by Batch Size")
        comparison_quality_plot = gr.Plot(label="Final Loss by Batch Size")
        comparison_convergence_plot = gr.Plot(label="Loss Convergence (All Batch Sizes)")
        comparison_table = gr.Dataframe(label="Detailed Results")

    gr.Markdown("---")

    # Event handlers

    # Robot type selection - updates both session and validation dropdowns
    def on_robot_type_change(robot_type):
        sessions_update = session_manager.refresh_sessions_for_type(robot_type)
        val_choices = session_manager.get_validation_session_choices(robot_type)
        # Clear validation when robot type changes
        state.clear_validation_session()
        return sessions_update, gr.Dropdown(choices=val_choices, value="None - No validation"), "No validation session selected"

    robot_type_dropdown.change(
        fn=on_robot_type_change,
        inputs=[robot_type_dropdown],
        outputs=[session_dropdown, validation_session_dropdown, validation_status]
    )

    # Refresh button - refreshes both dropdowns for current robot type
    def refresh_all_sessions():
        sessions_update = session_manager.refresh_sessions()
        val_choices = session_manager.get_validation_session_choices()
        return sessions_update, gr.Dropdown(choices=val_choices, value="None - No validation")

    refresh_btn.click(
        fn=refresh_all_sessions,
        inputs=[],
        outputs=[session_dropdown, validation_session_dropdown]
    )

    load_session_btn.click(
        fn=session_manager.load_session,
        inputs=[session_dropdown],
        outputs=[session_info, frame_image, frame_info, canvas_image, checkpoint_dropdown, counterfactual_action_radio]
    )

    # Validation session handlers
    validation_session_dropdown.change(
        fn=session_manager.load_validation_session,
        inputs=[validation_session_dropdown],
        outputs=[validation_status]
    )

    def clear_validation_and_reset_dropdown():
        state.clear_validation_session()
        return "No validation session selected", gr.Dropdown(value="None - No validation")

    clear_validation_btn.click(
        fn=clear_validation_and_reset_dropdown,
        inputs=[],
        outputs=[validation_status, validation_session_dropdown]
    )

    # Checkpoint management event handlers
    save_checkpoint_btn.click(
        fn=checkpoint_manager.save_model_weights,
        inputs=[save_checkpoint_name],
        outputs=[save_checkpoint_status, checkpoint_dropdown]
    )

    refresh_checkpoints_btn.click(
        fn=checkpoint_manager.refresh_checkpoints,
        inputs=[],
        outputs=[checkpoint_dropdown]
    )

    def load_checkpoint_and_update_resume(checkpoint_name):
        """Load checkpoint and update resume-related fields."""
        status = checkpoint_manager.load_model_weights(checkpoint_name)
        # Return the samples_seen from loaded metadata
        samples_seen = state.loaded_checkpoint_metadata.get('samples_seen', 0)
        return status, samples_seen

    load_checkpoint_btn.click(
        fn=load_checkpoint_and_update_resume,
        inputs=[checkpoint_dropdown],
        outputs=[load_checkpoint_status, starting_samples_input]
    )

    def jump_to_frame(frame_num):
        frame_num = int(frame_num)
        img, info, canvas = session_manager.update_frame(frame_num)
        return img, info, canvas

    jump_btn.click(
        fn=jump_to_frame,
        inputs=[frame_number_input],
        outputs=[frame_image, frame_info, canvas_image]
    )

    # Counterfactual testing
    run_counterfactual_btn.click(
        fn=inference.run_counterfactual_inference,
        inputs=[frame_number_input, counterfactual_action_radio],
        outputs=[
            counterfactual_status,
            counterfactual_true_canvas,
            counterfactual_cf_canvas,
            counterfactual_true_inference,
            counterfactual_cf_inference,
            counterfactual_diff_heatmap,
            counterfactual_stats,
        ]
    )

    # Note: evaluate_full_session now returns 5 values, but we only use 4 in the UI
    # The 5th value (stats dict) is used internally by run_world_model_batch
    evaluate_session_btn.click(
        fn=lambda: evaluation.evaluate_full_session()[:4],  # Take only first 4 return values
        inputs=[],
        outputs=[
            eval_status,
            eval_loss_over_time,
            eval_distribution,
            eval_statistics,
        ]
    )

    single_canvas_train_btn.click(
        fn=training.train_on_single_canvas,
        inputs=[frame_number_input, single_canvas_training_steps],
        outputs=[
            single_canvas_status,
            single_canvas_training_info,
            single_canvas_grad_diag,
            single_canvas_loss_history,
            single_canvas_training_canvas,
            single_canvas_training_masked,
            single_canvas_inpainting_full,
            single_canvas_inpainting_composite,
        ]
    )

    # Dynamic training info update handlers
    total_samples_input.change(
        fn=visualization.calculate_training_info,
        inputs=[total_samples_input, batch_size_input, sampling_mode_dropdown],
        outputs=[training_info_display]
    )

    batch_size_input.change(
        fn=visualization.calculate_training_info,
        inputs=[total_samples_input, batch_size_input, sampling_mode_dropdown],
        outputs=[training_info_display]
    )

    sampling_mode_dropdown.change(
        fn=visualization.calculate_training_info,
        inputs=[total_samples_input, batch_size_input, sampling_mode_dropdown],
        outputs=[training_info_display]
    )

    # Batch training handler (takes current slider value as input)
    run_batch_btn.click(
        fn=training.run_world_model_batch,
        inputs=[
            total_samples_input, batch_size_input, frame_number_input, update_interval_input,
            window_size_input, num_random_obs_input, num_best_models_input,
            enable_wandb_input, wandb_run_name_input,
            # Resume mode parameters
            resume_mode_checkbox, samples_mode_radio, starting_samples_input,
            preserve_optimizer_checkbox, preserve_scheduler_checkbox,
            # Learning rate parameters
            custom_lr_input, disable_lr_scaling_checkbox, custom_warmup_input, lr_min_ratio_input,
            resume_warmup_ratio_input,
            # Sampling mode
            sampling_mode_dropdown,
            # Loss-weighted sampling parameters
            loss_weight_temperature_input, loss_weight_refresh_input,
            # Divergence-based early stopping parameters
            stop_on_divergence_checkbox, divergence_gap_input, divergence_ratio_input,
            divergence_patience_input, divergence_min_updates_input,
            # Validation spike detection parameters
            val_spike_threshold_input, val_spike_window_input, val_spike_frequency_input,
            # ReduceLROnPlateau parameters
            plateau_factor_input,
        ],
        outputs=[
            batch_training_status,
            loss_vs_samples_plot,
            loss_vs_recent_plot,
            lr_vs_samples_plot,
            sample_weights_plot,  # Sample weights distribution (loss-weighted mode only)
            latest_eval_loss_plot,
            latest_eval_dist_plot,
            observation_samples_status,
            observation_samples_plot,
        ]
    )

    # Stop training button - sets flag that training loop checks
    stop_training_btn.click(
        fn=state.request_training_stop,
        inputs=[],
        outputs=[]
    )

    # Pre-flight summary button
    show_preflight_btn.click(
        fn=training.generate_preflight_summary,
        inputs=[
            total_samples_input, batch_size_input,
            resume_mode_checkbox, samples_mode_radio, starting_samples_input,
            preserve_optimizer_checkbox, preserve_scheduler_checkbox,
            custom_lr_input, disable_lr_scaling_checkbox, custom_warmup_input, lr_min_ratio_input,
            resume_warmup_ratio_input,
            sampling_mode_dropdown,
            # Loss-weighted sampling parameters
            loss_weight_temperature_input, loss_weight_refresh_input,
            # Divergence-based early stopping parameters
            stop_on_divergence_checkbox, divergence_gap_input, divergence_ratio_input,
            divergence_patience_input, divergence_min_updates_input,
            # Validation spike detection parameters
            val_spike_threshold_input, val_spike_window_input, val_spike_frequency_input,
            # ReduceLROnPlateau parameters
            plateau_factor_input,
        ],
        outputs=[preflight_summary]
    )

    generate_attn_btn.click(
        fn=attention.generate_attention_visualization,
        inputs=[
            frame_number_input,
            attn_selection_mode,
            attn_brightness_threshold,
            attn_manual_patches,
            attn_quantile,
            attn_layer0,
            attn_layer1,
            attn_layer2,
            attn_layer3,
            attn_layer4,
            attn_head0,
            attn_head1,
            attn_head2,
            attn_head3,
            attn_aggregation,
            attn_selected_aggregation,
            attn_viz_type,
        ],
        outputs=[
            attn_status,
            attn_plot,
            attn_stats,
        ]
    )

    run_comparison_btn.click(
        fn=training.run_batch_comparison,
        inputs=[batch_sizes_input, comparison_total_samples_input],
        outputs=[
            comparison_status,
            comparison_summary,
            comparison_time_plot,
            comparison_quality_plot,
            comparison_convergence_plot,
            comparison_table,
        ]
    )

    # Initialize session dropdown and checkpoint dropdown on load
    def initialize_ui():
        # Initialize sessions for default robot type (so101)
        sessions = session_manager.refresh_sessions_for_type("so101")
        checkpoints = checkpoint_manager.refresh_checkpoints()
        # Initialize training info with default values (Epoch-based is the default)
        training_info = visualization.calculate_training_info(10000000, 64, "Epoch-based (shuffle each epoch)")
        # Validation dropdown with "None" option
        val_choices = session_manager.get_validation_session_choices("so101")
        return sessions, gr.Dropdown(choices=val_choices, value="None - No validation"), checkpoints, training_info

    demo.load(
        fn=initialize_ui,
        inputs=[],
        outputs=[session_dropdown, validation_session_dropdown, checkpoint_dropdown, training_info_display]
    )
