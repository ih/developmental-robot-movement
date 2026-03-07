"""
Overfit Test Script

Tests whether a model can overfit small subsets of examples from a session.
For each subset, trains a fresh model and reports convergence behavior.

Usage:
    # Basic usage
    python overfit_test.py --session saved/sessions/so101/my_session --batch-size 4

    # With YAML config file
    python overfit_test.py --session saved/sessions/so101/my_session --config my_config.yaml

    # CLI args override config values
    python overfit_test.py --session saved/sessions/so101/my_session --config my_config.yaml \
        --batch-size 8 --learning-rate 1e-3

    # Focal loss overrides
    python overfit_test.py --session saved/sessions/so101/my_session --batch-size 4 \
        --focal-alpha 0.5 --focal-beta 10

    # Model type override
    python overfit_test.py --session saved/sessions/so101/my_session --batch-size 4 \
        --model-type dit --vae-type pretrained_sd
"""

import argparse
import base64
import gc
import io
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import world_model_utils
from staged_training import (
    load_session_for_training,
    setup_world_model,
    get_available_actions,
    fig_to_base64,
    format_duration,
    format_loss_safe,
    apply_model_config,
)
from staged_training_config import StagedTrainingConfig
from overfit_test_config import OverfitTestConfig
from concat_world_model_explorer import state
from concat_world_model_explorer import inference
from models.autoencoder_concat_predictor import (
    canvas_to_tensor,
    compute_randomized_patch_mask_for_last_slot_gpu,
)
from models.canvas_dataset import PrebuiltCanvasDataset, CanvasCollateFn


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OverfitResult:
    """Result of training a single subset to overfitting."""
    subset_idx: int
    frame_indices: tuple
    converged: bool
    final_loss: float
    best_loss: float
    iterations: int
    iterations_to_target: Optional[int]
    time_elapsed: float
    time_to_target: Optional[float]
    stop_reason: str  # "converged" | "plateau" | "max_iterations"
    loss_history: list
    best_checkpoint_path: Optional[str]


# =============================================================================
# Subset generation
# =============================================================================

def generate_subsets(
    valid_indices: list[int],
    batch_size: int,
    max_subsets: int,
    seed: Optional[int] = None,
) -> list[tuple[int, ...]]:
    """
    Generate subsets of valid frame indices for overfit testing.

    If the number of possible combinations <= max_subsets, enumerates all.
    Otherwise, randomly samples max_subsets unique subsets.

    Args:
        valid_indices: List of valid frame indices from the session
        batch_size: Number of frames per subset
        max_subsets: Maximum number of subsets to generate
        seed: Random seed for reproducibility

    Returns:
        List of tuples of frame indices
    """
    n = len(valid_indices)
    if batch_size > n:
        raise ValueError(
            f"batch_size ({batch_size}) exceeds number of valid frames ({n})"
        )

    total_combinations = math.comb(n, batch_size)
    print(f"Total possible subsets: C({n}, {batch_size}) = {total_combinations:,}")

    if total_combinations <= max_subsets:
        print(f"Enumerating all {total_combinations} subsets")
        subsets = [
            tuple(combo)
            for combo in itertools.combinations(valid_indices, batch_size)
        ]
    else:
        print(f"Randomly sampling {max_subsets} of {total_combinations:,} subsets")
        rng = random.Random(seed)

        # Generate unique random subsets by sampling index-combos
        seen = set()
        subsets = []
        sorted_indices = sorted(valid_indices)
        while len(subsets) < max_subsets:
            combo = tuple(sorted(rng.sample(sorted_indices, batch_size)))
            if combo not in seen:
                seen.add(combo)
                subsets.append(combo)

    return subsets


# =============================================================================
# Training loop
# =============================================================================

def train_until_overfit(
    session: dict,
    subset_idx: int,
    subset_indices: tuple[int, ...],
    max_iterations: int,
    target_loss: float,
    plateau_patience: int,
    plateau_threshold: float,
    learning_rate: float,
    checkpoint_dir: Path,
) -> OverfitResult:
    """
    Train a fresh model on a single subset until it overfits or stops.

    Args:
        session: Session state dict (with canvas_cache)
        subset_idx: Index of this subset (for naming)
        subset_indices: Frame indices to train on
        max_iterations: Maximum training iterations
        target_loss: Loss threshold for convergence
        plateau_patience: Iterations without improvement before stopping
        plateau_threshold: Minimum improvement to reset patience
        learning_rate: Fixed learning rate
        checkpoint_dir: Directory to save best checkpoint

    Returns:
        OverfitResult with training outcome
    """
    # Set up fresh model (no checkpoint)
    setup_world_model(session, checkpoint_path=None)

    model = state.world_model
    autoencoder = model.autoencoder
    optimizer = model.ae_optimizer

    # Override learning rate to fixed value (no scheduling)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # Build batch from canvas cache
    canvas_cache = session["canvas_cache"]
    dataset = PrebuiltCanvasDataset(canvas_cache, list(subset_indices))
    collate_fn = CanvasCollateFn(
        config.AutoencoderConcatPredictorWorldModelConfig,
        device=str(state.device),
        transfer_to_device=True,
    )

    # Pre-collate the single batch (we reuse it every iteration)
    all_samples = [dataset[i] for i in range(len(dataset))]
    canvas_tensor, patch_mask, frame_idx_list = collate_fn(all_samples)

    # Training state
    loss_history = []
    best_loss = float('inf')
    best_iteration = 0
    patience_counter = 0
    iterations_to_target = None
    time_to_target = None
    stop_reason = "max_iterations"
    checkpoint_path = None

    start_time = time.time()
    log_interval = max(1, max_iterations // 20)  # Log ~20 times

    autoencoder.train()

    for iteration in range(1, max_iterations + 1):
        # Forward + backward on the full subset
        loss_value, _ = autoencoder.train_on_canvas(
            canvas_tensor, patch_mask, optimizer
        )

        loss_history.append(loss_value)

        # Track best
        if loss_value < best_loss - plateau_threshold:
            best_loss = loss_value
            best_iteration = iteration
            patience_counter = 0

            # Save best checkpoint
            checkpoint_path = str(checkpoint_dir / f"subset_{subset_idx}_best.pth")
            torch.save({
                "model_state_dict": autoencoder.state_dict(),
                "loss": loss_value,
                "iteration": iteration,
                "subset_indices": subset_indices,
            }, checkpoint_path)
        else:
            patience_counter += 1

        # Check convergence
        if loss_value <= target_loss:
            if iterations_to_target is None:
                iterations_to_target = iteration
                time_to_target = time.time() - start_time
            stop_reason = "converged"
            # Keep training a bit more to see if it goes lower,
            # but stop once we've been at target for a while
            if patience_counter >= min(100, plateau_patience):
                break

        # Check plateau
        if patience_counter >= plateau_patience:
            stop_reason = "plateau"
            break

        # Progress logging
        if iteration % log_interval == 0 or iteration == 1:
            elapsed = time.time() - start_time
            print(
                f"  Subset {subset_idx} | iter {iteration}/{max_iterations} | "
                f"loss {loss_value:.6f} | best {best_loss:.6f} | "
                f"patience {patience_counter}/{plateau_patience} | "
                f"{elapsed:.1f}s"
            )

    elapsed = time.time() - start_time

    # If we reached target but never formally stopped for convergence
    if stop_reason == "max_iterations" and iterations_to_target is not None:
        stop_reason = "converged"

    converged = iterations_to_target is not None

    print(
        f"  Subset {subset_idx} DONE: {stop_reason} | "
        f"final={loss_history[-1]:.6f} best={best_loss:.6f} | "
        f"{len(loss_history)} iters | {elapsed:.1f}s"
    )

    return OverfitResult(
        subset_idx=subset_idx,
        frame_indices=subset_indices,
        converged=converged,
        final_loss=loss_history[-1] if loss_history else float('inf'),
        best_loss=best_loss,
        iterations=len(loss_history),
        iterations_to_target=iterations_to_target,
        time_elapsed=elapsed,
        time_to_target=time_to_target,
        stop_reason=stop_reason,
        loss_history=loss_history,
        best_checkpoint_path=checkpoint_path,
    )


# =============================================================================
# Inference visualization
# =============================================================================

def generate_subset_inference_images(
    session: dict,
    result: OverfitResult,
) -> dict[int, list[tuple[int, str]]]:
    """
    Generate counterfactual inference images for each frame in a subset.

    Loads the best checkpoint, then runs inference for all actions on each frame.

    Args:
        session: Session state dict
        result: OverfitResult containing checkpoint path and frame indices

    Returns:
        Dict mapping frame_idx -> list of (action, base64_image) tuples
    """
    if not result.best_checkpoint_path or not os.path.exists(result.best_checkpoint_path):
        return {}

    # Load best checkpoint
    setup_world_model(session, result.best_checkpoint_path)
    state.session_state = session

    available_actions = get_available_actions(session)
    frame_images = {}

    for frame_idx in result.frame_indices:
        action_images = []
        for action in available_actions:
            try:
                inf_result = inference.run_counterfactual_inference(frame_idx, action)
                if inf_result and len(inf_result) >= 7:
                    _, fig_true, fig_cf, fig_true_inf, fig_cf_inf, fig_diff, _ = inf_result
                    # Use the counterfactual inference composite (shows prediction for this action)
                    if fig_cf_inf is not None:
                        b64 = fig_to_base64(fig_cf_inf)
                        action_images.append((action, b64))
                    # Close other figures
                    for f in [fig_true, fig_cf, fig_true_inf, fig_diff]:
                        if f is not None:
                            plt.close(f)
            except Exception as e:
                print(f"    Error: inference frame {frame_idx} action {action}: {e}")

        frame_images[frame_idx] = action_images

    return frame_images


# =============================================================================
# Plot generation
# =============================================================================

def create_loss_curve(result: OverfitResult) -> Optional[plt.Figure]:
    """Create loss vs iteration plot for a single subset."""
    if not result.loss_history:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    iterations = list(range(1, len(result.loss_history) + 1))
    ax.plot(iterations, result.loss_history, linewidth=0.8, color='#1976D2')

    # Mark convergence point
    if result.iterations_to_target is not None:
        ax.axvline(
            x=result.iterations_to_target, color='green',
            linestyle='--', alpha=0.7, label=f'Target reached (iter {result.iterations_to_target})'
        )

    ax.axhline(y=result.best_loss, color='red', linestyle=':', alpha=0.5,
               label=f'Best loss: {result.best_loss:.6f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(f'Subset {result.subset_idx} - {result.stop_reason.upper()}')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_aggregate_loss_boxplot(results: list[OverfitResult]) -> Optional[plt.Figure]:
    """Create box plot of final losses across all subsets."""
    final_losses = [r.best_loss for r in results]
    if not final_losses:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    # Color by convergence status
    converged_losses = [r.best_loss for r in results if r.converged]
    failed_losses = [r.best_loss for r in results if not r.converged]

    data = []
    labels = []
    colors = []
    if converged_losses:
        data.append(converged_losses)
        labels.append(f'Converged ({len(converged_losses)})')
        colors.append('#4CAF50')
    if failed_losses:
        data.append(failed_losses)
        labels.append(f'Not converged ({len(failed_losses)})')
        colors.append('#f44336')

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Best Loss')
    ax.set_title('Best Loss Distribution by Convergence Status')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def create_iterations_bar_chart(results: list[OverfitResult]) -> Optional[plt.Figure]:
    """Create bar chart of iterations per subset."""
    if not results:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.6), 4))
    indices = list(range(len(results)))
    iterations = [r.iterations for r in results]
    colors = ['#4CAF50' if r.converged else '#f44336' for r in results]

    ax.bar(indices, iterations, color=colors, alpha=0.7, edgecolor='white')

    # Add target iterations markers
    for i, r in enumerate(results):
        if r.iterations_to_target is not None:
            ax.plot(i, r.iterations_to_target, 'v', color='green',
                    markersize=8, zorder=5)

    ax.set_xlabel('Subset Index')
    ax.set_ylabel('Iterations')
    ax.set_title('Training Iterations per Subset (green=converged, red=not)')
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def create_time_histogram(results: list[OverfitResult]) -> Optional[plt.Figure]:
    """Create histogram of time-to-target for converged subsets."""
    converged_times = [r.time_to_target for r in results if r.time_to_target is not None]
    if not converged_times:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(converged_times, bins=min(20, len(converged_times)),
            color='#4CAF50', alpha=0.7, edgecolor='white')
    ax.axvline(np.median(converged_times), color='red', linestyle='--',
               label=f'Median: {np.median(converged_times):.1f}s')
    ax.set_xlabel('Time to Target (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Time to Convergence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# =============================================================================
# HTML report generation
# =============================================================================

def generate_report(
    session: dict,
    results: list[OverfitResult],
    all_inference_images: dict[int, dict[int, list[tuple[int, str]]]],
    params: dict,
    output_dir: Path,
) -> str:
    """
    Generate HTML report for overfit test results.

    Args:
        session: Session state dict
        results: List of OverfitResult for each subset
        all_inference_images: {subset_idx: {frame_idx: [(action, b64), ...]}}
        params: CLI parameters dict
        output_dir: Output directory

    Returns:
        Path to generated report
    """
    # Aggregate statistics
    num_converged = sum(1 for r in results if r.converged)
    convergence_rate = num_converged / len(results) if results else 0
    best_losses = [r.best_loss for r in results]
    converged_times = [r.time_to_target for r in results if r.time_to_target is not None]

    # Generate aggregate plots
    images = {}

    fig_boxplot = create_aggregate_loss_boxplot(results)
    if fig_boxplot:
        images['loss_boxplot'] = fig_to_base64(fig_boxplot)

    fig_iters = create_iterations_bar_chart(results)
    if fig_iters:
        images['iterations_bar'] = fig_to_base64(fig_iters)

    fig_time = create_time_histogram(results)
    if fig_time:
        images['time_histogram'] = fig_to_base64(fig_time)

    # Generate per-subset loss curves
    subset_loss_images = {}
    for r in results:
        fig = create_loss_curve(r)
        if fig:
            subset_loss_images[r.subset_idx] = fig_to_base64(fig)

    # Build summary table rows
    summary_rows = ""
    for r in results:
        status_color = '#4CAF50' if r.converged else '#f44336'
        status_text = 'CONVERGED' if r.converged else r.stop_reason.upper()
        target_iter_str = str(r.iterations_to_target) if r.iterations_to_target else '-'
        target_time_str = f"{r.time_to_target:.1f}s" if r.time_to_target else '-'
        frames_str = ', '.join(str(f) for f in r.frame_indices)
        summary_rows += f"""
        <tr>
            <td>{r.subset_idx}</td>
            <td title="{frames_str}">[{frames_str}]</td>
            <td style="color: {status_color}; font-weight: bold;">{status_text}</td>
            <td>{format_loss_safe(r.best_loss)}</td>
            <td>{format_loss_safe(r.final_loss)}</td>
            <td>{r.iterations:,}</td>
            <td>{target_iter_str}</td>
            <td>{target_time_str}</td>
            <td>{format_duration(r.time_elapsed)}</td>
        </tr>"""

    # Build per-subset detail sections
    subset_sections = ""
    available_actions = get_available_actions(session)
    for r in results:
        status_text = 'CONVERGED' if r.converged else r.stop_reason.upper()
        status_emoji = '&#9989;' if r.converged else '&#10060;'
        frames_str = ', '.join(str(f) for f in r.frame_indices)
        iter_info = f"{r.iterations:,} iters, {format_duration(r.time_elapsed)}"

        # Loss curve
        loss_img = ''
        if r.subset_idx in subset_loss_images:
            loss_img = f'<img src="data:image/png;base64,{subset_loss_images[r.subset_idx]}" />'

        # Inference images
        inference_html = ''
        subset_inferences = all_inference_images.get(r.subset_idx, {})
        for frame_idx in r.frame_indices:
            frame_actions = subset_inferences.get(frame_idx, [])
            if frame_actions:
                inference_html += f'<h4>Frame {frame_idx}</h4><div class="inference-row">'
                for action, b64 in frame_actions:
                    if b64:
                        inference_html += (
                            f'<div class="inference-item">'
                            f'<span>Action {action}</span>'
                            f'<img src="data:image/png;base64,{b64}" />'
                            f'</div>'
                        )
                inference_html += '</div>'

        subset_sections += f"""
        <details {"open" if r.subset_idx < 3 else ""}>
            <summary>{status_emoji} Subset {r.subset_idx} - Frames [{frames_str}] - {status_text} ({iter_info})</summary>
            <div class="subset-detail">
                <div class="metric">
                    <strong>Best Loss:</strong> {format_loss_safe(r.best_loss)} |
                    <strong>Final Loss:</strong> {format_loss_safe(r.final_loss)} |
                    <strong>Stop Reason:</strong> {r.stop_reason}
                </div>
                {loss_img}
                <h3>Inference Visualizations</h3>
                {inference_html if inference_html else '<p>No inference images available</p>'}
            </div>
        </details>"""

    # Build config table
    config_rows = ""
    for key, val in sorted(params.items()):
        config_rows += f'<tr><td><code>{key}</code></td><td>{val}</td></tr>'

    # World model config
    WMConfig = config.AutoencoderConcatPredictorWorldModelConfig
    wm_rows = ""
    for attr in sorted(vars(WMConfig)):
        if attr.startswith('_'):
            continue
        val = getattr(WMConfig, attr)
        wm_rows += f'<tr><td><code>{attr}</code></td><td>{val}</td></tr>'

    # Assemble final HTML
    session_name = session.get('session_name', 'unknown')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Overfit Test Report - {session_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        h4 {{ color: #777; margin-top: 15px; }}
        .metric {{ background: #f0f7ff; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .metric strong {{ color: #1976D2; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .inference-row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .inference-item {{ flex: 1; min-width: 200px; text-align: center; }}
        .inference-item img {{ max-width: 300px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        details {{ margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }}
        details > summary {{ padding: 12px 15px; cursor: pointer; background: #fafafa; font-weight: bold; }}
        details > summary:hover {{ background: #f0f0f0; }}
        .subset-detail {{ padding: 15px; }}
        .convergence-good {{ color: #4CAF50; font-weight: bold; }}
        .convergence-bad {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Overfit Test Report</h1>
    <p style="color: #999;">{session_name} | {timestamp}</p>

    <section id="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Session:</strong> {session_name}<br>
            <strong>Batch Size:</strong> {params.get('batch_size', '?')}<br>
            <strong>Subsets Tested:</strong> {len(results)}<br>
            <strong>Convergence Rate:</strong>
            <span class="{"convergence-good" if convergence_rate >= 0.5 else "convergence-bad"}">
                {num_converged}/{len(results)} ({convergence_rate:.0%})
            </span><br>
            <strong>Target Loss:</strong> {params.get('target_loss', '?')}<br>
            <strong>Learning Rate:</strong> {params.get('learning_rate', '?')}<br>
            <strong>Median Best Loss:</strong> {format_loss_safe(float(np.median(best_losses)))}<br>
            <strong>Median Time to Target:</strong> {f"{np.median(converged_times):.1f}s" if converged_times else "N/A"}<br>
            <strong>Total Wall Time:</strong> {format_duration(sum(r.time_elapsed for r in results))}
        </div>

        <table>
            <tr>
                <th>#</th>
                <th>Frames</th>
                <th>Status</th>
                <th>Best Loss</th>
                <th>Final Loss</th>
                <th>Iterations</th>
                <th>Iter to Target</th>
                <th>Time to Target</th>
                <th>Total Time</th>
            </tr>
            {summary_rows}
        </table>
    </section>

    <section id="aggregate">
        <h2>Aggregate Results</h2>
        {'<img src="data:image/png;base64,' + images.get("loss_boxplot", "") + '" />' if images.get("loss_boxplot") else ""}
        {'<img src="data:image/png;base64,' + images.get("iterations_bar", "") + '" />' if images.get("iterations_bar") else ""}
        {'<img src="data:image/png;base64,' + images.get("time_histogram", "") + '" />' if images.get("time_histogram") else "<p>No subsets converged - no time histogram available.</p>"}
    </section>

    <section id="subsets">
        <h2>Per-Subset Results</h2>
        {subset_sections}
    </section>

    <section id="configuration">
        <h2>Configuration</h2>
        <details>
            <summary>Overfit Test Parameters</summary>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {config_rows}
            </table>
        </details>
        <details>
            <summary>World Model Architecture (config.py)</summary>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {wm_rows}
            </table>
        </details>
    </section>

</div>
</body>
</html>"""

    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(report_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test whether a model can overfit small subsets of examples"
    )
    parser.add_argument("--session", required=True, help="Path to session directory")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (overfit_test_config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Number of examples per subset")
    parser.add_argument("--max-subsets", type=int, default=None,
                        help="Max number of subsets to test")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max training iterations per subset")
    parser.add_argument("--target-loss", type=float, default=None,
                        help="Loss threshold for convergence")
    parser.add_argument("--plateau-patience", type=int, default=None,
                        help="Iterations without improvement before stopping")
    parser.add_argument("--plateau-threshold", type=float, default=None,
                        help="Minimum improvement to reset patience")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Fixed learning rate")
    parser.add_argument("--model-type", default=None,
                        help="Model type override (encoder_decoder, decoder_only, dit)")
    parser.add_argument("--vae-type", default=None,
                        help="VAE type override for DiT")
    parser.add_argument("--focal-alpha", type=float, default=None,
                        help="Override FOCAL_LOSS_ALPHA (blend ratio)")
    parser.add_argument("--focal-beta", type=float, default=None,
                        help="Override FOCAL_BETA (temperature)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight decay (0.0 for overfitting)")
    parser.add_argument("--encoder-embed-dim", type=int, default=None,
                        help="Override encoder embedding dimension")
    parser.add_argument("--encoder-depth", type=int, default=None,
                        help="Override encoder depth")
    parser.add_argument("--encoder-num-heads", type=int, default=None,
                        help="Override encoder number of heads")
    parser.add_argument("--decoder-embed-dim", type=int, default=None,
                        help="Override decoder embedding dimension")
    parser.add_argument("--decoder-depth", type=int, default=None,
                        help="Override decoder depth")
    parser.add_argument("--decoder-num-heads", type=int, default=None,
                        help="Override decoder number of heads")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default=None,
                        help="Report output directory")
    parser.add_argument("--run-id", default=None,
                        help="Unique run identifier (default: auto timestamp)")
    args = parser.parse_args()

    # Load config: YAML file -> defaults, then CLI overrides
    if args.config:
        print(f"Loading config from: {args.config}")
        test_config = OverfitTestConfig.from_yaml(args.config)
    else:
        test_config = OverfitTestConfig()

    # CLI args override config values (only when explicitly provided)
    cli_overrides = {
        'batch_size': args.batch_size,
        'max_subsets': args.max_subsets,
        'max_iterations': args.max_iterations,
        'target_loss': args.target_loss,
        'plateau_patience': args.plateau_patience,
        'plateau_threshold': args.plateau_threshold,
        'learning_rate': args.learning_rate,
        'model_type': args.model_type,
        'vae_type': args.vae_type,
        'focal_alpha': args.focal_alpha,
        'focal_beta': args.focal_beta,
        'weight_decay': args.weight_decay,
        'encoder_embed_dim': args.encoder_embed_dim,
        'encoder_depth': args.encoder_depth,
        'encoder_num_heads': args.encoder_num_heads,
        'decoder_embed_dim': args.decoder_embed_dim,
        'decoder_depth': args.decoder_depth,
        'decoder_num_heads': args.decoder_num_heads,
        'seed': args.seed,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            setattr(test_config, key, value)

    # batch_size is required - check after config merge
    if test_config.batch_size is None:
        parser.error("--batch-size is required (or set batch_size in config YAML)")

    # Generate run ID
    run_id = args.run_id or datetime.now().strftime("overfit_%Y%m%d_%H%M%S")

    # Set seed if provided
    if test_config.seed is not None:
        random.seed(test_config.seed)
        torch.manual_seed(test_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(test_config.seed)
        print(f"Random seed: {test_config.seed}")

    # Apply model config overrides if specified
    if test_config.model_type or test_config.vae_type:
        cfg = StagedTrainingConfig(
            model_type=test_config.model_type,
            vae_type=test_config.vae_type,
        )
        apply_model_config(cfg)

    # Apply focal loss overrides
    WMConfig = config.AutoencoderConcatPredictorWorldModelConfig
    if test_config.focal_alpha is not None:
        WMConfig.FOCAL_LOSS_ALPHA = test_config.focal_alpha
        print(f"FOCAL_LOSS_ALPHA overridden: {test_config.focal_alpha}")
    if test_config.focal_beta is not None:
        WMConfig.FOCAL_BETA = test_config.focal_beta
        print(f"FOCAL_BETA overridden: {test_config.focal_beta}")

    # Apply weight decay override
    if test_config.weight_decay is not None:
        WMConfig.WEIGHT_DECAY = test_config.weight_decay
        print(f"WEIGHT_DECAY overridden: {test_config.weight_decay}")

    # Apply encoder/decoder architecture overrides
    if test_config.encoder_embed_dim is not None:
        WMConfig.ENCODER_EMBED_DIM = test_config.encoder_embed_dim
        print(f"ENCODER_EMBED_DIM overridden: {test_config.encoder_embed_dim}")
    if test_config.encoder_depth is not None:
        WMConfig.ENCODER_DEPTH = test_config.encoder_depth
        print(f"ENCODER_DEPTH overridden: {test_config.encoder_depth}")
    if test_config.encoder_num_heads is not None:
        WMConfig.ENCODER_NUM_HEADS = test_config.encoder_num_heads
        print(f"ENCODER_NUM_HEADS overridden: {test_config.encoder_num_heads}")
    if test_config.decoder_embed_dim is not None:
        WMConfig.DECODER_EMBED_DIM = test_config.decoder_embed_dim
        print(f"DECODER_EMBED_DIM overridden: {test_config.decoder_embed_dim}")
    if test_config.decoder_depth is not None:
        WMConfig.DECODER_DEPTH = test_config.decoder_depth
        print(f"DECODER_DEPTH overridden: {test_config.decoder_depth}")
    if test_config.decoder_num_heads is not None:
        WMConfig.DECODER_NUM_HEADS = test_config.decoder_num_heads
        print(f"DECODER_NUM_HEADS overridden: {test_config.decoder_num_heads}")

    # Load session
    session_path = Path(args.session).resolve()
    print(f"\nLoading session: {session_path}")
    session = load_session_for_training(str(session_path))
    session_name = session.get("session_name", session_path.name)
    print(f"Session: {session_name}")
    print(f"Total observations: {len(session['observations'])}")
    print(f"Canvas cache size: {len(session['canvas_cache'])}")

    # Determine valid frame indices (those with pre-built canvases)
    valid_indices = sorted(session["canvas_cache"].keys())
    print(f"Valid frame indices: {len(valid_indices)}")

    if test_config.batch_size > len(valid_indices):
        print(f"ERROR: batch-size ({test_config.batch_size}) > valid frames ({len(valid_indices)})")
        return

    # Generate subsets
    print(f"\nGenerating subsets (batch_size={test_config.batch_size}, max={test_config.max_subsets})...")
    subsets = generate_subsets(valid_indices, test_config.batch_size, test_config.max_subsets, test_config.seed)
    print(f"Generated {len(subsets)} subsets")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("saved/overfit_reports") / session_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Collect params for the report
    params = {
        'session': str(session_path),
        'session_name': session_name,
        'batch_size': test_config.batch_size,
        'max_subsets': test_config.max_subsets,
        'num_subsets': len(subsets),
        'max_iterations': test_config.max_iterations,
        'target_loss': test_config.target_loss,
        'plateau_patience': test_config.plateau_patience,
        'plateau_threshold': test_config.plateau_threshold,
        'learning_rate': test_config.learning_rate,
        'model_type': test_config.model_type or config.AutoencoderConcatPredictorWorldModelConfig.MODEL_TYPE,
        'focal_alpha': WMConfig.FOCAL_LOSS_ALPHA,
        'focal_beta': WMConfig.FOCAL_BETA,
        'weight_decay': WMConfig.WEIGHT_DECAY,
        'encoder_embed_dim': WMConfig.ENCODER_EMBED_DIM,
        'encoder_depth': WMConfig.ENCODER_DEPTH,
        'encoder_num_heads': WMConfig.ENCODER_NUM_HEADS,
        'decoder_embed_dim': WMConfig.DECODER_EMBED_DIM,
        'decoder_depth': WMConfig.DECODER_DEPTH,
        'decoder_num_heads': WMConfig.DECODER_NUM_HEADS,
        'seed': test_config.seed,
        'run_id': run_id,
        'valid_frames': len(valid_indices),
    }

    # Save params and config YAML
    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)
    test_config.to_yaml(str(output_dir / "config.yaml"))

    # Train each subset
    print(f"\n{'='*60}")
    print(f"Starting overfit tests: {len(subsets)} subsets, batch_size={test_config.batch_size}")
    print(f"{'='*60}")

    results = []
    overall_start = time.time()

    for i, subset in enumerate(subsets):
        print(f"\n--- Subset {i}/{len(subsets)-1}: frames {list(subset)} ---")
        result = train_until_overfit(
            session=session,
            subset_idx=i,
            subset_indices=subset,
            max_iterations=test_config.max_iterations,
            target_loss=test_config.target_loss,
            plateau_patience=test_config.plateau_patience,
            plateau_threshold=test_config.plateau_threshold,
            learning_rate=test_config.learning_rate,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(result)

        # Free GPU memory between subsets
        if state.world_model is not None:
            del state.world_model
            state.world_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_start
    num_converged = sum(1 for r in results if r.converged)
    print(f"\n{'='*60}")
    print(f"Training complete: {num_converged}/{len(results)} converged in {format_duration(overall_elapsed)}")
    print(f"{'='*60}")

    # Generate inference images for each subset
    print("\nGenerating inference visualizations...")
    all_inference_images = {}
    for r in results:
        print(f"  Subset {r.subset_idx} ({r.stop_reason})...")
        frame_images = generate_subset_inference_images(session, r)
        all_inference_images[r.subset_idx] = frame_images

        # Free GPU memory
        if state.world_model is not None:
            del state.world_model
            state.world_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate report
    print("\nGenerating HTML report...")
    report_path = generate_report(session, results, all_inference_images, params, output_dir)
    print(f"Report saved to: {report_path}")

    # Save results JSON
    results_data = []
    for r in results:
        d = {
            'subset_idx': r.subset_idx,
            'frame_indices': list(r.frame_indices),
            'converged': r.converged,
            'final_loss': r.final_loss,
            'best_loss': r.best_loss,
            'iterations': r.iterations,
            'iterations_to_target': r.iterations_to_target,
            'time_elapsed': r.time_elapsed,
            'time_to_target': r.time_to_target,
            'stop_reason': r.stop_reason,
            'best_checkpoint_path': r.best_checkpoint_path,
        }
        results_data.append(d)

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'params': params,
            'results': results_data,
            'aggregate': {
                'num_subsets': len(results),
                'num_converged': num_converged,
                'convergence_rate': num_converged / len(results) if results else 0,
                'median_best_loss': float(np.median([r.best_loss for r in results])),
                'total_time': overall_elapsed,
            },
        }, f, indent=2, default=str)

    print(f"\nDone! Results in: {output_dir}")


if __name__ == "__main__":
    main()
