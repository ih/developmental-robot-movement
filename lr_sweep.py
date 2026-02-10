"""
Learning Rate Sweep Module

Time-budgeted learning rate sweep with two-phase optimization:
- Phase A: Broad exploration with many LRs, short time budgets
- Phase B: Deep validation with top survivors, multiple seeds

Supports parallel execution and resume from interrupted sweeps.
"""

import copy
import json
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure proper multiprocessing start method for CUDA compatibility
# On Windows/macOS, 'spawn' is default and required for CUDA
# On Linux, we force 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LRTrialResult:
    """Result from a single LR trial run."""
    lr: float
    seed: int
    phase: str  # "A" or "B"
    status: str  # "completed", "timeout", "diverged", "nan", "error"

    # Timing
    wall_time_sec: float
    samples_trained: int
    time_to_best_val_sec: float
    samples_to_best_val: int

    # Metrics
    best_val_loss: float
    final_val_loss: float
    best_train_loss: float
    final_train_loss: float

    # History (truncated to save space)
    loss_history: list  # [(samples, train_loss, val_loss), ...]

    # Checkpoint path (if saved)
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LRTrialResult":
        """Deserialize from JSON."""
        return cls(**data)


@dataclass
class LRAggregatedResult:
    """Aggregated results for a single LR across seeds."""
    lr: float
    phase: str
    num_seeds: int

    # Aggregated metrics
    median_best_val: float
    mean_best_val: float
    std_best_val: float
    min_best_val: float
    max_best_val: float

    # Timing aggregates
    median_wall_time_sec: float
    median_time_to_best_sec: float

    # Trial breakdown
    trials: list  # list[LRTrialResult]

    # Status counts
    completed_count: int
    timeout_count: int
    diverged_count: int
    nan_count: int
    error_count: int

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        d = asdict(self)
        d['trials'] = [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.trials]
        return d


@dataclass
class LRSweepPhaseResult:
    """Results from a complete LR sweep phase."""
    phase: str  # "A" or "B"
    stage_num: int

    # All LR results (aggregated)
    lr_results: list  # list[LRAggregatedResult]

    # Selected survivors (for Phase A) or winner (for Phase B)
    survivors: list  # list[float] - LR values
    winner: Optional[float]  # Single best LR (Phase B only)
    winner_best_val: Optional[float]  # Best val loss of winner
    winner_checkpoint_path: Optional[str] = None  # Checkpoint from winning trial

    # Timing
    total_wall_time_sec: float = 0.0

    def get_ranked_lrs(self, metric: str = "median_best_val") -> list:
        """Return LRs sorted by ranking metric (ascending loss)."""
        key_fn = {
            "median_best_val": lambda r: r.median_best_val,
            "mean_best_val": lambda r: r.mean_best_val,
            "min_best_val": lambda r: r.min_best_val,
        }.get(metric, lambda r: r.median_best_val)

        sorted_results = sorted(self.lr_results, key=key_fn)
        return [r.lr for r in sorted_results]

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            'phase': self.phase,
            'stage_num': self.stage_num,
            'lr_results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.lr_results],
            'survivors': self.survivors,
            'winner': self.winner,
            'winner_best_val': self.winner_best_val,
            'winner_checkpoint_path': self.winner_checkpoint_path,
            'total_wall_time_sec': self.total_wall_time_sec,
        }


@dataclass
class LRSweepStageResult:
    """Complete LR sweep results for a stage."""
    stage_num: int
    phase_a: LRSweepPhaseResult
    phase_b: LRSweepPhaseResult
    selected_lr: float
    total_wall_time_sec: float
    winning_checkpoint_path: Optional[str] = None  # Checkpoint from winning Phase B trial

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            'stage_num': self.stage_num,
            'phase_a': self.phase_a.to_dict(),
            'phase_b': self.phase_b.to_dict(),
            'selected_lr': self.selected_lr,
            'total_wall_time_sec': self.total_wall_time_sec,
            'winning_checkpoint_path': self.winning_checkpoint_path,
        }


@dataclass
class StageTiming:
    """Detailed timing breakdown for a stage."""
    stage_num: int

    # LR Sweep timing (in seconds)
    lr_sweep_total_sec: float
    lr_sweep_phase_a_sec: float
    lr_sweep_phase_b_sec: float
    lr_sweep_trial_count: int

    # Main training timing
    main_training_sec: float
    main_training_samples: int

    # Total
    total_stage_sec: float

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return asdict(self)

    def to_html_table_row(self) -> str:
        """Generate HTML table row for timing breakdown."""
        return f"""
        <tr>
            <td>Stage {self.stage_num}</td>
            <td>{format_duration(self.lr_sweep_phase_a_sec)}</td>
            <td>{format_duration(self.lr_sweep_phase_b_sec)}</td>
            <td>{format_duration(self.lr_sweep_total_sec)}</td>
            <td>{format_duration(self.main_training_sec)}</td>
            <td>{format_duration(self.total_stage_sec)}</td>
        </tr>
        """


# =============================================================================
# Utility Functions
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def cleanup_gpu_memory():
    """
    Force GPU memory cleanup after parallel training.

    Call this after ProcessPoolExecutor completes to free cached GPU memory
    that may not be immediately released when worker processes terminate.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_max_parallel_workers(verbose: bool = True) -> int:
    """
    Determine max parallel workers based on available GPU memory.

    Each worker needs ~2-4GB GPU memory depending on batch size.
    Returns conservative estimate to avoid OOM.
    """
    if not torch.cuda.is_available():
        cpu_count = mp.cpu_count()
        if verbose:
            print(f"[WORKERS] CUDA not available in main process, using CPU count: {cpu_count}")
        return cpu_count

    device_count = torch.cuda.device_count()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    memory_per_worker_gb = 3.0  # Conservative estimate
    max_gpu_workers = max(1, int(gpu_memory_gb / memory_per_worker_gb))

    # If multiple GPUs, can run more workers
    if device_count > 1:
        max_gpu_workers = max_gpu_workers * device_count

    result = min(max_gpu_workers, mp.cpu_count())

    if verbose:
        print(f"[WORKERS] CUDA available: {device_count} GPU(s), {gpu_memory_gb:.1f}GB memory")
        print(f"[WORKERS] Max parallel workers: {result} (GPU limit: {max_gpu_workers}, CPU count: {mp.cpu_count()})")

    return result


def _cuda_test_worker(worker_id: int):
    """Worker that tests CUDA access. Module-level for pickling."""
    import time as _time
    pid = os.getpid()
    start = _time.time()

    # Small delay to help distinguish parallel vs sequential execution
    _time.sleep(0.5)

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    result = {
        'worker_id': worker_id,
        'pid': pid,
        'start_time': start,
        'cuda_available': cuda_available,
        'device_count': device_count,
        'device_name': 'N/A',
        'memory_gb': 0,
    }

    if cuda_available:
        torch.cuda.init()
        result['device_name'] = torch.cuda.get_device_name(0)
        result['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return result


def test_parallel_cuda_workers(num_workers: int = 2, timeout: float = 30.0) -> bool:
    """
    Test that parallel workers can access CUDA.

    This is a diagnostic function to verify the parallel setup works correctly.

    Args:
        num_workers: Number of worker processes to test
        timeout: Timeout in seconds

    Returns:
        True if all workers can access CUDA, False otherwise
    """
    print(f"\n{'='*60}")
    print("Testing parallel CUDA workers...")
    print(f"{'='*60}")

    # Main process info
    print(f"\nMain process (PID {os.getpid()}):")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    print(f"\nSpawning {num_workers} worker processes...")

    results = []
    test_start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_cuda_test_worker, i): i for i in range(num_workers)}
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
                print(f"  Worker {result['worker_id']} (PID {result['pid']}): "
                      f"CUDA={result['cuda_available']}, devices={result['device_count']}")
            except Exception as e:
                print(f"  Worker failed: {e}")
                results.append({'cuda_available': False, 'error': str(e)})

    total_time = time.time() - test_start
    expected_sequential = 0.5 * num_workers  # Each worker sleeps 0.5s
    is_parallel = total_time < expected_sequential * 0.8  # Allow 20% margin

    print(f"\nTiming: {total_time:.2f}s (expected sequential: {expected_sequential:.1f}s)")
    print(f"Parallelism: {'[OK] Workers ran in parallel' if is_parallel else '[INFO] Workers may have run sequentially'}")

    all_cuda = all(r.get('cuda_available', False) for r in results)
    print(f"\nResult: {'[OK] All workers have CUDA access' if all_cuda else '[FAIL] Some workers lack CUDA access'}")
    print(f"{'='*60}\n")

    return all_cuda


# =============================================================================
# LR Sampling
# =============================================================================

def sample_learning_rates_log_uniform(lr_min: float, lr_max: float, n: int) -> list:
    """
    Generate n LR values in log-uniform distribution.

    Args:
        lr_min: Minimum learning rate
        lr_max: Maximum learning rate
        n: Number of LR values to generate

    Returns:
        List of n learning rates evenly spaced in log scale
    """
    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)
    log_values = np.linspace(log_min, log_max, n)
    return [10 ** v for v in log_values]


# =============================================================================
# Aggregation and Ranking
# =============================================================================

def aggregate_lr_trials(lr: float, trials: list, phase: str) -> LRAggregatedResult:
    """
    Aggregate multiple trials for the same LR.

    Args:
        lr: Learning rate value
        trials: List of LRTrialResult for this LR
        phase: Phase identifier ("A" or "B")

    Returns:
        LRAggregatedResult with aggregated metrics
    """
    # Filter out invalid results
    valid_trials = [t for t in trials if t.best_val_loss < float('inf')]
    best_vals = [t.best_val_loss for t in valid_trials]
    wall_times = [t.wall_time_sec for t in trials]
    times_to_best = [t.time_to_best_val_sec for t in valid_trials if t.time_to_best_val_sec > 0]

    # Count statuses
    status_counts = {
        'completed': sum(1 for t in trials if t.status == "completed"),
        'timeout': sum(1 for t in trials if t.status == "timeout"),
        'diverged': sum(1 for t in trials if t.status == "diverged"),
        'nan': sum(1 for t in trials if t.status == "nan"),
        'error': sum(1 for t in trials if t.status.startswith("error")),
    }

    return LRAggregatedResult(
        lr=lr,
        phase=phase,
        num_seeds=len(trials),
        median_best_val=float(np.median(best_vals)) if best_vals else float('inf'),
        mean_best_val=float(np.mean(best_vals)) if best_vals else float('inf'),
        std_best_val=float(np.std(best_vals)) if len(best_vals) > 1 else 0.0,
        min_best_val=float(min(best_vals)) if best_vals else float('inf'),
        max_best_val=float(max(best_vals)) if best_vals else float('inf'),
        median_wall_time_sec=float(np.median(wall_times)) if wall_times else 0.0,
        median_time_to_best_sec=float(np.median(times_to_best)) if times_to_best else 0.0,
        trials=trials,
        completed_count=status_counts['completed'],
        timeout_count=status_counts['timeout'],
        diverged_count=status_counts['diverged'],
        nan_count=status_counts['nan'],
        error_count=status_counts['error'],
    )


def select_survivors(
    aggregated_results: list,
    count: int,
    metric: str = "median_best_val",
) -> list:
    """
    Select top LRs by ranking metric.

    Args:
        aggregated_results: List of LRAggregatedResult
        count: Number of survivors to select
        metric: Ranking metric ("median_best_val", "mean_best_val", "min_best_val")

    Returns:
        List of LR values (floats) for top performers
    """
    # Filter out LRs with all failed trials
    valid_results = [r for r in aggregated_results if r.median_best_val < float('inf')]

    if not valid_results:
        # Fallback: return first `count` LRs even if they failed
        return [r.lr for r in aggregated_results[:count]]

    # Sort by metric (ascending - lower loss is better)
    key_fn = {
        "median_best_val": lambda r: r.median_best_val,
        "mean_best_val": lambda r: r.mean_best_val,
        "min_best_val": lambda r: r.min_best_val,
    }.get(metric, lambda r: r.median_best_val)

    sorted_results = sorted(valid_results, key=key_fn)

    num_survivors = min(count, len(sorted_results))
    return [r.lr for r in sorted_results[:num_survivors]]


# =============================================================================
# Sweep State Persistence (Resume Support)
# =============================================================================

def save_sweep_state(
    output_dir: Path,
    phase: str,
    stage_num: int,
    trials: dict,
    phase_a_result: Optional[LRSweepPhaseResult] = None,
) -> None:
    """
    Save intermediate sweep state for resume.

    Args:
        output_dir: Directory to save state
        phase: Current phase ("A" or "B")
        stage_num: Stage number
        trials: Dict mapping LR to list of LRTrialResult
        phase_a_result: Completed Phase A result (if in Phase B)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'phase': phase,
        'stage_num': stage_num,
        'completed_trials': {
            str(lr): [t.to_dict() for t in trial_list]
            for lr, trial_list in trials.items()
        },
        'timestamp': datetime.now().isoformat(),
    }

    if phase_a_result:
        state['phase_a_result'] = phase_a_result.to_dict()

    state_path = output_dir / "sweep_state.json"
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def load_sweep_state(output_dir: Path) -> Optional[dict]:
    """
    Load saved sweep state for resume.

    Args:
        output_dir: Directory containing sweep state

    Returns:
        State dict if exists, None otherwise
    """
    state_path = Path(output_dir) / "sweep_state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return None


# =============================================================================
# Trial Runner (will be implemented with training integration)
# =============================================================================

def run_lr_trial(
    lr: float,
    seed: int,
    phase: str,
    time_budget_min: float,
    train_session_path: str,
    val_session_path: str,
    starting_checkpoint: Optional[str],
    cfg_dict: dict,
    output_dir: str,
    run_id: str,
) -> LRTrialResult:
    """
    Run a single LR trial with time budget enforcement.

    This function runs in a separate process for parallel execution.
    It loads the sessions, creates the model, and trains with time budget.

    Args:
        lr: Learning rate to test
        seed: Random seed for reproducibility
        phase: Phase identifier ("A" or "B")
        time_budget_min: Time budget in minutes
        train_session_path: Path to training session
        val_session_path: Path to validation session
        starting_checkpoint: Optional checkpoint to load
        cfg_dict: Configuration dictionary
        output_dir: Output directory for this trial
        run_id: Run identifier

    Returns:
        LRTrialResult with trial metrics
    """
    # Import here to avoid circular imports and ensure each process has its own imports
    from staged_training_config import StagedTrainingConfig
    from staged_training import (
        load_session_for_training,
        setup_world_model,
    )
    from concat_world_model_explorer import state, training

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Reconstruct config with LR override (use from_dict to handle nested LRSweepConfig)
    cfg = StagedTrainingConfig.from_dict(cfg_dict)
    cfg.custom_lr = lr
    cfg.disable_lr_scaling = True

    # Initialize tracking
    start_time = time.time()
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    time_to_best_val = 0.0
    samples_to_best_val = 0
    loss_history = []
    history_update_counter = 0
    status = "completed"
    samples_trained = 0
    final_train_loss = float('inf')
    final_val_loss = float('inf')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load sessions
        train_session = load_session_for_training(train_session_path)
        val_session = load_session_for_training(val_session_path)

        # Setup world model
        setup_world_model(train_session, starting_checkpoint, cfg)
        state.validation_session_state = val_session

        # Compute effective samples (dynamic budget)
        from config import AutoencoderConcatPredictorWorldModelConfig
        canvas_history_size = AutoencoderConcatPredictorWorldModelConfig.CANVAS_HISTORY_SIZE
        num_valid_frames = len(train_session["observations"]) - (canvas_history_size - 1)

        if cfg.stage_samples_multiplier > 0:
            effective_total_samples = num_valid_frames * cfg.stage_samples_multiplier
        else:
            effective_total_samples = cfg.total_samples

        # Run training with time budget
        generator = training.run_world_model_batch(
            total_samples=effective_total_samples,
            batch_size=cfg.batch_size,
            current_observation_idx=cfg.selected_frame_offset,
            update_interval=cfg.update_interval,
            window_size=cfg.window_size,
            custom_lr=lr,
            disable_lr_scaling=True,
            custom_warmup=cfg.custom_warmup,
            lr_min_ratio=cfg.lr_min_ratio,
            resume_warmup_ratio=cfg.resume_warmup_ratio,
            sampling_mode=cfg.sampling_mode,
            loss_weight_temperature=cfg.loss_weight_temperature,
            loss_weight_refresh_interval=cfg.loss_weight_refresh_interval,
            stop_on_divergence=cfg.stop_on_divergence,
            divergence_gap=cfg.divergence_gap,
            divergence_ratio=cfg.divergence_ratio,
            divergence_patience=cfg.divergence_patience,
            divergence_min_updates=cfg.divergence_min_updates,
            val_spike_threshold=cfg.val_spike_threshold,
            val_spike_window=cfg.val_spike_window,
            val_spike_frequency=cfg.val_spike_frequency,
            # Disable plateau features in sweep trials - trials should train to time budget
            val_plateau_patience=0,  # Disable legacy plateau early stopping
            val_plateau_min_delta=cfg.val_plateau_min_delta,
            plateau_factor=cfg.plateau_factor,
            plateau_patience=cfg.plateau_patience,
            num_best_models_to_keep=1,
            preserve_optimizer=cfg.preserve_optimizer,
            preserve_scheduler=cfg.preserve_scheduler,
            samples_mode=cfg.samples_mode,
            resume_mode=starting_checkpoint is not None,
            time_budget_min=time_budget_min,
            min_samples_for_timeout=cfg.lr_sweep.min_samples_before_timeout if hasattr(cfg, 'lr_sweep') else 1000,
            # Explicitly disable plateau sweep in trial workers to prevent re-triggering
            plateau_sweep_enabled=False,
        )

        for result in generator:
            if result is None:
                continue

            # Extract metrics from cumulative_metrics
            samples_seen_list = state.cumulative_metrics.get('samples_seen', [])
            train_loss_list = state.cumulative_metrics.get('loss_at_sample', [])
            val_loss_list = state.cumulative_metrics.get('val_loss_at_sample', [])

            if samples_seen_list:
                samples_trained = samples_seen_list[-1]

            if train_loss_list:
                final_train_loss = train_loss_list[-1]
                if final_train_loss < best_train_loss:
                    best_train_loss = final_train_loss

            if val_loss_list:
                final_val_loss = val_loss_list[-1]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    time_to_best_val = time.time() - start_time
                    samples_to_best_val = samples_trained

            # Record history (sample every 10th update to avoid huge logs)
            history_update_counter += 1
            if history_update_counter % 10 == 0:
                train_val = train_loss_list[-1] if train_loss_list else 0
                val_val = val_loss_list[-1] if val_loss_list else 0
                loss_history.append((samples_trained, train_val, val_val))

            # Check for stop conditions in result
            if isinstance(result, tuple) and len(result) > 0:
                status_msg = result[0] if isinstance(result[0], str) else ""
                if "timeout" in status_msg.lower():
                    status = "timeout"
                elif "diverged" in status_msg.lower() or "divergence" in status_msg.lower():
                    status = "diverged"
                elif "nan" in status_msg.lower():
                    status = "nan"

    except Exception as e:
        status = f"error: {str(e)}"
        import traceback
        traceback.print_exc()

    wall_time = time.time() - start_time

    # Truncate history to last 100 points
    if len(loss_history) > 100:
        loss_history = loss_history[-100:]

    # Extract best checkpoint path from auto-saved checkpoints
    best_checkpoint_path = None
    try:
        auto_saved_checkpoints = state.cumulative_metrics.get('auto_saved_checkpoints', [])
        if auto_saved_checkpoints:
            # Sort by loss (ascending) - list contains (loss, filepath) tuples
            auto_saved_checkpoints.sort(key=lambda x: x[0])
            _, best_checkpoint_path = auto_saved_checkpoints[0]
    except Exception:
        pass  # Silently ignore if checkpoints not available

    # Cleanup GPU memory in worker process before returning
    try:
        if state.world_model is not None:
            del state.world_model
            state.world_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # Silently ignore cleanup errors

    return LRTrialResult(
        lr=lr,
        seed=seed,
        phase=phase,
        status=status,
        wall_time_sec=wall_time,
        samples_trained=samples_trained,
        time_to_best_val_sec=time_to_best_val,
        samples_to_best_val=samples_to_best_val,
        best_val_loss=best_val_loss,
        final_val_loss=final_val_loss,
        best_train_loss=best_train_loss,
        final_train_loss=final_train_loss,
        loss_history=loss_history,
        checkpoint_path=best_checkpoint_path,
    )


def _run_trial_worker(args: tuple) -> LRTrialResult:
    """
    Worker function for parallel trial execution.
    Runs in separate process with its own CUDA context.
    """
    import os

    # Extract worker_index from end of args tuple for GPU assignment
    *trial_args, worker_index = args
    worker_id = os.getpid()

    # Initialize CUDA in worker process
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        # Round-robin GPU assignment using worker index (not PID)
        if device_count > 1:
            device_id = worker_index % device_count
            torch.cuda.set_device(device_id)
            print(f"[Worker {worker_id}] Using CUDA device {device_id}/{device_count} (worker_index={worker_index})")
        else:
            print(f"[Worker {worker_id}] Using CUDA device 0 (single GPU)")
        # Force CUDA initialization
        torch.cuda.init()
    else:
        print(f"[Worker {worker_id}] CUDA not available, using CPU")

    return run_lr_trial(*trial_args)


# =============================================================================
# Phase Runner
# =============================================================================

def run_lr_sweep_phase(
    phase: str,
    stage_num: int,
    lr_candidates: list,
    seeds_per_lr: int,
    time_budget_min: float,
    train_session_path: str,
    val_session_path: str,
    starting_checkpoint: Optional[str],
    cfg_dict: dict,
    output_dir: Path,
    run_id: str,
    ranking_metric: str = "median_best_val",
    survivor_count: int = 5,
    save_state: bool = True,
    max_workers: Optional[int] = None,
) -> LRSweepPhaseResult:
    """
    Run a complete LR sweep phase with parallel execution.

    Args:
        phase: Phase identifier ("A" or "B")
        stage_num: Stage number
        lr_candidates: List of LR values to test
        seeds_per_lr: Number of seeds per LR
        time_budget_min: Time budget per trial in minutes
        train_session_path: Path to training session
        val_session_path: Path to validation session
        starting_checkpoint: Optional checkpoint to load
        cfg_dict: Configuration dictionary
        output_dir: Output directory
        run_id: Run identifier
        ranking_metric: Metric for ranking ("median_best_val", etc.)
        survivor_count: Number of survivors to select (Phase A)
        save_state: Whether to save intermediate state
        max_workers: Max parallel workers (None = auto)

    Returns:
        LRSweepPhaseResult with phase results
    """
    phase_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LR SWEEP Phase {phase}: Stage {stage_num}")
    print(f"Testing {len(lr_candidates)} LRs x {seeds_per_lr} seeds = {len(lr_candidates) * seeds_per_lr} trials")
    print(f"Time budget: {time_budget_min:.1f} min per trial")
    print(f"{'='*60}\n")

    # Build list of trial args (worker_index appended for GPU assignment)
    trial_args = []
    worker_index = 0
    for lr in lr_candidates:
        for seed in range(seeds_per_lr):
            trial_dir = output_dir / f"phase_{phase}" / f"lr_{lr:.2e}" / f"seed_{seed}"
            trial_args.append((
                lr, seed, phase, time_budget_min,
                train_session_path, val_session_path,
                starting_checkpoint, cfg_dict, str(trial_dir), run_id,
                worker_index
            ))
            worker_index += 1

    # Run trials in parallel
    all_trials: dict = {lr: [] for lr in lr_candidates}
    workers = max_workers or get_max_parallel_workers(verbose=True)

    # Cap workers for single-GPU systems to avoid OOM
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        # On single GPU, limit concurrency to avoid memory issues
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        safe_workers = max(1, int(gpu_memory_gb / 4.0))  # ~4GB per worker for safety
        if workers > safe_workers:
            print(f"[LR SWEEP] Reducing workers from {workers} to {safe_workers} for single-GPU safety")
            workers = safe_workers

    print(f"[LR SWEEP] Running {len(trial_args)} trials with {workers} parallel workers")
    print(f"[LR SWEEP] Start time: {datetime.now().strftime('%H:%M:%S')}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_trial_worker, args): args for args in trial_args}

        for i, future in enumerate(as_completed(futures)):
            args = futures[future]
            lr = args[0]
            seed = args[1]
            completion_time = datetime.now().strftime('%H:%M:%S')
            try:
                result = future.result()
                all_trials[lr].append(result)
                print(f"[LR SWEEP] [{i+1}/{len(trial_args)}] [{completion_time}] Phase {phase}: "
                      f"LR={lr:.2e} seed={seed} -> {result.status} "
                      f"(best_val={result.best_val_loss:.6f}, time={result.wall_time_sec:.1f}s)")
            except Exception as e:
                print(f"[LR SWEEP] [{i+1}/{len(trial_args)}] [{completion_time}] Phase {phase}: "
                      f"LR={lr:.2e} seed={seed} FAILED: {e}")
                # Create error result
                all_trials[lr].append(LRTrialResult(
                    lr=lr, seed=seed, phase=phase, status=f"error: {e}",
                    wall_time_sec=0, samples_trained=0, time_to_best_val_sec=0,
                    samples_to_best_val=0, best_val_loss=float('inf'),
                    final_val_loss=float('inf'), best_train_loss=float('inf'),
                    final_train_loss=float('inf'), loss_history=[],
                ))

            # Save intermediate state
            if save_state:
                save_sweep_state(output_dir, phase, stage_num, all_trials)

    # Force GPU memory cleanup after all workers complete
    cleanup_gpu_memory()

    # Aggregate results
    aggregated_results = [
        aggregate_lr_trials(lr, trials, phase)
        for lr, trials in all_trials.items()
    ]

    # Select survivors/winner
    survivors = select_survivors(aggregated_results, survivor_count, ranking_metric)
    winner = survivors[0] if phase == "B" and survivors else None

    # Get winner's best val loss and checkpoint path
    winner_best_val = None
    winner_checkpoint_path = None
    if winner:
        for r in aggregated_results:
            if r.lr == winner:
                winner_best_val = r.median_best_val
                # Find the trial with the best val loss to get its checkpoint
                if r.trials:
                    best_trial = min(r.trials, key=lambda t: t.best_val_loss)
                    winner_checkpoint_path = best_trial.checkpoint_path
                break

    total_time = time.time() - phase_start

    print(f"\n[LR SWEEP] Phase {phase} complete in {format_duration(total_time)}")
    if phase == "A":
        print(f"[LR SWEEP] Survivors: {[f'{lr:.2e}' for lr in survivors]}")
    else:
        checkpoint_info = f", checkpoint={winner_checkpoint_path}" if winner_checkpoint_path else ""
        print(f"[LR SWEEP] Winner: {winner:.2e} (median_best_val={winner_best_val:.6f}{checkpoint_info})")

    return LRSweepPhaseResult(
        phase=phase,
        stage_num=stage_num,
        lr_results=aggregated_results,
        survivors=survivors,
        winner=winner,
        winner_best_val=winner_best_val,
        winner_checkpoint_path=winner_checkpoint_path,
        total_wall_time_sec=total_time,
    )


# =============================================================================
# Stage-Level Sweep
# =============================================================================

def run_lr_sweep_for_stage(
    stage_num: int,
    train_session_path: str,
    val_session_path: str,
    starting_checkpoint: Optional[str],
    cfg_dict: dict,
    output_dir: Path,
    run_id: str,
    lr_min: float = 1e-6,
    lr_max: float = 1e-2,
    phase_a_num_candidates: int = 40,
    phase_a_seeds: int = 1,
    phase_a_time_budget_min: float = 3.0,
    phase_a_survivor_count: int = 5,
    phase_b_seeds: int = 3,
    phase_b_time_budget_min: float = 10.0,
    ranking_metric: str = "median_best_val",
    save_state: bool = True,
    max_workers: Optional[int] = None,
    time_budget_min: float = 0,  # Total sweep time budget (0 = unlimited)
) -> LRSweepStageResult:
    """
    Run complete two-phase LR sweep for a stage.

    Phase A: Test many LRs with short time budgets
    Phase B: Test top survivors with multiple seeds

    Args:
        stage_num: Stage number
        train_session_path: Path to training session
        val_session_path: Path to validation session
        starting_checkpoint: Optional checkpoint to load
        cfg_dict: Configuration dictionary
        output_dir: Output directory
        run_id: Run identifier
        lr_min: Minimum LR for search space
        lr_max: Maximum LR for search space
        phase_a_num_candidates: Number of LRs in Phase A
        phase_a_seeds: Seeds per LR in Phase A
        phase_a_time_budget_min: Time budget per trial in Phase A (minutes)
        phase_a_survivor_count: Number of survivors from Phase A
        phase_b_seeds: Seeds per LR in Phase B
        phase_b_time_budget_min: Time budget per trial in Phase B (minutes)
        ranking_metric: Metric for ranking
        save_state: Whether to save intermediate state
        max_workers: Max parallel workers
        time_budget_min: Total sweep time budget (0 = unlimited)

    Returns:
        LRSweepStageResult with complete sweep results
    """
    sweep_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing state (resume support)
    existing_state = load_sweep_state(output_dir)
    phase_a_result = None

    if existing_state:
        print(f"[LR SWEEP] Found existing state for stage {stage_num}, phase {existing_state.get('phase')}")
        # TODO: Implement resume logic
        # For now, we'll just start fresh
        existing_state = None

    # Phase A: Broad exploration
    lr_candidates = sample_learning_rates_log_uniform(lr_min, lr_max, phase_a_num_candidates)

    # Adjust phase budgets if total sweep budget specified
    actual_phase_a_budget = phase_a_time_budget_min
    actual_phase_b_budget = phase_b_time_budget_min

    if time_budget_min > 0:
        # Estimate time allocation
        workers = max_workers or get_max_parallel_workers()
        phase_a_wall_time = phase_a_time_budget_min * math.ceil(phase_a_num_candidates * phase_a_seeds / workers)
        phase_b_wall_time = phase_b_time_budget_min * math.ceil(phase_a_survivor_count * phase_b_seeds / workers)
        total_estimated = phase_a_wall_time + phase_b_wall_time

        if total_estimated > time_budget_min:
            # Scale down budgets proportionally
            scale = time_budget_min / total_estimated
            actual_phase_a_budget = phase_a_time_budget_min * scale
            actual_phase_b_budget = phase_b_time_budget_min * scale
            print(f"[LR SWEEP] Scaling budgets by {scale:.2f} to fit total budget of {time_budget_min:.1f} min")

    phase_a_result = run_lr_sweep_phase(
        phase="A",
        stage_num=stage_num,
        lr_candidates=lr_candidates,
        seeds_per_lr=phase_a_seeds,
        time_budget_min=actual_phase_a_budget,
        train_session_path=train_session_path,
        val_session_path=val_session_path,
        starting_checkpoint=starting_checkpoint,
        cfg_dict=cfg_dict,
        output_dir=output_dir,
        run_id=run_id,
        ranking_metric=ranking_metric,
        survivor_count=phase_a_survivor_count,
        save_state=save_state,
        max_workers=max_workers,
    )

    # Phase B: Deep validation with survivors
    phase_b_result = run_lr_sweep_phase(
        phase="B",
        stage_num=stage_num,
        lr_candidates=phase_a_result.survivors,
        seeds_per_lr=phase_b_seeds,
        time_budget_min=actual_phase_b_budget,
        train_session_path=train_session_path,
        val_session_path=val_session_path,
        starting_checkpoint=starting_checkpoint,
        cfg_dict=cfg_dict,
        output_dir=output_dir,
        run_id=run_id,
        ranking_metric=ranking_metric,
        survivor_count=1,  # Select single winner
        save_state=save_state,
        max_workers=max_workers,
    )

    selected_lr = phase_b_result.winner
    winning_checkpoint = phase_b_result.winner_checkpoint_path

    # Fallback if all Phase B trials failed (winner is None)
    if selected_lr is None:
        if phase_a_result.survivors:
            selected_lr = phase_a_result.survivors[0]
            print(f"[LR SWEEP] WARNING: All Phase B trials failed. Falling back to best Phase A survivor: {selected_lr:.2e}")
        else:
            # Extract default LR from config
            from staged_training_config import StagedTrainingConfig
            fallback_cfg = StagedTrainingConfig.from_dict(cfg_dict)
            selected_lr = fallback_cfg.custom_lr
            print(f"[LR SWEEP] WARNING: All sweep trials failed. Falling back to config default LR: {selected_lr:.2e}")

    # Fallback checkpoint: use Phase A winner's checkpoint if Phase B has none
    if winning_checkpoint is None and phase_a_result.lr_results:
        for r in phase_a_result.lr_results:
            if r.lr == selected_lr and r.trials:
                best_trial = min(r.trials, key=lambda t: t.best_val_loss)
                if best_trial.checkpoint_path:
                    winning_checkpoint = best_trial.checkpoint_path
                    print(f"[LR SWEEP] Using Phase A checkpoint as fallback: {winning_checkpoint}")
                break

    total_time = time.time() - sweep_start

    print(f"\n{'='*60}")
    print(f"LR SWEEP COMPLETE for Stage {stage_num}")
    print(f"Selected LR: {selected_lr:.2e}")
    if winning_checkpoint:
        print(f"Winning checkpoint: {winning_checkpoint}")
    print(f"Total sweep time: {format_duration(total_time)}")
    print(f"{'='*60}\n")

    # Final GPU memory cleanup after stage sweep
    cleanup_gpu_memory()

    return LRSweepStageResult(
        stage_num=stage_num,
        phase_a=phase_a_result,
        phase_b=phase_b_result,
        selected_lr=selected_lr,
        total_wall_time_sec=total_time,
        winning_checkpoint_path=winning_checkpoint,
    )


def run_plateau_triggered_sweep(
    sweep_number: int,
    train_session_path: str,
    val_session_path: str,
    current_checkpoint: str,
    cfg_dict: dict,
    output_dir: Path,
    run_id: str,
    max_workers: Optional[int] = None,
) -> tuple:
    """
    Run LR sweep triggered by plateau detection.

    This is a wrapper around run_lr_sweep_for_stage() for mid-training use.
    Uses the same Phase A → Phase B multi-phase sweep structure.

    Key differences from upfront sweep:
    1. REQUIRES current_checkpoint (starts from current weights, not fresh)
    2. Returns tuple of (selected_lr, winning_checkpoint_path, winner_best_val) for easy unpacking
    3. Uses phase budgets from lr_sweep config directly

    Args:
        sweep_number: Which sweep this is (1, 2, 3, ...)
        train_session_path: Path to training session
        val_session_path: Path to validation session
        current_checkpoint: Current checkpoint path (REQUIRED - starts from current weights)
        cfg_dict: Configuration dictionary (includes LRSweepConfig and PlateauSweepConfig)
        output_dir: Output directory for this sweep
        run_id: Run identifier
        max_workers: Maximum parallel workers

    Returns:
        Tuple of (selected_lr, winning_checkpoint_path, winner_best_val)
    """
    from staged_training_config import StagedTrainingConfig

    # Reconstruct config to get LR sweep parameters
    cfg = StagedTrainingConfig.from_dict(cfg_dict)

    print(f"\n{'='*60}")
    print(f"PLATEAU-TRIGGERED LR SWEEP #{sweep_number}")
    print(f"Starting from checkpoint: {current_checkpoint}")
    print(f"Phase A budget: {cfg.lr_sweep.phase_a_time_budget_min:.1f} min, Phase B budget: {cfg.lr_sweep.phase_b_time_budget_min:.1f} min")
    print(f"{'='*60}\n")

    # Create sweep output directory
    sweep_dir = output_dir / f"plateau_sweep_{sweep_number}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Run the full Phase A → Phase B sweep using phase budgets directly
    result = run_lr_sweep_for_stage(
        stage_num=0,  # Not used for plateau sweeps, use sweep_number for identification
        train_session_path=train_session_path,
        val_session_path=val_session_path,
        starting_checkpoint=current_checkpoint,
        cfg_dict=cfg_dict,
        output_dir=sweep_dir,
        run_id=f"{run_id}_plateau_sweep_{sweep_number}",
        lr_min=cfg.lr_sweep.lr_min,
        lr_max=cfg.lr_sweep.lr_max,
        phase_a_num_candidates=cfg.lr_sweep.phase_a_num_candidates,
        phase_a_seeds=cfg.lr_sweep.phase_a_seeds,
        phase_a_time_budget_min=cfg.lr_sweep.phase_a_time_budget_min,
        phase_a_survivor_count=cfg.lr_sweep.phase_a_survivor_count,
        phase_b_seeds=cfg.lr_sweep.phase_b_seeds,
        phase_b_time_budget_min=cfg.lr_sweep.phase_b_time_budget_min,
        ranking_metric=cfg.lr_sweep.ranking_metric,
        save_state=cfg.lr_sweep.save_sweep_state,
        max_workers=max_workers,
    )

    print(f"\n[PLATEAU SWEEP #{sweep_number}] Complete")
    print(f"  Selected LR: {result.selected_lr:.2e}")
    print(f"  Checkpoint: {result.winning_checkpoint_path}")
    print(f"  Total time: {format_duration(result.total_wall_time_sec)}")

    return (result.selected_lr, result.winning_checkpoint_path, result.phase_b.winner_best_val)


# =============================================================================
# Parallel Main Training
# =============================================================================

def run_main_training_parallel(
    stage_num: int,
    num_runs: int,
    train_session_path: str,
    val_session_path: str,
    starting_checkpoint: Optional[str],
    selected_lr: float,
    cfg_dict: dict,
    output_dir: Path,
    run_id: str,
    is_baseline: bool,
    time_budget_min: float = 0,
    max_workers: Optional[int] = None,
) -> list:
    """
    Run multiple training runs in parallel.

    Used for both:
    - Staged training: runs_per_stage runs with weight carryover
    - Baseline training: baseline_runs_per_stage runs with fresh weights

    Args:
        stage_num: Stage number
        num_runs: Number of runs
        train_session_path: Path to training session
        val_session_path: Path to validation session
        starting_checkpoint: Checkpoint to load (None for baseline)
        selected_lr: Learning rate from sweep
        cfg_dict: Configuration dictionary
        output_dir: Output directory
        run_id: Run identifier
        is_baseline: Whether this is baseline training
        time_budget_min: Time budget per run in minutes (0 = unlimited)
        max_workers: Max parallel workers

    Returns:
        List of StageResult from each run
    """
    from staged_training import StageResult

    output_dir = Path(output_dir)
    prefix = "baseline_" if is_baseline else ""

    print(f"\n[TRAINING] Running {num_runs} {prefix}runs in parallel for stage {stage_num}")
    print(f"[TRAINING] Selected LR: {selected_lr:.2e}")

    # Build list of run args (1-based indexing for consistency with staged_training.py)
    run_args = []
    for run_num in range(1, num_runs + 1):
        run_output_dir = output_dir / f"stage{stage_num}_{prefix}run{run_num}"
        seed = hash((run_id, stage_num, run_num, is_baseline)) % (2**32)

        # Create modified config with selected LR
        run_cfg = cfg_dict.copy()
        run_cfg['custom_lr'] = selected_lr
        run_cfg['disable_lr_scaling'] = True

        run_args.append((
            stage_num, run_num, train_session_path, val_session_path,
            starting_checkpoint if not is_baseline else None,
            run_cfg, str(run_output_dir), run_id, is_baseline,
            time_budget_min, seed
        ))

    # Run in parallel
    results = []
    workers = max_workers or get_max_parallel_workers(verbose=True)

    # Cap workers for single-GPU systems
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        safe_workers = max(1, int(gpu_memory_gb / 4.0))
        if workers > safe_workers:
            print(f"[TRAINING] Reducing workers from {workers} to {safe_workers} for single-GPU safety")
            workers = safe_workers

    print(f"[TRAINING] Using {workers} parallel workers, start time: {datetime.now().strftime('%H:%M:%S')}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_main_training_worker, args): args for args in run_args}

        for future in as_completed(futures):
            args = futures[future]
            run_num = args[1]
            try:
                result = future.result()
                results.append(result)
                print(f"[TRAINING] Stage {stage_num} {prefix}run {run_num} complete: "
                      f"loss={result.best_loss:.6f}, time={result.elapsed_time:.1f}s")
            except Exception as e:
                print(f"[TRAINING] Stage {stage_num} {prefix}run {run_num} FAILED: {e}")
                results.append(StageResult(
                    stage_num=stage_num,
                    run_num=run_num,
                    is_baseline=is_baseline,
                    best_checkpoint_path="",
                    best_loss=float('inf'),
                    stop_reason=f"error: {e}",
                    elapsed_time=0,
                    total_samples_trained=0,
                    cumulative_metrics={},
                    final_train_loss=float('inf'),
                    final_val_loss=None,
                ))

    return results


def _run_main_training_worker(args: tuple):
    """Worker function for parallel main training."""
    (stage_num, run_num, train_session_path, val_session_path,
     starting_checkpoint, cfg_dict, output_dir, run_id, is_baseline,
     time_budget_min, seed) = args

    # Import here to avoid circular imports
    from staged_training import run_stage_training, load_session_for_training
    from staged_training_config import StagedTrainingConfig

    worker_id = os.getpid()

    # Initialize CUDA in worker process
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = run_num % device_count
            torch.cuda.set_device(device_id)
            print(f"[Training Worker {worker_id}] Using CUDA device {device_id}/{device_count}")
        else:
            print(f"[Training Worker {worker_id}] Using CUDA device 0 (single GPU)")
        torch.cuda.init()
    else:
        print(f"[Training Worker {worker_id}] CUDA not available, using CPU")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Reconstruct config (use from_dict to handle nested LRSweepConfig)
    cfg = StagedTrainingConfig.from_dict(cfg_dict)

    # Load sessions
    train_session = load_session_for_training(train_session_path)
    val_session = load_session_for_training(val_session_path)

    # Run training
    return run_stage_training(
        stage_num=stage_num,
        run_num=run_num,
        train_session=train_session,
        val_session=val_session,
        checkpoint_path=starting_checkpoint,
        cfg=cfg,
        output_dir=Path(output_dir),
        run_id=run_id,
        is_baseline=is_baseline,
        time_budget_min=time_budget_min,
    )


# =============================================================================
# CLI for Diagnostics
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LR Sweep module diagnostics")
    parser.add_argument("--test-cuda", action="store_true",
                        help="Test parallel CUDA workers")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of workers to test (default: 2)")

    args = parser.parse_args()

    if args.test_cuda:
        test_parallel_cuda_workers(num_workers=args.workers)
    else:
        print("LR Sweep Module")
        print("="*40)
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"Max parallel workers: {get_max_parallel_workers()}")
        print()
        print("Run with --test-cuda to test parallel worker CUDA access")
