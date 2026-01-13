"""
Analyze training logs from ongoing or completed runs.

Usage:
    python analyze_training_logs.py --run-dir saved/training_logs/session_name/run_timestamp
    python analyze_training_logs.py --latest session_name
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def find_latest_run(session_name: str, log_dir: str = "saved/training_logs") -> Optional[Path]:
    """Find the most recent run directory for a session."""
    session_dir = Path(log_dir) / session_name
    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}")
        return None

    run_dirs = sorted(session_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run directories found in {session_dir}")
        return None

    return run_dirs[-1]


def load_config(run_dir: Path) -> Optional[Dict]:
    """Load training configuration."""
    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        return None


def load_interval_summaries(run_dir: Path) -> List[Dict]:
    """Load interval summaries from ongoing run."""
    summary_file = run_dir / "training_summary.jsonl"
    if not summary_file.exists():
        return []

    summaries = []
    try:
        with open(summary_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(json.loads(line))
    except Exception as e:
        print(f"Warning: Error reading summaries: {e}")

    return summaries


def load_recent_raw_metrics(run_dir: Path, num_batches: int = 100) -> List[Dict]:
    """Load most recent raw metrics from rotated files."""
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        return []

    # Find all metric files
    metric_files = sorted(raw_dir.glob("metrics_*.jsonl"))
    if not metric_files:
        return []

    # Read from most recent file(s)
    recent_metrics = []
    for metric_file in reversed(metric_files):
        try:
            with open(metric_file) as f:
                file_metrics = []
                for line in f:
                    line = line.strip()
                    if line:
                        file_metrics.append(json.loads(line))

                # Add to beginning (since we're reading in reverse file order)
                recent_metrics = file_metrics + recent_metrics

                if len(recent_metrics) >= num_batches:
                    break
        except Exception as e:
            print(f"Warning: Failed to read {metric_file}: {e}")
            continue

    return recent_metrics[-num_batches:]


def analyze_batch_composition(summaries: List[Dict]) -> Dict:
    """Analyze batch composition for averaging issues."""
    if not summaries:
        return {}

    # Look at recent intervals
    recent = summaries[-10:] if len(summaries) > 10 else summaries

    all_freqs = []
    all_samples = set()

    for s in recent:
        comp = s.get('batch_composition', {})
        freqs = comp.get('sample_frequencies', [])
        samples = comp.get('unique_samples_seen', [])

        if freqs:
            all_freqs.extend(freqs)
            all_samples.update(samples)

    if not all_freqs:
        return {}

    mean_freq = np.mean(all_freqs)
    std_freq = np.std(all_freqs)
    variance_ratio = std_freq / (mean_freq + 1e-8)

    return {
        'num_unique_samples': len(all_samples),
        'mean_frequency': mean_freq,
        'std_frequency': std_freq,
        'variance_ratio': variance_ratio,
        'averaging_suspected': variance_ratio > 0.3
    }


def analyze_loss_trajectory(summaries: List[Dict]) -> Dict:
    """Analyze loss trajectory for plateau detection."""
    if len(summaries) < 2:
        return {}

    losses = [s['loss_mean'] for s in summaries]
    samples_seen = [s['samples_seen'] for s in summaries]

    initial_loss = losses[0]
    current_loss = losses[-1]
    best_loss = min(losses)
    best_idx = losses.index(best_loss)

    # Check for plateau
    plateau_detected = False
    if len(summaries) >= 5:
        cutoff = max(1, len(summaries) // 5)
        recent = losses[-cutoff:]
        early_loss = losses[0]
        recent_start_loss = recent[0]
        final_loss = losses[-1]

        total_improvement = early_loss - final_loss
        recent_improvement = recent_start_loss - final_loss

        if total_improvement > 1e-8:
            improvement_ratio = recent_improvement / total_improvement
            plateau_detected = improvement_ratio < 0.05

    return {
        'initial_loss': initial_loss,
        'current_loss': current_loss,
        'best_loss': best_loss,
        'best_samples_seen': samples_seen[best_idx],
        'total_reduction_pct': ((initial_loss - current_loss) / initial_loss * 100) if initial_loss > 0 else 0,
        'plateau_detected': plateau_detected,
        'total_samples_seen': samples_seen[-1],
        'num_intervals': len(summaries)
    }


def analyze_gradients(summaries: List[Dict]) -> Dict:
    """Analyze gradient norms."""
    grad_norms = [s.get('grad_norm_mean', 0) for s in summaries if s.get('grad_norm_mean', 0) > 0]
    if not grad_norms:
        return {}

    mean_grad = np.mean(grad_norms)
    max_grad = np.max(grad_norms)
    min_grad = np.min(grad_norms)

    issues = []
    if mean_grad < 1e-6:
        issues.append("Very small gradients (possible vanishing)")
    if max_grad > 100:
        issues.append(f"Large gradients detected (max: {max_grad:.2e})")

    return {
        'mean_grad_norm': mean_grad,
        'max_grad_norm': max_grad,
        'min_grad_norm': min_grad,
        'issues': issues
    }


def format_summary(
    config: Optional[Dict],
    summaries: List[Dict],
    batch_comp: Dict,
    loss_traj: Dict,
    grad_analysis: Dict
) -> str:
    """Format analysis as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("TRAINING RUN ANALYSIS (ONGOING)")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    if config:
        lines.append("Configuration:")
        lines.append(f"  Batch size: {config.get('batch_size', 'N/A')}")
        lines.append(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        lines.append(f"  Total samples target: {config.get('total_samples', 'N/A')}")
        lines.append(f"  Num training examples: {config.get('num_training_examples', 'N/A')}")
        lines.append(f"  Warmup steps: {config.get('warmup_steps', 'N/A')}")
        lines.append(f"  Focal beta: {config.get('focal_beta', 'N/A')}")
        lines.append(f"  Focal alpha: {config.get('focal_alpha', 'N/A')}")
        lines.append("")

    # Training progress
    if loss_traj:
        lines.append("Training Progress:")
        lines.append(f"  Samples processed: {loss_traj.get('total_samples_seen', 'N/A'):,}")
        lines.append(f"  Intervals logged: {loss_traj.get('num_intervals', 'N/A')}")
        lines.append(f"  Initial loss: {loss_traj.get('initial_loss', 0):.6f}")
        lines.append(f"  Current loss: {loss_traj.get('current_loss', 0):.6f}")
        lines.append(f"  Best loss: {loss_traj.get('best_loss', 0):.6f} "
                    f"(at {loss_traj.get('best_samples_seen', 0):,} samples)")
        lines.append(f"  Reduction: {loss_traj.get('total_reduction_pct', 0):.1f}%")
        lines.append("")

    # Batch composition analysis
    if batch_comp:
        lines.append("Batch Composition Analysis:")
        lines.append(f"  Unique samples seen: {batch_comp.get('num_unique_samples', 'N/A')}")
        lines.append(f"  Mean sample frequency: {batch_comp.get('mean_frequency', 0):.1f}")
        lines.append(f"  Frequency std dev: {batch_comp.get('std_frequency', 0):.1f}")
        lines.append(f"  Variance ratio: {batch_comp.get('variance_ratio', 0):.2f}")

        if batch_comp.get('averaging_suspected'):
            lines.append("  ⚠️  BATCH AVERAGING SUSPECTED!")
            lines.append("     High variance in sample frequencies suggests some samples")
            lines.append("     appear more often than others, leading to biased gradients.")
        else:
            lines.append("  ✅ Batch composition looks balanced")
        lines.append("")

    # Gradient analysis
    if grad_analysis:
        lines.append("Gradient Analysis:")
        lines.append(f"  Mean gradient norm: {grad_analysis.get('mean_grad_norm', 0):.2e}")
        lines.append(f"  Max gradient norm: {grad_analysis.get('max_grad_norm', 0):.2e}")
        lines.append(f"  Min gradient norm: {grad_analysis.get('min_grad_norm', 0):.2e}")

        issues = grad_analysis.get('issues', [])
        if issues:
            for issue in issues:
                lines.append(f"  ⚠️  {issue}")
        else:
            lines.append("  ✅ Gradient flow appears healthy")
        lines.append("")

    # Diagnostics
    lines.append("Diagnostics:")

    if loss_traj.get('plateau_detected'):
        lines.append("  ⚠️  Loss plateau detected (< 5% improvement in final 20%)")
        lines.append("     Consider: increasing LR, changing architecture, or stopping")
    else:
        lines.append("  ✅ Loss still improving")

    lines.append("")

    # Recommendations
    lines.append("Recommendations:")

    if batch_comp.get('averaging_suspected'):
        lines.append("  1. Try batch_size=1 to eliminate gradient averaging")

    if loss_traj.get('plateau_detected'):
        lines.append("  2. Training may have converged - consider stopping or increasing LR")

    if grad_analysis.get('issues'):
        lines.append("  3. Address gradient flow issues (see gradient analysis above)")

    if not any([
        batch_comp.get('averaging_suspected'),
        loss_traj.get('plateau_detected'),
        grad_analysis.get('issues')
    ]):
        lines.append("  No major issues detected - training looks healthy!")

    lines.append("")
    lines.append("=" * 70)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training logs from ongoing or completed runs"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (e.g., saved/training_logs/session/run_20260112_103045)"
    )
    parser.add_argument(
        "--latest",
        type=str,
        help="Session name to find latest run for (e.g., session_so101_should_pan_500_part1_10_train)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="saved/training_logs",
        help="Root log directory (default: saved/training_logs)"
    )

    args = parser.parse_args()

    # Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.latest:
        run_dir = find_latest_run(args.latest, args.log_dir)
        if run_dir is None:
            sys.exit(1)
        print(f"Found latest run: {run_dir}")
        print()
    else:
        parser.print_help()
        sys.exit(1)

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        sys.exit(1)

    # Load data
    print("Loading training data...")
    config = load_config(run_dir)
    summaries = load_interval_summaries(run_dir)

    if not summaries:
        print("Warning: No interval summaries found yet.")
        print("Training may have just started or logging may not be enabled.")
        print()
        # Try to load raw metrics
        raw_metrics = load_recent_raw_metrics(run_dir, num_batches=10)
        if raw_metrics:
            print(f"Found {len(raw_metrics)} recent raw metric entries:")
            for i, m in enumerate(raw_metrics[-5:]):
                batch = m.get('batch', '?')
                loss = m.get('loss_hybrid', 0)
                samples = m.get('samples_seen', '?')
                print(f"  Batch {batch}: loss={loss:.6f}, samples={samples}")
        sys.exit(0)

    # Analyze
    print(f"Analyzing {len(summaries)} interval summaries...")
    batch_comp = analyze_batch_composition(summaries)
    loss_traj = analyze_loss_trajectory(summaries)
    grad_analysis = analyze_gradients(summaries)

    # Format and print
    summary = format_summary(config, summaries, batch_comp, loss_traj, grad_analysis)
    print()
    # Use UTF-8 encoding for Windows terminal compatibility
    import sys
    if sys.platform == 'win32':
        # Print to stdout with UTF-8 encoding on Windows
        sys.stdout.reconfigure(encoding='utf-8')
    print(summary)

    # Save to file
    output_file = run_dir / "analysis_snapshot.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print()
        print(f"Analysis saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Failed to save analysis: {e}")


if __name__ == "__main__":
    main()
