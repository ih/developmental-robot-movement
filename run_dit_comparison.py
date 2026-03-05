"""
Run DiT comparison experiment across different VAE/encoder backends.

Runs staged_training.py sequentially for each VAE type, then prints
a comparison summary.

Usage:
    python run_dit_comparison.py \
        --train-session .\saved\sessions\so101\session_so101_multiheight_part1_1345\ \
        --val-session .\saved\sessions\so101\session_so101_multiheight_part2_149\ \
        [--seed 42] \
        [--stage-time-budget-min 60]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Experiments to run: (vae_type, short_label, description)
EXPERIMENTS = [
    ("pretrained_sd", "SD", "Stable Diffusion VAE (4ch, 8x compression)"),
    ("pretrained_flux", "FLUX", "FLUX VAE (16ch, 8x compression)"),
]

# Arguments that are passed through directly to staged_training.py
PASSTHROUGH_ARGS = [
    "--config",
    "--runs-per-stage",
    "--stage-time-budget-min",
    "--lr-sweep-lr-min",
    "--lr-sweep-lr-max",
    "--lr-sweep-phase-a-candidates",
    "--lr-sweep-phase-a-budget-min",
    "--lr-sweep-phase-a-survivors",
    "--lr-sweep-phase-b-seeds",
    "--lr-sweep-phase-b-budget-min",
    "--original-session",
    "--checkpoint",
    "--root-session",
    "--stage",
    "--max-workers",
]

# Boolean flags that are passed through when present
PASSTHROUGH_FLAGS = [
    "--serial-runs",
    "--disable-plateau-sweep",
    "--disable-initial-sweep",
    "--disable-baseline",
]


def build_command(
    vae_type: str,
    run_id: str,
    args: argparse.Namespace,
    raw_args: list,
) -> list:
    """Build the staged_training.py command for one experiment."""
    cmd = [
        sys.executable, "staged_training.py",
        "--train-session", args.train_session,
        "--val-session", args.val_session,
        "--model-type", "dit",
        "--vae-type", vae_type,
        "--run-id", run_id,
    ]

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    # Pass through named arguments that were explicitly provided
    for arg_name in PASSTHROUGH_ARGS:
        # Convert --arg-name to args.arg_name for lookup
        attr_name = arg_name.lstrip("-").replace("-", "_")
        value = getattr(args, attr_name, None)
        if value is not None:
            cmd.extend([arg_name, str(value)])

    # Pass through boolean flags
    for flag in PASSTHROUGH_FLAGS:
        attr_name = flag.lstrip("-").replace("-", "_")
        if getattr(args, attr_name, False):
            cmd.append(flag)

    return cmd


def load_summary(output_dir: str) -> dict:
    """Load summary.json from a training run's output directory."""
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def print_comparison(results: list) -> None:
    """Print a comparison table of experiment results."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: DiT VAE Encoder Comparison")
    print("=" * 80)

    # Header
    print(f"\n{'VAE Type':<20} {'Best Val Loss':>14} {'Best Train Loss':>16} {'Time':>10} {'Samples':>10} {'Status':>10}")
    print("-" * 80)

    best_val = float("inf")
    best_idx = -1

    for i, r in enumerate(results):
        vae_type = r["vae_type"]
        label = r["label"]
        status = r["status"]

        if status == "completed" and r.get("summary"):
            summary = r["summary"]
            # Extract metrics from summary - format is summary["staged"]["stages"]
            staged = summary.get("staged", {})
            stages = staged.get("stages", [])
            # Best original loss is at the top level of "staged"
            val_loss = staged.get("best_loss_original", float("inf"))
            if stages:
                last_stage = stages[-1]
                train_loss = last_stage.get("train_loss", float("inf"))
                samples = last_stage.get("samples", 0)
            else:
                train_loss = float("inf")
                samples = 0

            elapsed = r.get("elapsed_sec", 0)
            time_str = f"{elapsed / 60:.1f}m" if elapsed > 0 else "N/A"

            if val_loss < best_val:
                best_val = val_loss
                best_idx = i

            print(f"{label:<20} {val_loss:>14.6f} {train_loss:>16.6f} {time_str:>10} {samples:>10,} {'OK':>10}")
        else:
            elapsed = r.get("elapsed_sec", 0)
            time_str = f"{elapsed / 60:.1f}m" if elapsed > 0 else "N/A"
            print(f"{label:<20} {'N/A':>14} {'N/A':>16} {time_str:>10} {'N/A':>10} {status:>10}")

    print("-" * 80)

    if best_idx >= 0:
        winner = results[best_idx]
        print(f"\nWinner: {winner['label']} (best val loss: {best_val:.6f})")
    else:
        print("\nNo completed experiments to compare.")

    # Print output directories
    print("\nOutput directories:")
    for r in results:
        print(f"  {r['label']}: {r['output_dir']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run DiT comparison experiment across different VAE/encoder backends"
    )
    parser.add_argument(
        "--train-session", required=True,
        help="Path to training session",
    )
    parser.add_argument(
        "--val-session", required=True,
        help="Path to validation session",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed for reproducibility (shared across experiments)",
    )
    parser.add_argument(
        "--output-dir",
        help="Base output directory (default: saved/staged_training_reports/{session_name})",
    )
    # Passthrough args (define them here so argparse doesn't error)
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--runs-per-stage", type=int, default=None)
    parser.add_argument("--stage-time-budget-min", type=float, default=None)
    parser.add_argument("--lr-sweep-lr-min", type=float, default=None)
    parser.add_argument("--lr-sweep-lr-max", type=float, default=None)
    parser.add_argument("--lr-sweep-phase-a-candidates", type=int, default=None)
    parser.add_argument("--lr-sweep-phase-a-budget-min", type=float, default=None)
    parser.add_argument("--lr-sweep-phase-a-survivors", type=int, default=None)
    parser.add_argument("--lr-sweep-phase-b-seeds", type=int, default=None)
    parser.add_argument("--lr-sweep-phase-b-budget-min", type=float, default=None)
    parser.add_argument("--original-session", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--root-session", default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Max parallel workers for LR sweeps (default: 2 for DiT models)")
    parser.add_argument("--serial-runs", action="store_true")
    parser.add_argument("--disable-plateau-sweep", action="store_true")
    parser.add_argument("--disable-initial-sweep", action="store_true")
    parser.add_argument("--disable-baseline", action="store_true")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = Path(args.train_session).name

    print("=" * 80)
    print("DiT VAE Encoder Comparison Experiment")
    print("=" * 80)
    print(f"Train session: {args.train_session}")
    print(f"Val session:   {args.val_session}")
    print(f"Timestamp:     {timestamp}")
    if args.seed is not None:
        print(f"Seed:          {args.seed}")
    print(f"\nExperiments ({len(EXPERIMENTS)}):")
    for vae_type, label, desc in EXPERIMENTS:
        print(f"  - {label}: {desc}")
    print()

    results = []
    overall_start = time.time()

    for i, (vae_type, label, desc) in enumerate(EXPERIMENTS):
        run_id = f"dit_{vae_type}_{timestamp}"

        # Determine output directory for this experiment
        if args.output_dir:
            output_dir = str(Path(args.output_dir) / run_id)
        else:
            output_dir = f"saved/staged_training_reports/{session_name}/{run_id}"

        print(f"\n{'#' * 80}")
        print(f"# Experiment {i + 1}/{len(EXPERIMENTS)}: {label} ({desc})")
        print(f"# Run ID: {run_id}")
        print(f"# Output: {output_dir}")
        print(f"{'#' * 80}\n")

        cmd = build_command(vae_type, run_id, args, sys.argv[1:])

        # Add output dir
        cmd.extend(["--output-dir", output_dir])

        print(f"Command: {' '.join(cmd)}\n")

        exp_start = time.time()
        try:
            result = subprocess.run(cmd, check=True)
            status = "completed"
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment {label} failed with return code {e.returncode}")
            status = "failed"
        except KeyboardInterrupt:
            print(f"\nExperiment {label} interrupted by user")
            status = "interrupted"

        exp_elapsed = time.time() - exp_start

        # Try to load summary
        summary = load_summary(output_dir)

        results.append({
            "vae_type": vae_type,
            "label": label,
            "description": desc,
            "run_id": run_id,
            "output_dir": output_dir,
            "status": status,
            "elapsed_sec": exp_elapsed,
            "summary": summary,
        })

        print(f"\n{label} finished in {exp_elapsed / 60:.1f} minutes (status: {status})")

        # Stop early if interrupted
        if status == "interrupted":
            print("Stopping remaining experiments due to interruption.")
            break

    overall_elapsed = time.time() - overall_start

    # Print comparison
    print_comparison(results)

    print(f"Total experiment time: {overall_elapsed / 60:.1f} minutes")

    # Save comparison results
    comparison_output = Path(args.output_dir) if args.output_dir else Path(f"saved/staged_training_reports/{session_name}")
    comparison_output.mkdir(parents=True, exist_ok=True)
    comparison_file = comparison_output / f"dit_comparison_{timestamp}.json"
    with open(comparison_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "train_session": args.train_session,
            "val_session": args.val_session,
            "seed": args.seed,
            "total_elapsed_sec": overall_elapsed,
            "experiments": results,
        }, f, indent=2, default=str)
    print(f"Comparison results saved to: {comparison_file}")


if __name__ == "__main__":
    main()
