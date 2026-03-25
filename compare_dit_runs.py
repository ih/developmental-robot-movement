#!/usr/bin/env python3
"""
Quick comparison tool for DiT experiments.

Extracts and compares key metrics from staged training reports.
Usage:
    python compare_dit_runs.py
"""

import json
from pathlib import Path
from typing import Dict, List

def analyze_run(run_dir: Path) -> Dict:
    """Extract key metrics from a staged training run."""
    run_dir = Path(run_dir)

    if not (run_dir / "summary.json").exists():
        return None

    try:
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        # Find the first stage metrics
        stage_dirs = sorted([d for d in run_dir.iterdir() if d.name.startswith("stage")])
        metrics = None

        for stage_dir in stage_dirs:
            metrics_file = stage_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                break

        if not metrics:
            return None

        return {
            "run_id": run_dir.name,
            "best_checkpoint": summary["staged"]["best_checkpoint"],
            "orig_loss": summary["staged"]["best_loss_original"],
            "orig_loss_median": metrics["original_eval_stats"]["hybrid"]["median"] if "original_eval_stats" in metrics else None,
            "cf_divergence": metrics["counterfactual_metrics"]["overall_mean_divergence"],
            "samples_trained": metrics["total_samples_trained"],
            "time_sec": metrics["elapsed_time_seconds"],
            "stop_reason": metrics["stop_reason"],
        }
    except Exception as e:
        print(f"Error analyzing {run_dir}: {e}")
        return None

def compare_runs(base_dir: str = "saved/staged_training_reports/session_so101_multiheight_part1_1345"):
    """Compare all runs in a directory."""
    base_dir = Path(base_dir)

    print(f"\n{'='*120}")
    print(f"Comparing DiT runs in: {base_dir}")
    print(f"{'='*120}\n")

    results = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        result = analyze_run(run_dir)
        if result:
            results.append(result)

    if not results:
        print("No results found!")
        return

    # Sort by original loss (lower is better)
    results.sort(key=lambda x: x["orig_loss"])

    # Print header
    print(f"{'Run ID':<45} | {'Loss (Orig)':<12} | {'Median':<12} | {'CF Div':<10} | {'Samples':<10} | {'Time':<10} | {'Stop':<15}")
    print(f"{'-'*45} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*15}")

    # Best result (for highlighting)
    best_loss = results[0]["orig_loss"]

    for r in results:
        loss_str = f"{r['orig_loss']:.6f}"
        median_str = f"{r['orig_loss_median']:.6f}" if r['orig_loss_median'] else "N/A"

        # Highlight improvement/degradation
        if abs(r["orig_loss"] - 0.0092) < 0.0001:  # Within Run 7's performance
            loss_marker = " ✓"
        elif r["orig_loss"] < 0.0092:
            loss_marker = " ↑↑"  # Better than Run 7
        else:
            loss_marker = ""

        time_min = r["time_sec"] / 60
        samples_k = r["samples_trained"] / 1000

        print(f"{r['run_id']:<45} | {loss_str}{loss_marker:<10} | {median_str:<12} | {r['cf_divergence']:<10.2f} | {samples_k:<10.1f}k | {time_min:<10.1f}m | {r['stop_reason']:<15}")

    print(f"\n{'='*120}")
    print(f"Run 7 baseline (decoder-only): Loss=0.0092, CF=19.71, Time=5.8m")
    print(f"Run 11 (DiT, no sweeps): Loss=0.0287 (3.1× worse), CF=27.19, Time=180m")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    compare_runs()
