#!/usr/bin/env python3
"""
Quick comparison of DiT iteration experiments.

Usage:
    python compare_dit_iterations.py
"""

import json
from pathlib import Path
from datetime import datetime

def analyze_dit_run(run_name: str) -> dict:
    """Extract metrics from a DiT iteration run."""
    report_dir = Path("saved/staged_training_reports/session_so101_multiheight_part1_1345") / run_name

    if not report_dir.exists():
        return None

    # Find stage1_run1 directory
    stage_dir = report_dir / "stage1_run1"
    if not (stage_dir / "metrics.json").exists():
        return None

    try:
        with open(stage_dir / "metrics.json") as f:
            metrics = json.load(f)

        with open(report_dir / "summary.json") as f:
            summary = json.load(f)

        return {
            "run_name": run_name,
            "orig_loss": summary["staged"]["best_loss_original"],
            "hybrid_mean": metrics["original_eval_stats"]["hybrid"]["mean"],
            "hybrid_median": metrics["original_eval_stats"]["hybrid"]["median"],
            "hybrid_std": metrics["original_eval_stats"]["hybrid"]["std"],
            "cf_divergence": metrics["counterfactual_metrics"]["overall_mean_divergence"],
            "samples": metrics["total_samples_trained"],
            "time_sec": metrics["elapsed_time_seconds"],
            "stop_reason": metrics["stop_reason"],
        }
    except Exception as e:
        print(f"Error reading {run_name}: {e}")
        return None

def main():
    """Compare all DiT iterations."""

    runs = [
        "multiheight_dit_iter0_baseline",
        "multiheight_dit_iter1_fewer_steps",
        "multiheight_dit_iter2_10steps",
    ]

    print("\n" + "="*100)
    print("DiT ITERATION COMPARISON")
    print("="*100)

    results = []
    for run in runs:
        result = analyze_dit_run(run)
        if result:
            results.append(result)

    if not results:
        print("[NO RESULTS YET] Experiments still running or not found")
        return

    # Sort by loss
    results.sort(key=lambda x: x["orig_loss"])

    # Print header
    print(f"\n{'Run':<40} | {'Loss':<10} | {'Median':<10} | {'Std':<8} | {'CF Div':<8} | {'Time':<8} | {'Stop Reason':<20}")
    print("-" * 100)

    # Baselines for reference
    run7_loss = 0.0092
    run11_loss = 0.0287

    for r in results:
        time_min = r["time_sec"] / 60

        # Determine if improvement
        if r["orig_loss"] < run7_loss:
            marker = " [BEATS RUN7!]"
        elif r["orig_loss"] < run11_loss:
            marker = " [better]"
        elif r["orig_loss"] < 0.05:
            marker = " [marginal]"
        else:
            marker = " [bad]"

        run_display = r["run_name"][:40]

        print(f"{run_display:<40} | {r['orig_loss']:>10.6f} | {r['hybrid_median']:>10.6f} | {r['hybrid_std']:>8.5f} | {r['cf_divergence']:>8.2f} | {time_min:>8.1f}m | {r['stop_reason']:<20}{marker}")

    print("\n" + "="*100)
    print(f"Reference (Run 7, decoder-only): Loss=0.0092, CF=19.71, Time=5.8m")
    print(f"Reference (Run 11, DiT 50 steps): Loss=0.0287, CF=27.19, Time=180m")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
