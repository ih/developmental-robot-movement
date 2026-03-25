#!/usr/bin/env python3
"""
Check if Iteration 1 results are ready and perform quick analysis.

Usage:
    python check_iteration_ready.py
"""

import json
from pathlib import Path
from datetime import datetime

def check_iteration1_ready():
    """Check if Iteration 1 (Run 14) results are available."""
    result_dir = Path("saved/staged_training_reports/session_so101_multiheight_part1_1345/multiheight_dit_run14")

    if not result_dir.exists():
        print(f"[NOT STARTED] Iteration 1 not yet started: {result_dir}")
        return False

    summary_file = result_dir / "summary.json"
    if not summary_file.exists():
        print(f"[RUNNING] Iteration 1 running... (summary.json not yet created)")
        return False

    # Load summary and first stage metrics
    try:
        with open(summary_file) as f:
            summary = json.load(f)

        # Find metrics from first stage
        stage_dir = result_dir / "stage1_run1"
        if not (stage_dir / "metrics.json").exists():
            print(f"[RUNNING] Iteration 1 running... (metrics not yet available)")
            return False

        with open(stage_dir / "metrics.json") as f:
            metrics = json.load(f)

        # Extract results
        orig_loss = summary["staged"]["best_loss_original"]
        cf_div = metrics["counterfactual_metrics"]["overall_mean_divergence"]
        samples = metrics["total_samples_trained"]
        time_sec = metrics["elapsed_time_seconds"]
        stop_reason = metrics["stop_reason"]

        print(f"\n{'='*80}")
        print(f"[SUCCESS] ITERATION 1 (RUN 14) COMPLETE!")
        print(f"{'='*80}")
        print(f"Original Loss:        {orig_loss:.6f}")
        print(f"Counterfactual Div:   {cf_div:.2f}")
        print(f"Samples Trained:      {samples:,}")
        print(f"Time:                 {time_sec:.1f}s ({time_sec/60:.1f}m)")
        print(f"Stop Reason:          {stop_reason}")
        print(f"{'='*80}")

        # Compare to baselines
        run7_loss = 0.0092
        run11_loss = 0.0287

        print(f"\nCOMPARISON TO BASELINES:")
        print(f"  Run 7 (decoder-only):  {run7_loss:.6f}")
        print(f"  Run 11 (DiT, no sweeps): {run11_loss:.6f}")
        print(f"  Run 14 (this):         {orig_loss:.6f}")

        improvement_vs_11 = (run11_loss - orig_loss) / run11_loss * 100
        print(f"\nImprovement vs Run 11: {improvement_vs_11:+.1f}%")

        if orig_loss < run7_loss:
            print(f"[EXCELLENT] BEATS RUN 7 by {(run7_loss - orig_loss)/run7_loss * 100:.1f}%!")
        elif orig_loss < 0.015:
            print(f"[GOOD] Close to Run 7 ({(orig_loss - run7_loss)/run7_loss * 100:+.1f}%)")
        elif orig_loss < 0.020:
            print(f"[OK] Significant improvement but not beating Run 7")
        elif orig_loss < 0.025:
            print(f"[MARGINAL] Marginal improvement")
        else:
            print(f"[FAILURE] Still worse than Run 11")

        print(f"\nCheckpoint: {summary['staged']['best_checkpoint']}")
        print(f"Report: file://{result_dir.absolute()}/final_report_*.html")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"[ERROR] Error reading results: {e}")
        return False

if __name__ == "__main__":
    ready = check_iteration1_ready()
    if not ready:
        print("\n[INFO] Run this script periodically to check for Iteration 1 completion")
        print("   Expected time: 2-3+ hours from start")
