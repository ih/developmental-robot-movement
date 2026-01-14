"""
Training logger for diagnostic data collection.

Provides multi-tier logging:
- Tier 1: Full batch-level metrics (rotated JSONL files)
- Tier 2: Interval summaries (aggregated statistics)
- Tier 3: Milestone snapshots (config, predictions, checkpoints)
- Tier 4: Auto-generated LLM-friendly plain text summary

Designed to keep LLM-consumable logs under 5KB regardless of training length.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class TrainingLogger:
    """
    Multi-tier training logger with automatic summarization.

    Logs comprehensive metrics while keeping LLM-consumable summaries compact.
    """

    def __init__(
        self,
        session_name: str,
        log_dir: str = "saved/training_logs",
        summary_interval: int = 100,
        enable_logging: bool = True
    ):
        """
        Initialize training logger.

        Args:
            session_name: Name of the session being trained
            log_dir: Root directory for training logs
            summary_interval: Number of batches per interval summary
            enable_logging: If False, logger becomes a no-op (for disabling without code changes)
        """
        self.enabled = enable_logging
        if not self.enabled:
            return

        # Create directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / session_name / f"run_{timestamp}"
        self.raw_dir = self.run_dir / "raw"
        self.milestone_dir = self.run_dir / "milestones"
        self.sample_dir = self.run_dir / "sample_predictions"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.milestone_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)

        # State
        self.summary_interval = summary_interval
        self.interval_buffer: List[Dict] = []
        self.sample_seen_counts: Dict[int, int] = {}
        self.training_start_time = datetime.now()

        # Tracking for diagnostics
        self.initial_loss: Optional[float] = None
        self.best_loss: Optional[float] = None
        self.best_loss_batch: Optional[int] = None

    def log_config(self, config_dict: Dict[str, Any]):
        """
        Save training configuration to JSON.

        Args:
            config_dict: Dictionary of hyperparameters and settings
        """
        if not self.enabled:
            return

        config_path = self.run_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def log_batch(self, metrics_dict: Dict[str, Any]):
        """
        Log full batch metrics (Tier 1) and buffer for interval summary (Tier 2).

        Args:
            metrics_dict: Dictionary containing batch metrics (loss, LR, grad norms, etc.)
                Required keys: 'batch', 'samples_seen', 'loss_hybrid'
                Optional keys: 'lr', 'grad_norm_total', 'batch_indices', etc.
        """
        if not self.enabled:
            return

        # Track initial and best loss
        loss = metrics_dict.get('loss_hybrid', 0)
        batch = metrics_dict.get('batch', 0)

        if self.initial_loss is None:
            self.initial_loss = loss

        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_batch = batch

        # Update sample seen counts
        batch_indices = metrics_dict.get('batch_indices', [])
        for idx in batch_indices:
            self.sample_seen_counts[idx] = self.sample_seen_counts.get(idx, 0) + 1

        # Add sample counts to metrics
        metrics_dict['sample_counts'] = [
            self.sample_seen_counts.get(idx, 0) for idx in batch_indices
        ]

        # Add per-sample loss statistics if available (for loss-weighted sampling)
        per_sample_losses = metrics_dict.get('per_sample_losses')
        if per_sample_losses and len(per_sample_losses) > 0:
            metrics_dict['per_sample_loss_mean'] = float(np.mean(per_sample_losses))
            metrics_dict['per_sample_loss_std'] = float(np.std(per_sample_losses))
            metrics_dict['per_sample_loss_min'] = float(np.min(per_sample_losses))
            metrics_dict['per_sample_loss_max'] = float(np.max(per_sample_losses))

        # Tier 1: Append to raw JSONL (rotated by 1000-batch chunks)
        chunk_id = batch // 1000
        raw_file = self.raw_dir / f"metrics_{chunk_id*1000:05d}-{(chunk_id+1)*1000:05d}.jsonl"

        try:
            with open(raw_file, 'a') as f:
                # Serialize with default handler for numpy types
                f.write(json.dumps(metrics_dict, default=self._json_default) + '\n')
        except Exception as e:
            print(f"[TrainingLogger WARNING] Failed to write batch metrics: {e}")

        # Tier 2: Buffer for interval summary
        self.interval_buffer.append(metrics_dict)
        if len(self.interval_buffer) >= self.summary_interval:
            self._write_interval_summary()

    def _write_interval_summary(self):
        """Aggregate interval buffer and write compact summary (Tier 2)."""
        if not self.enabled or not self.interval_buffer:
            return

        # Extract metrics from buffer
        losses = [b.get('loss_hybrid', 0) for b in self.interval_buffer]
        lrs = [b.get('lr', 0) for b in self.interval_buffer]
        grad_norms = [b.get('grad_norm_total', 0) for b in self.interval_buffer]

        # Get validation loss from last batch in interval (if available)
        val_loss = None
        for b in reversed(self.interval_buffer):
            if 'val_loss' in b and b['val_loss'] is not None:
                val_loss = b['val_loss']
                break

        # Aggregate batch composition
        sample_counts: Dict[int, int] = {}
        for b in self.interval_buffer:
            for idx in b.get('batch_indices', []):
                sample_counts[idx] = sample_counts.get(idx, 0) + 1

        # Create summary entry
        summary = {
            'interval_start': self.interval_buffer[0].get('batch', 0),
            'interval_end': self.interval_buffer[-1].get('batch', 0),
            'samples_seen': self.interval_buffer[-1].get('samples_seen', 0),
            'num_batches': len(self.interval_buffer),
            'loss_mean': float(np.mean(losses)),
            'loss_min': float(np.min(losses)),
            'loss_max': float(np.max(losses)),
            'loss_std': float(np.std(losses)),
            'lr_mean': float(np.mean(lrs)) if lrs and any(lrs) else 0,
            'grad_norm_mean': float(np.mean(grad_norms)) if grad_norms and any(grad_norms) else 0,
            'grad_norm_max': float(np.max(grad_norms)) if grad_norms and any(grad_norms) else 0,
        }

        # Add validation loss if available
        if val_loss is not None:
            summary['val_loss'] = float(val_loss)

        # Add batch composition if samples were tracked
        if sample_counts:
            summary['batch_composition'] = {
                'unique_samples_seen': sorted(sample_counts.keys()),
                'sample_frequencies': [sample_counts[k] for k in sorted(sample_counts.keys())]
            }

        # Append to summary file
        summary_file = self.run_dir / "training_summary.jsonl"
        try:
            with open(summary_file, 'a') as f:
                f.write(json.dumps(summary) + '\n')
        except Exception as e:
            print(f"[TrainingLogger WARNING] Failed to write interval summary: {e}")

        # Clear buffer
        self.interval_buffer.clear()

    def log_prediction_sample(
        self,
        frame_idx: int,
        iteration: int,
        pred_img: np.ndarray,
        target_img: np.ndarray,
        canvas_img: Optional[np.ndarray] = None
    ):
        """
        Save prediction visualization (Tier 3 - milestone snapshots).

        Creates a comparison image: [canvas | target | prediction | diff]

        Args:
            frame_idx: Index of the frame being predicted
            iteration: Current training iteration/sample count
            pred_img: Predicted frame (H, W, 3) uint8
            target_img: Target ground truth frame (H, W, 3) uint8
            canvas_img: Optional canvas image (H, W_canvas, 3) uint8
        """
        if not self.enabled:
            return

        try:
            # Compute difference image
            diff_img = np.abs(pred_img.astype(float) - target_img.astype(float)).astype(np.uint8)

            # Create figure
            num_cols = 4 if canvas_img is not None else 3
            fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))

            col = 0
            if canvas_img is not None:
                axes[col].imshow(canvas_img)
                axes[col].set_title('Canvas')
                axes[col].axis('off')
                col += 1

            axes[col].imshow(target_img)
            axes[col].set_title('Target')
            axes[col].axis('off')
            col += 1

            axes[col].imshow(pred_img)
            axes[col].set_title('Prediction')
            axes[col].axis('off')
            col += 1

            axes[col].imshow(diff_img)
            axes[col].set_title('Abs Difference')
            axes[col].axis('off')

            plt.tight_layout()

            # Save figure
            save_path = self.sample_dir / f"sample_{frame_idx:04d}_iter_{iteration:06d}.jpg"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"[TrainingLogger WARNING] Failed to save prediction sample: {e}")

    def generate_llm_summary(self, final_metrics: Optional[Dict] = None) -> str:
        """
        Generate plain-text summary for LLM consumption (Tier 4).

        Automatically analyzes training run and generates diagnostic report.

        Args:
            final_metrics: Optional dictionary with final training metrics

        Returns:
            Plain text summary string (< 2000 tokens)
        """
        if not self.enabled:
            return "Logging disabled - no summary available"

        # Flush any remaining interval buffer
        if self.interval_buffer:
            self._write_interval_summary()

        # Read interval summaries
        summary_file = self.run_dir / "training_summary.jsonl"
        summaries = []
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summaries = [json.loads(line) for line in f]
            except Exception as e:
                print(f"[TrainingLogger WARNING] Failed to read summaries: {e}")

        # Run diagnostics
        diagnostics = self._analyze_training_run(summaries, final_metrics or {})

        # Generate formatted summary
        summary_text = self._format_llm_summary(summaries, diagnostics, final_metrics or {})

        # Save to file
        try:
            with open(self.run_dir / "llm_summary.txt", 'w') as f:
                f.write(summary_text)
        except Exception as e:
            print(f"[TrainingLogger WARNING] Failed to write LLM summary: {e}")

        return summary_text

    def _analyze_training_run(
        self,
        summaries: List[Dict],
        final_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Detect common training issues from summary data.

        Args:
            summaries: List of interval summary dictionaries
            final_metrics: Final metrics from training

        Returns:
            Dictionary of diagnostic flags and observations
        """
        diagnostics = {
            'lr_issues': False,
            'gradient_issues': False,
            'batch_averaging_suspected': False,
            'plateau_detected': False,
            'observations': []
        }

        if not summaries:
            return diagnostics

        # Check for batch averaging (uneven sample frequencies)
        for s in summaries[-10:]:  # Last 10 intervals
            comp = s.get('batch_composition', {})
            freqs = comp.get('sample_frequencies', [])
            if freqs and len(freqs) > 1:
                # High variance in frequencies suggests some samples dominate
                mean_freq = np.mean(freqs)
                std_freq = np.std(freqs)
                variance_ratio = std_freq / (mean_freq + 1e-8)

                if variance_ratio > 0.3:
                    diagnostics['batch_averaging_suspected'] = True
                    diagnostics['observations'].append(
                        f"Sample frequency variance ratio: {variance_ratio:.2f} "
                        f"(frequencies: {freqs})"
                    )
                    break

        # Check for plateau (last 20% shows <5% loss improvement)
        if len(summaries) >= 5:
            cutoff = max(1, len(summaries) // 5)
            recent = summaries[-cutoff:]

            early_loss = summaries[0]['loss_mean']
            recent_start_loss = recent[0]['loss_mean']
            final_loss = summaries[-1]['loss_mean']

            total_improvement = early_loss - final_loss
            recent_improvement = recent_start_loss - final_loss

            if total_improvement > 1e-8:
                improvement_ratio = recent_improvement / total_improvement
                if improvement_ratio < 0.05:
                    diagnostics['plateau_detected'] = True
                    diagnostics['observations'].append(
                        f"Plateau detected: only {improvement_ratio*100:.1f}% improvement "
                        f"in final 20% of training"
                    )

        # Check gradient norms
        if summaries:
            grad_norms = [s.get('grad_norm_mean', 0) for s in summaries if s.get('grad_norm_mean', 0) > 0]
            if grad_norms:
                mean_grad = np.mean(grad_norms)
                max_grad = np.max(grad_norms)

                if mean_grad < 1e-6:
                    diagnostics['gradient_issues'] = True
                    diagnostics['observations'].append(
                        f"Very small gradients detected (mean: {mean_grad:.2e})"
                    )
                elif max_grad > 100:
                    diagnostics['gradient_issues'] = True
                    diagnostics['observations'].append(
                        f"Large gradients detected (max: {max_grad:.2e})"
                    )

        # Check LR schedule
        if summaries:
            lrs = [s.get('lr_mean', 0) for s in summaries]
            if lrs:
                lr_range = np.max(lrs) - np.min(lrs)
                if lr_range < 1e-8:
                    # Constant LR
                    diagnostics['observations'].append(f"Constant LR: {np.mean(lrs):.2e}")
                else:
                    # Varying LR
                    diagnostics['observations'].append(
                        f"LR schedule: {np.min(lrs):.2e} → {np.max(lrs):.2e}"
                    )

        return diagnostics

    def _format_llm_summary(
        self,
        summaries: List[Dict],
        diagnostics: Dict[str, Any],
        final_metrics: Dict
    ) -> str:
        """
        Format training run summary as plain text.

        Args:
            summaries: List of interval summaries
            diagnostics: Diagnostic findings
            final_metrics: Final training metrics

        Returns:
            Formatted plain text summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("TRAINING RUN SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        # Basic info
        lines.append(f"Run Directory: {self.run_dir}")
        lines.append(f"Started: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        duration = (datetime.now() - self.training_start_time).total_seconds()
        lines.append(f"Duration: {duration/60:.1f} minutes")
        lines.append("")

        # Configuration (read from file if available)
        config_path = self.run_dir / "training_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                lines.append("Configuration:")
                lines.append(f"- Batch size: {config.get('batch_size', 'N/A')}")
                lines.append(f"- Learning rate: {config.get('learning_rate', 'N/A')}")
                lines.append(f"- Total samples: {config.get('total_samples', 'N/A')}")
                lines.append(f"- Num training examples: {config.get('num_training_examples', 'N/A')}")
                lines.append(f"- Warmup steps: {config.get('warmup_steps', 'N/A')}")
                lines.append(f"- Focal beta: {config.get('focal_beta', 'N/A')}")
                lines.append(f"- Focal alpha: {config.get('focal_alpha', 'N/A')}")
                lines.append("")
            except Exception as e:
                lines.append(f"Configuration: (failed to load: {e})")
                lines.append("")

        # Training progress
        if summaries:
            initial_loss = summaries[0]['loss_mean']
            final_loss = summaries[-1]['loss_mean']
            reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0

            lines.append("Training Progress:")
            lines.append(f"- Initial loss: {initial_loss:.6f} (batch {summaries[0]['interval_start']})")
            lines.append(f"- Final loss: {final_loss:.6f} (batch {summaries[-1]['interval_end']})")

            if self.best_loss is not None:
                lines.append(f"- Best loss: {self.best_loss:.6f} (batch {self.best_loss_batch})")

            lines.append(f"- Loss reduction: {reduction:.1f}%")
            lines.append(f"- Total samples seen: {summaries[-1]['samples_seen']}")
            lines.append("")

        # Key observations
        if diagnostics.get('observations'):
            lines.append("Key Observations:")
            for obs in diagnostics['observations']:
                lines.append(f"- {obs}")
            lines.append("")

        # Diagnostics
        lines.append("Diagnostics:")

        if diagnostics.get('lr_issues'):
            lines.append("- ⚠️  Learning rate issues detected")
        else:
            lines.append("- ✅ Learning rate schedule appears normal")

        if diagnostics.get('gradient_issues'):
            lines.append("- ⚠️  Gradient flow issues detected")
        else:
            lines.append("- ✅ Gradients flowing normally")

        if diagnostics.get('batch_averaging_suspected'):
            lines.append("- ⚠️  Batch averaging suspected (uneven sample frequencies)")

        if diagnostics.get('plateau_detected'):
            lines.append("- ⚠️  Loss plateau detected in final 20% of training")

        lines.append("")

        # Recommendations
        lines.append("Recommendations:")

        if diagnostics.get('batch_averaging_suspected'):
            lines.append("1. Try batch_size=1 to avoid gradient averaging across samples")

        if diagnostics.get('plateau_detected'):
            lines.append("2. Consider training longer or increasing learning rate")

        if diagnostics.get('gradient_issues'):
            lines.append("3. Check gradient clipping, normalization, or architectural issues")

        if not any([diagnostics.get('batch_averaging_suspected'),
                   diagnostics.get('plateau_detected'),
                   diagnostics.get('gradient_issues')]):
            lines.append("- No major issues detected")

        lines.append("")
        lines.append(f"Full logs: {self.run_dir}")
        lines.append("=" * 70)

        return '\n'.join(lines)

    @staticmethod
    def _json_default(obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
