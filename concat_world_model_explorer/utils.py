"""
Utility functions for formatting and display.
"""


def format_loss(loss_value):
    """Format loss value for display"""
    if loss_value is None:
        return "--"
    if loss_value < 0.001:
        return f"{loss_value:.2e}"
    else:
        return f"{loss_value:.6f}"


def format_grad_diagnostics(grad_diag):
    """Format gradient diagnostics for display"""
    if grad_diag is None:
        return "No diagnostics available"

    lines = []
    lines.append("**Gradient Flow Diagnostics:**\n")
    lines.append(f"- Learning Rate: {grad_diag['lr']:.2e}")
    lines.append(f"- Decoder Head Weight Grad Norm: {format_loss(grad_diag['head_weight_norm'])}")
    lines.append(f"- Decoder Head Bias Grad Norm: {format_loss(grad_diag['head_bias_norm'])}")
    lines.append(f"- Mask Token Grad Norm: {format_loss(grad_diag['mask_token_norm'])}")
    lines.append(f"- First Decoder QKV Weight Grad Norm: {format_loss(grad_diag['qkv_weight_norm'])}")

    lines.append("\n**Loss Metrics:**\n")
    focal_beta = grad_diag.get('focal_beta', 10.0)
    focal_alpha = grad_diag.get('focal_alpha', 0.2)
    lines.append(f"- **Hybrid Loss (training)**: {format_loss(grad_diag.get('loss_hybrid'))} *[α={focal_alpha:.2f}]*")
    lines.append(f"  - Plain MSE Component: {format_loss(grad_diag.get('loss_plain'))}")
    lines.append(f"  - Focal MSE Component: {format_loss(grad_diag.get('loss_focal'))} *[β={focal_beta:.1f}]*")
    lines.append(f"- Standard Loss (for comparison): {format_loss(grad_diag.get('loss_standard'))}")

    lines.append("\n**Focal Weight Statistics:**\n")
    lines.append(f"- Mean Focal Weight: {grad_diag.get('focal_weight_mean', 1.0):.3f}")
    lines.append(f"- Max Focal Weight: {grad_diag.get('focal_weight_max', 1.0):.3f}")
    lines.append("  *(Focal weights adaptively upweight high-error pixels)*")

    lines.append("\n**Loss Dilution Diagnostics:**\n")
    lines.append(f"- Loss on Non-Black Pixels: {format_loss(grad_diag.get('loss_nonblack'))}")
    lines.append(f"- Black Baseline (if model predicted black): {format_loss(grad_diag.get('black_baseline'))}")
    lines.append(f"- Fraction of Non-Black Pixels: {grad_diag.get('frac_nonblack', 0):.6f} ({grad_diag.get('frac_nonblack', 0)*100:.2f}%)")

    # Add interpretation hint
    if grad_diag.get('loss_nonblack') is not None and grad_diag.get('black_baseline') is not None:
        loss_nonblack = grad_diag['loss_nonblack']
        baseline = grad_diag['black_baseline']
        if loss_nonblack is not None and baseline is not None:
            improvement_pct = (1 - loss_nonblack / baseline) * 100
            lines.append(f"\n**Dot Learning Progress:** {improvement_pct:.1f}% improvement over black baseline")

            if loss_nonblack >= baseline * 0.95:  # Loss is close to or above baseline
                lines.append("⚠️ **Loss on non-black pixels ≈ black baseline** → Model is NOT learning the dot (loss dilution)")
            elif loss_nonblack < baseline * 0.5:  # Clear improvement
                lines.append("✅ **Loss dropping below baseline** → Model IS learning the dot!")
            else:
                lines.append("📊 **Some progress** → Keep training to see if dot emerges")

    return "\n".join(lines)
