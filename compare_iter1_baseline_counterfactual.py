"""
Compare counterfactual predictions between Baseline (50 steps) and Iter 1 (20 steps)
Visualize predictions under different actions side-by-side
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from recording_reader import RecordingReader
from autoencoder_concat_predictor_world_model import AutoencoderConcatPredictorWorldModel
from config import Config
from concat_world_model_explorer.canvas_ops import build_canvas

def load_checkpoint(checkpoint_path):
    """Load a model checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderConcatPredictorWorldModel(device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.model.load_state_dict(checkpoint)
    
    model.model.eval()
    return model, device

def get_counterfactual_predictions(model, canvas, device):
    """Get predictions under all 3 actions"""
    with torch.no_grad():
        canvas_tensor = torch.from_numpy(canvas).unsqueeze(0).to(device).float() / 255.0
        predictions = {}
        
        for action in [0, 1, 2]:
            pred = model.model.forward_with_patch_mask(
                canvas_tensor, 
                patch_mask=None,  # Full mask on next frame slot
                action=action
            )
            predictions[action] = (pred[0].cpu().numpy() * 255).astype(np.uint8)
    
    return predictions

# Checkpoint paths
baseline_ckpt = Path("saved/checkpoints/so101/best_model_auto_session_so101_multiheight_part1_1345_multiheight_dit_iter0_solo_00106240_fresh_val_0.016683.pth")
iter1_ckpt = Path("saved/checkpoints/so101/best_model_auto_session_so101_multiheight_part1_1345_multiheight_dit_iter1_solo_00059136_fresh_val_0.030396.pth")

print("Loading checkpoints...")
if not baseline_ckpt.exists():
    print(f"❌ Baseline checkpoint not found: {baseline_ckpt}")
    sys.exit(1)
if not iter1_ckpt.exists():
    print(f"❌ Iter 1 checkpoint not found: {iter1_ckpt}")
    sys.exit(1)

baseline_model, device = load_checkpoint(baseline_ckpt)
iter1_model, _ = load_checkpoint(iter1_ckpt)
print("✓ Models loaded")

# Load validation session
val_session = Path("saved/sessions/so101/session_so101_multiheight_part2_149")
reader = RecordingReader(val_session)
print(f"✓ Loaded validation session with {reader.num_steps} frames")

# Sample a few frames from the middle
sample_indices = [20, 50, 80]
action_names = ["Stay (0)", "Move Right (1)", "Move Left (2)"]

for idx in sample_indices:
    if idx >= reader.num_steps:
        continue
    
    print(f"\n{'='*70}")
    print(f"Frame {idx}")
    print(f"{'='*70}")
    
    # Get frames for canvas
    frames = [reader.get_observation(max(0, idx - i)) for i in range(Config.CANVAS_HISTORY_SIZE)][::-1]
    actions = [reader.get_action(max(0, idx - i)) for i in range(Config.CANVAS_HISTORY_SIZE - 1)][::-1]
    
    # Build canvas
    canvas = build_canvas(frames, actions)
    
    # Get predictions
    baseline_preds = get_counterfactual_predictions(baseline_model, canvas, device)
    iter1_preds = get_counterfactual_predictions(iter1_model, canvas, device)
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    fig.suptitle(f"Counterfactual Comparison - Frame {idx}\n(Baseline 50 steps vs Iter1 20 steps)", 
                 fontsize=14, fontweight='bold')
    
    for action in [0, 1, 2]:
        # Baseline prediction
        ax = fig.add_subplot(gs[0, action])
        pred_img = baseline_preds[action].transpose(1, 2, 0) if baseline_preds[action].shape[0] == 3 else baseline_preds[action][0]
        ax.imshow(pred_img, cmap='gray' if len(pred_img.shape) == 2 else None)
        ax.set_title(f"Baseline: {action_names[action]}", fontweight='bold')
        ax.axis('off')
        
        # Iter 1 prediction
        ax = fig.add_subplot(gs[1, action])
        pred_img = iter1_preds[action].transpose(1, 2, 0) if iter1_preds[action].shape[0] == 3 else iter1_preds[action][0]
        ax.imshow(pred_img, cmap='gray' if len(pred_img.shape) == 2 else None)
        ax.set_title(f"Iter 1: {action_names[action]}", fontweight='bold')
        ax.axis('off')
    
    # Save figure
    output_path = f"counterfactual_comparison_frame_{idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

print("\n" + "="*70)
print("Counterfactual comparison complete!")
print("Images saved as: counterfactual_comparison_frame_*.png")
print("="*70)
