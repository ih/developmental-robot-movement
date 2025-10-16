"""
Session Explorer Library

This module contains utility functions and classes for exploring recorded robot sessions.
Used by session_explorer.ipynb and session_explorer.py for session loading, frame processing,
model management, and visualization.

Main sections:
- DynamicStubRobotInterface: Stub robot that adapts to session's action space
- Session Management: Loading and listing sessions
- Frame Operations: Loading and processing frame tensors
- Action Utilities: Action formatting and comparison
- Model Loading: Autoencoder and predictor model loading
- Attention Visualization: Computing and plotting attention metrics
- Predictor Sequence Building: Building sequences for predictor training
- Weight Visualization: Visualizing autoencoder and predictor weights
"""

import os
import io
import json
import glob
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

import config
from models import MaskedAutoencoderViT, TransformerActionConditionedPredictor
from robot_interface import RobotInterface


# =============================================================================
# DynamicStubRobotInterface
# =============================================================================

class DynamicStubRobotInterface(RobotInterface):
    """Stub robot interface that adapts to session's action space"""

    def __init__(self):
        self._action_space = []

    def set_action_space(self, action_space):
        """Set action space from session metadata"""
        self._action_space = action_space if action_space else []

    def get_observation(self):
        """Return a dummy observation - not used in training"""
        return {"frame": np.zeros((224, 224, 3), dtype=np.uint8)}

    def execute_action(self, action):
        """Do nothing - not used in training"""
        pass

    @property
    def action_space(self):
        return self._action_space

    def cleanup(self):
        pass


# =============================================================================
# Session Management
# =============================================================================

def list_session_dirs_from_base(base_dir):
    """
    Return sorted session directory names from a single base directory.

    Args:
        base_dir: Base directory to search for sessions

    Returns:
        List of session directory paths
    """
    if not os.path.exists(base_dir):
        return []
    entries = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("session_"):
            entries.append(os.path.join(base_dir, name))
    return entries


def list_session_dirs(jetbot_sessions_dir=None, toroidal_dot_sessions_dir=None, sessions_root_dir=None):
    """
    Return sorted session directory paths from all robot-specific directories.

    Args:
        jetbot_sessions_dir: Path to JetBot sessions directory (uses config default if None)
        toroidal_dot_sessions_dir: Path to toroidal dot sessions directory (uses config default if None)
        sessions_root_dir: Optional legacy root directory for backward compatibility

    Returns:
        List of all session directory paths
    """
    # Use config defaults if not provided
    if jetbot_sessions_dir is None:
        jetbot_sessions_dir = config.JETBOT_RECORDING_DIR
    if toroidal_dot_sessions_dir is None:
        toroidal_dot_sessions_dir = config.TOROIDAL_DOT_RECORDING_DIR

    all_sessions = []

    # Scan jetbot sessions
    all_sessions.extend(list_session_dirs_from_base(jetbot_sessions_dir))

    # Scan toroidal dot sessions
    all_sessions.extend(list_session_dirs_from_base(toroidal_dot_sessions_dir))

    # Also scan legacy root directory for backward compatibility
    if sessions_root_dir and os.path.exists(sessions_root_dir):
        for name in os.listdir(sessions_root_dir):
            path = os.path.join(sessions_root_dir, name)
            # Skip robot-specific subdirectories
            if os.path.isdir(path) and name.startswith("session_"):
                all_sessions.append(path)

    # Sort by session name
    all_sessions.sort()
    return all_sessions


def get_session_display_name(session_path):
    """
    Extract readable session name from full path.

    Args:
        session_path: Full path to session directory

    Returns:
        Session name (basename)
    """
    return os.path.basename(session_path)


def get_session_robot_type(session_path):
    """
    Infer robot type from session path.

    Args:
        session_path: Full path to session directory

    Returns:
        Robot type string ('jetbot', 'toroidal_dot', or 'unknown')
    """
    if "jetbot" in session_path.lower():
        return "jetbot"
    elif "toroidal" in session_path.lower():
        return "toroidal_dot"
    return "unknown"


def load_session_metadata(session_dir):
    """
    Load session metadata from session_meta.json.

    Args:
        session_dir: Path to session directory

    Returns:
        Dictionary of session metadata
    """
    meta_path = os.path.join(session_dir, "session_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def load_session_events(session_dir):
    """
    Load all events from shard files and sort them by step.

    Args:
        session_dir: Path to session directory

    Returns:
        List of event dictionaries sorted by step
    """
    pattern = os.path.join(session_dir, "events_shard_*.jsonl")
    shard_files = sorted(glob.glob(pattern))
    events = []
    for shard_path in shard_files:
        with open(shard_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
    events.sort(key=lambda evt: evt.get("step", 0))
    return events


def extract_observations(events, session_dir):
    """
    Extract observation entries from events.

    Args:
        events: List of event dictionaries
        session_dir: Path to session directory for constructing full paths

    Returns:
        List of observation dictionaries with metadata
    """
    observations = []
    for idx, event in enumerate(events):
        if event.get("type") != "observation":
            continue
        data = event.get("data", {})
        frame_path = data.get("frame_path")
        if not frame_path:
            continue
        observations.append({
            "observation_index": len(observations),
            "event_index": idx,
            "step": event.get("step", idx),
            "timestamp": event.get("timestamp"),
            "frame_path": frame_path,
            "full_path": os.path.join(session_dir, frame_path),
        })
    return observations


def extract_actions(events):
    """
    Extract action entries from events.

    Args:
        events: List of event dictionaries

    Returns:
        List of action dictionaries with metadata
    """
    actions = []
    for idx, event in enumerate(events):
        if event.get("type") != "action":
            continue
        actions.append({
            "action_index": len(actions),
            "event_index": idx,
            "step": event.get("step", idx),
            "timestamp": event.get("timestamp"),
            "action": event.get("data", {}),
        })
    return actions


# =============================================================================
# Frame Operations
# =============================================================================

def load_frame_bytes(full_path):
    """
    Load frame bytes from disk.

    Args:
        full_path: Full path to frame file

    Returns:
        Bytes of the image file
    """
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Frame file not found: {full_path}")
    with open(full_path, "rb") as f:
        return f.read()


def load_frame_image(full_path):
    """
    Load frame as PIL Image.

    Args:
        full_path: Full path to frame file

    Returns:
        PIL Image in RGB format
    """
    return Image.open(io.BytesIO(load_frame_bytes(full_path))).convert("RGB")


def get_frame_tensor(session_dir, frame_path, transform=None):
    """
    Return normalized (C,H,W) tensor for a frame.

    Args:
        session_dir: Path to session directory
        frame_path: Relative path to frame within session
        transform: Optional transform to apply (uses config default if None)

    Returns:
        Tensor of shape (C, H, W)
    """
    if transform is None:
        transform = config.TRANSFORM

    full_path = os.path.join(session_dir, frame_path)
    pil_img = load_frame_image(full_path)
    return transform(pil_img)


def tensor_to_numpy_image(tensor):
    """
    Convert tensor to numpy array for visualization.

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)

    Returns:
        Numpy array of shape (H, W, C) in [0, 1] range
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float()
    # Simply clamp to [0, 1] - no denormalization needed since model outputs [0, 1] range
    tensor = tensor.clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


# =============================================================================
# Action Utilities
# =============================================================================

def format_timestamp(ts):
    """
    Format timestamp for display.

    Args:
        ts: Unix timestamp or None

    Returns:
        ISO formatted timestamp string or "N/A"
    """
    if ts is None:
        return "N/A"
    try:
        return datetime.datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return str(ts)


def describe_action(action):
    """
    Create human-readable description of action.

    Args:
        action: Action dictionary

    Returns:
        String description of action
    """
    if not action:
        return "{}"
    parts = []
    for key in sorted(action.keys()):
        parts.append(f"{key}: {action[key]}")
    return ", ".join(parts)


def canonical_action_key(action):
    """
    Create canonical tuple key for action comparison.

    Args:
        action: Action dictionary

    Returns:
        Tuple of sorted (key, value) pairs
    """
    if not action:
        return ()
    return tuple(sorted(action.items()))


def get_action_space(session_state):
    """
    Extract action space from session metadata or derive from recorded actions.

    Args:
        session_state: Dictionary containing session metadata and actions

    Returns:
        List of action dictionaries
    """
    metadata_actions = session_state.get("metadata", {}).get("action_space") or []
    if metadata_actions:
        return metadata_actions
    unique = []
    seen = set()
    for action_entry in session_state.get("actions", []):
        action = action_entry.get("action", {})
        key = canonical_action_key(action)
        if key and key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def format_action_label(action):
    """
    Create compact action label for display.

    Args:
        action: Action dictionary

    Returns:
        String label with key=value pairs
    """
    if not action:
        return "{}"
    parts = []
    for key in sorted(action.keys()):
        parts.append(f"{key}={action[key]}")
    return ", ".join(parts)


def clone_action(action):
    """
    Create a deep copy of an action dictionary.

    Args:
        action: Action dictionary

    Returns:
        Cloned action dictionary
    """
    if not action:
        return {}
    return {key: float(value) if isinstance(value, (int, float)) else value for key, value in action.items()}


def actions_equal(action_a, action_b):
    """
    Check if two actions are equal.

    Args:
        action_a: First action dictionary
        action_b: Second action dictionary

    Returns:
        True if actions are equal, False otherwise
    """
    return canonical_action_key(action_a) == canonical_action_key(action_b)


# =============================================================================
# Model Loading
# =============================================================================

def load_autoencoder_model(path, device):
    """
    Load autoencoder model from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Torch device to load model on

    Returns:
        Loaded MaskedAutoencoderViT model in eval mode
    """
    model = MaskedAutoencoderViT()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_predictor_model(path, device):
    """
    Load predictor model from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Torch device to load model on

    Returns:
        Loaded TransformerActionConditionedPredictor model in eval mode
    """
    model = TransformerActionConditionedPredictor()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    level = checkpoint.get("level")
    if level is not None:
        model.level = level
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Action Variants
# =============================================================================

def build_action_variants(action, action_space=None):
    """
    Build action variants using the actual action space instead of percentage variations.

    Args:
        action: The recorded action (unused, kept for compatibility)
        action_space: List of all possible actions from the robot's action space

    Returns:
        List of action dictionaries from the action space
    """
    if action_space:
        return [clone_action(a) for a in action_space]
    # Fallback if no action_space provided
    if not action:
        return []
    return [clone_action(action)]


# =============================================================================
# Attention Visualization
# =============================================================================

def compute_attention_visual_data(attn_info):
    """
    Compute attention metrics and prepare data for visualization.

    Args:
        attn_info: Dictionary containing attention weights and token indices

    Returns:
        Dictionary with metrics, heatmap, token types, and breakdown data
    """
    if not attn_info or not attn_info.get('attn'):
        return None
    attn_list = attn_info['attn']
    last_layer_attn = attn_list[-1]
    if last_layer_attn is None or last_layer_attn.size(0) == 0:
        return None
    attn_last = last_layer_attn[0]
    future_idx = attn_info['future_idx']
    if future_idx.numel() == 0:
        return None

    attn_future = attn_last.mean(dim=0)[future_idx]
    total = attn_future.sum(dim=-1, keepdim=True) + 1e-8

    if attn_info['last_action_pos'] is not None:
        apa = attn_future[..., attn_info['last_action_pos']]
    else:
        apa = torch.zeros_like(attn_future[:, 0])

    if attn_info['last_frame_idx'].numel() > 0:
        alf = attn_future[..., attn_info['last_frame_idx']].sum(dim=-1)
    else:
        alf = torch.zeros_like(attn_future[:, 0])

    if attn_info['action_idx'].numel() > 0:
        action_mass = attn_future[..., attn_info['action_idx']].sum(dim=-1)
    else:
        action_mass = torch.zeros_like(attn_future[:, 0])

    if attn_info['frame_idx'].numel() > 0:
        frame_mass = attn_future[..., attn_info['frame_idx']].sum(dim=-1)
    else:
        frame_mass = torch.zeros_like(attn_future[:, 0])

    ttar = action_mass / (frame_mass + 1e-8)

    k = min(16, attn_future.shape[-1])
    recent_mass = attn_future[..., -k:].sum(dim=-1)
    ri_k = recent_mass / (total.squeeze(-1))

    P = (attn_future / total).clamp_min(1e-8)
    entropy = -(P * P.log()).sum(dim=-1)

    metrics = {
        'APA': apa.mean().item(),
        'ALF': alf.mean().item(),
        'TTAR': ttar.mean().item(),
        'RI@16': ri_k.mean().item(),
        'Entropy': entropy.mean().item(),
        'UniformBaseline': (1.0 / (future_idx.to(torch.float32) + 1.0)).mean().item(),
    }

    last_action_frac = (apa / total.squeeze(-1)).cpu().numpy()
    last_frame_frac = (alf / total.squeeze(-1)).cpu().numpy()
    rest_frac = np.clip(1.0 - last_action_frac - last_frame_frac, 0.0, 1.0)

    heatmap = attn_last[0][future_idx].cpu().numpy()
    token_types = attn_info['token_types'].cpu().numpy()

    return {
        'metrics': metrics,
        'heatmap': heatmap,
        'token_types': token_types,
        'breakdown': {
            'last_action': last_action_frac,
            'last_frame': last_frame_frac,
            'rest': rest_frac,
        },
        'future_indices': future_idx.cpu().numpy(),
    }


def plot_attention_heatmap(heatmap, token_types):
    """
    Plot attention heatmap with token type visualization.

    Args:
        heatmap: Attention weights array of shape (num_future_queries, num_tokens)
        token_types: Token type array (0=frame, 1=action, 2=future)
    """
    if heatmap.size == 0:
        return
    fig, (ax_heat, ax_types) = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [4, 0.4]}, sharex=True)
    im = ax_heat.imshow(heatmap, aspect='auto', cmap='viridis')
    ax_heat.set_ylabel('Future Query')
    ax_heat.set_title('Attention (last layer, head 0)')
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    type_cmap = ListedColormap(['#4c72b0', '#dd8452', '#55a868'])
    ax_types.imshow(token_types.reshape(1, -1), aspect='auto', cmap=type_cmap, vmin=0, vmax=2)
    ax_types.set_yticks([])
    ax_types.set_xlabel('Key Token Index')
    ax_types.set_title('Token Types (frame/action/future)')
    plt.tight_layout()
    plt.show()


def plot_attention_breakdown(breakdown):
    """
    Plot attention breakdown as stacked bar chart.

    Args:
        breakdown: Dictionary with 'last_action', 'last_frame', and 'rest' arrays
    """
    if not breakdown['last_action'].size:
        return
    indices = np.arange(len(breakdown['last_action']))
    width = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(indices - width, breakdown['last_action'], width, label='Last Action')
    plt.bar(indices, breakdown['last_frame'], width, label='Last Frame Block')
    plt.bar(indices + width, breakdown['rest'], width, label='Rest')
    plt.xlabel('Future Query Index')
    plt.ylabel('Attention Fraction')
    plt.ylim(0.0, 1.05)
    plt.title('Attention Distribution per Future Query')
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Predictor Sequence Building
# =============================================================================

def decode_features_to_image(autoencoder, predicted_features):
    """
    Decode latent features to image using autoencoder decoder.

    Args:
        autoencoder: MaskedAutoencoderViT model
        predicted_features: Latent features tensor

    Returns:
        Decoded image tensor
    """
    autoencoder.eval()
    with torch.no_grad():
        num_patches = autoencoder.patch_embed.num_patches
        ids_restore = torch.arange(num_patches, device=predicted_features.device).unsqueeze(0).repeat(predicted_features.shape[0], 1)
        pred_patches = autoencoder.forward_decoder(predicted_features, ids_restore)
        decoded = autoencoder.unpatchify(pred_patches)
    return decoded


def build_predictor_sequence(session_state, target_obs_index, desired_length):
    """
    Build a sequence of observations and actions for predictor training.

    Args:
        session_state: Dictionary containing observations and events
        target_obs_index: Index of target observation
        desired_length: Desired sequence length

    Returns:
        Tuple of (selected_observations, action_dicts, error_message)
    """
    observations = session_state.get("observations", [])
    events = session_state.get("events", [])
    if not observations:
        return [], [], "No observations loaded."
    if target_obs_index < 0 or target_obs_index >= len(observations):
        return [], [], "Selected observation is out of range."
    desired_length = max(2, desired_length)
    selected_obs = [observations[target_obs_index]]
    action_dicts = []
    current_idx = target_obs_index
    current_event_index = observations[target_obs_index]["event_index"]
    while len(selected_obs) < desired_length and current_idx > 0:
        prev_idx = current_idx - 1
        found = False
        while prev_idx >= 0:
            prev_obs = observations[prev_idx]
            prev_event_index = prev_obs["event_index"]
            actions_between = [events[i] for i in range(prev_event_index + 1, current_event_index) if events[i].get("type") == "action"]
            if len(actions_between) == 1:
                action_dicts.insert(0, actions_between[0].get("data", {}))
                selected_obs.insert(0, prev_obs)
                current_idx = prev_idx
                current_event_index = prev_event_index
                found = True
                break
            prev_idx -= 1
        if not found:
            break
    if len(selected_obs) < 2:
        return [], [], "Could not assemble a history with actions between frames. Choose a later frame."
    return selected_obs, action_dicts, None


def find_action_between_events(events, start_event_index, end_event_index):
    """
    Return the recorded action between two observation events, falling back to the prior action.

    Args:
        events: List of event dictionaries
        start_event_index: Starting event index
        end_event_index: Ending event index

    Returns:
        Tuple of (action_dict, source) where source is 'between' or 'previous'
    """
    between_actions = [
        event for event in events[start_event_index + 1:end_event_index]
        if event.get("type") == "action"
    ]
    if between_actions:
        return clone_action(between_actions[-1].get("data", {})), "between"

    for idx in range(start_event_index, -1, -1):
        event = events[idx]
        if event.get("type") == "action":
            return clone_action(event.get("data", {})), "previous"

    return None, None


def get_future_action_for_prediction(session_state, target_obs_index):
    """
    Return the action to pair with the next observation for prediction.

    Args:
        session_state: Dictionary containing observations and events
        target_obs_index: Index of current observation

    Returns:
        Tuple of (action_dict, source)
    """
    observations = session_state.get("observations", [])
    events = session_state.get("events", [])
    if target_obs_index < 0 or target_obs_index >= len(observations) - 1:
        return None, None
    current_obs = observations[target_obs_index]
    next_obs = observations[target_obs_index + 1]
    return find_action_between_events(events, current_obs["event_index"], next_obs["event_index"])


# =============================================================================
# Weight Visualization
# =============================================================================

def visualize_autoencoder_weights(autoencoder):
    """
    Visualize key autoencoder weights for monitoring changes.

    Args:
        autoencoder: MaskedAutoencoderViT model

    Returns:
        Dictionary of weight statistics
    """
    if autoencoder is None:
        return None

    with torch.no_grad():
        # Get patch embedding weights (first layer)
        patch_embed_weight = autoencoder.patch_embed.proj.weight.detach().cpu()

        # Get cls token and pos embed
        cls_token = autoencoder.cls_token.detach().cpu()
        pos_embed = autoencoder.pos_embed.detach().cpu()

        # Get some decoder weights
        decoder_embed_weight = autoencoder.decoder_embed.weight.detach().cpu() if hasattr(autoencoder, 'decoder_embed') else None

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Patch embedding weights - show first 16 filters
        patch_weights_vis = patch_embed_weight[:16].view(16, 3, 16, 16)
        patch_grid = torch.cat([torch.cat([patch_weights_vis[i*4+j] for j in range(4)], dim=2) for i in range(4)], dim=1)
        patch_grid = (patch_grid - patch_grid.min()) / (patch_grid.max() - patch_grid.min())
        axes[0, 0].imshow(patch_grid.permute(1, 2, 0))
        axes[0, 0].set_title("Patch Embed Weights (16 filters)")
        axes[0, 0].axis("off")

        # Patch embedding weight statistics
        axes[0, 1].hist(patch_embed_weight.flatten().numpy(), bins=50, alpha=0.7)
        axes[0, 1].set_title(f"Patch Embed Weight Distribution\nMean: {patch_embed_weight.mean():.6f}, Std: {patch_embed_weight.std():.6f}")
        axes[0, 1].set_xlabel("Weight Value")
        axes[0, 1].set_ylabel("Count")

        # CLS token visualization
        cls_reshaped = cls_token.view(-1).numpy()
        axes[0, 2].plot(cls_reshaped)
        axes[0, 2].set_title(f"CLS Token\nMean: {cls_token.mean():.6f}, Std: {cls_token.std():.6f}")
        axes[0, 2].set_xlabel("Dimension")
        axes[0, 2].set_ylabel("Value")

        # Position embedding visualization (first 100 dimensions)
        pos_vis = pos_embed[0, :, :100].numpy()
        im = axes[1, 0].imshow(pos_vis, aspect='auto', cmap='coolwarm')
        axes[1, 0].set_title(f"Position Embeddings (first 100 dims)\nShape: {pos_embed.shape}")
        axes[1, 0].set_xlabel("Embedding Dimension")
        axes[1, 0].set_ylabel("Position")
        plt.colorbar(im, ax=axes[1, 0])

        # Decoder embedding weights if available
        if decoder_embed_weight is not None:
            axes[1, 1].hist(decoder_embed_weight.flatten().numpy(), bins=50, alpha=0.7, color='orange')
            axes[1, 1].set_title(f"Decoder Embed Weight Distribution\nMean: {decoder_embed_weight.mean():.6f}, Std: {decoder_embed_weight.std():.6f}")
            axes[1, 1].set_xlabel("Weight Value")
            axes[1, 1].set_ylabel("Count")
        else:
            axes[1, 1].text(0.5, 0.5, "Decoder weights\nnot available", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Decoder Weights")

        # Weight norms across layers
        layer_norms = []
        layer_names = []
        for name, param in autoencoder.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                layer_norms.append(param.norm().item())
                layer_names.append(name.split('.')[-2] if '.' in name else name)

        if layer_norms:
            axes[1, 2].bar(range(len(layer_norms)), layer_norms)
            axes[1, 2].set_title("Layer Weight Norms")
            axes[1, 2].set_xlabel("Layer")
            axes[1, 2].set_ylabel("L2 Norm")
            axes[1, 2].set_xticks(range(0, len(layer_names), max(1, len(layer_names)//5)))
            axes[1, 2].set_xticklabels([layer_names[i] for i in range(0, len(layer_names), max(1, len(layer_names)//5))], rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, "No weight layers\nfound", ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title("Layer Weight Norms")

        plt.tight_layout()
        plt.show()

        # Return weight statistics for comparison
        stats = {
            'patch_embed_mean': patch_embed_weight.mean().item(),
            'patch_embed_std': patch_embed_weight.std().item(),
            'cls_token_mean': cls_token.mean().item(),
            'cls_token_std': cls_token.std().item(),
            'pos_embed_mean': pos_embed.mean().item(),
            'pos_embed_std': pos_embed.std().item(),
            'layer_norms': dict(zip(layer_names, layer_norms)) if layer_norms else {}
        }
        return stats


def visualize_predictor_weights(predictor):
    """
    Visualize key predictor (transformer) weights for monitoring changes.

    Args:
        predictor: TransformerActionConditionedPredictor model

    Returns:
        Dictionary of weight statistics
    """
    if predictor is None:
        return None

    with torch.no_grad():
        # Get various transformer weights based on actual architecture
        # Action embedding weights
        action_embed_weight = predictor.action_embedding.weight.detach().cpu() if hasattr(predictor, 'action_embedding') else None

        # Position embedding weights
        pos_embed_weight = predictor.position_embedding.weight.detach().cpu() if hasattr(predictor, 'position_embedding') else None

        # Token type embedding weights
        token_type_weight = predictor.token_type_embedding.weight.detach().cpu() if hasattr(predictor, 'token_type_embedding') else None

        # Future query parameter
        future_query_weight = predictor.future_query.detach().cpu() if hasattr(predictor, 'future_query') else None

        # Get first transformer layer weights (TransformerEncoderLayer structure)
        first_layer_self_attn_weight = None
        first_layer_linear1_weight = None
        first_layer_linear2_weight = None

        if hasattr(predictor, 'transformer_layers') and len(predictor.transformer_layers) > 0:
            first_layer = predictor.transformer_layers[0]

            # Self-attention weights (in_proj_weight contains Q, K, V)
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'in_proj_weight'):
                first_layer_self_attn_weight = first_layer.self_attn.in_proj_weight.detach().cpu()

            # MLP weights
            if hasattr(first_layer, 'linear1') and hasattr(first_layer.linear1, 'weight'):
                first_layer_linear1_weight = first_layer.linear1.weight.detach().cpu()
            if hasattr(first_layer, 'linear2') and hasattr(first_layer.linear2, 'weight'):
                first_layer_linear2_weight = first_layer.linear2.weight.detach().cpu()

        # Output head weights
        output_head_weight = predictor.output_head.weight.detach().cpu() if hasattr(predictor, 'output_head') and hasattr(predictor.output_head, 'weight') else None

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Action embedding weights
        if action_embed_weight is not None:
            axes[0, 0].hist(action_embed_weight.flatten().numpy(), bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title(f"Action Embed Weight Distribution\nMean: {action_embed_weight.mean():.6f}, Std: {action_embed_weight.std():.6f}")
            axes[0, 0].set_xlabel("Weight Value")
            axes[0, 0].set_ylabel("Count")
        else:
            axes[0, 0].text(0.5, 0.5, "Action embedding\nweights not found", ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("Action Embedding Weights")

        # Position embedding weights
        if pos_embed_weight is not None:
            axes[0, 1].hist(pos_embed_weight.flatten().numpy(), bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title(f"Position Embed Weight Distribution\nMean: {pos_embed_weight.mean():.6f}, Std: {pos_embed_weight.std():.6f}")
            axes[0, 1].set_xlabel("Weight Value")
            axes[0, 1].set_ylabel("Count")
        else:
            axes[0, 1].text(0.5, 0.5, "Position embedding\nweights not found", ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Position Embedding Weights")

        # First transformer layer self-attention weights
        if first_layer_self_attn_weight is not None:
            axes[0, 2].hist(first_layer_self_attn_weight.flatten().numpy(), bins=50, alpha=0.7, color='red')
            axes[0, 2].set_title(f"First Layer Self-Attn Weights\nMean: {first_layer_self_attn_weight.mean():.6f}, Std: {first_layer_self_attn_weight.std():.6f}")
            axes[0, 2].set_xlabel("Weight Value")
            axes[0, 2].set_ylabel("Count")
        else:
            axes[0, 2].text(0.5, 0.5, "Self-attention\nweights not found", ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title("First Layer Self-Attn")

        # First transformer layer linear1 (MLP) weights
        if first_layer_linear1_weight is not None:
            axes[1, 0].hist(first_layer_linear1_weight.flatten().numpy(), bins=50, alpha=0.7, color='purple')
            axes[1, 0].set_title(f"First Layer Linear1 Weights\nMean: {first_layer_linear1_weight.mean():.6f}, Std: {first_layer_linear1_weight.std():.6f}")
            axes[1, 0].set_xlabel("Weight Value")
            axes[1, 0].set_ylabel("Count")
        else:
            axes[1, 0].text(0.5, 0.5, "Linear1 weights\nnot found", ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("First Layer Linear1")

        # Output head weights
        if output_head_weight is not None:
            axes[1, 1].hist(output_head_weight.flatten().numpy(), bins=50, alpha=0.7, color='orange')
            axes[1, 1].set_title(f"Output Head Weight Distribution\nMean: {output_head_weight.mean():.6f}, Std: {output_head_weight.std():.6f}")
            axes[1, 1].set_xlabel("Weight Value")
            axes[1, 1].set_ylabel("Count")
        else:
            axes[1, 1].text(0.5, 0.5, "Output head\nweights not found", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Output Head Weights")

        # Weight norms across all layers
        layer_norms = []
        layer_names = []
        for name, param in predictor.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                layer_norms.append(param.norm().item())
                # Shorten layer names for display
                short_name = name.split('.')[-1] if '.' in name else name
                if len(short_name) > 15:
                    short_name = short_name[:12] + "..."
                layer_names.append(short_name)

        if layer_norms:
            axes[1, 2].bar(range(len(layer_norms)), layer_norms)
            axes[1, 2].set_title("Layer Weight Norms")
            axes[1, 2].set_xlabel("Layer")
            axes[1, 2].set_ylabel("L2 Norm")
            # Show every 5th label to avoid overcrowding
            step = max(1, len(layer_names)//8)
            axes[1, 2].set_xticks(range(0, len(layer_names), step))
            axes[1, 2].set_xticklabels([layer_names[i] for i in range(0, len(layer_names), step)], rotation=45, ha='right')
        else:
            axes[1, 2].text(0.5, 0.5, "No weight layers\nfound", ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title("Layer Weight Norms")

        plt.tight_layout()
        plt.show()

        # Return weight statistics for comparison
        stats = {}
        if action_embed_weight is not None:
            stats.update({
                'action_embed_mean': action_embed_weight.mean().item(),
                'action_embed_std': action_embed_weight.std().item(),
            })
        if pos_embed_weight is not None:
            stats.update({
                'pos_embed_mean': pos_embed_weight.mean().item(),
                'pos_embed_std': pos_embed_weight.std().item(),
            })
        if first_layer_self_attn_weight is not None:
            stats.update({
                'first_self_attn_mean': first_layer_self_attn_weight.mean().item(),
                'first_self_attn_std': first_layer_self_attn_weight.std().item(),
            })
        if first_layer_linear1_weight is not None:
            stats.update({
                'first_linear1_mean': first_layer_linear1_weight.mean().item(),
                'first_linear1_std': first_layer_linear1_weight.std().item(),
            })
        if output_head_weight is not None:
            stats.update({
                'output_head_mean': output_head_weight.mean().item(),
                'output_head_std': output_head_weight.std().item(),
            })
        if layer_norms:
            stats['layer_norms'] = dict(zip(layer_names, layer_norms))

        return stats
