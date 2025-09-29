#!/usr/bin/env python
# coding: utf-8

# # Session Explorer
# This notebook lets you explore recorded robot sessions, scrub through frames, run the autoencoder and predictor models on stored observations, and train the models on selected frames/history.

# ## How to Use
# 1. Pick a session and load it.
# 2. Use the playback controls to scrub through frames.
# 3. (Optional) Load model checkpoints, then run the Autoencoder and Predictor sections using the current frame selection.
# 4. (Optional) Use the Training sections to train models on current frame/history until a loss threshold is met or for a specified number of steps.

# In[1]:


import os
import io
import json
import glob
import datetime
from functools import lru_cache
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, Markdown

import config
from models import MaskedAutoencoderViT, TransformerActionConditionedPredictor
from adaptive_world_model import AdaptiveWorldModel
from robot_interface import RobotInterface

# Additional imports for training
import time
from tqdm.auto import tqdm


# In[2]:


SESSIONS_BASE_DIR = config.RECORDING_BASE_DIR
DEFAULT_AUTOENCODER_PATH = os.path.join(config.DEFAULT_CHECKPOINT_DIR, "autoencoder.pth")
DEFAULT_PREDICTOR_PATH = os.path.join(config.DEFAULT_CHECKPOINT_DIR, "predictor_0.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a stub robot interface for the AdaptiveWorldModel
class StubRobotInterface(RobotInterface):
    """Stub robot interface for notebook training purposes"""
    def get_observation(self):
        # Return a dummy observation - not used in training
        return {"frame": np.zeros((224, 224, 3), dtype=np.uint8)}
    
    def execute_action(self, action):
        # Do nothing - not used in training
        pass
    
    @property
    def action_space(self):
        # Return minimal action space - not used in training
        return [{"motor_left": 0, "motor_right": 0, "duration": 0.2}]
    
    def cleanup(self):
        pass

# Instantiate AdaptiveWorldModel for training access
stub_robot = StubRobotInterface()
adaptive_world_model = AdaptiveWorldModel(stub_robot, wandb_project="session_explorer", checkpoint_dir=config.DEFAULT_CHECKPOINT_DIR, interactive=False)


# In[3]:


# Utility helpers for loading sessions, caching frames, and preparing model inputs
def list_session_dirs(base_dir):
    # Return sorted session directory names.
    if not os.path.exists(base_dir):
        return []
    entries = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("session_"):
            entries.append(name)
    entries.sort()
    return entries

def load_session_metadata(session_dir):
    meta_path = os.path.join(session_dir, "session_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

def load_session_events(session_dir):
    # Load all events from shard files and sort them by step.
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

@lru_cache(maxsize=4096)
def load_frame_bytes(full_path):
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Frame file not found: {full_path}")
    with open(full_path, "rb") as f:
        return f.read()

def load_frame_image(full_path):
    return Image.open(io.BytesIO(load_frame_bytes(full_path))).convert("RGB")

tensor_cache = {}

def get_frame_tensor(session_dir, frame_path):
    # Return normalized (C,H,W) tensor for a frame, cached on CPU.
    key = (session_dir, frame_path)
    if key not in tensor_cache:
        full_path = os.path.join(session_dir, frame_path)
        pil_img = load_frame_image(full_path)
        tensor_cache[key] = config.TRANSFORM(pil_img)
    return tensor_cache[key]

def tensor_to_numpy_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float()
    tmin, tmax = float(tensor.min()), float(tensor.max())
    if tmin < -0.01 or tmax > 1.01:
        tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()

def format_timestamp(ts):
    if ts is None:
        return "N/A"
    try:
        return datetime.datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return str(ts)

def describe_action(action):
    if not action:
        return "{}"
    parts = []
    for key in sorted(action.keys()):
        parts.append(f"{key}: {action[key]}")
    return ", ".join(parts)

def canonical_action_key(action):
    if not action:
        return ()
    return tuple(sorted(action.items()))

def get_action_space(session_state):
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
    if not action:
        return "{}"
    parts = []
    for key in sorted(action.keys()):
        parts.append(f"{key}={action[key]}")
    return ", ".join(parts)

def clone_action(action):
    if not action:
        return {}
    return {key: float(value) if isinstance(value, (int, float)) else value for key, value in action.items()}

def actions_equal(action_a, action_b):
    return canonical_action_key(action_a) == canonical_action_key(action_b)

def load_autoencoder_model(path, device):
    model = MaskedAutoencoderViT()
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_predictor_model(path, device):
    model = TransformerActionConditionedPredictor()
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    level = checkpoint.get("level")
    if level is not None:
        model.level = level
    model.to(device)
    model.eval()
    return model

def decode_features_to_image(autoencoder, predicted_features):
    autoencoder.eval()
    with torch.no_grad():
        num_patches = autoencoder.patch_embed.num_patches
        ids_restore = torch.arange(num_patches, device=predicted_features.device).unsqueeze(0).repeat(predicted_features.shape[0], 1)
        pred_patches = autoencoder.forward_decoder(predicted_features, ids_restore)
        decoded = autoencoder.unpatchify(pred_patches)
    return decoded

def build_predictor_sequence(session_state, target_obs_index, desired_length):
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
    """Return the recorded action between two observation events, falling back to the prior action."""
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
    """Return the action to pair with the next observation for prediction."""
    observations = session_state.get("observations", [])
    events = session_state.get("events", [])
    if target_obs_index < 0 or target_obs_index >= len(observations) - 1:
        return None, None
    current_obs = observations[target_obs_index]
    next_obs = observations[target_obs_index + 1]
    return find_action_between_events(events, current_obs["event_index"], next_obs["event_index"])

def visualize_autoencoder_weights(autoencoder):
    """Visualize key autoencoder weights for monitoring changes"""
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
    """Visualize key predictor (transformer) weights for monitoring changes"""
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


# In[4]:


# Training helper functions using AdaptiveWorldModel
def train_autoencoder_step_wrapper(frame_tensor):
    """Single autoencoder training step using AdaptiveWorldModel"""
    # Convert tensor to numpy frame for AdaptiveWorldModel
    frame_numpy = tensor_to_numpy_image(frame_tensor)
    
    # Use AdaptiveWorldModel's train_autoencoder method
    loss = adaptive_world_model.train_autoencoder(frame_numpy)
    return loss

def train_predictor_step_wrapper(target_idx, history_features, history_actions):
    """Single predictor training step using AdaptiveWorldModel"""
    # Get target frame
    next_obs = session_state["observations"][target_idx + 1]
    target_tensor = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0).to(device)
    target_frame = tensor_to_numpy_image(target_tensor)
    
    # Set up prediction context in AdaptiveWorldModel
    adaptive_world_model.observation_history = []
    adaptive_world_model.action_history = []
    
    # Add history to the world model
    for i, (feat, action) in enumerate(zip(history_features, history_actions)):
        # Convert feature back to frame if needed
        obs = session_state["observations"][target_idx - len(history_features) + 1 + i]
        frame_tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0)
        frame_numpy = tensor_to_numpy_image(frame_tensor)
        
        adaptive_world_model.observation_history.append(frame_numpy)
        adaptive_world_model.action_history.append(action)
    
    # Train predictor level 0
    loss = adaptive_world_model.train_predictor(0, target_tensor)
    return loss

def format_loss(loss_value):
    """Format loss value for display"""
    if loss_value < 0.001:
        return f"{loss_value:.2e}"
    else:
        return f"{loss_value:.6f}"


# In[5]:


# Autoencoder Training Section using AdaptiveWorldModel
import asyncio
import time

training_widgets = {}
AUTOENCODER_TASK_KEY = "autoencoder"


def _prepare_autoencoder_training():
    """Validate prerequisites and build context for autoencoder training."""
    output = training_widgets["autoencoder_training_output"]
    autoencoder = adaptive_world_model.autoencoder
    if autoencoder is None:
        with output:
            output.clear_output()
            display(Markdown("Load the autoencoder checkpoint first."))
        set_training_status("autoencoder", "error", "Autoencoder checkpoint is not loaded.")
        update_training_loss("autoencoder", None)
        return None

    frame_slider = session_widgets.get("frame_slider")
    if frame_slider is None:
        with output:
            output.clear_output()
            display(Markdown("Load a session to select frames."))
        set_training_status("autoencoder", "error", "Load a session to select frames before training.")
        update_training_loss("autoencoder", None)
        return None

    observations = session_state.get("observations", [])
    if not observations:
        with output:
            output.clear_output()
            display(Markdown("No session loaded. Load a recording before training."))
        set_training_status("autoencoder", "error", "No session loaded.")
        update_training_loss("autoencoder", None)
        return None

    idx = frame_slider.value
    observation = observations[idx]
    frame_tensor = get_frame_tensor(
        session_state["session_dir"],
        observation["frame_path"],
    ).unsqueeze(0).to(device)

    return {
        "autoencoder": autoencoder,
        "frame_tensor": frame_tensor,
        "idx": idx,
        "observation": observation,
        "output": output,
    }


async def autoencoder_threshold_training(context, threshold, max_steps):
    autoencoder = context["autoencoder"]
    frame_tensor = context["frame_tensor"]
    idx = context["idx"]
    observation = context["observation"]
    output = context["output"]

    training_control["autoencoder_resume_data"] = None
    losses = []
    start_time = time.time()

    set_training_status("autoencoder", "running", f"Running to threshold {format_loss(threshold)}")
    update_training_loss("autoencoder", None)

    with output:
        output.clear_output()
        display(Markdown(
            f"**Training autoencoder using AdaptiveWorldModel on frame {idx + 1} (step {observation['step']})**"
        ))
        display(Markdown(f"Target threshold: {format_loss(threshold)}, Max steps: {max_steps}"))
        display(Markdown("### Pre-Training Network Weights"))
        pre_stats = visualize_autoencoder_weights(autoencoder)

        display(Markdown("**Using AdaptiveWorldModel.train_autoencoder() method with randomized masking**"))
        display(Markdown("**Tip: Click 'Pause' button to interrupt training at any time.**"))

        progress = tqdm(range(max_steps), desc="Training")
        try:
            for step in progress:
                await asyncio.sleep(0)
                if training_control["autoencoder_paused"]:
                    training_control["autoencoder_resume_data"] = {
                        "frame_tensor": frame_tensor,
                        "threshold": threshold,
                        "max_steps": max_steps,
                        "current_step": step,
                        "losses": losses,
                        "start_time": start_time,
                        "pre_stats": pre_stats,
                    }
                    set_training_status("autoencoder", "paused", f"Paused at step {step} of {max_steps}. Resume available.")
                    if losses:
                        update_training_loss("autoencoder", losses[-1], len(losses), state="paused")
                    else:
                        update_training_loss("autoencoder", None)
                    display(Markdown("**Training paused. Use Resume button to continue.**"))
                    return

                loss = train_autoencoder_step_wrapper(frame_tensor)
                losses.append(loss)
                update_training_loss("autoencoder", loss, len(losses))
                progress.set_postfix({"Loss": format_loss(loss), "Step": f"{step + 1}/{max_steps}"})

                if loss <= threshold:
                    break
        finally:
            progress.close()

        end_time = time.time()
        final_loss = losses[-1] if losses else float("inf")

        display(Markdown(f"**Training completed after {len(losses)} steps in {end_time - start_time:.1f}s**"))
        display(Markdown(f"Final loss: {format_loss(final_loss)}"))

        if final_loss <= threshold:
            display(Markdown(f"**Target threshold {format_loss(threshold)} achieved!**"))
        else:
            display(Markdown(f"**Target threshold {format_loss(threshold)} not reached after {max_steps} steps.**"))

        display(Markdown("### Post-Training Network Weights"))
        post_stats = visualize_autoencoder_weights(autoencoder)

        if pre_stats and post_stats:
            weight_changes = {
                "patch_embed_mean_change": abs(post_stats["patch_embed_mean"] - pre_stats["patch_embed_mean"]),
                "patch_embed_std_change": abs(post_stats["patch_embed_std"] - pre_stats["patch_embed_std"]),
                "cls_token_mean_change": abs(post_stats["cls_token_mean"] - pre_stats["cls_token_mean"]),
                "cls_token_std_change": abs(post_stats["cls_token_std"] - pre_stats["cls_token_std"]),
            }
            changes_text = f"""
**Weight Changes:**
- Patch Embed Mean: {weight_changes['patch_embed_mean_change']:.8f}
- Patch Embed Std: {weight_changes['patch_embed_std_change']:.8f}
- CLS Token Mean: {weight_changes['cls_token_mean_change']:.8f}
- CLS Token Std: {weight_changes['cls_token_std_change']:.8f}
            """
            display(Markdown(changes_text))

        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reconstruction Loss")
            ax.set_title("Autoencoder Training Progress (AdaptiveWorldModel)")
            ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Target: {format_loss(threshold)}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        adaptive_world_model.autoencoder.eval()
        with torch.no_grad():
            reconstructed = adaptive_world_model.autoencoder.reconstruct(frame_tensor)

        original_img = tensor_to_numpy_image(frame_tensor)
        reconstructed_img = tensor_to_numpy_image(reconstructed)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(reconstructed_img)
        axes[1].set_title(f"After Training (Loss: {format_loss(final_loss)})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if losses:
        update_training_loss("autoencoder", final_loss, len(losses), state="completed")
        set_training_status("autoencoder", "completed", f"Completed in {len(losses)} steps (final loss {format_loss(final_loss)}).")
    else:
        update_training_loss("autoencoder", None)
        set_training_status("autoencoder", "completed", "Completed without updating any steps.")

    training_control["autoencoder_resume_data"] = None


async def autoencoder_steps_training(context, num_steps):
    autoencoder = context["autoencoder"]
    frame_tensor = context["frame_tensor"]
    idx = context["idx"]
    observation = context["observation"]
    output = context["output"]

    training_control["autoencoder_resume_data"] = None
    losses = []
    start_time = time.time()

    set_training_status("autoencoder", "running", f"Running for {num_steps} steps")
    update_training_loss("autoencoder", None)

    with output:
        output.clear_output()
        display(Markdown(
            f"**Training autoencoder using AdaptiveWorldModel on frame {idx + 1} (step {observation['step']}) for {num_steps} steps**"
        ))
        display(Markdown("### Pre-Training Network Weights"))
        pre_stats = visualize_autoencoder_weights(autoencoder)

        display(Markdown("**Using AdaptiveWorldModel.train_autoencoder() method with randomized masking**"))
        display(Markdown("**Tip: Click 'Pause' button to interrupt training at any time.**"))

        progress = tqdm(range(num_steps), desc="Training")
        try:
            for step in progress:
                await asyncio.sleep(0)
                if training_control["autoencoder_paused"]:
                    set_training_status("autoencoder", "paused", f"Paused after {step} of {num_steps} steps; restart to continue.")
                    if losses:
                        update_training_loss("autoencoder", losses[-1], len(losses), state="paused")
                    else:
                        update_training_loss("autoencoder", None)
                    display(Markdown("**Training paused. Step-based training cannot be resumed.**"))
                    display(Markdown(f"**Completed {step} out of {num_steps} steps before pausing.**"))
                    return

                loss = train_autoencoder_step_wrapper(frame_tensor)
                losses.append(loss)
                progress.set_postfix({"Loss": format_loss(loss), "Step": f"{step + 1}/{num_steps}"})
        finally:
            progress.close()

        end_time = time.time()
        final_loss = losses[-1] if losses else float("inf")

        display(Markdown(f"**Training completed in {end_time - start_time:.1f}s**"))
        if losses:
            display(Markdown(f"Initial loss: {format_loss(losses[0])}, Final loss: {format_loss(final_loss)}"))
        else:
            display(Markdown("** **No training steps were executed before the run stopped.**"))

        display(Markdown("### Post-Training Network Weights"))
        post_stats = visualize_autoencoder_weights(autoencoder)

        if pre_stats and post_stats:
            weight_changes = {
                "patch_embed_mean_change": abs(post_stats["patch_embed_mean"] - pre_stats["patch_embed_mean"]),
                "patch_embed_std_change": abs(post_stats["patch_embed_std"] - pre_stats["patch_embed_std"]),
                "cls_token_mean_change": abs(post_stats["cls_token_mean"] - pre_stats["cls_token_mean"]),
                "cls_token_std_change": abs(post_stats["cls_token_std"] - pre_stats["cls_token_std"]),
            }
            changes_text = f"""
**Weight Changes:**
- Patch Embed Mean: {weight_changes['patch_embed_mean_change']:.8f}
- Patch Embed Std: {weight_changes['patch_embed_std_change']:.8f}
- CLS Token Mean: {weight_changes['cls_token_mean_change']:.8f}
- CLS Token Std: {weight_changes['cls_token_std_change']:.8f}
            """
            display(Markdown(changes_text))

        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reconstruction Loss")
            ax.set_title("Autoencoder Training Progress (AdaptiveWorldModel)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        adaptive_world_model.autoencoder.eval()
        with torch.no_grad():
            reconstructed = adaptive_world_model.autoencoder.reconstruct(frame_tensor)

        original_img = tensor_to_numpy_image(frame_tensor)
        reconstructed_img = tensor_to_numpy_image(reconstructed)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(reconstructed_img)
        axes[1].set_title(f"After Training (Loss: {format_loss(final_loss)})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if losses:
        update_training_loss("autoencoder", final_loss, len(losses), state="completed")
        set_training_status("autoencoder", "completed", f"Finished {len(losses)} steps (final loss {format_loss(final_loss)}).")
    else:
        update_training_loss("autoencoder", None)
        set_training_status("autoencoder", "completed", "Completed without updating any steps.")

    training_control["autoencoder_resume_data"] = None


async def autoencoder_resume_training(resume_data):
    output = training_widgets["autoencoder_training_output"]
    autoencoder = adaptive_world_model.autoencoder
    if autoencoder is None:
        with output:
            output.clear_output()
            display(Markdown("**Autoencoder checkpoint not loaded. Load it before resuming.**"))
        training_control["autoencoder_resume_data"] = None
        update_training_loss("autoencoder", None)
        return

    with output:
        output.clear_output()
        display(Markdown("**Resuming autoencoder training...**"))

        frame_tensor = resume_data["frame_tensor"]
        threshold = resume_data["threshold"]
        max_steps = resume_data["max_steps"]
        current_step = resume_data["current_step"]
        losses = resume_data["losses"]
        start_time = resume_data["start_time"]
        pre_stats = resume_data["pre_stats"]

        if losses:
            update_training_loss("autoencoder", losses[-1], len(losses))
        else:
            update_training_loss("autoencoder", None)

        set_training_status("autoencoder", "running", f"Resuming from step {current_step} of {max_steps}.")

        remaining_steps = max_steps - current_step
        if remaining_steps <= 0:
            display(Markdown("**Training already reached the requested number of steps.**"))
            if losses:
                update_training_loss("autoencoder", losses[-1], len(losses), state="completed")
            else:
                update_training_loss("autoencoder", None)
            set_training_status("autoencoder", "completed", "Requested number of steps already reached before resuming.")
            training_control["autoencoder_resume_data"] = None
            return

        progress = tqdm(range(remaining_steps), desc=f"Resuming from step {current_step}")
        try:
            for step_offset in progress:
                await asyncio.sleep(0)
                if training_control["autoencoder_paused"]:
                    resume_data.update({
                        "current_step": current_step + step_offset,
                        "losses": losses,
                    })
                    set_training_status("autoencoder", "paused", f"Paused at step {current_step + step_offset} of {max_steps}. Resume available.")
                    if losses:
                        update_training_loss("autoencoder", losses[-1], len(losses), state="paused")
                    else:
                        update_training_loss("autoencoder", None)
                    display(Markdown("**Training paused. Use Resume button to continue.**"))
                    return

                loss = train_autoencoder_step_wrapper(frame_tensor)
                losses.append(loss)
                update_training_loss("autoencoder", loss, len(losses))
                progress.set_postfix({"Loss": format_loss(loss)})

                if loss <= threshold:
                    break
        finally:
            progress.close()

        training_control["autoencoder_resume_data"] = None

        end_time = time.time()
        final_loss = losses[-1] if losses else float("inf")

        display(Markdown(f"**Training completed after {len(losses)} total steps in {end_time - start_time:.1f}s**"))
        display(Markdown(f"Final loss: {format_loss(final_loss)}"))

        if final_loss <= threshold:
            display(Markdown(f"**Target threshold {format_loss(threshold)} achieved!**"))
        else:
            display(Markdown(f"**Target threshold {format_loss(threshold)} not reached after {max_steps} steps.**"))

        display(Markdown("### Post-Training Network Weights"))
        post_stats = visualize_autoencoder_weights(adaptive_world_model.autoencoder)

        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reconstruction Loss")
            ax.set_title("Autoencoder Training Progress (Resumed)")
            ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Target: {format_loss(threshold)}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        if losses:
            update_training_loss("autoencoder", final_loss, len(losses), state="completed")
            set_training_status("autoencoder", "completed", f"Completed with final loss {format_loss(final_loss)} after {len(losses)} steps.")
        else:
            update_training_loss("autoencoder", None)
            set_training_status("autoencoder", "completed", "Completed without updating any steps.")

    # Resume function does not render reconstructions to keep output concise.


def on_train_autoencoder_threshold(_):
    context = _prepare_autoencoder_training()
    if context is None:
        return

    training_control["autoencoder_paused"] = False
    training_control["autoencoder_resume_data"] = None

    threshold = training_widgets["autoencoder_threshold"].value
    max_steps = training_widgets["autoencoder_max_steps"].value

    set_training_status("autoencoder", "running", f"Running to threshold {format_loss(threshold)}")

    start_training_task(
        AUTOENCODER_TASK_KEY,
        autoencoder_threshold_training(context, threshold, max_steps),
        context["output"],
    )


def on_train_autoencoder_steps(_):
    context = _prepare_autoencoder_training()
    if context is None:
        return

    training_control["autoencoder_paused"] = False
    training_control["autoencoder_resume_data"] = None

    num_steps = training_widgets["autoencoder_steps"].value

    set_training_status("autoencoder", "running", f"Running for {num_steps} steps")

    start_training_task(
        AUTOENCODER_TASK_KEY,
        autoencoder_steps_training(context, num_steps),
        context["output"],
    )


# Create autoencoder training widgets
autoencoder_threshold = widgets.FloatText(value=0.0005, description="Threshold", step=0.0001, style={'description_width': '100px'})
autoencoder_max_steps = widgets.IntText(value=1000, description="Max Steps", style={'description_width': '100px'})
autoencoder_steps = widgets.IntText(value=100, description="Steps", style={'description_width': '100px'})

train_autoencoder_threshold_button = widgets.Button(description="Train to Threshold", button_style="warning", icon="target")
train_autoencoder_steps_button = widgets.Button(description="Train N Steps", button_style="warning", icon="forward")
autoencoder_training_output = widgets.Output()

training_widgets.update({
    "autoencoder_threshold": autoencoder_threshold,
    "autoencoder_max_steps": autoencoder_max_steps,
    "autoencoder_steps": autoencoder_steps,
    "autoencoder_training_output": autoencoder_training_output,
})

train_autoencoder_threshold_button.on_click(on_train_autoencoder_threshold)
train_autoencoder_steps_button.on_click(on_train_autoencoder_steps)


# In[6]:


# Training control state for pause/resume functionality
import asyncio

AUTOENCODER_TASK_KEY = globals().get("AUTOENCODER_TASK_KEY", "autoencoder")
PREDICTOR_TASK_KEY = globals().get("PREDICTOR_TASK_KEY", "predictor")

training_control = {
    "autoencoder_paused": False,
    "predictor_paused": False,
    "autoencoder_resume_data": None,
    "predictor_resume_data": None,
}

training_tasks = {
    AUTOENCODER_TASK_KEY: None,
    PREDICTOR_TASK_KEY: None,
}


STATUS_STYLES = {
    "idle": "color: #6c757d;",
    "running": "color: #2e7d32;",
    "pausing": "color: #f9a825;",
    "paused": "color: #ef6c00;",
    "completed": "color: #1565c0;",
    "error": "color: #c62828;",
}

def _status_html(state: str, message: str) -> str:
    style = STATUS_STYLES.get(state, STATUS_STYLES["idle"])
    return f"<b>Status:</b> <span style='{style}'>{message}</span>"

def set_training_status(kind: str, state: str, message: str) -> None:
    widget = training_widgets.get(f"{kind}_status")
    if widget is not None:
        widget.value = _status_html(state, message)

LOSS_COLORS = {
    "running": "#2e7d32",
    "paused": "#ef6c00",
    "completed": "#1565c0",
    "error": "#c62828",
}

LOSS_DEFAULT = "<b>Loss:</b> <span style='color: #6c757d;'>--</span>"

def update_training_loss(kind: str, loss=None, step=None, state="running") -> None:
    widget = training_widgets.get(f"{kind}_loss")
    if widget is None:
        return
    if loss is None:
        widget.value = LOSS_DEFAULT
        return
    color = LOSS_COLORS.get(state, LOSS_COLORS["running"])
    step_text = f" (step {step})" if step is not None else ""
    widget.value = f"<b>Loss:</b> <span style='color: {color};'>{format_loss(loss)}</span>{step_text}"


TASK_NAME_TO_KIND = {
    AUTOENCODER_TASK_KEY: "autoencoder",
    PREDICTOR_TASK_KEY: "predictor",
}



def _get_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.get_event_loop()


def start_training_task(task_name, coroutine, output_widget=None):
    existing = training_tasks.get(task_name)
    if existing and not existing.done():
        if output_widget is not None:
            with output_widget:
                display(Markdown("**Training already in progress. Pause it before starting a new run.**"))
        return None

    loop = _get_event_loop()
    task = loop.create_task(coroutine)
    training_tasks[task_name] = task

    def _cleanup(future):
        training_tasks[task_name] = None
        if output_widget is None:
            return
        if future.cancelled():
            return
        exc = future.exception()
        if exc:
            with output_widget:
                display(Markdown(f"**Training error:** {exc}"))
            kind = TASK_NAME_TO_KIND.get(task_name)
            if kind:
                set_training_status(kind, "error", f"Error: {exc}")
                update_training_loss(kind, None, state="error")

    task.add_done_callback(_cleanup)
    return task


def on_pause_autoencoder(_):
    output = training_widgets["autoencoder_training_output"]
    task = training_tasks.get(AUTOENCODER_TASK_KEY)
    if task is None or task.done():
        with output:
            display(Markdown("**No autoencoder training is currently running.**"))
        set_training_status("autoencoder", "idle", "Idle")
        return

    training_control["autoencoder_paused"] = True
    set_training_status("autoencoder", "pausing", "Pause requested. Waiting for current step to finish...")
    with output:
        display(Markdown("**Pause requested. Waiting for current step to finish...**"))


def on_resume_autoencoder(_):
    output = training_widgets["autoencoder_training_output"]
    resume_data = training_control.get("autoencoder_resume_data")
    if resume_data is None:
        with output:
            display(Markdown("**No paused training to resume.**"))
        set_training_status("autoencoder", "idle", "Idle")
        return

    task = training_tasks.get(AUTOENCODER_TASK_KEY)
    if task and not task.done():
        with output:
            display(Markdown("**Autoencoder training already running. Pause it before resuming.**"))
        return

    training_control["autoencoder_paused"] = False
    set_training_status("autoencoder", "running", "Resuming training...")
    start_training_task(
        AUTOENCODER_TASK_KEY,
        autoencoder_resume_training(resume_data),
        output,
    )


def on_pause_predictor(_):
    output = training_widgets["predictor_training_output"]
    task = training_tasks.get(PREDICTOR_TASK_KEY)
    if task is None or task.done():
        with output:
            display(Markdown("**No predictor training is currently running.**"))
        set_training_status("predictor", "idle", "Idle")
        return

    training_control["predictor_paused"] = True
    set_training_status("predictor", "pausing", "Pause requested. Waiting for current step to finish...")
    with output:
        display(Markdown("**Pause requested. Waiting for current step to finish...**"))


def on_resume_predictor(_):
    output = training_widgets["predictor_training_output"]
    resume_data = training_control.get("predictor_resume_data")
    if resume_data is None:
        with output:
            display(Markdown("**No paused training to resume.**"))
        set_training_status("predictor", "idle", "Idle")
        return

    task = training_tasks.get(PREDICTOR_TASK_KEY)
    if task and not task.done():
        with output:
            display(Markdown("**Predictor training already running. Pause it before resuming.**"))
        return

    training_control["predictor_paused"] = False
    set_training_status("predictor", "running", "Resuming training...")
    start_training_task(
        PREDICTOR_TASK_KEY,
        predictor_resume_training(resume_data),
        output,
    )


# Create pause/resume buttons
pause_autoencoder_button = widgets.Button(description="Pause", button_style="warning", icon="pause")
resume_autoencoder_button = widgets.Button(description="Resume", button_style="info", icon="play")
pause_predictor_button = widgets.Button(description="Pause", button_style="warning", icon="pause")
resume_predictor_button = widgets.Button(description="Resume", button_style="info", icon="play")

# Connect pause/resume handlers
pause_autoencoder_button.on_click(on_pause_autoencoder)
resume_autoencoder_button.on_click(on_resume_autoencoder)
pause_predictor_button.on_click(on_pause_predictor)
resume_predictor_button.on_click(on_resume_predictor)

autoencoder_status = widgets.HTML(value=_status_html("idle", "Idle"))
autoencoder_loss = widgets.HTML(value=LOSS_DEFAULT)
predictor_status = widgets.HTML(value=_status_html("idle", "Idle"))
predictor_loss = widgets.HTML(value=LOSS_DEFAULT)

# Add to training_widgets
training_widgets.update({
    "pause_autoencoder_button": pause_autoencoder_button,
    "resume_autoencoder_button": resume_autoencoder_button,
    "pause_predictor_button": pause_predictor_button,
    "resume_predictor_button": resume_predictor_button,
    "autoencoder_status": autoencoder_status,
    "autoencoder_loss": autoencoder_loss,
    "predictor_status": predictor_status,
    "predictor_loss": predictor_loss,
})


# In[7]:


# Predictor Training Section using AdaptiveWorldModel
import asyncio
import time

PREDICTOR_TASK_KEY = "predictor"


def _prepare_predictor_training():
    """Validate prerequisites and gather tensors for predictor training."""
    output = training_widgets["predictor_training_output"]
    autoencoder = adaptive_world_model.autoencoder
    predictor = adaptive_world_model.predictors[0] if adaptive_world_model.predictors else None
    if autoencoder is None or predictor is None:
        with output:
            output.clear_output()
            display(Markdown("Load both autoencoder and predictor checkpoints first."))
        set_training_status("predictor", "error", "Load the autoencoder and predictor checkpoints first.")
        update_training_loss("predictor", None)
        return None

    frame_slider = session_widgets.get("frame_slider")
    if frame_slider is None:
        with output:
            output.clear_output()
            display(Markdown("Load a session to select frames."))
        set_training_status("predictor", "error", "Load a session to select frames before training.")
        update_training_loss("predictor", None)
        return None

    observations = session_state.get("observations", [])
    if not observations:
        with output:
            output.clear_output()
            display(Markdown("No session loaded. Load a recording before training."))
        set_training_status("predictor", "error", "No session loaded.")
        update_training_loss("predictor", None)
        return None

    target_idx = frame_slider.value
    history_slider_widget = session_widgets.get("history_slider")
    desired_history = history_slider_widget.value if history_slider_widget else 3

    selected_obs, action_dicts, error = build_predictor_sequence(session_state, target_idx, desired_history)
    if error:
        with output:
            output.clear_output()
            display(Markdown(f"**Cannot train predictor:** {error}"))
        set_training_status("predictor", "error", f"Cannot train: {error}")
        update_training_loss("predictor", None)
        return None

    if target_idx + 1 >= len(observations):
        with output:
            output.clear_output()
            display(Markdown("**Cannot train predictor:** No next frame available as training target."))
        set_training_status("predictor", "error", "No next frame available for training.")
        update_training_loss("predictor", None)
        return None

    next_obs = observations[target_idx + 1]
    target_tensor = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0).to(device)

    feature_history = []
    for obs in selected_obs:
        tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)
        autoencoder.eval()
        with torch.no_grad():
            encoded = autoencoder.encode(tensor).detach()
        feature_history.append(encoded)

    recorded_future_action, action_source = get_future_action_for_prediction(session_state, target_idx)
    info_message = None
    if recorded_future_action is None:
        info_message = "No recorded action between current and next frame; using empty action."
        recorded_future_action = {}
    elif action_source == "previous":
        info_message = "Using the most recent action prior to the current frame."

    history_actions_with_future = [clone_action(action) for action in action_dicts]
    history_actions_with_future.append(clone_action(recorded_future_action))

    return {
        "output": output,
        "autoencoder": autoencoder,
        "predictor": predictor,
        "target_idx": target_idx,
        "selected_obs": selected_obs,
        "info_message": info_message,
        "feature_history": feature_history,
        "history_actions_with_future": history_actions_with_future,
        "target_tensor": target_tensor,
        "next_obs": next_obs,
    }


def _display_predictor_weight_changes(pre_stats, post_stats):
    if not (pre_stats and post_stats):
        return

    weight_changes = {}
    if "action_embed_mean" in pre_stats and "action_embed_mean" in post_stats:
        weight_changes.update({
            "action_embed_mean_change": abs(post_stats["action_embed_mean"] - pre_stats["action_embed_mean"]),
            "action_embed_std_change": abs(post_stats["action_embed_std"] - pre_stats["action_embed_std"]),
        })
    if "pos_embed_mean" in pre_stats and "pos_embed_mean" in post_stats:
        weight_changes.update({
            "pos_embed_mean_change": abs(post_stats["pos_embed_mean"] - pre_stats["pos_embed_mean"]),
            "pos_embed_std_change": abs(post_stats["pos_embed_std"] - pre_stats["pos_embed_std"]),
        })
    if "first_self_attn_mean" in pre_stats and "first_self_attn_mean" in post_stats:
        weight_changes.update({
            "first_self_attn_mean_change": abs(post_stats["first_self_attn_mean"] - pre_stats["first_self_attn_mean"]),
            "first_self_attn_std_change": abs(post_stats["first_self_attn_std"] - pre_stats["first_self_attn_std"]),
        })
    if "first_linear1_mean" in pre_stats and "first_linear1_mean" in post_stats:
        weight_changes.update({
            "first_linear1_mean_change": abs(post_stats["first_linear1_mean"] - pre_stats["first_linear1_mean"]),
            "first_linear1_std_change": abs(post_stats["first_linear1_std"] - pre_stats["first_linear1_std"]),
        })

    if not weight_changes:
        return

    lines = ["**Weight Changes:**"]
    if "action_embed_mean_change" in weight_changes:
        lines.extend([
            f"- Action Embed Mean: {weight_changes['action_embed_mean_change']:.8f}",
            f"- Action Embed Std: {weight_changes['action_embed_std_change']:.8f}",
        ])
    if "pos_embed_mean_change" in weight_changes:
        lines.extend([
            f"- Position Embed Mean: {weight_changes['pos_embed_mean_change']:.8f}",
            f"- Position Embed Std: {weight_changes['pos_embed_std_change']:.8f}",
        ])
    if "first_self_attn_mean_change" in weight_changes:
        lines.extend([
            f"- First Self-Attn Mean: {weight_changes['first_self_attn_mean_change']:.8f}",
            f"- First Self-Attn Std: {weight_changes['first_self_attn_std_change']:.8f}",
        ])
    if "first_linear1_mean_change" in weight_changes:
        lines.extend([
            f"- First Linear1 Mean: {weight_changes['first_linear1_mean_change']:.8f}",
            f"- First Linear1 Std: {weight_changes['first_linear1_std_change']:.8f}",
        ])

    display(Markdown("\n".join(lines)))


async def predictor_threshold_training(context, threshold, max_steps):
    output = context["output"]
    autoencoder = context["autoencoder"]
    predictor = context["predictor"]
    target_idx = context["target_idx"]
    selected_obs = context["selected_obs"]
    info_message = context["info_message"]
    feature_history = context["feature_history"]
    history_actions_with_future = context["history_actions_with_future"]
    target_tensor = context["target_tensor"]
    next_obs = context["next_obs"]

    training_control["predictor_resume_data"] = None
    losses = []
    start_time = time.time()

    set_training_status("predictor", "running", f"Running to threshold {format_loss(threshold)}")
    update_training_loss("predictor", None)

    with output:
        output.clear_output()
        display(Markdown(
            f"**Training predictor using AdaptiveWorldModel on history ending at frame {target_idx + 1} (step {selected_obs[-1]['step']})**"
        ))
        display(Markdown(f"Target threshold: {format_loss(threshold)}, Max steps: {max_steps}"))
        display(Markdown(f"History length: {len(selected_obs)} frames"))
        if info_message:
            display(Markdown(info_message))

        display(Markdown("### Pre-Training Predictor Network Weights"))
        pre_stats = visualize_predictor_weights(predictor)

        display(Markdown("**Using AdaptiveWorldModel.train_predictor() method with joint training**"))
        display(Markdown("**Tip: Click 'Pause' button to interrupt training at any time.**"))

        progress = tqdm(range(max_steps), desc="Training")
        try:
            for step in progress:
                await asyncio.sleep(0)
                if training_control["predictor_paused"]:
                    training_control["predictor_resume_data"] = {
                        "target_tensor": target_tensor,
                        "feature_history": feature_history,
                        "history_actions_with_future": history_actions_with_future,
                        "threshold": threshold,
                        "max_steps": max_steps,
                        "current_step": step,
                        "losses": losses,
                        "start_time": start_time,
                    }
                    set_training_status("predictor", "paused", f"Paused at step {step} of {max_steps}. Resume available.")
                    if losses:
                        update_training_loss("predictor", losses[-1], len(losses), state="paused")
                    else:
                        update_training_loss("predictor", None)
                    display(Markdown("**Training paused. Use Resume button to continue.**"))
                    return

                try:
                    predicted_features = predictor(feature_history, history_actions_with_future)
                    loss = adaptive_world_model.train_predictor(
                        level=0,
                        current_frame_tensor=target_tensor,
                        predicted_features=predicted_features,
                        history_features=feature_history,
                        history_actions=history_actions_with_future,
                    )
                except Exception as exc:
                    display(Markdown(f"**Training error:** {exc}"))
                    set_training_status("predictor", "error", f"Error: {exc}")
                    training_control["predictor_resume_data"] = None
                    update_training_loss("predictor", None, state="error")
                    return

                losses.append(loss)
                update_training_loss("predictor", loss, len(losses))
                progress.set_postfix({"Loss": format_loss(loss), "Step": f"{step + 1}/{max_steps}"})

                if loss <= threshold:
                    break
        finally:
            progress.close()

        end_time = time.time()
        final_loss = losses[-1] if losses else float("inf")

        display(Markdown(f"**Training completed after {len(losses)} steps in {end_time - start_time:.1f}s**"))
        display(Markdown(f"Final total loss: {format_loss(final_loss)}"))

        if final_loss <= threshold:
            display(Markdown(f"**Target threshold {format_loss(threshold)} achieved!**"))
        else:
            display(Markdown(f"**Target threshold {format_loss(threshold)} not reached after {max_steps} steps.**"))

        display(Markdown("### Post-Training Predictor Network Weights"))
        post_stats = visualize_predictor_weights(predictor)
        _display_predictor_weight_changes(pre_stats, post_stats)

        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses, label="Total Loss")
            ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Target: {format_loss(threshold)}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Predictor Training Progress (AdaptiveWorldModel)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        predictor.eval()
        autoencoder.eval()
        with torch.no_grad():
            predicted_features = predictor(feature_history, history_actions_with_future)
            predicted_frame = decode_features_to_image(autoencoder, predicted_features)

        predicted_img = tensor_to_numpy_image(predicted_frame)
        target_img = tensor_to_numpy_image(target_tensor)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(predicted_img)
        axes[0].set_title(f"Predicted (Loss: {format_loss(final_loss)})")
        axes[0].axis("off")
        axes[1].imshow(target_img)
        axes[1].set_title(f"Actual (step {next_obs['step']})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if losses:
        update_training_loss("predictor", final_loss, len(losses), state="completed")
        set_training_status("predictor", "completed", f"Completed in {len(losses)} steps (final loss {format_loss(final_loss)}).")
    else:
        update_training_loss("predictor", None)
        set_training_status("predictor", "completed", "Completed without updating any steps.")

    training_control["predictor_resume_data"] = None


async def predictor_steps_training(context, num_steps):
    output = context["output"]
    autoencoder = context["autoencoder"]
    predictor = context["predictor"]
    target_idx = context["target_idx"]
    selected_obs = context["selected_obs"]
    info_message = context["info_message"]
    feature_history = context["feature_history"]
    history_actions_with_future = context["history_actions_with_future"]
    target_tensor = context["target_tensor"]
    next_obs = context["next_obs"]

    training_control["predictor_resume_data"] = None
    losses = []
    start_time = time.time()

    set_training_status("predictor", "running", f"Running for {num_steps} steps")
    update_training_loss("predictor", None)

    with output:
        output.clear_output()
        display(Markdown(
            f"**Training predictor using AdaptiveWorldModel on history ending at frame {target_idx + 1} (step {selected_obs[-1]['step']}) for {num_steps} steps**"
        ))
        display(Markdown(f"History length: {len(selected_obs)} frames"))
        if info_message:
            display(Markdown(info_message))

        display(Markdown("### Pre-Training Predictor Network Weights"))
        pre_stats = visualize_predictor_weights(predictor)

        display(Markdown("**Using AdaptiveWorldModel.train_predictor() method with joint training**"))
        display(Markdown("**Tip: Click 'Pause' button to interrupt training at any time.**"))

        progress = tqdm(range(num_steps), desc="Training")
        try:
            for step in progress:
                await asyncio.sleep(0)
                if training_control["predictor_paused"]:
                    set_training_status("predictor", "paused", f"Paused after {step} of {num_steps} steps; restart to continue.")
                    if losses:
                        update_training_loss("predictor", losses[-1], len(losses), state="paused")
                    else:
                        update_training_loss("predictor", None)
                    display(Markdown("**Training paused. Step-based training cannot be resumed.**"))
                    display(Markdown(f"**Completed {step} out of {num_steps} steps before pausing.**"))
                    return

                try:
                    predicted_features = predictor(feature_history, history_actions_with_future)
                    loss = adaptive_world_model.train_predictor(
                        level=0,
                        current_frame_tensor=target_tensor,
                        predicted_features=predicted_features,
                        history_features=feature_history,
                        history_actions=history_actions_with_future,
                    )
                except Exception as exc:
                    display(Markdown(f"**Training error:** {exc}"))
                    set_training_status("predictor", "error", f"Error: {exc}")
                    update_training_loss("predictor", None, state="error")
                    return

                losses.append(loss)
                update_training_loss("predictor", loss, len(losses))
                progress.set_postfix({"Loss": format_loss(loss), "Step": f"{step + 1}/{num_steps}"})
        finally:
            progress.close()

        end_time = time.time()
        final_loss = losses[-1] if losses else float("inf")

        display(Markdown(f"**Training completed in {end_time - start_time:.1f}s**"))
        if losses:
            display(Markdown(f"Initial total loss: {format_loss(losses[0])}, Final total loss: {format_loss(final_loss)}"))
        else:
            display(Markdown("** **No training steps were executed before the run stopped.**"))

        display(Markdown("### Post-Training Predictor Network Weights"))
        post_stats = visualize_predictor_weights(predictor)
        _display_predictor_weight_changes(pre_stats, post_stats)

        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Total Loss")
            ax.set_title("Predictor Training Progress (AdaptiveWorldModel)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        predictor.eval()
        autoencoder.eval()
        with torch.no_grad():
            predicted_features = predictor(feature_history, history_actions_with_future)
            predicted_frame = decode_features_to_image(autoencoder, predicted_features)

        predicted_img = tensor_to_numpy_image(predicted_frame)
        target_img = tensor_to_numpy_image(target_tensor)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(predicted_img)
        axes[0].set_title(f"Predicted (Loss: {format_loss(final_loss)})")
        axes[0].axis("off")
        axes[1].imshow(target_img)
        axes[1].set_title(f"Actual (step {next_obs['step']})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if losses:
        update_training_loss("predictor", final_loss, len(losses), state="completed")
        set_training_status("predictor", "completed", f"Finished {len(losses)} steps (final loss {format_loss(final_loss)}).")
    else:
        update_training_loss("predictor", None)
        set_training_status("predictor", "completed", "Completed without updating any steps.")

    training_control["predictor_resume_data"] = None


async def predictor_resume_training(resume_data):
    output = training_widgets["predictor_training_output"]
    autoencoder = adaptive_world_model.autoencoder
    if autoencoder is None or not adaptive_world_model.predictors:
        with output:
            output.clear_output()
            display(Markdown("**Required models are not loaded. Load checkpoints before resuming.**"))
        set_training_status("predictor", "error", "Cannot resume because checkpoints are not loaded.")
        training_control["predictor_resume_data"] = None
        return

    predictor = adaptive_world_model.predictors[0]

    with output:
        output.clear_output()
        display(Markdown("**Resuming predictor training...**"))

        target_tensor = resume_data["target_tensor"]
        feature_history = resume_data["feature_history"]
        history_actions_with_future = resume_data["history_actions_with_future"]
        threshold = resume_data["threshold"]
        max_steps = resume_data["max_steps"]
        current_step = resume_data["current_step"]
        losses = resume_data["losses"]
        start_time = resume_data["start_time"]

        if losses:
            update_training_loss("predictor", losses[-1], len(losses))
        else:
            update_training_loss("predictor", None)

        set_training_status("predictor", "running", f"Resuming from step {current_step} of {max_steps}.")

        remaining_steps = max_steps - current_step
        if remaining_steps <= 0:
            display(Markdown("**Training already reached the requested number of steps.**"))
            if losses:
                update_training_loss("predictor", losses[-1], len(losses), state="completed")
            else:
                update_training_loss("predictor", None)
            set_training_status("predictor", "completed", "Requested number of steps already reached before resuming.")
            training_control["predictor_resume_data"] = None
            return

        progress = tqdm(range(remaining_steps), desc=f"Resuming from step {current_step}")
        try:
            for step_offset in progress:
                await asyncio.sleep(0)
                if training_control["predictor_paused"]:
                    resume_data.update({
                        "current_step": current_step + step_offset,
                        "losses": losses,
                    })
                    display(Markdown("** **Training paused. Use Resume button to continue.**"))
                    return

                try:
                    predicted_features = predictor(feature_history, history_actions_with_future)
                    loss = adaptive_world_model.train_predictor(
                        level=0,
                        current_frame_tensor=target_tensor,
                        predicted_features=predicted_features,
                        history_features=feature_history,
                        history_actions=history_actions_with_future,
                    )
                except Exception as exc:
                    display(Markdown(f"**Training error:** {exc}"))
                    set_training_status("predictor", "error", f"Error: {exc}")
                    training_control["predictor_resume_data"] = None
                    update_training_loss("predictor", None, state="error")
                    return

                losses.append(loss)
                update_training_loss("predictor", loss, len(losses))
                progress.set_postfix({"Loss": format_loss(loss)})

                if loss <= threshold:
                    break
        finally:
            progress.close()

        training_control["predictor_resume_data"] = None

        end_time = time.time()
        final_loss = losses[-1] if losses else float("inf")

        display(Markdown(f"**Training completed after {len(losses)} total steps in {end_time - start_time:.1f}s**"))
        display(Markdown(f"Final total loss: {format_loss(final_loss)}"))

        if final_loss <= threshold:
            display(Markdown(f"**Target threshold {format_loss(threshold)} achieved!**"))
        else:
            display(Markdown(f"**Target threshold {format_loss(threshold)} not reached after {max_steps} steps.**"))

        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses, label="Total Loss")
            ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Target: {format_loss(threshold)}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Predictor Training Progress (Resumed)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        if losses:
            update_training_loss("predictor", final_loss, len(losses), state="completed")
            set_training_status("predictor", "completed", f"Completed with final loss {format_loss(final_loss)} after {len(losses)} steps.")
        else:
            update_training_loss("predictor", None)
            set_training_status("predictor", "completed", "Completed without updating any steps.")


def on_train_predictor_threshold(_):
    context = _prepare_predictor_training()
    if context is None:
        return

    training_control["predictor_paused"] = False
    training_control["predictor_resume_data"] = None

    threshold = training_widgets["predictor_threshold"].value
    max_steps = training_widgets["predictor_max_steps"].value

    set_training_status("predictor", "running", f"Running to threshold {format_loss(threshold)}")

    start_training_task(
        PREDICTOR_TASK_KEY,
        predictor_threshold_training(context, threshold, max_steps),
        context["output"],
    )


def on_train_predictor_steps(_):
    context = _prepare_predictor_training()
    if context is None:
        return

    training_control["predictor_paused"] = False
    training_control["predictor_resume_data"] = None

    num_steps = training_widgets["predictor_steps"].value

    set_training_status("predictor", "running", f"Running for {num_steps} steps")

    start_training_task(
        PREDICTOR_TASK_KEY,
        predictor_steps_training(context, num_steps),
        context["output"],
    )


# Create predictor training widgets (same as before)
predictor_threshold = widgets.FloatText(value=0.0005, description="Threshold", step=0.0001, style={'description_width': '100px'})
predictor_max_steps = widgets.IntText(value=1000, description="Max Steps", style={'description_width': '100px'})
predictor_steps = widgets.IntText(value=100, description="Steps", style={'description_width': '100px'})

train_predictor_threshold_button = widgets.Button(description="Train to Threshold", button_style="danger", icon="target")
train_predictor_steps_button = widgets.Button(description="Train N Steps", button_style="danger", icon="forward")
predictor_training_output = widgets.Output()

training_widgets.update({
    "predictor_threshold": predictor_threshold,
    "predictor_max_steps": predictor_max_steps,
    "predictor_steps": predictor_steps,
    "predictor_training_output": predictor_training_output,
})

train_predictor_threshold_button.on_click(on_train_predictor_threshold)
train_predictor_steps_button.on_click(on_train_predictor_steps)


# In[8]:


# Main Session Explorer Interface
session_state = {}
session_widgets = {}

# Model saving functionality
def on_save_autoencoder(_):
    """Save the current autoencoder model"""
    autoencoder = adaptive_world_model.autoencoder
    save_path = session_widgets["autoencoder_save_path"].value

    with session_widgets["save_output"]:
        session_widgets["save_output"].clear_output()

        if autoencoder is None:
            display(Markdown("❌ **No autoencoder model loaded to save**"))
            return

        if not save_path.strip():
            display(Markdown("❌ **Please specify a save path for the autoencoder**"))
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save model using the same format as AdaptiveWorldModel
            checkpoint = {
                'model_state_dict': autoencoder.state_dict(),
                'step': adaptive_world_model.step if hasattr(adaptive_world_model, 'step') else 0,
            }
            torch.save(checkpoint, save_path)

            display(Markdown(f"✅ **Autoencoder saved successfully to:** `{save_path}`"))

        except Exception as e:
            display(Markdown(f"❌ **Error saving autoencoder:** {str(e)}"))

def on_save_predictor(_):
    """Save the current predictor model"""
    predictor = adaptive_world_model.predictors[0] if adaptive_world_model.predictors else None
    save_path = session_widgets["predictor_save_path"].value

    with session_widgets["save_output"]:
        session_widgets["save_output"].clear_output()

        if predictor is None:
            display(Markdown("❌ **No predictor model loaded to save**"))
            return

        if not save_path.strip():
            display(Markdown("❌ **Please specify a save path for the predictor**"))
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save model using the same format as AdaptiveWorldModel
            checkpoint = {
                'model_state_dict': predictor.state_dict(),
                'level': getattr(predictor, 'level', 0),
                'step': adaptive_world_model.step if hasattr(adaptive_world_model, 'step') else 0,
            }
            torch.save(checkpoint, save_path)

            display(Markdown(f"✅ **Predictor saved successfully to:** `{save_path}`"))

        except Exception as e:
            display(Markdown(f"❌ **Error saving predictor:** {str(e)}"))

def on_save_both(_):
    """Save both autoencoder and predictor models"""
    autoencoder = adaptive_world_model.autoencoder
    predictor = adaptive_world_model.predictors[0] if adaptive_world_model.predictors else None

    with session_widgets["save_output"]:
        session_widgets["save_output"].clear_output()

        if autoencoder is None and predictor is None:
            display(Markdown("❌ **No models loaded to save**"))
            return

        saved_models = []
        errors = []

        # Save autoencoder if loaded and path specified
        if autoencoder is not None:
            autoencoder_path = session_widgets["autoencoder_save_path"].value
            if autoencoder_path.strip():
                try:
                    os.makedirs(os.path.dirname(autoencoder_path), exist_ok=True)
                    checkpoint = {
                        'model_state_dict': autoencoder.state_dict(),
                        'step': adaptive_world_model.step if hasattr(adaptive_world_model, 'step') else 0,
                    }
                    torch.save(checkpoint, autoencoder_path)
                    saved_models.append(f"Autoencoder → `{autoencoder_path}`")
                except Exception as e:
                    errors.append(f"Autoencoder: {str(e)}")
            else:
                errors.append("Autoencoder: No save path specified")

        # Save predictor if loaded and path specified
        if predictor is not None:
            predictor_path = session_widgets["predictor_save_path"].value
            if predictor_path.strip():
                try:
                    os.makedirs(os.path.dirname(predictor_path), exist_ok=True)
                    checkpoint = {
                        'model_state_dict': predictor.state_dict(),
                        'level': getattr(predictor, 'level', 0),
                        'step': adaptive_world_model.step if hasattr(adaptive_world_model, 'step') else 0,
                    }
                    torch.save(checkpoint, predictor_path)
                    saved_models.append(f"Predictor → `{predictor_path}`")
                except Exception as e:
                    errors.append(f"Predictor: {str(e)}")
            else:
                errors.append("Predictor: No save path specified")

        # Display results
        if saved_models:
            display(Markdown("✅ **Successfully saved:**"))
            for model in saved_models:
                display(Markdown(f"- {model}"))

        if errors:
            display(Markdown("❌ **Errors occurred:**"))
            for error in errors:
                display(Markdown(f"- {error}"))

def on_load_session_change(change):
    selected_session = change["new"]
    if not selected_session:
        return
    session_dir = os.path.join(SESSIONS_BASE_DIR, selected_session)
    if not os.path.exists(session_dir):
        return

    # Load session data
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)
    metadata = load_session_metadata(session_dir)

    # Update session state
    session_state.clear()
    session_state.update({
        "session_dir": session_dir,
        "session_name": selected_session,
        "events": events,
        "observations": observations,
        "actions": actions,
        "metadata": metadata,
    })

    # Update widgets
    frame_slider = session_widgets.get("frame_slider")
    frame_input = session_widgets.get("frame_input")
    history_slider = session_widgets.get("history_slider")

    if frame_slider:
        frame_slider.max = max(0, len(observations) - 1)
        frame_slider.value = min(frame_slider.value, frame_slider.max)
        frame_slider.description = f"Frame (0-{frame_slider.max})"

    if frame_input:
        frame_input.min = 0
        frame_input.max = max(0, len(observations) - 1)
        target_value = frame_slider.value if frame_slider else frame_input.value
        frame_input.value = max(frame_input.min, min(frame_input.max, target_value))

    if history_slider:
        history_slider.max = min(10, len(observations))
        history_slider.value = min(history_slider.value, history_slider.max)

    with session_widgets["output"]:
        session_widgets["output"].clear_output()
        display(Markdown(f"**Loaded session:** {selected_session}"))
        display(Markdown(f"**Observations:** {len(observations)}"))
        display(Markdown(f"**Actions:** {len(actions)}"))
        if metadata:
            display(Markdown(f"**Metadata:** {len(metadata)} keys"))

def render_frame(idx):
    observations = session_state.get("observations", [])
    if not observations or idx < 0 or idx >= len(observations):
        return

    observation = observations[idx]
    frame_tensor = get_frame_tensor(session_state["session_dir"], observation["frame_path"])
    frame_image = tensor_to_numpy_image(frame_tensor.unsqueeze(0))

    with session_widgets["output"]:
        session_widgets["output"].clear_output()

        display(Markdown(f"**Frame {idx+1}/{len(observations)}** (step {observation['step']})"))
        display(Markdown(f"**Timestamp:** {format_timestamp(observation.get('timestamp'))}"))

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(frame_image)
        ax.set_title(f"Observation {idx+1} (step {observation['step']})")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def on_frame_change(change):
    idx = change["new"]
    frame_input = session_widgets.get("frame_input")
    if frame_input is not None and frame_input.value != idx:
        frame_input.value = idx
    render_frame(idx)


def on_frame_input_change(change):
    frame_slider = session_widgets.get("frame_slider")
    frame_input = session_widgets.get("frame_input")
    if frame_slider is None or frame_input is None:
        return

    new_value = change["new"]
    if new_value is None:
        return

    idx = int(new_value)
    idx = max(frame_slider.min, min(frame_slider.max, idx))

    if frame_input.value != idx:
        frame_input.value = idx

    if frame_slider.value != idx:
        frame_slider.value = idx
    else:
        render_frame(idx)

def on_load_models(_):
    # Load models into AdaptiveWorldModel
    autoencoder_path = session_widgets["autoencoder_path"].value
    predictor_path = session_widgets["predictor_path"].value

    with session_widgets["model_output"]:
        session_widgets["model_output"].clear_output()

        try:
            # Load autoencoder if specified and exists
            if autoencoder_path and os.path.exists(autoencoder_path):
                adaptive_world_model.autoencoder = load_autoencoder_model(autoencoder_path, device)
                display(Markdown(f"✅ **Loaded autoencoder** from {autoencoder_path}"))
            elif autoencoder_path:
                display(Markdown(f"❌ **Autoencoder file not found:** {autoencoder_path}"))
            else:
                display(Markdown("⚠️ **No autoencoder path specified**"))

            # Load predictor if specified and exists
            if predictor_path and os.path.exists(predictor_path):
                predictor = load_predictor_model(predictor_path, device)
                adaptive_world_model.predictors = [predictor]
                display(Markdown(f"✅ **Loaded predictor** from {predictor_path}"))
            elif predictor_path:
                display(Markdown(f"❌ **Predictor file not found:** {predictor_path}"))
            else:
                display(Markdown("⚠️ **No predictor path specified**"))

        except Exception as e:
            display(Markdown(f"❌ **Error loading models:** {str(e)}"))

def on_run_autoencoder(_):
    autoencoder = adaptive_world_model.autoencoder
    frame_slider = session_widgets.get("frame_slider")

    with session_widgets["autoencoder_output"]:
        session_widgets["autoencoder_output"].clear_output()

        if autoencoder is None:
            display(Markdown("Load the autoencoder checkpoint first."))
            return
        if frame_slider is None:
            display(Markdown("Load a session to select frames."))
            return

        idx = frame_slider.value
        observation = session_state.get("observations", [])[idx]
        frame_tensor = get_frame_tensor(session_state["session_dir"], observation["frame_path"]).unsqueeze(0).to(device)

        autoencoder.eval()
        with torch.no_grad():
            reconstructed = autoencoder.reconstruct(frame_tensor)
            loss = torch.nn.functional.mse_loss(reconstructed, frame_tensor).item()

        original_img = tensor_to_numpy_image(frame_tensor)
        reconstructed_img = tensor_to_numpy_image(reconstructed)

        display(Markdown(f"**Autoencoder Inference on Frame {idx+1} (step {observation['step']})**"))
        display(Markdown(f"**Reconstruction Loss:** {loss:.6f}"))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(reconstructed_img)
        axes[1].set_title(f"Reconstructed (Loss: {loss:.6f})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

        # Show autoencoder weight visualization
        display(Markdown("### Autoencoder Network Weight Visualization"))
        autoencoder_weight_stats = visualize_autoencoder_weights(autoencoder)

        # Display autoencoder weight statistics
        if autoencoder_weight_stats:
            stats_text = f"""
**Autoencoder Weight Statistics:**
- Patch Embed: Mean={autoencoder_weight_stats['patch_embed_mean']:.6f}, Std={autoencoder_weight_stats['patch_embed_std']:.6f}
- CLS Token: Mean={autoencoder_weight_stats['cls_token_mean']:.6f}, Std={autoencoder_weight_stats['cls_token_std']:.6f}
- Position Embed: Mean={autoencoder_weight_stats['pos_embed_mean']:.6f}, Std={autoencoder_weight_stats['pos_embed_std']:.6f}
            """
            display(Markdown(stats_text))

def on_run_predictor(_):
    autoencoder = adaptive_world_model.autoencoder
    predictor = adaptive_world_model.predictors[0] if adaptive_world_model.predictors else None
    frame_slider = session_widgets.get("frame_slider")

    with session_widgets["predictor_output"]:
        session_widgets["predictor_output"].clear_output()

        if autoencoder is None or predictor is None:
            display(Markdown("Load both autoencoder and predictor checkpoints first."))
            return
        if frame_slider is None:
            display(Markdown("Load a session to select frames."))
            return

        target_idx = frame_slider.value
        history_slider_widget = session_widgets.get("history_slider")
        desired_history = history_slider_widget.value if history_slider_widget else 3

        selected_obs, action_dicts, error = build_predictor_sequence(session_state, target_idx, desired_history)
        if error:
            display(Markdown(f"**Cannot run predictor:** {error}"))
            return

        # Get features for history frames
        feature_history = []
        for obs in selected_obs:
            tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)
            autoencoder.eval()
            with torch.no_grad():
                encoded = autoencoder.encode(tensor).detach()
            feature_history.append(encoded)

        display(Markdown(f"**Predictor Inference on History ending at Frame {target_idx+1} (step {selected_obs[-1]['step']})**"))
        display(Markdown(f"**History length:** {len(selected_obs)} frames"))

        # Show history sequence
        history_images = []
        for obs in selected_obs:
            tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"])
            history_images.append(tensor_to_numpy_image(tensor.unsqueeze(0)))

        # Visualize history
        fig, axes = plt.subplots(1, len(history_images), figsize=(3 * len(history_images), 3))
        if len(history_images) == 1:
            axes = [axes]
        for i, img in enumerate(history_images):
            axes[i].imshow(img)
            axes[i].set_title(f"Frame {selected_obs[i]['observation_index']+1}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

        # Get recorded action between current and next frame if available
        recorded_action, action_source = get_future_action_for_prediction(session_state, target_idx)
        if recorded_action is None:
            display(Markdown("*No recorded action between current and next frame; using empty action.*"))
            recorded_action = {}
        elif action_source == "previous":
            display(Markdown("*Using the most recent action prior to the current frame.*"))

        # Generate predictions for all possible actions
        action_space = get_action_space(session_state)
        all_predictions = []

        if action_space:
            for action in action_space:
                history_actions_with_future = [clone_action(a) for a in action_dicts]
                history_actions_with_future.append(clone_action(action))

                predictor.eval()
                with torch.no_grad():
                    predicted_features = predictor(feature_history, history_actions_with_future)
                    predicted_frame = decode_features_to_image(autoencoder, predicted_features)

                predicted_img = tensor_to_numpy_image(predicted_frame)

                # Calculate MSE if next frame exists
                mse_loss = None
                if target_idx + 1 < len(session_state.get("observations", [])):
                    next_obs = session_state["observations"][target_idx + 1]
                    next_tensor = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        mse_loss = torch.nn.functional.mse_loss(predicted_frame, next_tensor).item()

                all_predictions.append({
                    "action": action,
                    "image": predicted_img,
                    "label": format_action_label(action),
                    "mse": mse_loss
                })

        display(Markdown("### Predictions for All Actions"))

        if all_predictions:
            cols = min(4, len(all_predictions))
            rows = math.ceil(len(all_predictions) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
            axes = np.array(axes).reshape(rows, cols)
            for idx, prediction in enumerate(all_predictions):
                ax = axes[idx // cols][idx % cols]
                ax.imshow(prediction["image"])
                title = prediction["label"]
                if actions_equal(prediction["action"], recorded_action):
                    title += " (recorded)"
                if prediction["mse"] is not None:
                    title += f" MSE: {prediction['mse']:.6f}"
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            for idx in range(len(all_predictions), rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            plt.tight_layout()
            plt.show()
        else:
            display(Markdown("No actions available to visualize predictions."))

        # Show predictor weight visualization
        display(Markdown("### Predictor Network Weight Visualization"))
        predictor_weight_stats = visualize_predictor_weights(predictor)

        # Display predictor weight statistics
        if predictor_weight_stats:
            stats_lines = ["**Predictor Weight Statistics:**"]
            if 'action_embed_mean' in predictor_weight_stats:
                stats_lines.append(f"- Action Embed: Mean={predictor_weight_stats['action_embed_mean']:.6f}, Std={predictor_weight_stats['action_embed_std']:.6f}")
            if 'pos_embed_mean' in predictor_weight_stats:
                stats_lines.append(f"- Position Embed: Mean={predictor_weight_stats['pos_embed_mean']:.6f}, Std={predictor_weight_stats['pos_embed_std']:.6f}")
            if 'first_self_attn_mean' in predictor_weight_stats:
                stats_lines.append(f"- First Self-Attn: Mean={predictor_weight_stats['first_self_attn_mean']:.6f}, Std={predictor_weight_stats['first_self_attn_std']:.6f}")
            if 'first_linear1_mean' in predictor_weight_stats:
                stats_lines.append(f"- First Linear1: Mean={predictor_weight_stats['first_linear1_mean']:.6f}, Std={predictor_weight_stats['first_linear1_std']:.6f}")
            if 'output_head_mean' in predictor_weight_stats:
                stats_lines.append(f"- Output Head: Mean={predictor_weight_stats['output_head_mean']:.6f}, Std={predictor_weight_stats['output_head_std']:.6f}")
            display(Markdown("\n".join(stats_lines)))

# Create model saving widgets
autoencoder_save_path = widgets.Text(
    value=os.path.join(config.DEFAULT_CHECKPOINT_DIR, "autoencoder_trained.pth"),
    description="Save to:",
    style={'description_width': '100px'}
)
predictor_save_path = widgets.Text(
    value=os.path.join(config.DEFAULT_CHECKPOINT_DIR, "predictor_0_trained.pth"),
    description="Save to:",
    style={'description_width': '100px'}
)

save_autoencoder_button = widgets.Button(description="Save Autoencoder", button_style="primary", icon="save")
save_predictor_button = widgets.Button(description="Save Predictor", button_style="primary", icon="save")
save_both_button = widgets.Button(description="Save Both Models", button_style="success", icon="save")

save_output = widgets.Output()

# Create session selection widgets
session_dirs = list_session_dirs(SESSIONS_BASE_DIR)
load_session_dropdown = widgets.Dropdown(
    options=[""] + session_dirs,
    value="",
    description="Session:",
    style={'description_width': '100px'}
)

frame_slider = widgets.IntSlider(value=0, min=0, max=0, description="Frame (0-0)", style={'description_width': '100px'})
frame_input = widgets.BoundedIntText(value=0, min=0, max=0, description="Go to:", layout=widgets.Layout(width="140px"), style={'description_width': '60px'})
history_slider = widgets.IntSlider(value=3, min=2, max=10, description="History", style={'description_width': '100px'})

# Create model loading widgets
autoencoder_path = widgets.Text(value=DEFAULT_AUTOENCODER_PATH, description="Autoencoder:", style={'description_width': '100px'})
predictor_path = widgets.Text(value=DEFAULT_PREDICTOR_PATH, description="Predictor:", style={'description_width': '100px'})
load_models_button = widgets.Button(description="Load Models", button_style="primary", icon="download")

# Create inference buttons
run_autoencoder_button = widgets.Button(description="Run Autoencoder", button_style="success", icon="play")
run_predictor_button = widgets.Button(description="Run Predictor", button_style="info", icon="play")

# Create output widgets
session_output = widgets.Output()
model_output = widgets.Output()
autoencoder_output = widgets.Output()
predictor_output = widgets.Output()

# Store widgets in global dict
session_widgets.update({
    "load_session_dropdown": load_session_dropdown,
    "frame_slider": frame_slider,
    "frame_input": frame_input,
    "history_slider": history_slider,
    "autoencoder_path": autoencoder_path,
    "predictor_path": predictor_path,
    "autoencoder_save_path": autoencoder_save_path,
    "predictor_save_path": predictor_save_path,
    "save_output": save_output,
    "output": session_output,
    "model_output": model_output,
    "autoencoder_output": autoencoder_output,
    "predictor_output": predictor_output,
})

# Connect event handlers
load_session_dropdown.observe(on_load_session_change, names="value")
frame_slider.observe(on_frame_change, names="value")
frame_input.observe(on_frame_input_change, names="value")
load_models_button.on_click(on_load_models)
run_autoencoder_button.on_click(on_run_autoencoder)
run_predictor_button.on_click(on_run_predictor)
save_autoencoder_button.on_click(on_save_autoencoder)
save_predictor_button.on_click(on_save_predictor)
save_both_button.on_click(on_save_both)

# Display interface
display(Markdown("## Session Explorer Interface"))

display(Markdown("### Session Selection"))
display(widgets.HBox([load_session_dropdown]))
display(widgets.HBox([frame_slider, frame_input, history_slider]))
display(session_output)

display(Markdown("### Model Loading"))
display(widgets.VBox([autoencoder_path, predictor_path]))
display(load_models_button)
display(model_output)

display(Markdown("### Model Saving"))
display(widgets.VBox([autoencoder_save_path, predictor_save_path]))
display(widgets.HBox([save_autoencoder_button, save_predictor_button, save_both_button]))
display(save_output)

display(Markdown("### Inference"))
display(widgets.HBox([run_autoencoder_button, run_predictor_button]))

display(Markdown("### Autoencoder Results"))
display(autoencoder_output)

display(Markdown("### Predictor Results"))
display(predictor_output)

display(Markdown("### Training Sections"))
display(Markdown("#### Autoencoder Training"))
display(widgets.HBox([training_widgets["autoencoder_threshold"], training_widgets["autoencoder_max_steps"]]))
display(widgets.HBox([training_widgets["autoencoder_steps"]]))
display(widgets.HBox([train_autoencoder_threshold_button, train_autoencoder_steps_button]))
display(widgets.HBox([training_widgets["pause_autoencoder_button"], training_widgets["resume_autoencoder_button"]]))
display(training_widgets["autoencoder_status"])
display(training_widgets["autoencoder_loss"])
display(training_widgets["autoencoder_training_output"])

display(Markdown("#### Predictor Training"))
display(widgets.HBox([training_widgets["predictor_threshold"], training_widgets["predictor_max_steps"]]))
display(widgets.HBox([training_widgets["predictor_steps"]]))
display(widgets.HBox([train_predictor_threshold_button, train_predictor_steps_button]))
display(widgets.HBox([training_widgets["pause_predictor_button"], training_widgets["resume_predictor_button"]]))
display(training_widgets["predictor_status"])
display(training_widgets["predictor_loss"])
display(training_widgets["predictor_training_output"])


# In[ ]:





# In[ ]:





# In[ ]:




