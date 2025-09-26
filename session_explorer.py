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
adaptive_world_model = AdaptiveWorldModel(stub_robot, wandb_project=None, checkpoint_dir=config.DEFAULT_CHECKPOINT_DIR, interactive=False)


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
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0.0, 1.0)
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
training_widgets = {}

def on_train_autoencoder_threshold(_):
    """Train autoencoder until threshold is met using AdaptiveWorldModel"""
    autoencoder = adaptive_world_model.autoencoder
    if autoencoder is None:
        with training_widgets["autoencoder_training_output"]:
            training_widgets["autoencoder_training_output"].clear_output()
            display(Markdown("Load the autoencoder checkpoint first."))
        return
    
    frame_slider = session_widgets.get("frame_slider")
    if frame_slider is None:
        with training_widgets["autoencoder_training_output"]:
            training_widgets["autoencoder_training_output"].clear_output()
            display(Markdown("Load a session to select frames."))
        return
    
    # Get training parameters
    threshold = training_widgets["autoencoder_threshold"].value
    max_steps = training_widgets["autoencoder_max_steps"].value
    
    # Setup for training
    idx = frame_slider.value
    observation = session_state.get("observations", [])[idx]
    frame_tensor = get_frame_tensor(session_state["session_dir"], observation["frame_path"]).unsqueeze(0).to(device)
    
    with training_widgets["autoencoder_training_output"]:
        training_widgets["autoencoder_training_output"].clear_output()
        
        # Show pre-training weights
        display(Markdown(f"**Training autoencoder using AdaptiveWorldModel on frame {idx+1} (step {observation['step']})**"))
        display(Markdown(f"Target threshold: {format_loss(threshold)}, Max steps: {max_steps}"))
        display(Markdown("### Pre-Training Network Weights"))
        pre_stats = visualize_autoencoder_weights(autoencoder)
        
        display(Markdown("**Using AdaptiveWorldModel.train_autoencoder() method with randomized masking**"))
        
        losses = []
        start_time = time.time()
        
        # Create progress bar
        progress = tqdm(range(max_steps), desc="Training")
        
        for step in progress:
            loss = train_autoencoder_step_wrapper(frame_tensor)
            losses.append(loss)
            
            progress.set_postfix({"Loss": format_loss(loss)})
            
            # Check if threshold met
            if loss <= threshold:
                progress.close()
                break
        else:
            progress.close()
        
        end_time = time.time()
        final_loss = losses[-1] if losses else float('inf')
        
        # Display results
        display(Markdown(f"**Training completed after {len(losses)} steps in {end_time-start_time:.1f}s**"))
        display(Markdown(f"Final loss: {format_loss(final_loss)}"))
        
        if final_loss <= threshold:
            display(Markdown(f"✅ **Target threshold {format_loss(threshold)} achieved!**"))
        else:
            display(Markdown(f"⚠️ **Target threshold {format_loss(threshold)} not reached after {max_steps} steps**"))
        
        # Show post-training weights
        display(Markdown("### Post-Training Network Weights"))
        post_stats = visualize_autoencoder_weights(autoencoder)
        
        # Compare weight changes
        if pre_stats and post_stats:
            weight_changes = {
                'patch_embed_mean_change': abs(post_stats['patch_embed_mean'] - pre_stats['patch_embed_mean']),
                'patch_embed_std_change': abs(post_stats['patch_embed_std'] - pre_stats['patch_embed_std']),
                'cls_token_mean_change': abs(post_stats['cls_token_mean'] - pre_stats['cls_token_mean']),
                'cls_token_std_change': abs(post_stats['cls_token_std'] - pre_stats['cls_token_std']),
            }
            changes_text = f"""
**Weight Changes:**
- Patch Embed Mean: {weight_changes['patch_embed_mean_change']:.8f}
- Patch Embed Std: {weight_changes['patch_embed_std_change']:.8f}
- CLS Token Mean: {weight_changes['cls_token_mean_change']:.8f}
- CLS Token Std: {weight_changes['cls_token_std_change']:.8f}
            """
            display(Markdown(changes_text))
        
        # Plot training progress
        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reconstruction Loss")
            ax.set_title("Autoencoder Training Progress (AdaptiveWorldModel)")
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Target: {format_loss(threshold)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Show final reconstruction
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

def on_train_autoencoder_steps(_):
    """Train autoencoder for specified number of steps using AdaptiveWorldModel"""
    autoencoder = adaptive_world_model.autoencoder
    if autoencoder is None:
        with training_widgets["autoencoder_training_output"]:
            training_widgets["autoencoder_training_output"].clear_output()
            display(Markdown("Load the autoencoder checkpoint first."))
        return
    
    frame_slider = session_widgets.get("frame_slider")
    if frame_slider is None:
        with training_widgets["autoencoder_training_output"]:
            training_widgets["autoencoder_training_output"].clear_output()
            display(Markdown("Load a session to select frames."))
        return
    
    # Get training parameters
    num_steps = training_widgets["autoencoder_steps"].value
    
    # Setup for training
    idx = frame_slider.value
    observation = session_state.get("observations", [])[idx]
    frame_tensor = get_frame_tensor(session_state["session_dir"], observation["frame_path"]).unsqueeze(0).to(device)
    
    with training_widgets["autoencoder_training_output"]:
        training_widgets["autoencoder_training_output"].clear_output()
        
        # Show pre-training weights
        display(Markdown(f"**Training autoencoder using AdaptiveWorldModel on frame {idx+1} (step {observation['step']}) for {num_steps} steps**"))
        display(Markdown("### Pre-Training Network Weights"))
        pre_stats = visualize_autoencoder_weights(autoencoder)
        
        display(Markdown("**Using AdaptiveWorldModel.train_autoencoder() method with randomized masking**"))
        
        losses = []
        start_time = time.time()
        
        # Create progress bar
        progress = tqdm(range(num_steps), desc="Training")
        
        for step in progress:
            loss = train_autoencoder_step_wrapper(frame_tensor)
            losses.append(loss)
            progress.set_postfix({"Loss": format_loss(loss)})
        
        progress.close()
        end_time = time.time()
        final_loss = losses[-1] if losses else float('inf')
        
        # Display results
        display(Markdown(f"**Training completed in {end_time-start_time:.1f}s**"))
        display(Markdown(f"Initial loss: {format_loss(losses[0])}, Final loss: {format_loss(final_loss)}"))
        
        # Show post-training weights
        display(Markdown("### Post-Training Network Weights"))
        post_stats = visualize_autoencoder_weights(autoencoder)
        
        # Compare weight changes
        if pre_stats and post_stats:
            weight_changes = {
                'patch_embed_mean_change': abs(post_stats['patch_embed_mean'] - pre_stats['patch_embed_mean']),
                'patch_embed_std_change': abs(post_stats['patch_embed_std'] - pre_stats['patch_embed_std']),
                'cls_token_mean_change': abs(post_stats['cls_token_mean'] - pre_stats['cls_token_mean']),
                'cls_token_std_change': abs(post_stats['cls_token_std'] - pre_stats['cls_token_std']),
            }
            changes_text = f"""
**Weight Changes:**
- Patch Embed Mean: {weight_changes['patch_embed_mean_change']:.8f}
- Patch Embed Std: {weight_changes['patch_embed_std_change']:.8f}
- CLS Token Mean: {weight_changes['cls_token_mean_change']:.8f}
- CLS Token Std: {weight_changes['cls_token_std_change']:.8f}
            """
            display(Markdown(changes_text))
        
        # Plot training progress
        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reconstruction Loss")
            ax.set_title("Autoencoder Training Progress (AdaptiveWorldModel)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Show final reconstruction
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


# Predictor Training Section using AdaptiveWorldModel
def on_train_predictor_threshold(_):
    """Train predictor until threshold is met using AdaptiveWorldModel"""
    autoencoder = adaptive_world_model.autoencoder
    predictor = adaptive_world_model.predictors[0] if adaptive_world_model.predictors else None
    frame_slider = session_widgets.get("frame_slider")
    
    with training_widgets["predictor_training_output"]:
        training_widgets["predictor_training_output"].clear_output()
        
        if autoencoder is None or predictor is None:
            display(Markdown("Load both autoencoder and predictor checkpoints first."))
            return
        if frame_slider is None:
            display(Markdown("Load a session to select frames."))
            return
        
        # Get training parameters
        threshold = training_widgets["predictor_threshold"].value
        max_steps = training_widgets["predictor_max_steps"].value
        
        target_idx = frame_slider.value
        history_slider_widget = session_widgets.get("history_slider")
        desired_history = history_slider_widget.value if history_slider_widget else 3
        
        selected_obs, action_dicts, error = build_predictor_sequence(session_state, target_idx, desired_history)
        if error:
            display(Markdown(f"**Cannot train predictor:** {error}"))
            return
        
        # Check if we have a next frame for training target
        if target_idx + 1 >= len(session_state["observations"]):
            display(Markdown("**Cannot train predictor:** No next frame available as training target."))
            return
        
        # Get target frame tensor
        next_obs = session_state["observations"][target_idx + 1]
        target_tensor = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0).to(device)
        
        # Get feature history and setup context
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
        if info_message:
            display(Markdown(info_message))

        history_actions_with_future = [clone_action(action) for action in action_dicts]
        history_actions_with_future.append(clone_action(recorded_future_action))

        display(Markdown(f"**Training predictor using AdaptiveWorldModel on history ending at frame {target_idx+1} (step {selected_obs[-1]['step']})**"))
        display(Markdown(f"Target threshold: {format_loss(threshold)}, Max steps: {max_steps}"))
        display(Markdown(f"History length: {len(selected_obs)} frames"))
        
        # Show pre-training weights
        display(Markdown("### Pre-Training Predictor Network Weights"))
        pre_stats = visualize_predictor_weights(predictor)
        
        display(Markdown(f"**Using AdaptiveWorldModel.train_predictor() method with joint training**"))
        
        losses = []
        start_time = time.time()
        
        # Create progress bar
        progress = tqdm(range(max_steps), desc="Training")
        
        for step in progress:
            # Use AdaptiveWorldModel's train_predictor method
            try:
                # Set up prediction context for fresh predictions
                predicted_features = predictor(feature_history, history_actions_with_future)
                
                loss = adaptive_world_model.train_predictor(
                    level=0,
                    current_frame_tensor=target_tensor,
                    predicted_features=predicted_features,
                    history_features=feature_history,
                    history_actions=history_actions_with_future
                )
                losses.append(loss)
                
                progress.set_postfix({"Loss": format_loss(loss)})
                
                # Check if threshold met
                if loss <= threshold:
                    progress.close()
                    break
            except Exception as e:
                progress.close()
                display(Markdown(f"**Training error:** {str(e)}"))
                return
        else:
            progress.close()
        
        end_time = time.time()
        final_loss = losses[-1] if losses else float('inf')
        
        # Display results
        display(Markdown(f"**Training completed after {len(losses)} steps in {end_time-start_time:.1f}s**"))
        display(Markdown(f"Final total loss: {format_loss(final_loss)}"))
        
        if final_loss <= threshold:
            display(Markdown(f"✅ **Target threshold {format_loss(threshold)} achieved!**"))
        else:
            display(Markdown(f"⚠️ **Target threshold {format_loss(threshold)} not reached after {max_steps} steps**"))
        
        # Show post-training weights
        display(Markdown("### Post-Training Predictor Network Weights"))
        post_stats = visualize_predictor_weights(predictor)
        
        # Compare weight changes
        if pre_stats and post_stats:
            weight_changes = {}
            if 'action_embed_mean' in pre_stats and 'action_embed_mean' in post_stats:
                weight_changes.update({
                    'action_embed_mean_change': abs(post_stats['action_embed_mean'] - pre_stats['action_embed_mean']),
                    'action_embed_std_change': abs(post_stats['action_embed_std'] - pre_stats['action_embed_std']),
                })
            if 'pos_embed_mean' in pre_stats and 'pos_embed_mean' in post_stats:
                weight_changes.update({
                    'pos_embed_mean_change': abs(post_stats['pos_embed_mean'] - pre_stats['pos_embed_mean']),
                    'pos_embed_std_change': abs(post_stats['pos_embed_std'] - pre_stats['pos_embed_std']),
                })
            if 'first_self_attn_mean' in pre_stats and 'first_self_attn_mean' in post_stats:
                weight_changes.update({
                    'first_self_attn_mean_change': abs(post_stats['first_self_attn_mean'] - pre_stats['first_self_attn_mean']),
                    'first_self_attn_std_change': abs(post_stats['first_self_attn_std'] - pre_stats['first_self_attn_std']),
                })
            if 'first_linear1_mean' in pre_stats and 'first_linear1_mean' in post_stats:
                weight_changes.update({
                    'first_linear1_mean_change': abs(post_stats['first_linear1_mean'] - pre_stats['first_linear1_mean']),
                    'first_linear1_std_change': abs(post_stats['first_linear1_std'] - pre_stats['first_linear1_std']),
                })
            
            if weight_changes:
                changes_lines = ["**Weight Changes:**"]
                if 'action_embed_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- Action Embed Mean: {weight_changes['action_embed_mean_change']:.8f}",
                        f"- Action Embed Std: {weight_changes['action_embed_std_change']:.8f}",
                    ])
                if 'pos_embed_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- Position Embed Mean: {weight_changes['pos_embed_mean_change']:.8f}",
                        f"- Position Embed Std: {weight_changes['pos_embed_std_change']:.8f}",
                    ])
                if 'first_self_attn_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- First Self-Attn Mean: {weight_changes['first_self_attn_mean_change']:.8f}",
                        f"- First Self-Attn Std: {weight_changes['first_self_attn_std_change']:.8f}",
                    ])
                if 'first_linear1_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- First Linear1 Mean: {weight_changes['first_linear1_mean_change']:.8f}",
                        f"- First Linear1 Std: {weight_changes['first_linear1_std_change']:.8f}",
                    ])
                display(Markdown("\n".join(changes_lines)))
        
        # Plot training progress
        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses, label="Total Loss")
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Target: {format_loss(threshold)}')
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Predictor Training Progress (AdaptiveWorldModel)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Show prediction comparison
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

def on_train_predictor_steps(_):
    """Train predictor for specified number of steps using AdaptiveWorldModel"""
    autoencoder = adaptive_world_model.autoencoder
    predictor = adaptive_world_model.predictors[0] if adaptive_world_model.predictors else None
    frame_slider = session_widgets.get("frame_slider")
    
    with training_widgets["predictor_training_output"]:
        training_widgets["predictor_training_output"].clear_output()
        
        if autoencoder is None or predictor is None:
            display(Markdown("Load both autoencoder and predictor checkpoints first."))
            return
        if frame_slider is None:
            display(Markdown("Load a session to select frames."))
            return
        
        # Get training parameters
        num_steps = training_widgets["predictor_steps"].value
        
        target_idx = frame_slider.value
        history_slider_widget = session_widgets.get("history_slider")
        desired_history = history_slider_widget.value if history_slider_widget else 3
        
        selected_obs, action_dicts, error = build_predictor_sequence(session_state, target_idx, desired_history)
        if error:
            display(Markdown(f"**Cannot train predictor:** {error}"))
            return
        
        # Check if we have a next frame for training target
        if target_idx + 1 >= len(session_state["observations"]):
            display(Markdown("**Cannot train predictor:** No next frame available as training target."))
            return
        
        # Get target frame tensor
        next_obs = session_state["observations"][target_idx + 1]
        target_tensor = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0).to(device)
        
        # Get feature history and setup context
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
        if info_message:
            display(Markdown(info_message))

        history_actions_with_future = [clone_action(action) for action in action_dicts]
        history_actions_with_future.append(clone_action(recorded_future_action))

        display(Markdown(f"**Training predictor using AdaptiveWorldModel on history ending at frame {target_idx+1} (step {selected_obs[-1]['step']}) for {num_steps} steps**"))
        display(Markdown(f"History length: {len(selected_obs)} frames"))
        
        # Show pre-training weights
        display(Markdown("### Pre-Training Predictor Network Weights"))
        pre_stats = visualize_predictor_weights(predictor)
        
        display(Markdown(f"**Using AdaptiveWorldModel.train_predictor() method with joint training**"))
        
        losses = []
        start_time = time.time()
        
        # Create progress bar
        progress = tqdm(range(num_steps), desc="Training")
        
        for step in progress:
            # Use AdaptiveWorldModel's train_predictor method
            try:
                # Set up prediction context for fresh predictions
                predicted_features = predictor(feature_history, history_actions_with_future)
                
                loss = adaptive_world_model.train_predictor(
                    level=0,
                    current_frame_tensor=target_tensor,
                    predicted_features=predicted_features,
                    history_features=feature_history,
                    history_actions=history_actions_with_future
                )
                losses.append(loss)
                
                progress.set_postfix({"Loss": format_loss(loss)})
            except Exception as e:
                progress.close()
                display(Markdown(f"**Training error:** {str(e)}"))
                return
        
        progress.close()
        end_time = time.time()
        final_loss = losses[-1] if losses else float('inf')
        
        # Display results
        display(Markdown(f"**Training completed in {end_time-start_time:.1f}s**"))
        display(Markdown(f"Initial total loss: {format_loss(losses[0])}, Final total loss: {format_loss(final_loss)}"))
        
        # Show post-training weights
        display(Markdown("### Post-Training Predictor Network Weights"))
        post_stats = visualize_predictor_weights(predictor)
        
        # Compare weight changes
        if pre_stats and post_stats:
            weight_changes = {}
            if 'action_embed_mean' in pre_stats and 'action_embed_mean' in post_stats:
                weight_changes.update({
                    'action_embed_mean_change': abs(post_stats['action_embed_mean'] - pre_stats['action_embed_mean']),
                    'action_embed_std_change': abs(post_stats['action_embed_std'] - pre_stats['action_embed_std']),
                })
            if 'pos_embed_mean' in pre_stats and 'pos_embed_mean' in post_stats:
                weight_changes.update({
                    'pos_embed_mean_change': abs(post_stats['pos_embed_mean'] - pre_stats['pos_embed_mean']),
                    'pos_embed_std_change': abs(post_stats['pos_embed_std'] - pre_stats['pos_embed_std']),
                })
            if 'first_self_attn_mean' in pre_stats and 'first_self_attn_mean' in post_stats:
                weight_changes.update({
                    'first_self_attn_mean_change': abs(post_stats['first_self_attn_mean'] - pre_stats['first_self_attn_mean']),
                    'first_self_attn_std_change': abs(post_stats['first_self_attn_std'] - pre_stats['first_self_attn_std']),
                })
            if 'first_linear1_mean' in pre_stats and 'first_linear1_mean' in post_stats:
                weight_changes.update({
                    'first_linear1_mean_change': abs(post_stats['first_linear1_mean'] - pre_stats['first_linear1_mean']),
                    'first_linear1_std_change': abs(post_stats['first_linear1_std'] - pre_stats['first_linear1_std']),
                })
            
            if weight_changes:
                changes_lines = ["**Weight Changes:**"]
                if 'action_embed_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- Action Embed Mean: {weight_changes['action_embed_mean_change']:.8f}",
                        f"- Action Embed Std: {weight_changes['action_embed_std_change']:.8f}",
                    ])
                if 'pos_embed_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- Position Embed Mean: {weight_changes['pos_embed_mean_change']:.8f}",
                        f"- Position Embed Std: {weight_changes['pos_embed_std_change']:.8f}",
                    ])
                if 'first_self_attn_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- First Self-Attn Mean: {weight_changes['first_self_attn_mean_change']:.8f}",
                        f"- First Self-Attn Std: {weight_changes['first_self_attn_std_change']:.8f}",
                    ])
                if 'first_linear1_mean_change' in weight_changes:
                    changes_lines.extend([
                        f"- First Linear1 Mean: {weight_changes['first_linear1_mean_change']:.8f}",
                        f"- First Linear1 Std: {weight_changes['first_linear1_std_change']:.8f}",
                    ])
                display(Markdown("\n".join(changes_lines)))
        
        # Plot training progress
        if len(losses) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Total Loss")
            ax.set_title("Predictor Training Progress (AdaptiveWorldModel)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Show prediction comparison
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


# In[7]:


# Main Session Explorer Interface
session_state = {}
session_widgets = {}

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
    history_slider = session_widgets.get("history_slider")

    if frame_slider:
        frame_slider.max = max(0, len(observations) - 1)
        frame_slider.value = min(frame_slider.value, frame_slider.max)
        frame_slider.description = f"Frame (0-{frame_slider.max})"

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

def on_frame_change(change):
    idx = change["new"]
    observations = session_state.get("observations", [])
    if not observations or idx >= len(observations):
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

# Create session selection widgets
session_dirs = list_session_dirs(SESSIONS_BASE_DIR)
load_session_dropdown = widgets.Dropdown(
    options=[""] + session_dirs,
    value="",
    description="Session:",
    style={'description_width': '100px'}
)

frame_slider = widgets.IntSlider(value=0, min=0, max=0, description="Frame (0-0)", style={'description_width': '100px'})
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
    "history_slider": history_slider,
    "autoencoder_path": autoencoder_path,
    "predictor_path": predictor_path,
    "output": session_output,
    "model_output": model_output,
    "autoencoder_output": autoencoder_output,
    "predictor_output": predictor_output,
})

# Connect event handlers
load_session_dropdown.observe(on_load_session_change, names="value")
frame_slider.observe(on_frame_change, names="value")
load_models_button.on_click(on_load_models)
run_autoencoder_button.on_click(on_run_autoencoder)
run_predictor_button.on_click(on_run_predictor)

# Display interface
display(Markdown("## Session Explorer Interface"))

display(Markdown("### Session Selection"))
display(widgets.HBox([load_session_dropdown]))
display(widgets.HBox([frame_slider, history_slider]))
display(session_output)

display(Markdown("### Model Loading"))
display(widgets.VBox([autoencoder_path, predictor_path]))
display(load_models_button)
display(model_output)

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
display(training_widgets["autoencoder_training_output"])

display(Markdown("#### Predictor Training"))
display(widgets.HBox([training_widgets["predictor_threshold"], training_widgets["predictor_max_steps"]]))
display(widgets.HBox([training_widgets["predictor_steps"]]))
display(widgets.HBox([train_predictor_threshold_button, train_predictor_steps_button]))
display(training_widgets["predictor_training_output"])


# In[ ]:




