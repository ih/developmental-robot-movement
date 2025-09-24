#!/usr/bin/env python
# coding: utf-8

# # Session Viewer
# This notebook lets you explore recorded robot sessions, scrub through frames, and run the autoencoder and predictor models on stored observations.
# 

# ## How to Use
# 1. Pick a session and load it.
# 2. Use the playback controls to scrub through frames.
# 3. (Optional) Load model checkpoints, then run the Autoencoder and Predictor sections using the current frame selection.
# 

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


# In[2]:


SESSIONS_BASE_DIR = config.RECORDING_BASE_DIR
DEFAULT_AUTOENCODER_PATH = os.path.join(config.DEFAULT_CHECKPOINT_DIR, "autoencoder.pth")
DEFAULT_PREDICTOR_PATH = os.path.join(config.DEFAULT_CHECKPOINT_DIR, "predictor_0.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


# In[4]:


# Interactive controls and callbacks
session_state = {
    "session_name": None,
    "session_dir": None,
    "metadata": {},
    "events": [],
    "observations": [],
    "actions": [],
    "autoencoder": None,
    "predictor": None,
    "feature_cache": {},
    "action_space": [],
}

session_widgets = {}

def reset_feature_cache():
    session_state["feature_cache"] = {}

def on_refresh_sessions(_=None):
    options = list_session_dirs(SESSIONS_BASE_DIR)
    current = session_widgets["session_dropdown"].value if "session_dropdown" in session_widgets else None
    session_widgets["session_dropdown"].options = options
    if not options:
        session_widgets["session_dropdown"].value = None
    elif current in options:
        session_widgets["session_dropdown"].value = current
    else:
        session_widgets["session_dropdown"].value = options[-1]

def on_load_session(_):
    dropdown = session_widgets["session_dropdown"]
    session_name = dropdown.value
    if not session_name:
        return
    session_dir = os.path.join(SESSIONS_BASE_DIR, session_name)
    metadata = load_session_metadata(session_dir)
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)

    session_state.update({
        "session_name": session_name,
        "session_dir": session_dir,
        "metadata": metadata,
        "events": events,
        "observations": observations,
        "actions": actions,
    })
    session_state["action_space"] = get_action_space(session_state)
    reset_feature_cache()
    tensor_cache.clear()
    load_frame_bytes.cache_clear()

    with session_widgets["session_area"]:
        session_widgets["session_area"].clear_output()
        if not observations:
            display(Markdown(f"**{session_name}** has no observation frames."))
            return
        details = [
            f"**Session:** {session_name}",
            f"**Total events:** {len(events)}",
            f"**Observations:** {len(observations)}",
            f"**Actions:** {len(actions)}",
        ]
        if metadata:
            start_time = metadata.get("start_time")
            if start_time:
                details.append(f"**Start:** {start_time}")
            robot_type = metadata.get("robot_type")
            if robot_type:
                details.append(f"**Robot:** {robot_type}")
        display(Markdown("<br>".join(details)))

        frame_slider = widgets.IntSlider(value=0, min=0, max=len(observations) - 1, description="Frame", continuous_update=False)
        play_widget = widgets.Play(interval=100, value=0, min=0, max=len(observations) - 1, step=1, description="Play")
        widgets.jslink((play_widget, "value"), (frame_slider, "value"))

        frame_image = widgets.Image(format="jpg")
        frame_image.layout.width = "448px"
        frame_info = widgets.HTML()
        history_preview = widgets.Output()

        session_widgets["frame_slider"] = frame_slider
        session_widgets["play_widget"] = play_widget
        session_widgets["frame_image"] = frame_image
        session_widgets["frame_info"] = frame_info
        session_widgets["history_preview"] = history_preview

        def update_history_preview(idx):
            if "history_preview" not in session_widgets:
                return
            history_slider_widget = session_widgets.get("history_slider")
            requested = history_slider_widget.value if history_slider_widget else 3
            requested = max(1, requested)
            requested = min(requested, idx + 1)
            start = max(0, idx - requested + 1)
            obs_slice = observations[start: idx + 1]
            events_local = session_state.get("events", [])
            display_items = []
            for offset, obs in enumerate(obs_slice):
                frame_bytes = load_frame_bytes(obs["full_path"])
                border_color = "#4caf50" if (start + offset) == idx else "#cccccc"
                image = widgets.Image(value=frame_bytes, format="jpg", layout=widgets.Layout(width="160px", height="120px", border=f"2px solid {border_color}"))
                label_text = f"Step {obs['step']}"
                if (start + offset) == idx:
                    label_text += " (current)"
                label = widgets.HTML(value=f"<div style='text-align:center; font-size:10px'>{label_text}</div>")
                display_items.append(widgets.VBox([image, label]))
                if offset < len(obs_slice) - 1:
                    next_obs = obs_slice[offset + 1]
                    actions_between = [events_local[i] for i in range(obs["event_index"] + 1, next_obs["event_index"]) if events_local[i].get("type") == "action"]
                    if actions_between:
                        action_text = "; ".join(format_action_label(act.get("data", {})) for act in actions_between)
                    else:
                        action_text = "No action"
                    action_label = widgets.HTML(value=f"<div style='font-size:10px; padding:0 6px;'>Action: {action_text}</div>", layout=widgets.Layout(height="120px", display="flex", align_items="center", justify_content="center"))
                    display_items.append(action_label)
            session_widgets["history_preview"].clear_output()
            with session_widgets["history_preview"]:
                if display_items:
                    layout = widgets.Layout(display="flex", flex_flow="row", align_items="center")
                    display(widgets.HBox(display_items, layout=layout))
                else:
                    display(Markdown("History preview unavailable for this frame."))

        session_widgets["update_history_preview"] = update_history_preview

        def update_frame(change):
            idx_local = change["new"] if isinstance(change, dict) else change
            observation = observations[idx_local]
            frame_image.value = load_frame_bytes(observation["full_path"])
            frame_info.value = f"<b>Observation {idx_local + 1} / {len(observations)}</b><br>Step: {observation['step']}<br>Timestamp: {format_timestamp(observation['timestamp'])}"
            update_history_preview(idx_local)

        frame_slider.observe(update_frame, names="value")
        update_frame({"new": frame_slider.value})

        display(widgets.VBox([
            widgets.HBox([play_widget, frame_slider]),
            frame_image,
            frame_info,
            widgets.HTML("<b>History preview</b>"),
            history_preview,
        ]))

    session_widgets["model_status"].value = ""
    session_widgets["autoencoder_output"].clear_output()
    session_widgets["predictor_output"].clear_output()

def on_load_models(_):
    messages = []
    auto_path = session_widgets["autoencoder_path"].value.strip()
    predictor_path = session_widgets["predictor_path"].value.strip()

    if auto_path:
        if os.path.exists(auto_path):
            try:
                session_state["autoencoder"] = load_autoencoder_model(auto_path, device)
                reset_feature_cache()
                messages.append(f"Autoencoder loaded from `{auto_path}`.")
            except Exception as exc:
                session_state["autoencoder"] = None
                messages.append(f"<span style='color:red'>Failed to load autoencoder: {exc}</span>")
        else:
            session_state["autoencoder"] = None
            messages.append(f"<span style='color:red'>Autoencoder path not found: {auto_path}</span>")
    else:
        session_state["autoencoder"] = None
        messages.append("Autoencoder path is empty; skipping load.")

    if predictor_path:
        if os.path.exists(predictor_path):
            try:
                session_state["predictor"] = load_predictor_model(predictor_path, device)
                messages.append(f"Predictor loaded from `{predictor_path}`.")
            except Exception as exc:
                session_state["predictor"] = None
                messages.append(f"<span style='color:red'>Failed to load predictor: {exc}</span>")
        else:
            session_state["predictor"] = None
            messages.append(f"<span style='color:red'>Predictor path not found: {predictor_path}</span>")
    else:
        session_state["predictor"] = None
        messages.append("Predictor path is empty; skipping load.")

    session_widgets["model_status"].value = "<br>".join(messages)

def on_run_autoencoder(_):
    autoencoder = session_state.get("autoencoder")
    if autoencoder is None:
        with session_widgets["autoencoder_output"]:
            session_widgets["autoencoder_output"].clear_output()
            display(Markdown("Load the autoencoder checkpoint first."))
        return
    frame_slider = session_widgets.get("frame_slider")
    if frame_slider is None:
        with session_widgets["autoencoder_output"]:
            session_widgets["autoencoder_output"].clear_output()
            display(Markdown("Load a session to select frames."))
        return

    idx = frame_slider.value
    observation = session_state.get("observations", [])[idx]
    frame_tensor = get_frame_tensor(session_state["session_dir"], observation["frame_path"]).unsqueeze(0).to(device)

    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder.reconstruct(frame_tensor)
    mse = torch.nn.functional.mse_loss(reconstructed, frame_tensor).item()

    original_img = tensor_to_numpy_image(frame_tensor)
    reconstructed_img = tensor_to_numpy_image(reconstructed)

    with session_widgets["autoencoder_output"]:
        session_widgets["autoencoder_output"].clear_output()
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original_img)
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(reconstructed_img)
        axes[1].set_title(f"Reconstruction MSE: {mse:.6f}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

def on_run_predictor(_):
    autoencoder = session_state.get("autoencoder")
    predictor = session_state.get("predictor")
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

        actual_history = len(selected_obs)
        if history_slider_widget and actual_history != history_slider_widget.value:
            history_slider_widget.value = actual_history

        feature_history = []
        for obs in selected_obs:
            cached = session_state["feature_cache"].get(obs["frame_path"])
            if cached is None:
                tensor = get_frame_tensor(session_state["session_dir"], obs["frame_path"]).unsqueeze(0).to(device)
                autoencoder.eval()
                with torch.no_grad():
                    encoded = autoencoder.encode(tensor)
                session_state["feature_cache"][obs["frame_path"]] = encoded.detach().cpu()
                cached = session_state["feature_cache"][obs["frame_path"]]
            feature_history.append(cached)

        feature_history_gpu = [feat.to(device) for feat in feature_history]

        recorded_action = clone_action(action_dicts[-1])

        predictor.eval()
        autoencoder.eval()

        def predict_for_action(action_dict):
            history_actions = [clone_action(act) for act in action_dicts]
            history_actions[-1] = clone_action(action_dict)
            pred_features = predictor(feature_history_gpu, history_actions)
            decoded_candidate = decode_features_to_image(autoencoder, pred_features)
            return decoded_candidate

        next_obs = session_state["observations"][target_idx + 1] if target_idx + 1 < len(session_state["observations"]) else None
        actual_tensor_cpu = None
        actual_tensor_gpu = None
        actual_img = None
        if next_obs is not None:
            actual_tensor_cpu = get_frame_tensor(session_state["session_dir"], next_obs["frame_path"]).unsqueeze(0)
            actual_tensor_gpu = actual_tensor_cpu.to(device)
            actual_img = tensor_to_numpy_image(actual_tensor_cpu)

        all_predictions = []
        with torch.no_grad():
            recorded_pred_tensor = predict_for_action(recorded_action)
            recorded_img = tensor_to_numpy_image(recorded_pred_tensor)
            recorded_mse = None
            if actual_tensor_gpu is not None:
                recorded_mse = torch.nn.functional.mse_loss(recorded_pred_tensor, actual_tensor_gpu).item()
            all_predictions.append({
                "label": "Recorded action",
                "action": clone_action(recorded_action),
                "image": recorded_img,
                "mse": recorded_mse,
            })

            for idx, action in enumerate(session_state.get("action_space", [])):
                if actions_equal(action, recorded_action):
                    continue
                pred_tensor = predict_for_action(action)
                pred_img = tensor_to_numpy_image(pred_tensor)
                mse_value = None
                if actual_tensor_gpu is not None:
                    mse_value = torch.nn.functional.mse_loss(pred_tensor, actual_tensor_gpu).item()
                all_predictions.append({
                    "label": f"{idx + 1}. {format_action_label(action)}",
                    "action": clone_action(action),
                    "image": pred_img,
                    "mse": mse_value,
                })

        history_steps_text = ", ".join(str(obs["step"]) for obs in selected_obs)
        action_lines = []
        for idx, action in enumerate(action_dicts, 1):
            action_lines.append(f"{idx}. {format_action_label(action)}")
        if not action_lines:
            action_lines.append("(No actions in window)")

        display(Markdown(f"**History steps:** {history_steps_text}"))
        display(Markdown("**Recorded actions:**<br>" + "<br>".join(action_lines)))

        history_fig, history_axes = plt.subplots(1, len(selected_obs), figsize=(3 * len(selected_obs), 3))
        if isinstance(history_axes, np.ndarray):
            axes_list = history_axes.flatten()
        else:
            axes_list = [history_axes]
        for idx, (obs, ax) in enumerate(zip(selected_obs, axes_list)):
            img = np.array(load_frame_image(obs["full_path"]))
            ax.imshow(img)
            ax.set_title(f"Step {obs['step']}")
            ax.axis("off")
            if idx < len(action_dicts):
                ax.set_xlabel(format_action_label(action_dicts[idx]), fontsize=9)
        plt.tight_layout()
        plt.show()

        if actual_img is not None and all_predictions:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(all_predictions[0]["image"])
            axes[0].set_title("Predicted (recorded action)")
            axes[0].axis("off")
            title = f"Actual next frame (step {next_obs['step']})"
            if all_predictions[0]["mse"] is not None:
                title += f"MSE: {all_predictions[0]['mse']:.6f}"
            axes[1].imshow(actual_img)
            axes[1].set_title(title)
            axes[1].axis("off")
            plt.tight_layout()
            plt.show()
        elif all_predictions:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.imshow(all_predictions[0]["image"])
            ax.set_title("Predicted next frame (recorded action)")
            ax.axis("off")
            plt.tight_layout()
            plt.show()

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
                    title += f"MSE: {prediction['mse']:.6f}"
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            for idx in range(len(all_predictions), rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            plt.tight_layout()
            plt.show()
        else:
            display(Markdown("No actions available to visualize predictions."))

def on_history_slider_change(_):
    if "frame_slider" in session_widgets and "update_history_preview" in session_widgets:
        session_widgets["update_history_preview"](session_widgets["frame_slider"].value)

session_dropdown = widgets.Dropdown(description="Session", layout=widgets.Layout(width="300px"))
session_widgets["session_dropdown"] = session_dropdown

refresh_button = widgets.Button(description="Refresh", icon="refresh")
load_session_button = widgets.Button(description="Load Session", button_style="primary")

session_area = widgets.Output()
session_widgets["session_area"] = session_area

autoencoder_path = widgets.Text(value=DEFAULT_AUTOENCODER_PATH, description="Autoencoder", layout=widgets.Layout(width="520px"))
predictor_path = widgets.Text(value=DEFAULT_PREDICTOR_PATH, description="Predictor", layout=widgets.Layout(width="520px"))
session_widgets["autoencoder_path"] = autoencoder_path
session_widgets["predictor_path"] = predictor_path

model_status = widgets.HTML()
session_widgets["model_status"] = model_status

run_autoencoder_button = widgets.Button(description="Run Autoencoder", button_style="success", icon="play")
autoencoder_output = widgets.Output()
session_widgets["autoencoder_output"] = autoencoder_output

history_slider = widgets.IntSlider(value=3, min=2, max=8, description="History", continuous_update=False)
session_widgets["history_slider"] = history_slider
history_slider.observe(on_history_slider_change, names="value")

run_predictor_button = widgets.Button(description="Run Predictor", button_style="info", icon="forward")
predictor_output = widgets.Output()
session_widgets["predictor_output"] = predictor_output

refresh_button.on_click(on_refresh_sessions)
load_session_button.on_click(on_load_session)
load_models_button = widgets.Button(description="Load Models", button_style="primary", icon="upload")
load_models_button.on_click(on_load_models)
run_autoencoder_button.on_click(on_run_autoencoder)
run_predictor_button.on_click(on_run_predictor)

on_refresh_sessions()

display(widgets.VBox([
    widgets.HBox([session_dropdown, refresh_button, load_session_button]),
    session_area,
    widgets.HTML("<hr><b>Model Checkpoints</b>"),
    autoencoder_path,
    predictor_path,
    load_models_button,
    model_status,
    widgets.HTML("<hr>"),
    widgets.VBox([
        widgets.HTML("<b>Autoencoder Inference</b>"),
        widgets.HTML("Uses the currently selected frame."),
        run_autoencoder_button,
        autoencoder_output,
    ]),
    widgets.HTML("<hr>"),
    widgets.VBox([
        widgets.HTML("<b>Predictor Inference</b>"),
        widgets.HTML("History uses frames leading up to the current selection to predict the next observation."),
        history_slider,
        run_predictor_button,
        predictor_output,
    ]),
]))


# In[ ]:




