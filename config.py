import torchvision.transforms as transforms

# Image transformation pipeline
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Adaptive World Model Parameters
class AdaptiveWorldModelConfig:
    # Core parameters
    LOOKAHEAD = 1
    MAX_LOOKAHEAD_MARGIN = 5
    PREDICTION_HISTORY_SIZE = 3
    UNCERTAINTY_THRESHOLD = 0.7
    RECONSTRUCTION_THRESHOLD = 0.001
    PREDICTION_THRESHOLD = 0.001  # Threshold for prediction errors (slightly higher than reconstruction)

    # Training parameters
    AUTOENCODER_LR = 1e-4
    PREDICTOR_LR = 1e-4
    MASK_RATIO_MIN = 0.3  # Minimum mask ratio for randomized masking
    MASK_RATIO_MAX = 0.85  # Maximum mask ratio for randomized masking

    # Prediction loss weights
    PRED_PATCH_W = 1  # Weight for patch-space reconstruction loss
    PRED_LATENT_W = .2  # Weight for latent-space prediction loss
    PRED_ACTION_W = 0  # Weight for action reconstruction loss (classifying which action was taken)

    # Logging and checkpointing
    LOG_INTERVAL = 20  # Log every N steps
    VISUALIZATION_UPLOAD_INTERVAL = 100  # Upload visualizations every N steps
    CHECKPOINT_SAVE_INTERVAL = 500  # Save every N predictor training steps
    DISPLAY_TRAINING_INTERVAL = 25  # Show training visualization every N steps
    DISPLAY_INTERVAL = 10  # Update display every N main loop iterations
    
    # History management
    MAX_HISTORY_SIZE = 1000
    CHECKPOINT_HISTORY_LIMIT = 100
    
    # Action execution
    ACTION_DELAY = 0  # Delay in seconds between actions

# Root auxiliary directory for checkpoints and recordings
AUX_DIR = "saved"
# Recording and Replay Parameters
RECORDING_MODE = False  # Set to False for online mode without recording
REPLAY_SESSION_DIR = f"{AUX_DIR}/sessions/session_20250921_142133"
RECORDING_BASE_DIR = f"{AUX_DIR}/sessions"  # Base directory for new recordings

# Recording Configuration
RECORDING_SHARD_SIZE = 1000  # Number of steps per shard before rotating
RECORDING_MAX_DISK_GB = 10.0  # Maximum disk space in GB for all recordings (older sessions deleted when exceeded)
RECORDING_SESSION_NAME = None  # Auto-generate if None (timestamp-based)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = f"{AUX_DIR}/checkpoints"

# Interactive mode setting
INTERACTIVE_MODE = True  # Set to True to enable interactive action selection

# Action Normalization and FiLM Configuration
ACTION_CHANNELS = ["motor_left", "motor_right", "duration"]
ACTION_RANGES = {  # min, max for scaling to [-1, 1]
    "motor_left":  (0.0,  0.0),    # left motor fixed at 0
    "motor_right": (0.0,  0.12),   # forward-only speeds used in sessions
    "duration":    (0.2,  0.2),    # step duration in seconds
}
ACTION_EMBED_DIM = 64          # learned action embedding dimension
FILM_HIDDEN_DIM  = 128         # hidden width inside Action MLP
FILM_BLOCK_IDS   = [0, 2]      # apply FiLM in early & mid transformer layers
DELTA_LATENT     = True        # switch on residual prediction


