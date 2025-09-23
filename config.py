import torchvision.transforms as transforms

# Image transformation pipeline
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Adaptive World Model Parameters
class AdaptiveWorldModelConfig:
    # Core parameters
    LOOKAHEAD = 1
    MAX_LOOKAHEAD_MARGIN = 5
    PREDICTION_HISTORY_SIZE = 10
    UNCERTAINTY_THRESHOLD = 0.7
    RECONSTRUCTION_THRESHOLD = 0.0005
    PREDICTION_THRESHOLD = 0.0005  # Threshold for prediction errors (slightly higher than reconstruction)

    # Training parameters
    AUTOENCODER_LR = 1e-4
    PREDICTOR_LR = 1e-4
    MASK_RATIO_MIN = 0.3  # Minimum mask ratio for randomized masking
    MASK_RATIO_MAX = 0.85  # Maximum mask ratio for randomized masking

    # Prediction loss weights
    PRED_PATCH_W = 1.0  # Weight for patch-space reconstruction loss
    PRED_LATENT_W = 0.1  # Weight for latent-space prediction loss

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
RECORDING_MODE = True  # Set to False for online mode without recording
REPLAY_SESSION_DIR = f"{AUX_DIR}/sessions/session_20250921_142133"
RECORDING_BASE_DIR = f"{AUX_DIR}/sessions"  # Base directory for new recordings

# Recording Configuration
RECORDING_SHARD_SIZE = 1000  # Number of steps per shard before rotating
RECORDING_MAX_DISK_GB = 10.0  # Maximum disk space in GB for all recordings (older sessions deleted when exceeded)
RECORDING_SESSION_NAME = None  # Auto-generate if None (timestamp-based)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = f"{AUX_DIR}/checkpoints"

# Interactive mode setting
INTERACTIVE_MODE = False  # Set to True to enable interactive action selection

