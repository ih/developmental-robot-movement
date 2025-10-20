import torchvision.transforms as transforms

# Image transformation pipeline
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Shared Autoencoder Training Parameters
MASK_RATIO_MIN = .3   # Minimum mask ratio for randomized masking
MASK_RATIO_MAX = 1  # Maximum mask ratio for randomized masking

# Autoencoder Latent Predictor World Model Parameters
class AutoencoderLatentPredictorWorldModelConfig:
    # Architecture selection
    AUTOENCODER_TYPE = "vit"  # Options: "vit" (Vision Transformer MAE), "cnn" (Convolutional)
    PREDICTOR_TYPE = "transformer"  # Options: "transformer", "lstm"

    # Core parameters
    LOOKAHEAD = 1
    MAX_LOOKAHEAD_MARGIN = 5
    PREDICTION_HISTORY_SIZE = 3
    UNCERTAINTY_THRESHOLD = 0.7
    RECONSTRUCTION_THRESHOLD = 0.001
    PREDICTION_THRESHOLD = 0.001  # Threshold for prediction errors (slightly higher than reconstruction)

    # Training parameters
    AUTOENCODER_LR = 1e-4
    PREDICTOR_LR = 1.5e-4  # Base learning rate for transformer/latent predictors
    WEIGHT_DECAY = 0.01  # AdamW weight decay (decoupled, excludes bias/LayerNorm)
    WARMUP_STEPS = 600  # Warmup steps (2-5% of total steps, or 1k-5k)
    LR_MIN_RATIO = 1e-2  # η_min = η₀ × LR_MIN_RATIO, or 1e-6 minimum

    # Prediction loss weights
    PRED_PATCH_W = 0.8  # Weight for patch-space reconstruction loss (visual quality anchor)
    PRED_LATENT_W = 0.2  # Weight for latent-space prediction loss (main training signal)
    PRED_ACTION_W = 0  # Weight for action reconstruction loss (disabled for now)

    # Action classification
    ENABLE_ACTION_CLASSIFIER = False  # Enable separate action classification module

    # Logging and checkpointing
    LOG_INTERVAL = 20  # Log every N steps
    VISUALIZATION_UPLOAD_INTERVAL = 100  # Upload visualizations every N steps
    CHECKPOINT_SAVE_INTERVAL = 500  # Save every N predictor training steps
    DISPLAY_TRAINING_INTERVAL = 1  # Show training visualization every N steps
    DISPLAY_INTERVAL = 1  # Update display every N main loop iterations
    
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

# Robot-specific recording directories
JETBOT_RECORDING_DIR = f"{AUX_DIR}/sessions/jetbot"
TOROIDAL_DOT_RECORDING_DIR = f"{AUX_DIR}/sessions/toroidal_dot"

# Legacy - kept for backward compatibility
RECORDING_BASE_DIR = f"{AUX_DIR}/sessions"  # Base directory for new recordings

# Recording Configuration
RECORDING_SHARD_SIZE = 1000  # Number of steps per shard before rotating
RECORDING_MAX_DISK_GB = 10.0  # Maximum disk space in GB for all recordings (older sessions deleted when exceeded)
RECORDING_SESSION_NAME = None  # Auto-generate if None (timestamp-based)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = f"{AUX_DIR}/checkpoints"

# Robot-specific checkpoint directories (separate due to different action spaces)
JETBOT_CHECKPOINT_DIR = f"{DEFAULT_CHECKPOINT_DIR}/jetbot"
TOROIDAL_DOT_CHECKPOINT_DIR = f"{DEFAULT_CHECKPOINT_DIR}/toroidal_dot"

# Interactive mode setting
INTERACTIVE_MODE = False  # Set to True to enable interactive action selection

# Toroidal Dot Environment Configuration
class ToroidalDotConfig:
    # Environment parameters
    IMG_SIZE = 224              # Size of square image
    DOT_RADIUS = 5              # Radius of white dot in pixels
    DOT_MOVE_PIXELS = 27         # Horizontal movement per action=1
    DOT_ACTION_DELAY = 0.0      # Delay between actions in seconds

    # Action space for toroidal dot (use these when running with ToroidalDotRobot)
    ACTION_CHANNELS_DOT = ["action"]
    ACTION_RANGES_DOT = {
        "action": (0, 1)        # Binary action: 0=stay, 1=move right
    }

# Action Normalization and FiLM Configuration
# NOTE: Switch to ToroidalDotConfig.ACTION_CHANNELS_DOT and ACTION_RANGES_DOT when using ToroidalDotRobot
ACTION_CHANNELS = ["motor_left", "motor_right", "duration"]
ACTION_RANGES = {  # min, max for scaling to [-1, 1]
    "motor_left":  (0.0,  0.0),    # left motor fixed at 0
    "motor_right": (0.0,  0.12),   # forward-only speeds used in sessions
    "duration":    (0.2,  0.2),    # step duration in seconds
}
ACTION_EMBED_DIM = 64          # learned action embedding dimension
FILM_HIDDEN_DIM  = 128         # hidden width inside Action MLP
FILM_BLOCK_IDS   = [0, 2]      # apply FiLM in early & mid transformer layers
DELTA_LATENT     = False        # switch on residual prediction

# Autoencoder Concat Predictor World Model Configuration
class AutoencoderConcatPredictorWorldModelConfig:
    # Canvas construction
    FRAME_SIZE = (224, 224)        # Size of each frame in canvas
    SEPARATOR_WIDTH = 8            # Width of action-colored separator between frames
    CANVAS_HISTORY_SIZE = 3        # Number of frames to keep in history

    # Training thresholds
    CANVAS_INPAINTING_THRESHOLD = 0.0016  # Threshold for stopping autoencoder training (loss on masked patches only)

    # Optimizer parameters
    AUTOENCODER_LR = 1e-4          # Learning rate for autoencoder training
    WEIGHT_DECAY = 0.01            # AdamW weight decay
    WARMUP_STEPS = 600             # Warmup steps for learning rate scheduler
    LR_MIN_RATIO = 0.01            # Minimum LR as ratio of base LR


