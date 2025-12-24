import torchvision.transforms as transforms

# Image transformation pipeline
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Shared Autoencoder Training Parameters
MASK_RATIO_MIN = 1   # Minimum mask ratio for randomized masking
MASK_RATIO_MAX = 1  # Maximum mask ratio for randomized masking

# Root auxiliary directory for checkpoints and recordings
AUX_DIR = "saved"

# Robot-specific recording directories
JETBOT_RECORDING_DIR = f"{AUX_DIR}/sessions/jetbot"
TOROIDAL_DOT_RECORDING_DIR = f"{AUX_DIR}/sessions/toroidal_dot"

# Robot-specific checkpoint directories
JETBOT_CHECKPOINT_DIR = f"{AUX_DIR}/checkpoints/jetbot"
TOROIDAL_DOT_CHECKPOINT_DIR = f"{AUX_DIR}/checkpoints/toroidal_dot"
SO101_CHECKPOINT_DIR = f"{AUX_DIR}/checkpoints/so101"

# SO-101 robot-specific recording directory
SO101_RECORDING_DIR = f"{AUX_DIR}/sessions/so101"

# Recording configuration
RECORDING_SHARD_SIZE = 1000           # Steps per recording shard
RECORDING_MAX_DISK_GB = 10.0          # Maximum disk space for recordings (per robot type)

# Toroidal Dot Environment Configuration
class ToroidalDotConfig:
    # Environment parameters
    IMG_SIZE = 224              # Size of square image
    DOT_RADIUS = 5              # Radius of white dot in pixels
    DOT_MOVE_PIXELS = 27        # Horizontal movement per action=1
    DOT_ACTION_DELAY = 0.0      # Delay between actions in seconds

    # Action space for toroidal dot
    ACTION_CHANNELS_DOT = ["action"]
    ACTION_RANGES_DOT = {
        "action": (0, 1)        # Binary action: 0=stay, 1=move right
    }

# Autoencoder Concat Predictor World Model Configuration
class AutoencoderConcatPredictorWorldModelConfig:
    # Canvas construction
    FRAME_SIZE = (224, 224)        # Size of each frame in canvas
    SEPARATOR_WIDTH = 16            # Width of action-colored separator between frames
    CANVAS_HISTORY_SIZE = 3        # Number of frames to keep in history

    # Model architecture
    PATCH_SIZE = 16                # Size of patches for Vision Transformer (WARNING: changing requires retraining)
    BATCH_SIZE = 1                 # Training batch size (1=online learning, >1=mini-batch)

    # Optimizer parameters
    AUTOENCODER_LR = 2e-5          # Learning rate for autoencoder training
    WEIGHT_DECAY = 0.01            # AdamW weight decay
    WARMUP_STEPS = 1000             # Warmup steps for learning rate scheduler
    LR_MIN_RATIO = 0.01            # Minimum LR as ratio of base LR

    # Focal loss parameters (for loss dilution)
    FOCAL_BETA = 5              # Temperature for exponential weighting (try 5-15)
    FOCAL_LOSS_ALPHA = 0.1          # Blend ratio: alpha * plain_mse + (1-alpha) * focal_mse

    # Gradio UI parameters
    GRADIO_UPDATE_INTERVAL = 1      # Update visualization every N iterations during training


# SO-101 Robot Arm Configuration
class SO101Config:
    """Configuration for SO-101 follower arm with single-joint control."""

    # Joint names (standard SO-101 6-DOF arm)
    JOINT_NAMES = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ]

    # Default movement parameters
    DEFAULT_ACTION_DURATION = 0.5    # Duration of move actions in seconds
    DEFAULT_POSITION_DELTA = 0.1     # Position change per discrete action in radians

    # Discrete action space (3 actions per controlled joint)
    ACTION_SPACE = [
        {"action": 0, "duration": 0.0},         # Stay (no movement)
        {"action": 1, "duration": 0.5},         # Move positive for duration
        {"action": 2, "duration": 0.5},         # Move negative for duration
    ]

    # Frame settings for dual-camera stacked view
    # base_0_rgb (224x224) stacked on top of left_wrist_0_rgb (224x224) = 448x224
    FRAME_SIZE = (448, 224)          # (H, W) - vertically stacked cameras
    CAMERAS = ["base_0_rgb", "left_wrist_0_rgb"]

    # Canvas dimensions with stacked frames
    # 3 frames (each 448x224) + 2 separators (16px) = 448 x 720
    CANVAS_HEIGHT = 448
    CANVAS_WIDTH = 720               # 224*3 + 16*2

