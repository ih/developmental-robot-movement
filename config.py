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

# Autoencoder Concat Predictor World Model Configuration
class AutoencoderConcatPredictorWorldModelConfig:
    # Canvas construction
    FRAME_SIZE = (224, 224)        # Size of each frame in canvas
    SEPARATOR_WIDTH = 8            # Width of action-colored separator between frames
    CANVAS_HISTORY_SIZE = 3        # Number of frames to keep in history

    # Optimizer parameters
    AUTOENCODER_LR = 1e-4          # Learning rate for autoencoder training
    WEIGHT_DECAY = 0.01            # AdamW weight decay
    WARMUP_STEPS = 600             # Warmup steps for learning rate scheduler
    LR_MIN_RATIO = 0.01            # Minimum LR as ratio of base LR

    # Focal loss parameters (for loss dilution)
    FOCAL_BETA = 5              # Temperature for exponential weighting (try 5-15)
    FOCAL_LOSS_ALPHA = 0.2          # Blend ratio: alpha * plain_mse + (1-alpha) * focal_mse

    # Gradio UI parameters
    GRADIO_UPDATE_INTERVAL = 1      # Update visualization every N iterations during training


