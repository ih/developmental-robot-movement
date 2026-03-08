import torchvision.transforms as transforms

# Image transformation pipeline
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Shared Autoencoder Training Parameters
MASK_RATIO_MIN = 1   # Eval/inference mask ratio (always full masking)
MASK_RATIO_MAX = 1   # Eval/inference mask ratio (always full masking)
TRAIN_MASK_RATIO_MIN = 1.0  # Training mask ratio minimum (variable masking for better gradients)
TRAIN_MASK_RATIO_MAX = 1.0  # Training mask ratio maximum

# Root auxiliary directory for checkpoints and recordings
AUX_DIR = "saved"

# Training logs directory
TRAINING_LOG_DIR = f"{AUX_DIR}/training_logs"

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
    SEPARATOR_WIDTH = 32            # Width of action-colored separator between frames
    CANVAS_HISTORY_SIZE = 3        # Number of frames to keep in history

    # Model architecture
    MODEL_TYPE = "decoder_only" # "encoder_decoder" (MAE), "decoder_only" (GPT-style), or "dit" (latent diffusion)
    PATCH_SIZE = 16                # Size of patches for Vision Transformer (WARNING: changing requires retraining)
    BATCH_SIZE = 1                 # Training batch size (1=online learning, >1=mini-batch)

    # Model capacity - encoder (encoder-decoder only)
    ENCODER_EMBED_DIM = 512        # Embedding dimension for encoder
    ENCODER_NUM_HEADS = 8          # Number of attention heads for encoder
    ENCODER_DEPTH = 5              # Number of transformer blocks in encoder

    # Model capacity - decoder (encoder-decoder decoder, or decoder-only single stack)
    DECODER_EMBED_DIM = 256        # Embedding dimension for decoder
    DECODER_NUM_HEADS = 8          # Number of attention heads in decoder
    DECODER_DEPTH = 12             # Number of transformer blocks in decoder

    # Optimizer parameters
    AUTOENCODER_LR = 3e-4          # Learning rate for autoencoder training
    WEIGHT_DECAY = 0.01            # AdamW weight decay
    WARMUP_STEPS = 1000             # Warmup steps for learning rate scheduler
    LR_MIN_RATIO = 0.001           # Minimum LR as ratio of base LR

    # Focal loss parameters (for loss dilution)
    FOCAL_BETA = 5             # Temperature for exponential weighting (try 5-15)
    FOCAL_LOSS_ALPHA = 1.0          # Blend ratio: alpha * plain_mse + (1-alpha) * focal_mse (1.0 = pure MSE)

    # Perceptual loss (VGG feature space loss for sharper predictions)
    PERCEPTUAL_LOSS_WEIGHT = 0.01   # Weight for perceptual loss (0.0 = disabled, no VGG loaded)

    # --- VAE/Encoder configuration (used when MODEL_TYPE == "dit") ---
    VAE_TYPE = "pretrained_sd"               # "custom", "pretrained_sd", "pretrained_flux", "dinov2"
    VAE_CHECKPOINT = None             # Path to trained VAE/decoder checkpoint (None = download pretrained)
    VAE_LATENT_CHANNELS = 4           # Latent channels (auto-set for pretrained: SD=4, FLUX=16, DINOv2=768)
    VAE_COMPRESSION_FACTOR = 8        # Spatial downsampling (auto-set for pretrained: SD/FLUX=8, DINOv2=14)
    VAE_MODE = "vae"                  # "vae" (KL divergence) or "rae" (L2 regularization) - custom only
    DINOV2_VARIANT = "vitb14"         # DINOv2 model variant: vits14, vitb14, vitl14, vitg14

    # --- DiT architecture (used when MODEL_TYPE == "dit") ---
    DIT_EMBED_DIM = 256               # Token embedding dimension
    DIT_DEPTH = 12                    # Number of DiT blocks
    DIT_NUM_HEADS = 4                 # Number of attention heads
    DIT_LATENT_PATCH_SIZE = 2         # Patch size in latent space
    DIT_PREDICTION_TYPE = "epsilon"   # "epsilon" (predict noise) or "sample" (predict clean)
    DIT_TRAINING_MODE = "unconditional" # "conditional" (noise on masked patches only) or "unconditional" (noise on all, RePaint inference)

    # --- Diffusion schedule (used when MODEL_TYPE == "dit") ---
    DIT_NUM_TRAIN_TIMESTEPS = 1000    # Number of diffusion timesteps during training
    DIT_NUM_INFERENCE_STEPS = 50      # Number of DDIM denoising steps at inference
    DIT_BETA_START = 0.0001           # Start of beta schedule
    DIT_BETA_END = 0.02               # End of beta schedule
    DIT_BETA_SCHEDULE = "linear"      # "linear" or "cosine"

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
    CANVAS_WIDTH = 736               # 224*3 + 32*2

