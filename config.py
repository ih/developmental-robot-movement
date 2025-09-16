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
    RECONSTRUCTION_THRESHOLD = 0.0022
    
    # Training parameters
    AUTOENCODER_LR = 1e-4
    PREDICTOR_LR = 1e-4
    MASK_RATIO_MIN = 0.3  # Minimum mask ratio for randomized masking
    MASK_RATIO_MAX = 0.85  # Maximum mask ratio for randomized masking
    
    # Logging and checkpointing
    LOG_INTERVAL = 100  # Log every N steps
    VISUALIZATION_UPLOAD_INTERVAL = 5000  # Upload visualizations every N steps
    CHECKPOINT_SAVE_INTERVAL = 500  # Save every N predictor training steps
    DISPLAY_TRAINING_INTERVAL = 25  # Show training visualization every N steps
    DISPLAY_INTERVAL = 10  # Update display every N main loop iterations
    
    # History management
    MAX_HISTORY_SIZE = 1000
    CHECKPOINT_HISTORY_LIMIT = 100
    
    # Action execution
    ACTION_DELAY = 0.1  # Delay in seconds between actions

