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
    MASK_RATIO = 0.75
    
    # Logging and checkpointing
    LOG_INTERVAL = 10  # Log every N steps
    VISUALIZATION_UPLOAD_INTERVAL = 10  # Upload visualizations every N steps
    CHECKPOINT_SAVE_INTERVAL = 10  # Save every N predictor training steps
    DISPLAY_TRAINING_INTERVAL = 10  # Show training visualization every N steps
    
    # History management
    MAX_HISTORY_SIZE = 1000
    CHECKPOINT_HISTORY_LIMIT = 100

