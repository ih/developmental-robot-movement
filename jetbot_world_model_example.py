#!/usr/bin/env python3
"""
Example script showing how to integrate JetBot with AdaptiveWorldModel
using the RobotInterface abstraction.
"""

from jetbot_interface import JetBotInterface
from adaptive_world_model import AdaptiveWorldModel
import logging
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress debug messages from 3rd party packages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('rpyc').setLevel(logging.WARNING)

def main():
    # Configuration
    JETBOT_IP = '192.168.68.51'  # Replace with your JetBot's IP address
    CHECKPOINT_DIR = "jetbot_checkpoints"  # Directory to save learning progress
    
    # Create JetBot interface
    logger.info("Connecting to JetBot...")
    jetbot = JetBotInterface(JETBOT_IP)
    
    # Create world model with JetBot interface (with wandb logging and checkpoints)
    logger.info("Initializing AdaptiveWorldModel...")
    world_model = AdaptiveWorldModel(jetbot, interactive=False, wandb_project="jetbot-developmental-movement", checkpoint_dir=CHECKPOINT_DIR)
    
    try:
        logger.info("Starting world model main loop...")
        logger.info("Press Ctrl+C to stop")
        world_model.main_loop()
        
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    
    finally:
        logger.info("Saving final checkpoint...")
        world_model.save_checkpoint()
        logger.info("Cleaning up...")
        jetbot.cleanup()
        # Clean up wandb run
        if world_model.wandb_enabled:
            wandb.finish()

if __name__ == "__main__":
    main()