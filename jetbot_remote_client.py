#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rpyc
import logging
import time
import cv2
import numpy as np
import base64
from IPython.display import display, Image  # No need for clear_output here
import ipywidgets as widgets
import os
import csv
import datetime
import torchvision.transforms as transforms
from PIL import Image
import random
import config
from tqdm.auto import tqdm



# --- Setup Logging ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('JetBotClient')

# --- Image Transformation ---
# Transformations *before* saving to disk (for consistency with training)
transform = config.TRANSFORM


class RemoteJetBot:
    def __init__(self, ip_address, port=18861):
        logger.info(f"Connecting to JetBot at {ip_address}:{port}")
        try:
            self.conn = rpyc.connect(
                ip_address,
                port,
                config={
                    'sync_request_timeout': 30,
                    'allow_all_attrs': True
                }
            )
            logger.info("Connected successfully!")
            # Initialize video window
            self.image_widget = widgets.Image(
                format='jpeg',
                width=400,
                height=300,
            )
            display(self.image_widget)
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise

    def get_frame(self):
        """Get a single frame from the camera and display it"""
        try:
            # Get frame from server
            jpg_as_text = self.conn.root.get_camera_frame()
            if jpg_as_text:
                # Decode base64 string directly to bytes
                jpg_bytes = base64.b64decode(jpg_as_text)
                # Update the image widget
                self.image_widget.value = jpg_bytes

                # Convert to NumPy array (for saving)
                npimg = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                return frame  # Return the frame as a NumPy array
            return None

        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def set_motors(self, left_speed, right_speed):
        try:
            logger.debug(f"Sending motor command: left={left_speed}, right={right_speed}")
            result = self.conn.root.set_motors(float(left_speed), float(right_speed))
            logger.debug("Command sent successfully")
            return result
        except Exception as e:
            logger.error(f"Error sending motor command: {str(e)}")
            raise

    def cleanup(self):
        try:
            logger.debug("Cleaning up connection")
            if hasattr(self, 'conn'):
                self.set_motors(0, 0)  # Stop motors
                self.conn.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def generate_random_actions(num_actions, possible_speeds, min_duration, max_duration):
    actions = []
    for _ in range(num_actions):
        speed = random.choice(possible_speeds)
        duration = random.uniform(min_duration, max_duration)  # Use uniform for continuous range
        actions.append((speed, duration))
    return actions



# In[5]:


if __name__ == "__main__":
    # --- Configuration ---
    JETBOT_IP = '192.168.68.51'  # Replace with your Jetbot's IP address

    jetbot = RemoteJetBot(JETBOT_IP)
    
    try:
        print("Starting live feed from JetBot...")
        print("Press Ctrl+C to stop")
        
        # Display live feed
        while True:
            frame = jetbot.get_frame()
            if frame is not None:
                # Convert BGR to RGB for proper display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Display using OpenCV
                cv2.imshow('JetBot Live Feed', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nLive feed stopped by user")
    finally:
        cv2.destroyAllWindows()
        jetbot.cleanup()  # Stop motors and close connection


# In[ ]:




