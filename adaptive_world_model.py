# Adaptive World Model with Hierarchical Action Learning
# Based on Day 6 outline with future generalizations from the document

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class AdaptiveWorldModel:
    def __init__(self, interactive=False):
        # Core components
        self.image_encoder = ImageEncoder()
        self.image_decoder = ImageDecoder()
        self.predictors = [ActionConditionedPredictor(level=0)]  # Start with one predictor
        self.action_encoders = []  # For future hierarchical actions
        self.action_decoders = []
        
        # Parameters
        self.lookahead = 1  # Start with 1-step lookahead
        self.max_lookahead_margin = 5  # Buffer before creating new level
        self.prediction_history_size = 10  # Fixed lookback window
        self.uncertainty_threshold = 0.7
        self.interactive = interactive
        
        # Matplotlib figure for persistent display
        self.fig = None
        self.axes = None
        
        # Action space (initially low-level) - cross product of motor actions
        motor_values = [-1, 0, +1]
        self.base_actions = []
        for m1 in motor_values:
            for m2 in motor_values:
                self.base_actions.append({'motor_1': m1, 'motor_2': m2})
        
        # History buffers
        self.frame_features_history = []
        self.action_history = []
        self.prediction_buffer = []
        
    def main_loop(self):
        while True:
            # Step 1: Capture and encode current frame
            current_frame = capture_image()
            current_features = self.image_encoder(current_frame)
            
            # Step 2: Validate image encoding quality
            decoded_frame = self.image_decoder(current_features)
            reconstruction_loss = calculate_loss(decoded_frame, current_frame)
            
            threshold = 0.5  # Define threshold locally
            if reconstruction_loss > threshold:
                # Train encoder/decoder if reconstruction is poor
                self.train_image_autoencoder(current_frame)
            
            # Step 3: Check past predictions accuracy (if any)
            prediction_errors = []
            if self.prediction_buffer:
                prediction_errors = self.evaluate_predictions(current_features)
                
                if all(error < threshold for error in prediction_errors):
                    # All predictions accurate - increase lookahead
                    self.lookahead += 1
                    
                    # Check if we need a new hierarchical level
                    if self.lookahead > self.get_max_predictor_lookahead() + self.max_lookahead_margin:
                        self.create_new_hierarchy_level()
                else:
                    # Train predictors that had errors
                    for level, error in enumerate(prediction_errors):
                        if error > threshold:
                            self.train_predictor(level, current_features)
                    
                    # Adjust lookahead based on accuracy horizon
                    self.lookahead = self.find_accurate_prediction_horizon()
            
            # Step 4: Select action based on uncertainty maximization
            best_action, all_action_predictions = self.select_action_by_uncertainty(current_features)
            
            # Interactive mode: show information and get user input
            if self.interactive:
                action_to_execute = self.interactive_prompt(
                    current_frame, decoded_frame, reconstruction_loss, 
                    prediction_errors, best_action, all_action_predictions
                )
            else:
                action_to_execute = best_action
            
            # Step 5: Take action and make predictions
            execute_action(action_to_execute)
            self.make_predictions(current_features, action_to_execute)
            
            # Step 6: Update history buffers
            self.frame_features_history.append(current_features)
            self.action_history.append(action_to_execute)
            self.maintain_history_window()
    
    def interactive_prompt(self, current_frame, decoded_frame, reconstruction_loss, 
                          prediction_errors, best_action, all_action_predictions):
        """Interactive mode: display info and get user input"""
        print("\n" + "="*60, flush=True)
        print("INTERACTIVE MODE", flush=True)
        print("="*60, flush=True)
        
        # Print metrics first
        print(f"Reconstruction Loss: {reconstruction_loss:.4f}", flush=True)
        if prediction_errors:
            print(f"Prediction Errors: {[f'{e:.4f}' for e in prediction_errors]}", flush=True)
        else:
            print("Prediction Errors: None (no previous predictions)", flush=True)
        print(f"Proposed Best Action: {best_action}", flush=True)
        print(f"Current Lookahead: {self.lookahead}", flush=True)
        print(f"History Size: {len(self.frame_features_history)}", flush=True)
        
        # Display frames (non-blocking)
        try:
            self.display_frames(current_frame, decoded_frame, all_action_predictions)
        except Exception as e:
            print(f"Warning: Could not display frames: {e}", flush=True)
        
        # Get user input
        while True:
            user_input = input("\nOptions:\n"
                             "1. Continue with proposed action (press Enter)\n"
                             "2. Replace action (type new action as dict, e.g., {'motor_1': 1})\n"
                             "3. Stop (type 'stop')\n"
                             "Choice: ").strip()
            
            if user_input == "":
                return best_action
            elif user_input.lower() == "stop":
                raise KeyboardInterrupt("Stopped by user")
            else:
                try:
                    # Try to parse as action dict
                    new_action = eval(user_input)
                    if isinstance(new_action, dict):
                        print(f"Using custom action: {new_action}")
                        return new_action
                    else:
                        print("Invalid action format. Please use dict format like {'motor_1': 1}")
                except:
                    print("Invalid input. Please try again.")
    
    def display_frames(self, current_frame, decoded_frame, all_action_predictions):
        """Display current frame, decoded frame, and predictions for all actions"""
        # Calculate grid layout: current + decoded + all action predictions
        total_predictions = sum(len(pred_data['predictions']) for pred_data in all_action_predictions)
        num_cols = 2 + total_predictions  # current + decoded + all predictions
        
        # Create figure on first call or if layout changed
        if self.fig is None or len(self.axes) != num_cols:
            if self.fig is not None:
                plt.close(self.fig)
            
            self.fig, self.axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 4))
            
            if num_cols == 1:
                self.axes = [self.axes]
            
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
        
        # Clear previous content
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        col_idx = 0
        
        # Current frame
        self.axes[col_idx].imshow(current_frame)
        self.axes[col_idx].set_title("Current Frame", fontsize=10)
        col_idx += 1
        
        # Decoded frame
        self.axes[col_idx].imshow(decoded_frame)
        self.axes[col_idx].set_title("Decoded Frame", fontsize=10)
        col_idx += 1
        
        # Predictions for each action
        for action_data in all_action_predictions:
            action = action_data['action']
            uncertainty = action_data['uncertainty']
            predictions = action_data['predictions']
            
            for level, pred_data in enumerate(predictions):
                if col_idx < len(self.axes):
                    pred_frame = pred_data['frame']
                    predictor_level = pred_data['level']
                    
                    self.axes[col_idx].imshow(pred_frame)
                    
                    # Multi-line title with action, level, and uncertainty
                    title = f"Action: {action}\nLevel {predictor_level}\nUncertainty: {uncertainty:.3f}"
                    self.axes[col_idx].set_title(title, fontsize=8)
                    
                    # Add border color based on uncertainty (higher = more red)
                    border_color = plt.cm.Reds(min(uncertainty, 1.0))
                    for spine in self.axes[col_idx].spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2)
                    
                    col_idx += 1
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to ensure rendering
    
    def select_action_by_uncertainty(self, current_features):
        """Choose action with most uncertain outcome (highest entropy)"""
        max_uncertainty = -float('inf')
        best_action = None
        all_action_predictions = []
        
        # Get relevant history for prediction
        history_features, history_actions = self.get_prediction_context()
        
        for action in self.get_available_actions():
            # Predict next states for this action
            predictions = []
            prediction_frames = []
            
            for predictor in self.predictors:
                next_state = predictor.predict(
                    history_features, 
                    history_actions + [action],
                    self.lookahead
                )
                predictions.append(next_state)
                
                # Generate frame for this prediction
                pred_frame = self.image_decoder(next_state)
                prediction_frames.append({
                    'frame': pred_frame,
                    'level': predictor.level,
                    'features': next_state
                })
            
            # Calculate uncertainty (entropy) over predictions
            uncertainty = calculate_entropy(predictions)
            
            # Store all prediction data for this action
            all_action_predictions.append({
                'action': action,
                'uncertainty': uncertainty,
                'predictions': prediction_frames
            })
            
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                best_action = action
        
        return best_action, all_action_predictions
    
    def train_predictor(self, level, current_features):
        """Train predictor at specified level"""
        predictor = self.predictors[level]
        
        # Get appropriate history window
        history_features, history_actions = self.get_prediction_context()
        
        # Calculate loss between prediction and actual
        predicted = self.prediction_buffer[level]['prediction']
        actual = current_features
        loss = calculate_loss(predicted, actual)
        
        # Backpropagate through predictor and encoders
        predictor.update_weights()
        
        # Also update image encoder/decoder if needed
        high_threshold = 1.0
        if loss > high_threshold:
            self.image_encoder.update_weights()
            self.image_decoder.update_weights()
    
    def train_image_autoencoder(self, ground_truth_frame):
        """Train using masked autoencoding"""
        # Create multiple masked versions
        masked_frames = create_masked_versions(ground_truth_frame)
        
        for masked_frame in masked_frames:
            # Encode masked version
            features = self.image_encoder(masked_frame)
            
            # Decode and calculate loss
            reconstructed = self.image_decoder(features)
            loss = calculate_loss(reconstructed, ground_truth_frame)
            
            # Single update step
            self.image_encoder.update_weights()
            self.image_decoder.update_weights()
    
    def create_new_hierarchy_level(self):
        """Create new predictor for higher-level actions"""
        # Create action encoder for current action sequences
        action_encoder = ActionEncoder(
            sequence_length=self.lookahead,
            input_actions=self.get_current_level_actions()
        )
        action_decoder = ActionDecoder()
        
        # Train on recent action sequences
        recent_sequences = self.extract_action_sequences(self.lookahead)
        action_encoder.train(recent_sequences)
        
        # Create new predictor for encoded actions
        new_predictor = ActionConditionedPredictor(
            level=len(self.predictors),
            action_space=action_encoder.latent_space
        )
        
        # Add to hierarchy
        self.predictors.append(new_predictor)
        self.action_encoders.append(action_encoder)
        self.action_decoders.append(action_decoder)
        
        print(f"Created hierarchy level {len(self.predictors)}")
    
    def get_available_actions(self):
        """Get actions at appropriate abstraction level"""
        if not self.action_encoders:
            return self.base_actions
        
        # Could return mix of low and high level actions
        # or just highest level based on current lookahead
        current_level = min(self.lookahead // self.max_lookahead_margin, 
                          len(self.action_encoders))
        
        if current_level == 0:
            return self.base_actions
        else:
            return self.action_encoders[current_level-1].get_abstract_actions()
    
    def get_prediction_context(self):
        """Get appropriate history window for prediction"""
        # Need enough history for lookahead + context
        context_size = self.prediction_history_size + self.lookahead
        
        # Get last context_size frames and actions
        start_idx = max(0, len(self.frame_features_history) - context_size)
        
        features = self.frame_features_history[start_idx:]
        actions = self.action_history[start_idx:]
        
        return features, actions
    
    def find_accurate_prediction_horizon(self):
        """Find how far ahead predictions remain accurate"""
        threshold = 0.5
        for horizon in range(1, self.lookahead + 1):
            if horizon-1 < len(self.prediction_buffer) and self.prediction_buffer[horizon-1].get('error', 0) > threshold:
                return max(1, horizon - 1)
        return self.lookahead
    
    def maintain_history_window(self):
        """Keep history buffers at reasonable size"""
        max_history = 1000  # Configurable
        
        if len(self.frame_features_history) > max_history:
            self.frame_features_history = self.frame_features_history[-max_history:]
            self.action_history = self.action_history[-max_history:]
    
    def make_predictions(self, current_features, action):
        """Generate predictions for future timesteps"""
        self.prediction_buffer = []
        
        history_features, history_actions = self.get_prediction_context()
        history_actions.append(action)
        
        for predictor in self.predictors:
            prediction = predictor.predict(
                history_features,
                history_actions, 
                self.lookahead
            )
            self.prediction_buffer.append({
                'prediction': prediction,
                'level': predictor.level,
                'timestamp': current_time()
            })
    
    def evaluate_predictions(self, actual_features):
        """Compare past predictions with actual outcome"""
        errors = []
        
        for pred_data in self.prediction_buffer:
            prediction = pred_data['prediction']
            error = calculate_loss(prediction, actual_features)
            errors.append(error)
            
        return errors
    
    def get_max_predictor_lookahead(self):
        """Get maximum lookahead capability of current predictors"""
        return max([p.max_lookahead for p in self.predictors])
    
    def get_current_level_actions(self):
        """Get actions from current abstraction level"""
        return self.base_actions
    
    def extract_action_sequences(self, length):
        """Extract action sequences of given length from history"""
        sequences = []
        for i in range(len(self.action_history) - length + 1):
            sequences.append(self.action_history[i:i+length])
        return sequences


# Supporting functions and classes (stub implementations)

class ImageEncoder:
    """Transformer-based encoder using masked token training"""
    def __call__(self, image): 
        # Return dummy features vector
        return np.random.randn(512)
    
    def update_weights(self): 
        # Stub weight update
        pass

class ImageDecoder:
    """Reconstructs images from encoded features"""
    def __call__(self, features): 
        # Return dummy image (224x224x3)
        return np.random.rand(224, 224, 3)
    
    def update_weights(self): 
        # Stub weight update
        pass

class ActionConditionedPredictor:
    """Predicts future states given past states and actions"""
    def __init__(self, level, action_space=None): 
        self.level = level
        self.action_space = action_space
        self.max_lookahead = 10  # Default max lookahead
    
    def predict(self, feature_history, action_history, lookahead): 
        # Return dummy prediction
        return np.random.randn(512)
    
    def update_weights(self): 
        # Stub weight update
        pass

class ActionEncoder:
    """Encodes sequences of actions into abstract actions"""
    def __init__(self, sequence_length, input_actions): 
        self.sequence_length = sequence_length
        self.input_actions = input_actions
        self.latent_space = ['abstract_action_1', 'abstract_action_2']
    
    def train(self, action_sequences): 
        # Stub training
        pass
    
    def encode(self, action_sequence): 
        # Return dummy encoded action
        return 'encoded_action'
    
    def get_abstract_actions(self): 
        return self.latent_space

class ActionDecoder:
    """Decodes abstract actions back to primitive sequences"""
    def decode(self, abstract_action): 
        # Return dummy action sequence
        return [{'motor_1': 0}, {'motor_2': 0}]

def calculate_entropy(predictions):
    """Calculate uncertainty/entropy over predicted distributions"""
    # Higher entropy = more uncertainty = more interesting to explore
    if not predictions:
        return 0.0
    # Simple variance-based uncertainty measure
    predictions_array = np.array([p.flatten() if hasattr(p, 'flatten') else [p] for p in predictions])
    return np.var(predictions_array)

def calculate_loss(predicted, actual):
    """Compute prediction error"""
    # Simple MSE loss
    if isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray):
        return np.mean((predicted - actual) ** 2)
    else:
        # For non-array types, return random loss
        return np.random.rand()

def capture_image():
    """Get current frame from robot camera"""
    # Return dummy 224x224x3 image with some pattern
    image = np.random.rand(224, 224, 3)
    # Add some visual pattern for better visualization
    image[50:150, 50:150] = [0.8, 0.2, 0.2]  # Red square
    return image

def execute_action(action):
    """Send motor commands to robot"""
    # Stub action execution
    print(f"Executing action: {action}")
    time.sleep(0.1)  # Simulate action duration

def create_masked_versions(image, num_masks=5):
    """Create multiple masked versions for training"""
    masked_versions = []
    for _ in range(num_masks):
        # Create a copy and mask random patches
        masked = image.copy()
        h, w = masked.shape[:2]
        mask_h, mask_w = h//4, w//4
        start_h, start_w = np.random.randint(0, h-mask_h), np.random.randint(0, w-mask_w)
        masked[start_h:start_h+mask_h, start_w:start_w+mask_w] = 0
        masked_versions.append(masked)
    return masked_versions

def current_time():
    """Get current timestamp"""
    return time.time()

# Example usage
if __name__ == "__main__":
    # Interactive mode
    model = AdaptiveWorldModel(interactive=True)
    try:
        model.main_loop()
    except KeyboardInterrupt:
        print("\nStopped by user")