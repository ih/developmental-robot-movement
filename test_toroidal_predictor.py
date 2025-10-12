"""
Test script for verifying the transformer predictor works with toroidal dot robot.
"""

import torch
from models import create_transformer_predictor_for_robot, get_action_config_for_robot

def test_toroidal_dot_predictor():
    """Test that the transformer predictor works with toroidal dot action space."""

    print("Testing transformer predictor for toroidal dot robot...")
    print("=" * 60)

    # Get action config for toroidal dot
    action_channels, action_ranges = get_action_config_for_robot("toroidal_dot")
    print(f"\nAction channels: {action_channels}")
    print(f"Action ranges: {action_ranges}")

    # Create predictor for toroidal dot
    predictor = create_transformer_predictor_for_robot(
        "toroidal_dot",
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        num_actions=2,  # Binary action space: 0 (stay), 1 (move right)
        level=0
    )

    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = predictor.to(device)

    print(f"\nPredictor created successfully!")
    print(f"Device: {device}")
    print(f"Predictor action channels: {predictor.action_channels}")
    print(f"Predictor action ranges: {predictor.action_ranges}")

    # Test action normalization
    print("\n" + "=" * 60)
    print("Testing action normalization...")

    test_actions = [
        {"action": 0},  # Stay
        {"action": 1},  # Move right
    ]

    for action in test_actions:
        normalized = predictor._normalize_action(action, device)
        print(f"Action {action} -> Normalized: {normalized.cpu().numpy()}")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")

    # Create dummy encoder features (simulate MAE encoder output)
    # Shape: [batch_size=1, num_patches+1=197, embed_dim=256]
    batch_size = 1
    num_patches = 196  # 14x14 patches for 224x224 image with 16x16 patches

    # Create a short history of 3 frames
    encoder_features_history = [
        torch.randn(batch_size, num_patches + 1, 256, device=device)
        for _ in range(3)
    ]

    # Create corresponding actions (2 actions between 3 frames)
    actions = [
        {"action": 1},  # Move right
        {"action": 0},  # Stay
    ]

    # Run forward pass
    with torch.no_grad():
        # Get last features for delta prediction
        last_features = encoder_features_history[-1]

        # Normalize the action for FiLM conditioning
        action_normalized = predictor._normalize_action(actions[-1], device)

        # Forward pass
        predicted_features = predictor.forward(
            encoder_features_history,
            actions,
            action_normalized=action_normalized,
            last_features=last_features
        )

    print(f"Input history length: {len(encoder_features_history)} frames")
    print(f"Actions: {actions}")
    print(f"Predicted features shape: {predicted_features.shape}")
    print(f"Expected shape: [1, {num_patches + 1}, 256]")

    # Verify shape
    assert predicted_features.shape == (batch_size, num_patches + 1, 256), \
        f"Unexpected shape: {predicted_features.shape}"

    print("\n[PASS] All tests passed!")
    print("=" * 60)

    return predictor


def test_jetbot_predictor():
    """Test that the transformer predictor still works with JetBot action space."""

    print("\n\nTesting transformer predictor for JetBot robot...")
    print("=" * 60)

    # Get action config for JetBot
    action_channels, action_ranges = get_action_config_for_robot("jetbot")
    print(f"\nAction channels: {action_channels}")
    print(f"Action ranges: {action_ranges}")

    # Create predictor for JetBot
    predictor = create_transformer_predictor_for_robot(
        "jetbot",
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        num_actions=2,  # Stop and forward
        level=0
    )

    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = predictor.to(device)

    print(f"\nPredictor created successfully!")
    print(f"Device: {device}")
    print(f"Predictor action channels: {predictor.action_channels}")

    # Test action normalization
    print("\n" + "=" * 60)
    print("Testing action normalization...")

    test_actions = [
        {"motor_left": 0, "motor_right": 0, "duration": 0.2},    # Stop
        {"motor_left": 0, "motor_right": 0.12, "duration": 0.2}, # Forward
    ]

    for action in test_actions:
        normalized = predictor._normalize_action(action, device)
        print(f"Action {action} -> Normalized: {normalized.cpu().numpy()}")

    print("\n[PASS] JetBot predictor tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Test toroidal dot predictor
    toroidal_predictor = test_toroidal_dot_predictor()

    # Test JetBot predictor (backward compatibility)
    test_jetbot_predictor()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("The transformer predictor now works with both robot types!")
    print("=" * 60)
