# SimpleJoint Policy for LeRobot

A simple non-learned policy for controlling a single joint of the SO-101 robot arm with 3 discrete actions.

## Action Space

| Action | Description |
|--------|-------------|
| 0 | Stay (no movement) |
| 1 | Move in positive direction for `move_duration` seconds |
| 2 | Move in negative direction for `move_duration` seconds |

## Installation

```bash
cd lerobot_policy_simple_joint
pip install -e .
```

## Usage with lerobot-record

### Random Exploration

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_so101 \
    --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
    --policy.type=simple_joint \
    --policy.joint_name=shoulder_pan.pos \
    --policy.move_duration=0.5 \
    --policy.move_speed=0.2 \
    --policy.use_random_policy=true \
    --dataset.repo_id=${HF_USER}/so101-single-joint \
    --dataset.num_episodes=10 \
    --dataset.single_task="Single joint random movement"
```

### Fixed Action Sequence

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_so101 \
    --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --policy.type=simple_joint \
    --policy.joint_name=shoulder_pan.pos \
    --policy.move_duration=0.5 \
    --policy.action_sequence="[1, 1, 0, 2, 2, 0]" \
    --dataset.repo_id=${HF_USER}/so101-sequence \
    --dataset.num_episodes=5 \
    --dataset.single_task="Single joint sequence movement"
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `joint_name` | `"shoulder_pan.pos"` | Joint to control. Options: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` |
| `move_duration` | `0.5` | Duration of movement in seconds |
| `move_speed` | `0.2` | Movement speed in radians/second |
| `use_random_policy` | `false` | If true, randomly select actions |
| `action_sequence` | `null` | Optional list of actions to cycle through (e.g., `[1, 0, 2, 0]`) |
| `random_seed` | `null` | Seed for reproducible random actions |

## Converting to concat_world_model_explorer Format

After recording with this policy, use the `convert_lerobot_to_explorer.py` script to convert the dataset:

```bash
python convert_lerobot_to_explorer.py \
    --lerobot-path ~/.cache/huggingface/lerobot/${HF_USER}/so101-single-joint \
    --output-dir saved/sessions/so101 \
    --cameras base_0_rgb left_wrist_0_rgb \
    --stack-cameras vertical \
    --joint-name shoulder_pan.pos
```
