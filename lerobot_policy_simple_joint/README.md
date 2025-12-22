# SimpleJoint Policy for LeRobot

A simple non-learned policy for controlling a single joint of the SO-101 robot arm with 3 discrete actions.

## Action Space

| Action | Description |
|--------|-------------|
| 0 | Stay (no movement) |
| 1 | Move in positive direction by `position_delta` |
| 2 | Move in negative direction by `position_delta` |

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
    --policy.action_duration=0.5 \
    --policy.position_delta=10 \
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
    --policy.action_duration=0.5 \
    --policy.position_delta=10 \
    --policy.action_sequence="[1, 1, 0, 2, 2, 0]" \
    --dataset.repo_id=${HF_USER}/so101-sequence \
    --dataset.num_episodes=5 \
    --dataset.single_task="Single joint sequence movement"
```

**Note on Sequence Behavior:**
- The policy will execute the action sequence exactly once
- After completing the sequence, it will output action 0 (stay) for the remainder of the episode
- The sequence does NOT wrap or repeat
- This is ideal for recording specific movement patterns without unintended repetition

For infinite random actions, use `use_random_policy=true` instead.

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `joint_name` | `"shoulder_pan.pos"` | Joint to control. Options: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` |
| `action_duration` | `0.5` | How long each discrete action lasts before selecting the next action (seconds) |
| `position_delta` | `0.1` | How far to move the joint when action 1 or 2 is selected (radians) |
| `use_random_policy` | `false` | If true, randomly select actions indefinitely |
| `action_sequence` | `null` | Optional list of actions to execute once (e.g., `[1, 0, 2, 0]`). No wrapping. |
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
