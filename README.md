# Wheel-Legged Robot

This project explores reinforcement learning and classical control for a wheel-legged robot. It combines physics simulation with modern RL techniques to achieve dynamic locomotion.

## Reinforcement Learning with PD Control

The control strategy mixes a learned policy with proportional-derivative (PD) control for joint actuation. The policy outputs target positions and velocities, while a PD controller translates them into torques. This hybrid approach helps stabilize learning and improves robustness.

## Setup

### Isaac Gym
1. Install [Isaac Gym](https://developer.nvidia.com/isaac-gym) following NVIDIA's instructions.
2. Verify that Python bindings work by running a sample environment.

### MuJoCo
1. Install [MuJoCo](https://mujoco.org/) and set the `MUJOCO_PY_MJKEY_PATH` and `LD_LIBRARY_PATH` as required.
2. Test the installation with the provided MuJoCo examples.

## Training

Run training with an RL framework of choice. Example using `python`:

```bash
python src/train.py --env wheel_legged --num-steps 1000000
```

## Testing

After training, evaluate the policy in simulation:

```bash
python src/test.py --env wheel_legged --model checkpoints/latest.pt
```

These commands assume that training and evaluation scripts are implemented under `src/` and that environments and controllers live in `envs/` and `controllers/` respectively.

