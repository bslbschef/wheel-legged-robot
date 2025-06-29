# Wheel Legged Robot

This repository contains a minimal setup for training a wheel-legged robot in Isaac Gym using a PPO algorithm and a simple PD controller.

## Training

The training script expects `stable-baselines3` and `isaacgym` to be installed. Run:

```bash
python src/train_wheel_legged.py --timesteps 1000000
```

The trained policy will be saved to `ppo_wheel_legged.zip` by default.
