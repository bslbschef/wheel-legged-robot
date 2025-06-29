"""Training script for the wheel-legged robot using PPO."""

import argparse

import gym
import numpy as np

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "stable-baselines3 is required to run the training script"
    ) from exc

from envs.wheel_legged_env import WheelLeggedEnv, PDController


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Wheel-Legged Robot")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of training timesteps")
    parser.add_argument("--kp", type=float, default=1.0, help="Proportional gain for PD controller")
    parser.add_argument("--kd", type=float, default=0.1, help="Derivative gain for PD controller")
    parser.add_argument("--save-path", type=str, default="ppo_wheel_legged.zip", help="Where to save the trained model")
    return parser.parse_args()


def main():
    args = parse_args()
    controller = PDController(kp=args.kp, kd=args.kd)
    env = WheelLeggedEnv(controller=controller)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    env.close()


if __name__ == "__main__":
    main()
