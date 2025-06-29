import argparse

from stable_baselines3 import PPO

from envs.wheel_legged_env import WheelLeggedEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the wheel-legged robot")
    parser.add_argument("--num-steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--model-path", default="checkpoints/latest", help="Where to save the trained model")
    args = parser.parse_args()

    env = WheelLeggedEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.num_steps)
    model.save(args.model_path)


if __name__ == "__main__":
    main()
