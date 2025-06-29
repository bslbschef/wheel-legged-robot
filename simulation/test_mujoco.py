import argparse

try:  # Optional runtime dependencies
    import gym
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - handled at runtime
    gym = None
    PPO = None


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained policy inside a MuJoCo simulation")
    parser.add_argument(
        "--env",
        default="HalfCheetah-v4",
        help="Gym environment ID for the MuJoCo task",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained policy file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of environment steps to execute",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment using the viewer",
    )
    args = parser.parse_args()

    env = gym.make(args.env)
    model = PPO.load(args.model)

    obs = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if args.render:
            env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
