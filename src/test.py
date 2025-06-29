import argparse

from stable_baselines3 import PPO

from envs.wheel_legged_env import WheelLeggedEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument("--model", required=True, help="Path to the trained policy")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()

    env = WheelLeggedEnv(render_mode="human" if args.render else None)
    model = PPO.load(args.model)

    obs, _ = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if args.render:
            env.render()
        if done:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
