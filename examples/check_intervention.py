import time

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def main() -> None:
    env = gym.make(
        "gym_hil/PandaPickCubeKeyboard-v0",
        render_mode="human",
        image_obs=True,
        max_episode_steps=1000,
    )

    obs, info = env.reset()

    # 前三维为末端位置动作，最后一维 1 表示夹爪保持不动
    policy_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(policy_action)

            if info.get("is_intervention", False):
                print(
                    "Intervention:",
                    info.get("teleop_action"),
                    "reward:",
                    reward,
                )

            if terminated or truncated:
                print("Episode finished:", info)
                obs, info = env.reset()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
