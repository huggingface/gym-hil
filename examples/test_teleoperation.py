#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Control Franka robot interactively")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for movement in meters")
    parser.add_argument(
        "--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="Rendering mode"
    )
    parser.add_argument("--use-keyboard", action="store_true", help="Use keyboard control")
    parser.add_argument(
        "--reset-delay",
        type=float,
        default=2.0,
        help="Delay in seconds when resetting the environment (0.0 means no delay)",
    )
    parser.add_argument(
        "--controller-config", type=str, default=None, help="Path to controller configuration JSON file"
    )
    args = parser.parse_args()

    # Create Franka environment - Use base environment first to debug
    env = gym.make(
        "gym_hil/PandaPickCubeBase-v0",  # Use the base environment for debugging
        render_mode=args.render_mode,
        image_obs=True,
    )

    # Print observation space for debugging
    print("Observation space:", env.observation_space)

    # Reset and check observation structure
    obs, _ = env.reset()
    print("Observation keys:", list(obs.keys()))
    if "pixels" in obs:
        print("Pixels keys:", list(obs["pixels"].keys()))

    # Now try with the wrapped version
    print("\nTrying wrapped environment...")
    env_id = "gym_hil/PandaPickCubeKeyboard-v0" if args.use_keyboard else "gym_hil/PandaPickCubeGamepad-v0"
    env = gym.make(
        env_id,
        render_mode=args.render_mode,
        image_obs=True,
        use_gamepad=not args.use_keyboard,
        max_episode_steps=1000,  # 100 seconds * 10Hz
    )

    # Print observation space for the wrapped environment
    print("Wrapped observation space:", env.observation_space)

    # Reset and check wrapped observation structure
    obs, _ = env.reset()
    print("Wrapped observation keys:", list(obs.keys()))

    # Reset environment
    obs, _ = env.reset()
    dummy_action = np.zeros(4, dtype=np.float32)
    # This ensures the "stay gripper" action is set when the intervention button is not pressed
    dummy_action[-1] = 1

    try:
        while True:
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            # Print some feedback
            if info.get("succeed", False):
                print("\nSuccess! Block has been picked up.")

            # If auto-reset is disabled, manually reset when episode ends
            if terminated or truncated:
                print("Episode ended, resetting environment")
                obs, _ = env.reset()

            # Add a small delay to control update rate
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()
        print("Session ended")


if __name__ == "__main__":
    main()
