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
import mujoco
import numpy as np

import gym_hil  # noqa: F401
try:
    from mujoco.glfw import glfw
except ImportError:
    print("GLFW not found. Please `pip install glfw`")
    exit(1)

# Utility for manual keyboard control mode
from gym_hil.wrappers.intervention_utils import MuJoCoViewerController


def gripper_scalar(cmd: str) -> float:
    """Maps gripper command to the action space in manual mode."""
    if cmd == "open": return -1.0
    if cmd == "close": return 1.0
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="Control Franka robot interactively")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for movement in meters")
    parser.add_argument(
        "--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="Rendering mode"
    )
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

    # Mode selection based on --use-keyboard flag
    if args.use_keyboard:
        # MANUAL KEYBOARD CONTROL (runs with `python`)
        print("Running in MANUAL KEYBOARD mode. This can be run with the standard `python` interpreter.")
        
        # Create the BASE environment, without any problematic rendering wrappers
        env = gym.make("gym_hil/PandaPickCubeBase-v0", max_episode_steps=5000) # Increase default steps to give user lot of time and to manually stop episode recording
        model = env.unwrapped.model
        data = env.unwrapped.data
        obs, _ = env.reset()

        # Initialize the viewer and controller manually
        glfw.init()
        window = glfw.create_window(1280, 720, "HIL Teleoperation (Manual Keyboard Mode)", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        
        controller = MuJoCoViewerController(window, x_step_size=args.step_size, y_step_size=args.step_size, z_step_size=args.step_size)

        # Setup MuJoCo rendering
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=10000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth = 90; cam.elevation = -35; cam.distance = 1.5
        cam.lookat = np.array([0.45, 0, 0.3])
        
        print("\nControls (Viewer window must be focused)\n"
              "  ↑ ↓ ← →          move end-effector in X / Y\n"
              "  L-Shift/R-Shift  move down/up\n"
              "  L-Ctrl/R-Ctrl    close/open gripper\n")

        try:
            with controller:
                while not glfw.window_should_close(window):
                    dx, dy, dz = controller.get_deltas()
                    grip_action = gripper_scalar(controller.gripper_command())
                    action = np.array([dx, dy, dz, 0, 0, 0, grip_action], dtype=np.float32)

                    obs, reward, terminated, truncated, info = env.step(action)

                    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
                    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(viewport, scene, context)
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    
                    if terminated or truncated or controller.get_episode_end_status():
                        print("Episode finished – resetting…")
                        time.sleep(args.reset_delay)
                        obs, _ = env.reset()
                        controller.reset()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            env.close()
            glfw.terminate()
            print("Session ended")

    else:
        # DEFAULT GAMEPAD CONTROL (requires `mjpython` on macOS)
        print("Running in DEFAULT GAMEPAD mode.")
        print("WARNING: This mode requires the `mjpython` interpreter on macOS due to its use of `mujoco.viewer.launch_passive`.")
        
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
        env = gym.make(
            "gym_hil/PandaPickCubeGamepad-v0",
            render_mode=args.render_mode,
            image_obs=True,
            use_gamepad=True,
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
