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

import logging
import sys
import time

import gymnasium as gym
import numpy as np

from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND


class EEActionSpaceParams:
    def __init__(self, x_step_size, y_step_size, z_step_size):
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        info["discrete_penalty"] = 0.0
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["discrete_penalty"] = self.penalty

        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return observation, reward, terminated, truncated, info


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_space_params=None, use_gripper=False):
        super().__init__(env)
        self.ee_action_space_params = ee_action_space_params
        self.use_gripper = use_gripper

        self._ee_step_size = np.array(
            [
                ee_action_space_params.x_step_size,
                ee_action_space_params.y_step_size,
                ee_action_space_params.z_step_size,
            ]
        )
        num_actions = 3

        if self.use_gripper:
            action_space_bounds_min = np.concatenate([-self._ee_step_size, [-1.0]])
            action_space_bounds_max = np.concatenate([self._ee_step_size, [1.0]])
            num_actions += 1

        ee_action_space = gym.spaces.Box(
            low=action_space_bounds_min,
            high=action_space_bounds_max,
            shape=(num_actions,),
            dtype=np.float32,
        )
        self.action_space = ee_action_space

    def action(self, action):
        """
        Mujoco env is expecting a 7D action space
        [x, y, z, rx, ry, rz, gripper_open]
        For the moment we only control the x, y, z, gripper
        """

        action_xyz = action[:3]
        # TODO: Extend to enable orientation control
        actions_orn = np.zeros(3)

        gripper_open_command = [-1.0]
        if self.use_gripper:
            # NOTE: Normalize gripper action from [0, 2] -> [-1, 1]
            gripper_open_command = [action[-1]]

        action = np.concatenate([action_xyz, actions_orn, gripper_open_command])
        return action


class AddJointVelocityToObservation(gym.ObservationWrapper):
    def __init__(self, env, joint_velocity_limits=100.0, fps=30):
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["agent_pos"].low
        old_high = self.observation_space["agent_pos"].high
        old_shape = self.observation_space["agent_pos"].shape

        self.last_joint_positions = np.zeros(old_shape)

        new_low = np.concatenate([old_low, np.ones_like(old_low) * -joint_velocity_limits])
        new_high = np.concatenate([old_high, np.ones_like(old_high) * joint_velocity_limits])

        new_shape = (old_shape[0] * 2,)

        self.observation_space["agent_pos"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

        self.dt = 1.0 / fps

    def observation(self, observation):
        joint_velocities = (observation["agent_pos"] - self.last_joint_positions) / self.dt
        self.last_joint_positions = observation["agent_pos"].copy()
        observation["agent_pos"] = np.concatenate([observation["agent_pos"], joint_velocities], axis=-1)
        return observation


class EEObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, ee_pose_limits):
        super().__init__(env)

        # Extend observation space to include end effector pose
        prev_space = self.observation_space["agent_pos"]

        self.observation_space["agent_pos"] = gym.spaces.Box(
            low=np.concatenate([prev_space.low, ee_pose_limits["min"]]),
            high=np.concatenate([prev_space.high, ee_pose_limits["max"]]),
            shape=(prev_space.shape[0] + 3,),
            dtype=np.float32,
        )

    def observation(self, observation):
        current_ee_pos = self.env.unwrapped.data.sensor("2f85/pinch_pos").data
        observation["agent_pos"] = np.concatenate([observation["agent_pos"], current_ee_pos], axis=-1)
        return observation


class InputsControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
        use_gamepad=True,
        controller_config_path=None,
    ):
        """
        Initialize the inputs controller wrapper.

        Args:
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            use_gripper: Whether to use gripper control
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
            use_gamepad: Whether to use gamepad or keyboard control
            controller_config_path: Path to the controller configuration JSON file
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import (
            GamepadController,
            GamepadControllerHID,
            KeyboardController,
        )

        # use HidApi for macos
        if use_gamepad:
            if sys.platform == "darwin":
                self.controller = GamepadControllerHID(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                    config_path=controller_config_path,
                )
            else:
                self.controller = GamepadController(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                    config_path=controller_config_path,
                )
        else:
            self.controller = KeyboardController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )

        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [1.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [-1.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [0.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        if is_intervention:
            action = gamepad_action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        info["is_intervention"] = is_intervention
        action_intervention = action

        info["action_intervention"] = action_intervention
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        return self.env.reset(**kwargs)

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class ResetDelayWrapper(gym.Wrapper):
    """
    Wrapper that adds a time delay when resetting the environment.

    This can be useful for adding a pause between episodes to allow for human observation.
    """

    def __init__(self, env, delay_seconds=1.0):
        """
        Initialize the time delay reset wrapper.

        Args:
            env: The environment to wrap
            delay_seconds: The number of seconds to delay during reset
        """
        super().__init__(env)
        self.delay_seconds = delay_seconds

    def reset(self, **kwargs):
        """Reset the environment with a time delay."""
        # Add the time delay
        logging.info(f"Reset delay of {self.delay_seconds} seconds")
        time.sleep(self.delay_seconds)

        # Call the parent reset method
        return self.env.reset(**kwargs)
