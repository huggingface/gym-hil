#!/usr/bin/env python

import gymnasium as gym

from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.hil_wrappers import (
    EEActionSpaceParams,
    EEActionWrapper,
    GripperPenaltyWrapper,
    InputsControlWrapper,
    ResetDelayWrapper,
)
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


def wrap_env(
    env: gym.Env,
    use_viewer: bool = False,
    use_gamepad: bool = False,
    use_gripper: bool = True,
    auto_reset: bool = False,
    step_size: float = 0.01,
    show_ui: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
) -> gym.Env:
    """Apply wrappers to an environment based on configuration.

    Args:
        env: The base environment to wrap
        use_viewer: Whether to add a passive viewer
        use_gamepad: Whether to use gamepad instead of keyboard controls
        use_gripper: Whether to enable gripper control
        auto_reset: Whether to automatically reset the environment when episode ends
        step_size: Step size for movement in meters
        show_ui: Whether to show UI panels in the viewer
        gripper_penalty: Penalty for using the gripper
        controller_config_path: Path to the controller configuration JSON file

    Returns:
        The wrapped environment
    """
    # Apply wrappers in the correct order
    if use_viewer:
        env = PassiveViewerWrapper(env, show_left_ui=show_ui, show_right_ui=show_ui)

    if use_gripper:
        env = GripperPenaltyWrapper(env, penalty=gripper_penalty)

    ee_params = EEActionSpaceParams(step_size, step_size, step_size)
    env = EEActionWrapper(env, ee_action_space_params=ee_params, use_gripper=True)

    # Apply control wrappers last
    env = InputsControlWrapper(
        env,
        x_step_size=step_size,
        y_step_size=step_size,
        z_step_size=step_size,
        use_gripper=use_gripper,
        auto_reset=auto_reset,
        use_gamepad=use_gamepad,
    )

    # Apply time delay wrapper
    env = ResetDelayWrapper(env, delay_seconds=reset_delay_seconds)

    return env


def make_env(
    env_id: str,
    use_viewer: bool = False,
    use_gamepad: bool = False,
    use_gripper: bool = True,
    auto_reset: bool = False,
    step_size: float = 0.01,
    show_ui: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
    **kwargs,
) -> gym.Env:
    """Create and wrap an environment in a single function.

    Args:
        env_id: The ID of the base environment to create
        use_viewer: Whether to add a passive viewer
        use_gamepad: Whether to use gamepad instead of keyboard controls
        use_gripper: Whether to enable gripper control
        auto_reset: Whether to automatically reset the environment when episode ends
        step_size: Step size for movement in meters
        show_ui: Whether to show UI panels in the viewer
        gripper_penalty: Penalty for using the gripper
        controller_config_path: Path to the controller configuration JSON file
        **kwargs: Additional arguments to pass to the base environment

    Returns:
        The wrapped environment
    """
    # Create the base environment directly
    if env_id == "gym_hil/PandaPickCubeBase-v0":
        env = PandaPickCubeGymEnv(**kwargs)
    else:
        raise ValueError(f"Environment ID {env_id} not supported")

    return wrap_env(
        env,
        use_viewer=use_viewer,
        use_gamepad=use_gamepad,
        use_gripper=use_gripper,
        auto_reset=auto_reset,
        step_size=step_size,
        show_ui=show_ui,
        gripper_penalty=gripper_penalty,
        reset_delay_seconds=reset_delay_seconds,
    )
