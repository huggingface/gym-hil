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

import gymnasium as gym

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec, MujocoGymEnv
from gym_hil.wrappers.factory import make_env, wrap_env
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

__all__ = [
    "MujocoGymEnv",
    "FrankaGymEnv",
    "GymRenderingSpec",
    "PassiveViewerWrapper",
    "make_env",
    "wrap_env",
]

from gymnasium.envs.registration import register

# Register the base environment directly
register(
    id="gym_hil/PandaPickCubeBase-v0",  # This is the base environment
    entry_point="gym_hil.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)

register(
    id="gym_hil/PandaArrangeBoxesBase-v0",  # This is the base environment
    entry_point="gym_hil.envs:PandaArrangeBoxesGymEnv",
    max_episode_steps=100,
)

# Register the viewer wrapper
register(
    id="gym_hil/PandaPickCubeViewer-v0",
    entry_point=lambda **kwargs: PassiveViewerWrapper(gym.make("gym_hil/PandaPickCubeBase-v0", **kwargs)),
    max_episode_steps=100,
)

register(
    id="gym_hil/PandaArrangeBoxesViewer-v0",
    entry_point=lambda **kwargs: PassiveViewerWrapper(gym.make("gym_hil/PandaArrangeBoxesBase-v0", **kwargs)),
    max_episode_steps=100,
)

# Register wrapped versions using the factory - NOTE: these now use PandaPickCubeBase-v0
register(
    id="gym_hil/PandaPickCube-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",  # Use the base environment
    },
)

register(
    id="gym_hil/PandaPickCubeGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",  # Use the base environment
        "use_viewer": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/PandaPickCubeKeyboard-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",  # Use the base environment
        "use_viewer": True,
        "gripper_penalty": -0.05,
    },
)

register(
    id="gym_hil/PandaArrangeBoxes-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaArrangeBoxesBase-v0",  # Use the base environment
    },
)

register(
    id="gym_hil/PandaArrangeBoxesGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaArrangeBoxesBase-v0",  # Use the base environment
        "use_viewer": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/PandaArrangeBoxesKeyboard-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={"env_id": "gym_hil/PandaArrangeBoxesBase-v0", "use_viewer": True, "gripper_penalty": -0.05},
)
