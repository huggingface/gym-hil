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
import numpy as np
from gymnasium.spaces import Box

from gym_hil.wrappers.hil_wrappers import DEFAULT_EE_STEP_SIZE, EEActionWrapper


# https://github.com/huggingface/gym-hil/issues/6
class MockEnv(gym.Env):
    """Mock environment for testing the EEActionWrapper."""

    def __init__(self):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def test_init_with_gripper():
    """Test initialization with gripper enabled."""
    env = MockEnv()

    wrapped_env = EEActionWrapper(env, ee_action_step_size=DEFAULT_EE_STEP_SIZE, use_gripper=True)

    # Check action space has correct shape (3 for position + 1 for gripper)
    assert wrapped_env.action_space.shape == (4,)

    # Check bounds are correct using assert_allclose to handle floating point precision
    expected_low = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
    expected_high = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(wrapped_env.action_space.low, expected_low)
    np.testing.assert_allclose(wrapped_env.action_space.high, expected_high)


def test_init_without_gripper():
    """Test initialization without gripper."""
    env = MockEnv()

    wrapped_env = EEActionWrapper(env, ee_action_step_size=DEFAULT_EE_STEP_SIZE, use_gripper=False)

    # Check action space has correct shape (3 for position only)
    assert wrapped_env.action_space.shape == (3,)

    # Check bounds are correct using assert_allclose to handle floating point precision
    expected_low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    expected_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(wrapped_env.action_space.low, expected_low)
    np.testing.assert_allclose(wrapped_env.action_space.high, expected_high)


def test_action_transformation_with_gripper():
    """Test that actions are correctly transformed with gripper."""
    env = MockEnv()

    wrapped_env = EEActionWrapper(env, ee_action_step_size=DEFAULT_EE_STEP_SIZE, use_gripper=True)
    transformed_action = wrapped_env.action(np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32))

    # Check transformed action has correct shape and values
    assert transformed_action.shape == (7,)
    expected_action = np.array(
        [DEFAULT_EE_STEP_SIZE["x"], -DEFAULT_EE_STEP_SIZE["y"], 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    np.testing.assert_allclose(transformed_action, expected_action)


def test_action_transformation_without_gripper():
    """Test that actions are correctly transformed without gripper."""
    env = MockEnv()

    wrapped_env = EEActionWrapper(env, ee_action_step_size=DEFAULT_EE_STEP_SIZE, use_gripper=False)
    transformed_action = wrapped_env.action(np.array([1.0, -1.0, 0.0], dtype=np.float32))

    # Check transformed action has correct shape and values
    assert transformed_action.shape == (7,)
    expected_action = np.array(
        [DEFAULT_EE_STEP_SIZE["x"], -DEFAULT_EE_STEP_SIZE["y"], 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    np.testing.assert_allclose(transformed_action, expected_action)
