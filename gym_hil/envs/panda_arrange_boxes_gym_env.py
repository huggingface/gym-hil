#!/usr/bin/env python

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec

_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.5, 0], [0.6, 0.5, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaArrangeBoxesGymEnv(FrankaGymEnv):
    """Environment for a Panda robot picking up a cube."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
    ):
        self.reward_type = reward_type

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            xml_path=Path(__file__).parent.parent / "assets" / "arrange_boxes_scene.xml",
        )

        # Task-specific setup
        self._block_z = self._model.geom("block1").size[2]

        # Setup observation space properly to match what _compute_observation returns
        # Observation space design:
        #   - "state":  agent (robot) configuration as a single Box
        #   - "environment_state": block position in the world as a single Box
        #   - "pixels": (optional) dict of camera views if image observations are enabled

        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.no_blocks = self._get_no_boxes()
        self.block_range = 0.3

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def _get_no_boxes(self):
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)
        ]
        block_names = list(filter(lambda joint: "block" in joint, joint_names))
        return len(block_names)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Ensure gymnasium internal RNG is initialized when a seed is provided
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        # Reset the robot to home position
        self.reset_robot()

        positions_coords = np.linspace(-self.block_range, self.block_range, self.no_blocks)
        np.random.shuffle(positions_coords)

        # Sample a new block position
        blocks = [f"block{i}" for i in range(1, self.no_blocks + 1)]
        np.random.shuffle(blocks)

        for block, pos in zip(blocks, positions_coords, strict=False):
            block_x_coord = self._data.joint(block).qpos[0]
            block_coords = np.array([block_x_coord, pos])
            self._data.joint(block).qpos[:3] = (*block_coords, self._block_z)

        mujoco.mj_forward(self._model, self._data)
        obs = self._compute_observation()

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply the action to the robot
        self.apply_action(action)

        # Compute observation, reward and termination
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        # Check if block is outside bounds
        block_pos = self._data.sensor("block1_pos").data
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - self.block_range - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + self.block_range + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        # Create the dictionary structure that matches our observation space
        observation = {}

        # Get robot state
        robot_state = self.get_robot_state().astype(np.float32)

        # Assemble observation respecting the newly defined observation_space
        block_pos = self._data.sensor("block1_pos").data.astype(np.float32)

        if self.image_obs:
            # Image observations
            front_view, wrist_view = self.render()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            # State-only observations
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }

        return observation

    def _get_sensors(self) -> Tuple[list, list]:
        """Retrieve block and target positions."""
        return (
            [self._data.sensor(f"block{i}_pos") for i in range(1, self.no_blocks + 1)],
            [self._data.sensor(f"target{i}_pos") for i in range(1, self.no_blocks + 1)],
        )

    def _compute_reward(self) -> float:
        """Compute the current reward based on block-target distances."""
        block_sensors, target_sensors = self._get_sensors()
        distances = [
            np.linalg.norm(block.data - target.data)
            for block, target in zip(block_sensors, target_sensors, strict=False)
        ]

        if self.reward_type == "dense":
            return sum(np.exp(-20 * d) for d in distances)
        else:
            return float(all(d < 0.03 for d in distances))

    def _is_success(self) -> bool:
        """Check if the task is successfully completed."""
        block_sensors, target_sensors = self._get_sensors()

        distances = [
            np.linalg.norm(block.data - target.data)
            for block, target in zip(block_sensors, target_sensors, strict=False)
        ]

        return all(dist < 0.03 for dist in distances)
