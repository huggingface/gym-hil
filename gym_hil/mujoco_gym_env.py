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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.controllers import opspace

MAX_GRIPPER_COMMAND = 255


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        """Release graphics resources if they exist.

        In MuJoCo < 2.3.0 `mujoco.Renderer` had no `close()` member.  Calling
        it unconditionally therefore raises `AttributeError`.  We check for
        the attribute first and fall back to a no-op, keeping compatibility
        across MuJoCo versions.
        """

        viewer = self._viewer
        if viewer is None:
            return

        if hasattr(viewer, "close") and callable(viewer.close):
            try:  # noqa: SIM105
                viewer.close()
            except Exception:
                # Ignore errors coming from already freed OpenGL contexts or
                # older MuJoCo builds.
                pass

        self._viewer = None

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random


class FrankaGymEnv(MujocoGymEnv):
    """Base class for Franka Panda robot environments."""

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)),  # noqa: B008
        cartesian_bounds: np.ndarray = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]]),  # noqa: B008
    ):
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "gym_hil" / "assets" / "scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs

        # Setup cameras
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)

        # Cache robot IDs
        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id

        # Setup observation and action spaces
        self._setup_observation_space()
        self._setup_action_space()

        # Only initialize the offscreen renderer if we are not in "human" mode. This prevents the context conflict.
        if self.render_mode == "rgb_array":
            self._viewer = mujoco.Renderer(self.model, height=render_spec.height, width=render_spec.width)
            self._viewer.render()

    def _setup_observation_space(self):
        """Setup the observation space for the Franka environment."""
        base_obs_space = {
            "agent_pos": spaces.Dict(
                {
                    "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                    "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                }
            )
        }

        self.observation_space = spaces.Dict(base_obs_space)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    **base_obs_space,
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

    def _setup_action_space(self):
        """Setup the action space for the Franka environment."""
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset_robot(self):
        """Reset the robot to home position."""
        self._data.qpos[self._panda_dof_ids] = self._home_position
        self._data.ctrl[self._panda_ctrl_ids] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

    def apply_action(self, action):
        """Apply the action to the robot."""
        x, y, z, rx, ry, rz, grasp_command = action

        # Set the mocap position
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z])
        npos = np.clip(pos + dpos, *self._cartesian_bounds)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp
        g = self._data.ctrl[self._gripper_ctrl_id] / MAX_GRIPPER_COMMAND
        ng = np.clip(g + grasp_command, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * MAX_GRIPPER_COMMAND

        # Apply operational space control
        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=self._home_position,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

    def get_robot_state(self):
        """Get the current state of the robot."""
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        # tcp_quat = self._data.sensor("2f85/pinch_quat").data
        # tcp_vel = self._data.sensor("2f85/pinch_vel").data
        # tcp_angvel = self._data.sensor("2f85/pinch_angvel").data
        qpos = self.data.qpos[self._panda_dof_ids].astype(np.float32)
        qvel = self.data.qvel[self._panda_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()

        return np.concatenate([qpos, qvel, gripper_pose, tcp_pos])

    def render(self):
        """Render the environment based on the specified mode."""
        # If we are in human mode, we expect a wrapper to handle rendering.
        # This method is now primarily for rgb_array mode.
        if self.render_mode == "rgb_array":
            if self._viewer is None:
                self._viewer = mujoco.Renderer(self.model, height=self._render_specs.height, width=self._render_specs.width)

            rendered_frames = []
            for cam_id in self.camera_id:
                self._viewer.update_scene(self.data, camera=cam_id)
                rendered_frames.append(self._viewer.render())
            return rendered_frames
        
        # In 'human' mode, the PassiveViewerWrapper's sync() method handles rendering,
        # so this method can do nothing.
        return None

    def get_gripper_pose(self):
        """Get the current pose of the gripper."""
        return np.array([self._data.ctrl[self._gripper_ctrl_id]], dtype=np.float32)
