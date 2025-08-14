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

from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw


class PassiveViewerWrapper(gym.Wrapper):
    """Gym wrapper that opens a passive MuJoCo viewer automatically.

    The wrapper starts a MuJoCo viewer in passive mode as soon as the
    environment is created so the user no longer needs to use
    ``mujoco.viewer.launch_passive`` or any context–manager boiler-plate.

    The viewer is kept in sync after every ``reset`` and ``step`` call and is
    closed automatically when the environment itself is closed or deleted.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
    ) -> None:
        super().__init__(env)

        # Launch the interactive viewer.  We expose *model* and *data* from the
        # *unwrapped* environment to make sure we operate on the base MuJoCo
        # objects even if other wrappers have been applied before this one.
        self._viewer = mujoco.viewer.launch_passive(
            env.unwrapped.model,
            env.unwrapped.data,
            # show_left_ui=show_left_ui,
            # show_right_ui=show_right_ui,
        )

        # Make sure the first frame is rendered.
        self._viewer.sync()

    # ---------------------------------------------------------------------
    # Gym API overrides

    def reset(self, **kwargs):  # type: ignore[override]
        observation, info = self.env.reset(**kwargs)
        self._viewer.sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._viewer.sync()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        """Close both the passive viewer and the underlying gym environment.

        MuJoCo's `Renderer` gained a `close()` method only in recent versions
        (>= 2.3.0).  When running with an older MuJoCo build the renderer
        instance stored inside `env.unwrapped._viewer` does not provide this
        method which causes `AttributeError` when the environment is closed.

        To remain version-agnostic we:
          1. Manually dispose of the underlying viewer *only* if it exposes a
             `close` method.
          2. Remove the reference from the environment so that a subsequent
             call to `env.close()` will not fail.
          3. Close our own passive viewer handle.
          4. Finally forward the `close()` call to the wrapped environment so
             that any other resources are released.
        """

        # 1. Tidy up the renderer managed by the wrapped environment (if any).
        base_env = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(base_env, "_viewer"):
            viewer = base_env._viewer
            if viewer is not None and hasattr(viewer, "close") and callable(viewer.close):
                try:  # noqa: SIM105
                    viewer.close()
                except Exception:
                    # Ignore errors coming from older MuJoCo versions or
                    # already-freed contexts.
                    pass
            # Prevent the underlying env from trying to close it again.
            base_env._viewer = None

        # 2. Close the passive viewer launched by this wrapper.
        try:  # noqa: SIM105
            self._viewer.close()
        except Exception:  # pragma: no cover
            # Defensive: avoid propagating viewer shutdown errors.
            pass

        # 3. Let the wrapped environment perform its own cleanup.
        self.env.close()

    def __del__(self):
        # "close" may raise if called during interpreter shutdown; guard just
        # in case.
        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass


class DualViewportWrapper(gym.Wrapper):
    """
    A dual viewport wrapper that uses GLFW for the operator view (dual viewport)
    and the MuJoCo Renderer for image-based observations.

    Args:
        env (gym.Env): The environment to wrap.
        view_camera_left (str): The name of the camera for the left viewport.
        view_camera_right (str): The name of the camera for the right viewport.
        observation_camera_names (tuple[str, str]): A tuple containing the names
            of the cameras used for generating observations.
        observation_image_sizes (tuple[tuple[int, int], tuple[int, int]]):
            A tuple of tuples, where each inner tuple specifies the (height, width)
            of the observation images for the corresponding camera.
        window_title (str): The title of the GLFW window.
    """

    def __init__(
        self,
        env: gym.Env,
        view_camera_left: str = "front",
        view_camera_right: str = "handcam_rgb",
        observation_camera_names: tuple[str, str] = ("front", "wrist"),
        observation_image_sizes: tuple[tuple[int, int], tuple[int, int]] = ((128, 128), (128, 128)),
        window_title: str = "MuJoCo — Dual Viewports",
    ) -> None:
        super().__init__(env)

        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.observation_camera_names = observation_camera_names

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        glfw.window_hint(glfw.SAMPLES, value=4)  # 4x MSAA
        self._window = glfw.create_window(1280, 720, window_title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._opt)
        self._scn = mujoco.MjvScene(self.model, maxgeom=2000)
        self._con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        self._camera_left = self._make_camera(view_camera_left)
        self._camera_right = self._make_camera(view_camera_right)

        self._closed = False
        glfw.set_window_close_callback(self._window, self._on_close)

        self.has_images = (
            hasattr(env.observation_space, "spaces") and "pixels" in env.observation_space.spaces
        )

        if self.has_images:
            self._renderers = {}
            for name, (height, width) in zip(observation_camera_names, observation_image_sizes, strict=True):
                self.model.vis.global_.offwidth = width
                self.model.vis.global_.offheight = height
                self._renderers[name] = mujoco.Renderer(self.model, width=width, height=height)

            self._observation_camera_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                for name in observation_camera_names
            ]

    def _on_close(self, window) -> None:
        self._closed = True

    def _make_camera(self, name: str) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        return camera

    def _capture_observation_images(self) -> dict[str, any]:
        """Render small images for reinforcement learning, keyed by camera names."""
        pixels = {}
        for name, camera_id in zip(self.observation_camera_names, self._observation_camera_ids, strict=True):
            renderer = self._renderers[name]
            renderer.update_scene(self.data, camera=camera_id)
            pixels[name] = renderer.render()
        return pixels

    def _draw_dual_view(self) -> None:
        if self._window is None or self._closed:
            return
        glfw.make_context_current(self._window)
        mujoco.mjr_setBuffer(int(mujoco.mjtFramebuffer.mjFB_WINDOW), self._con)
        width, height = glfw.get_framebuffer_size(self._window)
        left = mujoco.MjrRect(0, 0, width // 2, height)
        right = mujoco.MjrRect(width // 2, 0, width - width // 2, height)
        # left
        mujoco.mjv_updateScene(
            self.model, self.data, self._opt, None, self._camera_left, mujoco.mjtCatBit.mjCAT_ALL, self._scn
        )
        mujoco.mjr_render(left, self._scn, self._con)
        # right
        mujoco.mjv_updateCamera(self.model, self.data, self._camera_right, self._scn)
        mujoco.mjr_render(right, self._scn, self._con)
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    # ---------------------------------------------------------------------
    # Gym API overrides

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.has_images:
            pixels = self._capture_observation_images()
            observation["pixels"] = pixels
        self._draw_dual_view()  # first draw
        if self._closed:
            info = dict(info)
            info["viewer_closed"] = True
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.has_images:
            pixels = self._capture_observation_images()
            observation["pixels"] = pixels
        self._draw_dual_view()
        if self._closed:
            info = dict(info)
            info["viewer_closed"] = True
            # trunc = True # Stop the episode
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources and close the environment."""
        if hasattr(self, "_renderers"):
            for renderer in self._renderers.values():
                try: # noqa: SIM105
                    renderer.close()
                except Exception:
                    pass

        if self._window:
            try:
                glfw.set_window_close_callback(self._window, None)
                glfw.destroy_window(self._window)
            except Exception:
                pass
            finally:
                self._window = None
                glfw.terminate()
        super().close()
