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


class SplitViewportWindow:
    """Owns a GLFW window + shared MjrContext/Scene and draws two cameras side-by-side."""
    def __init__(self, model, data, title="gym-hil — Dual View", msaa=4):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        glfw.window_hint(glfw.SAMPLES, msaa)
        self.window = glfw.create_window(1280, 720, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.m = model
        self.d = data

        self.opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(self.opt)
        self.scn = mujoco.MjvScene(self.m, maxgeom=2000)
        self.con = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_100)

        def fixed_cam(name: str):
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id >= 0:
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cam_id
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            return cam

        self.cam_left = fixed_cam("front")
        self.cam_right = fixed_cam("handcam_rgb")

    def render_dual(self):
        w, h = glfw.get_framebuffer_size(self.window)
        left  = mujoco.MjrRect(0, 0, w//2, h)
        right = mujoco.MjrRect(w//2, 0, w - w//2, h)

        mujoco.mjr_setBuffer(int(mujoco.mjtFramebuffer.mjFB_WINDOW), self.con)
        mujoco.mjv_updateScene(self.m, self.d, self.opt, None, self.cam_left,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(left, self.scn, self.con)
        mujoco.mjv_updateCamera(self.m, self.d, self.cam_right, self.scn)
        mujoco.mjr_render(right, self.scn, self.con)

    def make_current(self):
        if glfw.get_current_context() is not self.window:
            glfw.make_context_current(self.window)

    def is_open(self):
        return not glfw.window_should_close(self.window)

    def swap(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        try:
            # No mjr_freeContext in your build; let GC handle it
            pass
        finally:
            if self.window:
                glfw.destroy_window(self.window)
                glfw.terminate()
                self.window = None


class OffscreenObsMixin:
    """Capture RGB images from named cameras via the SAME render context."""
    def __init__(self, split_window: SplitViewportWindow, size=(128, 128)):
        self.win = split_window
        self.h, self.w = size  # (H, W)

        # Prepare capture cameras
        self._obs_cams = {}
        for name in ("front", "handcam_rgb"):
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam_id = mujoco.mj_name2id(self.win.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id >= 0:
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cam_id
            self._obs_cams[name] = cam

        # Ensure offscreen size, then recreate the context (no free call)
        self.win.m.vis.global_.offwidth  = self.w
        self.win.m.vis.global_.offheight = self.h
        # Recreate MjrContext bound to current GL context
        self.win.make_current()
        # Optionally help GC: old = self.win.con; del old
        self.win.con = mujoco.MjrContext(self.win.m, mujoco.mjtFontScale.mjFONTSCALE_100)

    def capture_images(self):
        self.win.make_current()
        mujoco.mjr_setBuffer(int(mujoco.mjtFramebuffer.mjFB_OFFSCREEN), self.win.con)

        out = {}
        rect = mujoco.MjrRect(0, 0, self.win.con.offWidth, self.win.con.offHeight)
        rgb = np.empty((self.win.con.offHeight, self.win.con.offWidth, 3), dtype=np.uint8)
        depth = np.empty((self.win.con.offHeight, self.win.con.offWidth), dtype=np.float32)

        for name, cam in self._obs_cams.items():
            mujoco.mjv_updateScene(self.win.m, self.win.d, self.win.opt, None, cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.win.scn)
            mujoco.mjr_render(rect, self.win.scn, self.win.con)
            mujoco.mjr_readPixels(rgb, depth, rect, self.win.con)
            out[name] = rgb.copy()

        # Caller will switch back to WINDOW before on-screen draw
        return out


class DualViewportWrapper(gym.Wrapper):
    """
    Runs a GLFW split-viewport window, and injects image obs ('front','wrist')
    captured offscreen using the SAME MuJoCo context.
    """
    def __init__(self, env: gym.Env, image_size=(128, 128)):
        super().__init__(env)
        m, d = env.unwrapped.model, env.unwrapped.data
        self.win = SplitViewportWindow(m, d)
        self.cap = OffscreenObsMixin(self.win, size=image_size)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        imgs = self.cap.capture_images()
        obs = self._inject_images(obs, imgs)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        imgs = self.cap.capture_images()
        obs = self._inject_images(obs, imgs)

        # Draw dual view
        self.win.make_current()
        self.win.render_dual()
        self.win.swap()
        return obs, r, term, trunc, info

    def close(self):
        try:
            self.win.close()
        finally:
            return super().close()

    @staticmethod
    def _ensure_pixels_dict(obs):
        if not isinstance(obs, dict):
            obs = {"state": obs}
        if "pixels" not in obs:
            obs["pixels"] = {}
        return obs

    @staticmethod
    def _inject_images(obs, imgs):
        obs = DualViewportWrapper._ensure_pixels_dict(obs)
        obs["pixels"]["front"] = imgs.get("front")
        obs["pixels"]["wrist"] = imgs.get("handcam_rgb")
        return obs
