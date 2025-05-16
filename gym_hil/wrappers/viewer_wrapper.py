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


class PassiveViewerWrapper(gym.Wrapper):
    """Gym wrapper that opens a passive MuJoCo viewer automatically.

    The wrapper starts a MuJoCo viewer in passive mode as soon as the
    environment is created so the user no longer needs to use
    ``mujoco.viewer.launch_passive`` or any context–manager boiler-plate.

    Args:
        viewer_type: Whether to use a single or dual viewer. If dual, two 
        viewers will be opened. You can switch between fixed cameras using 
        the [, ] keys or access a free camera mode for manual adjustments using the Esc key 
        show_left_ui: Whether to show the left UI
        show_right_ui: Whether to show the right UI

    The viewer is kept in sync after every ``reset`` and ``step`` call and is
    closed automatically when the environment itself is closed or deleted.
    """

    def __init__(
        self,
        env: gym.Env,
        viewer_type: str = "single",
        show_left_ui: bool = False,
        show_right_ui: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.viewer_type = viewer_type
        # Launch the interactive viewer.  We expose *model* and *data* from the
        # *unwrapped* environment to make sure we operate on the base MuJoCo
        # objects even if other wrappers have been applied before this one.
        if self.viewer_type == "single":
            self._viewer = mujoco.viewer.launch_passive(
                env.unwrapped.model,
                env.unwrapped.data,
                # show_left_ui=show_left_ui,
                # show_right_ui=show_right_ui,
            )
        elif self.viewer_type == "dual":
            self._viewer_1 = mujoco.viewer.launch_passive(
                env.unwrapped.model,
                env.unwrapped.data,
                # show_left_ui=show_left_ui,
                # show_right_ui=show_right_ui,
            )
            self._viewer_2 = mujoco.viewer.launch_passive(
                env.unwrapped.model,
                env.unwrapped.data,
                # show_left_ui=show_left_ui,
                # show_right_ui=show_right_ui,
            )
        # Make sure the first frame is rendered.
        self._sync()

    # ---------------------------------------------------------------------
    # Gym API overrides

    def reset(self, **kwargs):  # type: ignore[override]
        observation, info = self.env.reset(**kwargs)
        self._sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._sync()
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
            if self.viewer_type == "single":
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
            if self.viewer_type == "single":
                self._viewer.close()
            elif self.viewer_type == "dual":
                self._viewer_1.close()
                self._viewer_2.close()
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
                if self.viewer_type == "single":
                    self._viewer.close()
                elif self.viewer_type == "dual":
                    self._viewer_1.close()
                    self._viewer_2.close()
            except Exception:
                pass

    def _sync(self):
        if self.viewer_type == "single":
            self._viewer.sync()
        elif self.viewer_type == "dual":
            self._viewer_1.sync()
            self._viewer_2.sync()

