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

from gym_hil.wrappers.intervention_utils import GamepadController, load_controller_config


def test_load_controller_config():
    """Test that load_controller_config loads correctly."""
    # Test loading default controller with no config_path (should use bundled config)
    config = load_controller_config("default")

    assert config is not None
    assert isinstance(config, dict)
    assert "axes" in config
    assert "buttons" in config
    assert "axis_inversion" in config

    # Test loading existing controller with no config_path (should use bundled config)
    xbox_config = load_controller_config("Xbox 360 Controller")
    assert xbox_config is not None
    assert isinstance(xbox_config, dict)
    assert "axes" in xbox_config
    assert "buttons" in xbox_config
    assert "axis_inversion" in xbox_config

    # Test loading Logitech RumblePad 2 USB controller from bundled config
    rumblepad_config = load_controller_config("Logitech RumblePad 2 USB")
    assert rumblepad_config is not None
    assert isinstance(rumblepad_config, dict)
    assert "axes" in rumblepad_config
    assert "buttons" in rumblepad_config
    assert "axis_inversion" in rumblepad_config

    # Test loading non-existent controller returns default
    unknown_config = load_controller_config("Unknown Controller")
    assert unknown_config == config

    # Test that when config_path is None, it falls back to bundled package config
    config_with_none = load_controller_config("default", None)
    assert config_with_none == config


def test_gamepad_controller_no_device_is_noop():
    """update() and get_deltas() must not crash when no gamepad was detected.

    When no gamepad is connected, start() sets running=False and leaves
    controller_config=None, so both methods should degrade gracefully instead
    of raising AttributeError on the missing config.
    """
    controller = GamepadController()
    # Reproduce the state start() leaves behind when no gamepad is present.
    controller.running = False
    assert controller.controller_config is None

    controller.update()
    assert controller.get_deltas() == (0.0, 0.0, 0.0)
    assert controller.should_intervene() is False
