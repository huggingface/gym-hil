# Controller Configuration

This document explains how to use the controller configuration feature to customize gamepad button and axis mappings.

## Overview

The gym-hil package now supports loading gamepad button and axis mappings from a JSON configuration file. This allows you to customize the controls for different types of gamepads.

## Configuration File Format

The configuration file is a JSON file with the following structure:

```json
{
  "default": {
    "axes": {
      "left_x": 0,
      "left_y": 1,
      "right_x": 2,
      "right_y": 3
    },
    "buttons": {
      "a": 1,
      "b": 2,
      "x": 0,
      "y": 3,
      "lb": 4,
      "rb": 5,
      "lt": 6,
      "rt": 7
    },
    "axis_inversion": {
      "left_x": false,
      "left_y": true,
      "right_x": false,
      "right_y": true
    }
  },
  "Xbox 360 Controller": {
    "axes": {
      "left_x": 0,
      "left_y": 1,
      "right_x": 3,
      "right_y": 4
    },
    "buttons": {
      "a": 0,
      "b": 1,
      "x": 2,
      "y": 3,
      "lb": 4,
      "rb": 5,
      "lt": 6,
      "rt": 7
    },
    "axis_inversion": {
      "left_x": false,
      "left_y": true,
      "right_x": false,
      "right_y": true
    }
  }
}
```

The configuration file contains mappings for different controller types, identified by their name. The `default` mapping is used if no specific mapping is found for the connected controller.

Each controller mapping contains:

- `axes`: Mapping of axis names to axis indices
- `buttons`: Mapping of button names to button indices
- `axis_inversion`: Whether each axis should be inverted

## Using the Configuration File

You can specify the path to the configuration file when running the teleoperation script:

```bash
python examples/test_teleoperation.py --controller-config path/to/controller_config.json
```

### Using with Installed Package

When gym-hil is installed as a package with pip, the system will look for the controller configuration file in the following locations (in order):

1. The path specified with the `--controller-config` argument
2. A file named `controller_config.json` in the current working directory
3. The default configuration file included with the package

This allows you to create your own configuration file without modifying the installed package. Simply create a file named `controller_config.json` in your project directory, or specify a custom path with the `--controller-config` argument.

### Using in Your Own Code

When using gym-hil in your own code, you can pass the controller configuration path to the environment:

```python
import gymnasium as gym
import gym_hil

env = gym.make(
    "gym_hil/PandaPickCubeGamepad-v0",
    controller_config_path="path/to/controller_config.json",
    # other parameters...
)
```

If no configuration file is specified, the system will look for a file named `controller_config.json` in the current working directory, and then fall back to the default configuration included with the package.

## Default Controls

The default controls are:

- Left analog stick: Move in X-Y plane
- Right analog stick (vertical): Move in Z axis
- RB button: Toggle intervention mode
- LT button: Close gripper
- RT button: Open gripper
- Y/Triangle button: End episode with SUCCESS
- A/Cross button: End episode with FAILURE
- X/Square button: Rerecord episode

## Adding New Controller Mappings

To add a new controller mapping:

1. Connect your controller and run the teleoperation script
2. Note the controller name printed in the console
3. Add a new entry to the configuration file with the controller name
4. Customize the axis and button mappings as needed
5. Run the teleoperation script with the updated configuration file

## Troubleshooting

If your controller is not working correctly:

1. Check that the controller is properly connected
2. Verify that the controller name in the configuration file matches the name printed in the console
3. Adjust the axis and button mappings as needed
4. Try using the `--use-keyboard` option to use keyboard controls instead
