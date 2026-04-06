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

"""
Run a SmolVLA checkpoint inside a gym-hil environment.

This example expects a SmolVLA checkpoint that was fine-tuned on a dataset whose
observations and actions are compatible with gym-hil. The foundation checkpoint
`lerobot/smolvla_base` is a base model, so it usually needs fine-tuning on a
gym-hil dataset before it can solve these tasks.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

import gym_hil  # noqa: F401

try:
    from lerobot.configs.types import FeatureType
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.observation_processor import VanillaObservationProcessorStep
except ImportError as exc:
    raise SystemExit(
        "This example requires LeRobot with SmolVLA dependencies. "
        "Install it with `pip install \"lerobot[smolvla]\"` "
        "or from a local checkout with `pip install -e ../lerobot[smolvla]`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SmolVLA checkpoint inside gym-hil")
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Hugging Face Hub id or local path to a SmolVLA checkpoint fine-tuned for gym-hil.",
    )
    parser.add_argument(
        "--env-id",
        default="gym_hil/PandaPickCube-v0",
        help=(
            "Gym environment id. For human interventions, use "
            "`gym_hil/PandaPickCubeKeyboard-v0` or `gym_hil/PandaPickCubeGamepad-v0`."
        ),
    )
    parser.add_argument(
        "--task",
        default="Pick up the cube.",
        help="Language instruction passed to SmolVLA during inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for SmolVLA (`auto`, `cuda`, or `cpu`).",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=100,
        help="Maximum number of steps to run per episode.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Optional sleep in seconds after each environment step.",
    )
    parser.add_argument(
        "--camera-key-map",
        action="append",
        default=[],
        metavar="SRC=DST",
        help=(
            "Rename processed camera keys before preprocessing, for example "
            "`front=observation.images.camera1`. This is useful if the checkpoint "
            "was trained with camera names different from gym-hil's `front`/`wrist`."
        ),
    )
    parser.add_argument(
        "--controller-config",
        type=str,
        default=None,
        help="Optional gamepad controller mapping JSON passed through to gym-hil.",
    )
    parser.add_argument(
        "--render-mode",
        default="none",
        choices=("none", "human", "rgb_array"),
        help="Environment render mode. Use `none` for headless quantitative evaluation.",
    )
    parser.add_argument(
        "--fill-missing-cameras",
        action="store_true",
        help=(
            "If the checkpoint expects more image keys than gym-hil provides after camera remapping, "
            "synthesize black images for the missing keys."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested_device)


def normalize_image_key(key: str) -> str:
    if key.startswith("observation.images."):
        return key
    return f"observation.images.{key}"


def parse_camera_key_map(pairs: list[str]) -> dict[str, str]:
    camera_key_map: dict[str, str] = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid camera mapping `{pair}`. Expected the format `source=destination`."
            )
        source, target = pair.split("=", maxsplit=1)
        source = normalize_image_key(source.strip())
        target = normalize_image_key(target.strip())

        if source in camera_key_map and camera_key_map[source] != target:
            raise ValueError(
                f"Camera key `{source}` is mapped more than once: "
                f"`{camera_key_map[source]}` and `{target}`."
            )
        camera_key_map[source] = target

    return camera_key_map


def get_expected_visual_keys(policy: SmolVLAPolicy) -> set[str]:
    input_features = policy.config.input_features or {}
    return {
        key
        for key, feature in input_features.items()
        if feature.type == FeatureType.VISUAL
    }


def get_expected_action_dim(policy: SmolVLAPolicy) -> int | None:
    output_features = policy.config.output_features or {}
    action_feature = output_features.get("action")
    if action_feature is None:
        return None
    return action_feature.shape[0]


def build_policy_batch(
    observation: dict,
    task: str,
    observation_processor: VanillaObservationProcessorStep,
    camera_key_map: dict[str, str],
    expected_visual_keys: set[str],
    fill_missing_cameras: bool,
) -> dict:
    processed_observation = observation_processor.observation(observation)
    remapped_observation: dict[str, torch.Tensor] = {}

    for key, value in processed_observation.items():
        target_key = camera_key_map.get(key, key)
        if target_key in remapped_observation and target_key != key:
            raise ValueError(
                f"Multiple cameras map to the same target key `{target_key}`. "
                "Please provide a one-to-one `--camera-key-map`."
            )
        remapped_observation[target_key] = value

    available_visual_keys = {
        key for key in remapped_observation if key.startswith("observation.images.")
    }
    missing_visual_keys = expected_visual_keys - available_visual_keys
    if fill_missing_cameras and missing_visual_keys:
        if not available_visual_keys:
            raise ValueError(
                "Cannot synthesize missing cameras because no image observations are available in the batch."
            )

        template_key = sorted(available_visual_keys)[0]
        template_value = remapped_observation[template_key]
        for missing_key in sorted(missing_visual_keys):
            remapped_observation[missing_key] = torch.zeros_like(template_value)

    remapped_observation["task"] = task
    return remapped_observation


def validate_policy_compatibility(
    policy: SmolVLAPolicy,
    env: gym.Env,
    sample_batch: dict,
) -> None:
    expected_visual_keys = get_expected_visual_keys(policy)
    available_visual_keys = {
        key for key in sample_batch if key.startswith("observation.images.")
    }
    missing_visual_keys = expected_visual_keys - available_visual_keys
    if missing_visual_keys:
        raise ValueError(
            "The SmolVLA checkpoint expects image keys that the prepared gym-hil batch does not provide.\n"
            f"Missing keys: {sorted(missing_visual_keys)}\n"
            f"Available keys: {sorted(available_visual_keys)}\n"
            "If the checkpoint was trained with different camera names, pass "
            "`--camera-key-map source=target` to rename gym-hil cameras."
        )

    expected_action_dim = get_expected_action_dim(policy)
    if expected_action_dim is not None:
        env_action_dim = int(np.prod(env.action_space.shape))
        if expected_action_dim != env_action_dim:
            raise ValueError(
                "The SmolVLA checkpoint action dimension does not match the selected gym-hil environment.\n"
                f"Checkpoint action dim: {expected_action_dim}\n"
                f"Environment action dim: {env_action_dim}\n"
                "Use a checkpoint fine-tuned on a gym-hil dataset recorded with the same action space."
            )

    if "observation.state" not in sample_batch:
        raise ValueError("The processed gym-hil observation does not contain `observation.state`.")


def action_tensor_to_numpy(action: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(action, np.ndarray):
        return action.astype(np.float32, copy=False)

    action = action.detach().cpu()
    if action.ndim == 2:
        action = action.squeeze(0)
    return action.numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    camera_key_map = parse_camera_key_map(args.camera_key_map)

    print(f"Loading SmolVLA checkpoint from: {args.policy_path}")
    print(f"Using device: {device}")

    env_kwargs = {"image_obs": True}
    if args.render_mode != "none":
        env_kwargs["render_mode"] = args.render_mode
    if args.controller_config:
        env_kwargs["controller_config_path"] = args.controller_config

    env = gym.make(args.env_id, **env_kwargs)
    observation_processor = VanillaObservationProcessorStep()

    try:
        policy = SmolVLAPolicy.from_pretrained(args.policy_path)
        policy.to(device)
        policy.eval()

        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            args.policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        expected_visual_keys = get_expected_visual_keys(policy)
        sample_observation, _ = env.reset(seed=args.seed)
        sample_batch = build_policy_batch(
            sample_observation,
            args.task,
            observation_processor,
            camera_key_map,
            expected_visual_keys,
            args.fill_missing_cameras,
        )
        validate_policy_compatibility(policy, env, sample_batch)

        print(f"Running {args.env_id} for {args.episodes} episode(s).")
        print(f"Expected visual keys: {sorted(expected_visual_keys)}")
        print(f"Environment action space: {env.action_space}")
        successes = 0
        returns: list[float] = []
        completed_episodes = 0

        for episode_idx in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_idx)
            policy.reset()
            episode_return = 0.0

            for step_idx in range(args.steps_per_episode):
                policy_batch = build_policy_batch(
                    observation,
                    args.task,
                    observation_processor,
                    camera_key_map,
                    expected_visual_keys,
                    args.fill_missing_cameras,
                )
                policy_input = preprocess(policy_batch)
                action = policy.select_action(policy_input)
                action = postprocess(action)
                action = action_tensor_to_numpy(action)

                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward

                if info.get("is_intervention", False):
                    print(f"[episode {episode_idx + 1} step {step_idx + 1}] human intervention active")

                if terminated or truncated:
                    success = bool(info.get("succeed", False))
                    print(
                        f"Episode {episode_idx + 1} finished after {step_idx + 1} step(s) "
                        f"with return={episode_return:.3f}, success={success}"
                    )
                    returns.append(episode_return)
                    completed_episodes += 1
                    successes += int(success)
                    break

                if args.step_delay > 0:
                    time.sleep(args.step_delay)
            else:
                print(
                    f"Episode {episode_idx + 1} reached the step limit "
                    f"with return={episode_return:.3f}"
                )
                returns.append(episode_return)
                completed_episodes += 1
        if completed_episodes:
            success_rate = successes / completed_episodes
            average_return = float(np.mean(returns))
            print(
                f"Summary: episodes={completed_episodes}, successes={successes}, "
                f"success_rate={success_rate:.3f}, average_return={average_return:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
