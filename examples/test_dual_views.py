import time

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401

env = gym.make("gym_hil/PandaPickCubeDualViewGamepad-v0")
obs, info = env.reset()
dummy_action = np.zeros(4, dtype=np.float32)
dummy_action[-1] = 1

try:
    while True:
        obs, r, term, trunc, info = env.step(dummy_action)

        if info.get("viewer_closed"):
            break

        if term or trunc:
            print("Episode ended, resetting environment")
            obs, _ = env.reset()

        time.sleep(0.02)
finally:
    env.close()
