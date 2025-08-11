import time
import numpy as np
import gymnasium as gym
import gym_hil

env = gym.make("gym_hil/PandaPickCubeDualView-v0", render_mode="rgb_array", image_obs=True)
obs, info = env.reset()
a = np.zeros(env.action_space.shape, dtype=np.float32)

try:
    while True:
        obs, r, term, trunc, info = env.step(a)

        if term or trunc:
            obs, info = env.reset()
        time.sleep(0.02)
finally:
    env.close()
