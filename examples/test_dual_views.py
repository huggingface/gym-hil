import time
import numpy as np
import gymnasium as gym
import gym_hil

from gym_hil.wrappers.viewer_wrapper import DualViewportWrapper


base = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode=None, image_obs=False)
env = DualViewportWrapper(base, image_size=(128, 128))

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
