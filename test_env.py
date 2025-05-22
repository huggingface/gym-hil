# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import SyncVectorEnv

import gym_hil  # noqa: F401


def test_franka(image_obs):
    env = gym.make("gym_hil/PandaPickCube-v0", image_obs=image_obs)
    check_env(env.unwrapped)


def _wrap():
    env = gym.make("gym_hil/PandaPickCubeViewer-v0", image_obs=True)
    return env


if __name__ == "__main__":
    # env = gym.make("gym_franka/PandaPickCube-v0", image_obs=True)
    env = SyncVectorEnv([_wrap])
    print(env.observation_space)
    obs, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(obs, reward, done, truncated, info)
    env.close()
    # test_franka(image_obs=True)
