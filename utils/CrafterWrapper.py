import crafter
import gymnasium as gym
import numpy
import cv2


class CommonWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.state = None
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=numpy.float32)

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        self.state = self._wrap_state(state)
        return self.state, None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.state = self._wrap_state(state)
        info = info["inventory"]
        return self.state, reward, done, None, info

    def render(self, size=512):
        im = numpy.swapaxes(self.state, 2, 0)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("crafter", im)
        cv2.waitKey(1)

    def _wrap_state(self, x):
        x = numpy.swapaxes(x, 0, 2)
        x = numpy.array(x / 255.0, dtype=numpy.float32)
        return x


def WrapperCrafter(env_name="crafter"):
    env = crafter.Env()
    env = CommonWrapper(env)

    return env
