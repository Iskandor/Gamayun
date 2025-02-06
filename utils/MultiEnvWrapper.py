import threading
import multiprocessing
import time
from concurrent.futures.thread import ThreadPoolExecutor

import numpy
import gym
import torch


class MultiEnvParallel:
    def __init__(self, envs_list, envs_count, thread_count=4):
        dummy_env = envs_list[0]

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

        self.envs_list = envs_list
        self.envs_count = envs_count
        self.threads_count = thread_count
        if envs_count >= thread_count:
            self.envs_per_thread = envs_count // thread_count
        else:
            self.envs_per_thread = envs_count

        self.observations = numpy.zeros((envs_count,) + self.observation_space.shape, dtype=numpy.float32)
        # a = [None] * n_env
        self.rewards = numpy.zeros((envs_count, 1), dtype=numpy.float32)
        self.dones = numpy.zeros((envs_count, 1), dtype=numpy.float32)
        self.truncs = numpy.zeros((envs_count, 1), dtype=numpy.float32)
        self.infos = [None] * envs_count

    def close(self):
        for i in range(self.envs_count):
            self.envs_list[i].close()

    def reset(self, env_id):
        result = self.envs_list[env_id].reset()

        if not isinstance(result, tuple):
            result = (result, None)

        return result

    def render(self, env_id):
        pass

    def _step(self, param):
        index, action = param
        obs, reward, done, trunc, info = self.envs_list[index].step(action)

        self.observations[index] = obs
        self.rewards[index] = reward
        self.dones[index] = done
        self.truncs[index] = trunc
        self.infos[index] = info

    def step(self, actions):
        p = [(i, a) for i, a in zip(range(self.envs_count), actions)]
        with ThreadPoolExecutor(max_workers=self.threads_count) as executor:
            executor.map(self._step, p)

        obs = self.observations
        reward = self.rewards
        done = self.dones
        trunc = self.truncs
        info = {}
        for i in self.infos:
            if i is not None:
                for k in i:
                    if k not in info:
                        info[k] = []
                    info[k].append(i[k])

        for k in info:
            info[k] = numpy.stack(info[k])

        return obs, reward, done, trunc, info

    def __str__(self):
        result = ''

        result += "MultiEnvParallel\n"
        result += "envs_count      = {0}\n".format(self.envs_count)
        result += "threads_count   = {0}\n".format(self.threads_count)
        result += "envs_per_thread = {0}\n".format(self.envs_per_thread)
        result += "\n\n"

        return result
