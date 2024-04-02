import time
from enum import Enum

import numpy
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from agents import ActorType
from algorithms.ReplayBuffer import GenericTrajectoryBuffer
from utils.RunningAverage import StepCounter, RunningAverageWindow
from utils.TimeEstimator import PPOTimeEstimator


class PPOAgent:
    def __init__(self, state_dim, action_dim, config, action_type):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.network = None
        self.memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.algorithm = None
        self.action_type = action_type

        self.step_counter = StepCounter(int(config.steps * 1e6))
        self.reward_avg = RunningAverageWindow(100)
        self.time_estimator = PPOTimeEstimator(self.step_counter.limit)
        self.best_agent_score = 0.

    def encode_state(self, state):
        raise NotImplementedError

    def get_action(self, state):
        value, action, probs = self.network(state)

        return value.detach(), action, probs.detach()

    def convert_action(self, action):
        if self.action_type == ActorType.discrete:
            a = torch.argmax(action, dim=1).numpy()
            return a
        if self.action_type == ActorType.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == ActorType.multibinary:
            return torch.argmax(action, dim=1).numpy()

    def initialize_analysis(self):
        raise NotImplementedError

    def step(self, env, agent_state):
        raise NotImplementedError

    def check_terminal_states(self, env, agent_state, analytics, trial, name):
        env_indices = numpy.nonzero(numpy.squeeze(agent_state.done, axis=1))[0]
        stats = analytics.reset(env_indices)
        self.step_counter.update(self.config.n_env)

        for i, index in enumerate(env_indices):
            agent_score = stats['re'].sum[i] / stats['re'].step[i]
            self.reward_avg.update(stats['re'].sum[i])

            if self.best_agent_score < agent_score:
                self.best_agent_score = agent_score
                self.save('./models/{0:s}'.format(name))

            self.print_step_info(trial, stats, i)
            print(self.time_estimator)
            agent_state.next_state[i], metadata = env.reset(index)

    def print_step_info(self, trial, stats, i):
        raise NotImplementedError

    def update_analysis(self, agent_state, analytics):
        raise NotImplementedError

    def train(self, agent_state):
        raise NotImplementedError

    def training_loop(self, env, name, trial, agent_state):
        s = numpy.zeros((self.config.n_env,) + env.observation_space.shape, dtype=numpy.float32)
        for i in range(self.config.n_env):
            s[i], metadata = env.reset(i)

        agent_state.state = self.encode_state(s)

        analytics = self.initialize_analysis()

        while self.step_counter.running():
            self.step(env, agent_state)
            self.check_terminal_states(env, agent_state, analytics, trial, name)
            self.train(agent_state)
            self.update_analysis(agent_state, analytics)
            self.time_estimator.update(self.config.n_env)

        print('Saving data...{0:s}'.format(name))
        analytics.reset(numpy.array(range(self.config.n_env)))
        save_data = analytics.finalize()
        numpy.save('ppo_{0:s}'.format(name), save_data)
        analytics.clear()
        env.close()

    def inference_loop(self, env, name, trial, agent_state):
        s = numpy.zeros((self.config.n_env,) + env.observation_space.shape, dtype=numpy.float32)
        for i in range(self.config.n_env):
            s[i], metadata = env.reset(i)

        agent_state.state = self.encode_state(s)

        analytics = self.initialize_analysis()

        while self.step_counter.running():
            self.step(env, agent_state)
            self.check_terminal_states(env, agent_state, analytics, trial, name)
            self.time_estimator.update(self.config.n_env)

        analytics.clear()
        env.close()

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth', map_location='cpu'))


class AgentMode(Enum):
    TRAINING = 0
    INFERENCE = 1


class PPOAgentBase:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.algorithm = None
        self.optimizer = None

        self.name = None
        self.action_type = None
        self.analytics = None
        self.info = None

        self.memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)

        self.step_counter = StepCounter(int(config.steps * 1e6))
        self.reward_avg = RunningAverageWindow(100)
        self.time_estimator = PPOTimeEstimator(self.step_counter.limit)
        self.best_agent_score = -1.

    def _initialize_env(self, env):
        s = numpy.zeros((self.config.n_env,) + env.observation_space.shape, dtype=numpy.float32)
        for i in range(self.config.n_env):
            s[i], metadata = env.reset(i)

        return s

    def _encode_state(self, state):
        raise NotImplementedError

    def _initialize_analysis(self):
        raise NotImplementedError

    def _initialize_info(self, trial):
        raise NotImplementedError

    def _step(self, env, trial, state, mode: AgentMode):
        raise NotImplementedError

    def _check_terminal_states(self, env, mode, done, next_state):
        env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
        self.step_counter.update(self.config.n_env)
        stats = self.analytics.reset(env_indices)

        for i, index in enumerate(env_indices):
            agent_score = stats['re'].sum[i] / stats['re'].step[i]
            self.reward_avg.update(stats['re'].sum[i])

            if mode == AgentMode.TRAINING and self.best_agent_score < agent_score:
                self.best_agent_score = agent_score
                self.save('./models/{0:s}'.format(self.name))

            self.info.print(stats, i)
            print(self.time_estimator)
            next_state[i], metadata = env.reset(index)

    # def _convert_action(self, action):
    #     if self.action_type == ActorType.discrete:
    #         a = torch.argmax(action, dim=1).numpy()
    #         return a
    #     if self.action_type == ActorType.continuous:
    #         return action.squeeze(0).numpy()
    #     if self.action_type == ActorType.multibinary:
    #         return torch.argmax(action, dim=1).numpy()

    def training_loop(self, env, name, trial):
        self.name = name
        self.info = self._initialize_info(trial)
        self.analytics = self._initialize_analysis()

        state = self._encode_state(self._initialize_env(env))

        while self.step_counter.running():
            state, done = self._step(env, trial, state, AgentMode.TRAINING)
            self.time_estimator.update(self.config.n_env)

        print('Saving data...{0:s}'.format(name))
        self.analytics.reset(numpy.array(range(self.config.n_env)))
        save_data = self.analytics.finalize()
        numpy.save('ppo_{0:s}'.format(name), save_data)
        self.analytics.clear()
        env.close()

    def inference_loop(self, env, name, trial):
        video_path = name + '.mp4'
        video_recorder = VideoRecorder(env.envs_list[0], video_path, enabled=video_path is not None)

        self.info = self._initialize_info(trial)
        self.analytics = self._initialize_analysis()

        state = self._encode_state(self._initialize_env(env))
        stop = False

        while not stop:
            env.render(0)
            video_recorder.capture_frame()
            state, done = self._step(env, trial, state, AgentMode.INFERENCE)
            stop = done.item() == 0.
            self.time_estimator.update(self.config.n_env)

        video_recorder.close()

        env.close()

    # video_path = 'ppo_{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
    # video_recorder = VideoRecorder(self._env, video_path, enabled=video_path is not None)
    # state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
    # done = False
    #
    # while not done:
    #     self._env.render()
    #     video_recorder.capture_frame()
    #     _, action0, _ = agent.get_action(state0)
    #     next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
    #     state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
    # video_recorder.close()

    def save(self, path):
        torch.save(self.model.state_dict(), path + '.pth')

    def load(self, path):
        self.model.load_state_dict(torch.load(path + '.pth', map_location='cpu'))

    # Experimental feature
    def _train_ppo(self, memory, indices):
        start = time.time()
        sample = memory.sample(indices, False)

        states = sample.state
        values = sample.value
        actions = sample.action
        probs = sample.prob
        rewards = sample.reward
        dones = sample.mask

        ref_values, adv_values = self.algorithm.calc_advantage(values, rewards, dones, self.config.gamma, self.config.n_env)

        permutation = torch.randperm(self.config.trajectory_size)

        states = states.reshape(-1, *states.shape[2:])[permutation].reshape(-1, self.config.batch_size, *states.shape[2:])
        actions = actions.reshape(-1, *actions.shape[2:])[permutation].reshape(-1, self.config.batch_size, *actions.shape[2:])
        probs = probs.reshape(-1, *probs.shape[2:])[permutation].reshape(-1, self.config.batch_size, *probs.shape[2:])
        adv_values = adv_values.reshape(-1, *adv_values.shape[2:])[permutation].reshape(-1, self.config.batch_size, *adv_values.shape[2:])
        ref_values = ref_values.reshape(-1, *ref_values.shape[2:])[permutation].reshape(-1, self.config.batch_size, *ref_values.shape[2:])

        n = states.shape[0]

        for epoch in range(self.config.ppo_epochs):
            for i in range(n):
                new_values, new_probs = self._ppo_eval(states[i].to(self.config.device))
                self.optimizer.zero_grad()
                loss = self.algorithm.loss(
                    new_values,
                    new_probs,
                    ref_values[i].to(self.config.device),
                    adv_values[i].to(self.config.device),
                    actions[i].to(self.config.device),
                    probs[i].to(self.config.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        end = time.time()
        print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self.config.trajectory_size, self.config.batch_size, self.config.ppo_epochs, end - start))

    def _ppo_eval(self, state):
        raise NotImplementedError
