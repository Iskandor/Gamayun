import numpy
import torch

from agents import TYPE
from algorithms.ReplayBuffer import GenericTrajectoryBuffer
from modules.PPO_Modules import PPOSimpleNetwork
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

    def get_action(self, state):
        value, action, probs = self.network(state)

        return value.detach(), action, probs.detach()

    def convert_action(self, action):
        if self.action_type == TYPE.discrete:
            a = torch.argmax(action, dim=1).numpy()
            return a
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == TYPE.multibinary:
            return torch.argmax(action, dim=1).numpy()

    def initialize_analysis(self):
        raise NotImplementedError

    def step(self, env, agent_state):
        raise NotImplementedError

    def check_terminal_states(self, env, agent_state, analysis, trial, name):
        env_indices = numpy.nonzero(numpy.squeeze(agent_state.done, axis=1))[0]
        stats = analysis.reset(env_indices)
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

    def update_analysis(self, agent_state, analysis):
        raise NotImplementedError

    def train(self, agent_state):
        raise NotImplementedError

    def training_loop(self, env, name, trial, agent_state):
        s = numpy.zeros((self.config.n_env,) + env.observation_space.shape, dtype=numpy.float32)
        for i in range(self.config.n_env):
            s[i], metadata = env.reset(i)

        agent_state.state = self.config.encode_state(s)

        analytic = self.initialize_analysis()

        while self.step_counter.running():
            self.step(env, agent_state)
            self.check_terminal_states(env, agent_state, analytic, trial, name)
            self.train(agent_state)
            self.update_analysis(agent_state, analytic)
            self.time_estimator.update(self.config.n_env)

        print('Saving data...{0:s}'.format(name))
        analytic.reset(numpy.array(range(self.config.n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0:s}'.format(name), save_data)
        analytic.clear()
        env.close()

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth', map_location='cpu'))
