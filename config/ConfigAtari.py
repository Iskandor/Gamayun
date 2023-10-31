import torch

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariAgent
from config.ConfigBase import ConfigPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel


class ConfigMontezumaBaseline(ConfigPPO):
    def __init__(self, num_threads, device, shift=0):
        super().__init__(steps=32, lr=0.0001, n_env=32, gamma=0.99, device=device)

        trial = 1
        env_name = 'MontezumaRevengeNoFrameskip-v4'

        print('Creating {0:d} environments'.format(self.n_env))
        env = MultiEnvParallel([WrapperHardAtari(env_name) for _ in range(self.n_env)], self.n_env, num_threads)

        input_shape = env.observation_space.shape
        action_dim = env.action_space.n

        print('Start training')
        experiment = ExperimentNEnvPPO(env_name, env, self)

        experiment.add_preprocess(self.encode_state)
        agent = PPOAtariAgent(input_shape, action_dim, self, TYPE.discrete)
        experiment.run_baseline(agent, trial, shift)

        env.close()

    @staticmethod
    def encode_state(state):
        return torch.tensor(state, dtype=torch.float32)
