import torch

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariAgent, PPOAtariRNDAgent, PPOAtariSNDAgent
from config.ConfigBase import ConfigPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel


class ConfigAtari(ConfigPPO):
    def __init__(self, env_name, steps, lr, n_env, gamma, num_threads, device, shift=0):
        super().__init__(steps=steps, lr=lr, n_env=n_env, gamma=gamma, device=device)

        self.num_threads = num_threads
        self.shift = shift

        self.env_name = env_name

        self.input_shape = None
        self.action_dim = None
        self.experiment = None

        self.init_experiment()

    def init_experiment(self):
        print('Creating {0:d} environments'.format(self.n_env))
        env = MultiEnvParallel([WrapperHardAtari(self.env_name) for _ in range(self.n_env)], self.n_env, self.num_threads)

        self.input_shape = env.observation_space.shape
        self.action_dim = env.action_space.n
        self.experiment = ExperimentNEnvPPO(self.env_name, env, self)
        self.experiment.add_preprocess(self.encode_state)

    @staticmethod
    def encode_state(state):
        return torch.tensor(state, dtype=torch.float32)


class ConfigMontezumaBaseline(ConfigAtari):
    def __init__(self, num_threads, device, shift=0):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=0.99, num_threads=num_threads, device=device, shift=shift)

    def run(self, trial):
        agent = PPOAtariAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        self.experiment.run_baseline(agent, trial, self.shift)


class ConfigMontezumaRND(ConfigAtari):
    def __init__(self, num_threads, device, shift=0):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1

    def run(self, trial):
        agent = PPOAtariRNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        self.experiment.run_rnd_model(agent, trial, self.shift)


class ConfigMontezumaSND(ConfigAtari):
    def __init__(self, num_threads, device, shift=0):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1
        self.type = 'vicreg'
        self.desc = 'vicreg'

    def run(self, trial):
        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        self.experiment.run_snd_model(agent, trial, self.shift)
