import torch

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariAgent, PPOAtariRNDAgent, PPOAtariSNDAgent, PPOAtariICMAgent, PPOAtariSPAgent
from config.ConfigBase import ConfigPPO
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel


class ConfigAtari(ConfigPPO):
    def __init__(self, env_name, steps, lr, n_env, gamma, num_threads, device, shift):
        super().__init__(steps=steps, lr=lr, n_env=n_env, gamma=gamma, device=device)

        self.num_threads = num_threads
        self.shift = shift

        self.env_name = env_name

        self.input_shape = None
        self.action_dim = None
        self.env = None
        self.experiment = None

        self.init_environment()

    def init_environment(self):
        print('Creating {0:d} environments'.format(self.n_env))
        self.env = MultiEnvParallel([WrapperHardAtari(self.env_name) for _ in range(self.n_env)], self.n_env, self.num_threads)

        self.input_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

    def encode_state(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.device)


class ConfigMontezumaBaseline(ConfigAtari):
    def __init__(self, num_threads, device, shift):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=0.99, num_threads=num_threads, device=device, shift=shift)

    def run(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariAgent.AgentState())


class ConfigMontezumaRND(ConfigAtari):
    def __init__(self, num_threads, device, shift):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1

    def run(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariRNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariRNDAgent.AgentState())


class ConfigMontezumaSND(ConfigAtari):
    def __init__(self, num_threads, device, shift):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift)

        self.motivation_lr = 1e-4
        self.motivation_eta = .25
        self.type = 'vicreg'

    def run(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, 'big_full_random_aug', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())


class ConfigMontezumaICM(ConfigAtari):
    def __init__(self, num_threads, device, shift):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1

    def run(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariICMAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariICMAgent.AgentState())


class ConfigMontezumaSP(ConfigAtari):
    def __init__(self, num_threads, device, shift):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1

    def run(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariSPAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSPAgent.AgentState())

