import torch

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariAgent, PPOAtariRNDAgent, PPOAtariSNDAgent, PPOAtariICMAgent, PPOAtariSPAgent
from config.ConfigBase import ConfigPPO
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel


class ConfigAtari(ConfigPPO):
    def __init__(self, env_name, steps, lr, n_env, gamma, num_threads, device, shift, path):
        super().__init__(steps=steps, lr=lr, n_env=n_env, gamma=gamma, device=device)

        self.num_threads = num_threads
        self.shift = shift
        self.path = path

        self.env_name = env_name

        self.input_shape = None
        self.action_dim = None
        self.env = None

        self.init_environment()

    def init_environment(self):
        print('Creating {0:d} environments'.format(self.n_env))
        self.env = MultiEnvParallel([WrapperHardAtari(self.env_name) for _ in range(self.n_env)], self.n_env, self.num_threads)

        self.input_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n


class ConfigMontezumaBaseline(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=0.99, num_threads=num_threads, device=device, shift=shift, path=path)

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariAgent.AgentState())


class ConfigMontezumaRND(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=0.5, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariRNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariRNDAgent.AgentState())


class ConfigMontezumaSNDBaseline(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=128, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = 0.25
        self.type = 'vicreg'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, 'v1', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())


class ConfigMontezumaSND2(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=8, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = 0.25
        self.type = 'vicreg2'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.training_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())


class ConfigMontezumaSND_VICL(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=0.5, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = 0.25
        self.type = 'vicregl'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())


class ConfigMontezumaSND_TP(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = 0.5
        self.type = 'tp'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '02', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())


class ConfigMontezumaSNDSpac(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=128, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = .25
        self.type = 'spacvicreg'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, 'std_spacvicreg', trial)
        print(name)

        agent = PPOAtariSNDAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSNDAgent.AgentState())


class ConfigMontezumaICM(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = 1

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariICMAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariICMAgent.AgentState())


class ConfigMontezumaSEER(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_eta = .1
        self.type = 'seer'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariSPAgent(self.input_shape, self.action_dim, self, TYPE.discrete)
        agent.training_loop(self.env, name, trial, PPOAtariSPAgent.AgentState())
