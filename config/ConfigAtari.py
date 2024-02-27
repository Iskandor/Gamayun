from agents.atari.PPOAtariAgent import PPOAtariAgent
from agents.atari.PPOAtariICMAgent import PPOAtariICMAgent
from agents.atari.PPOAtariRNDAgent import PPOAtariRNDAgent
from agents.atari.PPOAtariSEERAgent import PPOAtariSEERAgent
from agents.atari.PPOAtariSNDAgent import PPOAtariSNDAgent
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

        agent = PPOAtariAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaRND(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=0.5, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 1
        self.feature_dim = 512

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariRNDAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaSNDBaseline(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=128, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.feature_dim = 512
        self.motivation_lr = 1e-4
        self.motivation_scale = 0.25
        self.type = 'vicreg'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, 'v1', trial)
        print(name)

        agent = PPOAtariSNDAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaSND2(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=8, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.feature_dim = 512
        self.motivation_lr = 1e-4
        self.motivation_scale = 0.25
        self.type = 'vicreg2'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)
        print(name)

        agent = PPOAtariSNDAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)
        print(name)

        agent = PPOAtariSNDAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, trial)


class ConfigMontezumaSND_TP(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.feature_dim = 512
        self.motivation_lr = 1e-4
        self.motivation_scale = 0.5
        self.type = 'tp'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '02', trial)
        print(name)

        agent = PPOAtariSNDAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaSNDSpac(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=128, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.feature_dim = 512
        self.motivation_lr = 1e-4
        self.motivation_scale = .25
        self.type = 'spacvicreg'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, 'std_spacvicreg', trial)
        print(name)

        agent = PPOAtariSNDAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaICM(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.feature_dim = 512
        self.motivation_lr = 1e-4
        self.motivation_scale = 1

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariICMAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaSEER(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.feature_dim = 512
        self.hidden_dim = 128
        self.motivation_lr = 1e-4
        self.motivation_scale = 1
        self.type = 'seer'

        self.pi = 1.
        self.eta = 0.01

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariSEERAgent(self)
        agent.training_loop(self.env, name, trial)
