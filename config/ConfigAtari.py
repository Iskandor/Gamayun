import yaml

from agents.atari.PPOAtariA2Agent import PPOAtariA2Agent
from agents.atari.PPOAtariAgent import PPOAtariAgent
from agents.atari.PPOAtariDPMAgent import PPOAtariDPMAgent
from agents.atari.PPOAtariFMAgent import PPOAtariFMAgent, ArchitectureType
from agents.atari.PPOAtariICMAgent import PPOAtariICMAgent
from agents.atari.PPOAtariRNDAgent import PPOAtariRNDAgent
from agents.atari.PPOAtariSEERAgent import PPOAtariSEERAgent
from agents.atari.PPOAtariSNDAgent import PPOAtariSNDAgent
from agents.atari.PPOAtariSNDv2Agent import PPOAtariSNDv2Agent
from agents.atari.PPOAtariSNDv3Agent import PPOAtariSNDv3Agent
from agents.atari.PPOAtariSNDv4Agent import PPOAtariSNDv4Agent
from config.ConfigBase import ConfigPPO
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel
from utils.WrapperMontezuma import WrapperMontezuma


class ConfigAtari(ConfigPPO):
    def __init__(self, env_name, steps, lr, n_env, gamma, num_threads, device, shift, path):
        super().__init__(path='ppo_atari.yaml', steps=steps, lr=lr, n_env=n_env, gamma=gamma, device=device)

        self.num_threads = num_threads
        self.shift = shift
        self.path = path

        self.env_name = env_name
        self.render_mode = None

        self.input_shape = None
        self.action_dim = None
        self.feature_dim = 512
        self.ppo_feature_dim = self.feature_dim
        self.env = None

        self.init_environment()

    def init_environment(self):
        print('Creating {0:d} environments'.format(self.n_env))
        self.env = MultiEnvParallel([WrapperHardAtari(self.env_name, render_mode=self.render_mode) for _ in range(self.n_env)], self.n_env, self.num_threads)

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
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=0.5, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 1

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariRNDAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaSNDBaseline(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=8, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 0.25
        self.type = 'vicreg'

    def train(self, trial):
        super().train(trial)

        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)

        agent = PPOAtariSNDAgent(self)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)


class ConfigMontezumaSNDv2(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=8, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 0.25
        self.type = 'm5'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDv2Agent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDv2Agent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)


class ConfigMontezumaSNDv3(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=128, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 1.
        self.type = 'v1m1'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDv3Agent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDv3Agent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)


class ConfigMontezumaSNDv4(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=8, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 0.25
        self.type = 'v1m1'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDv4Agent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSNDv4Agent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)


class ConfigMontezumaICM(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 1

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOAtariICMAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaFMSTDIM(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4',
                         steps=32,
                         lr=1e-4,
                         n_env=64,
                         gamma=[0.998, 0.99],
                         num_threads=num_threads,
                         device=device,
                         shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.eta = 0.01
        self.type = 'st-dim_fm'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(f"Starting training: {name}")

        agent = PPOAtariFMAgent(self, _type = ArchitectureType.ST_DIM)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaFMSTDIM_0_01_32_32(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4',
                         steps=32,
                         lr=1e-4,
                         n_env=32,
                         gamma=[0.998, 0.99],
                         num_threads=num_threads,
                         device=device,
                         shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.eta = 0.01
        self.type = 'st-dim_fm'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(f"Starting training: {name}")

        agent = PPOAtariFMAgent(self, _type = ArchitectureType.ST_DIM)
        agent.training_loop(self.env, name, trial)

class ConfigMontezumaFMSTDIM_0_05_32_32(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4',
                         steps=32,
                         lr=1e-4,
                         n_env=32,
                         gamma=[0.998, 0.99],
                         num_threads=num_threads,
                         device=device,
                         shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.eta = 0.05
        self.type = 'st-dim_fm'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(f"Starting training: {name}")

        agent = PPOAtariFMAgent(self, _type = ArchitectureType.ST_DIM)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaFMSTDIM_0_05_32_64(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4',
                         steps=32,
                         lr=1e-4,
                         n_env=64,
                         gamma=[0.998, 0.99],
                         num_threads=num_threads,
                         device=device,
                         shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.eta = 0.05
        self.type = 'st-dim_fm'

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(f"Starting training: {name}")

        agent = PPOAtariFMAgent(self, _type = ArchitectureType.ST_DIM)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaFMIJEPA_0_01_32_32(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4',
                         steps=32,
                         lr=1e-4,
                         n_env=32,
                         gamma=[0.998, 0.99],
                         num_threads=num_threads,
                         device=device,
                         shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.eta = 0.01
        self.type = 'st-dim_fm'
        self.hidden_dim = self.feature_dim // 4

    def train(self, trial):
        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(f"Starting training: {name}")

        agent = PPOAtariFMAgent(self, _type = ArchitectureType.I_JEPA)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaSEER(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        # render_mode='rgb_array'
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=128, lr=1e-4, n_env=128, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift,
                         path=path)

        self.learned_projection_dim = self.feature_dim
        self.forward_model_dim = self.feature_dim * 8
        self.hidden_dim = self.feature_dim // 4
        # self.hidden_dim = self.feature_dim // 2
        # self.ppo_feature_dim = self.feature_dim * 4

        self.motivation_lr = 1e-4
        self.distillation_scale = 0.25
        self.forward_scale = 0.025
        self.forward_threshold = 1
        self.type = 'asym_v5m4f1'

        self.delta = 0.5
        # self.beta = 0.25
        self.pi = 0.5
        self.eta = 0.01

    def train(self, trial):
        super().train(trial)

        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)

        agent = PPOAtariSEERAgent(self)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariSEERAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)

    def analysis(self, task):
        name = '{0:s}_{1:s}'.format(self.__class__.__name__, self.type)
        print(name)

        agent = PPOAtariSEERAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.analytic_loop(self.env, name, task)


class ConfigMontezumaA2(ConfigAtari):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=32, lr=1e-4, n_env=32, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.hidden_dim = self.feature_dim // 4
        self.motivation_lr = 1e-4
        self.motivation_scale = 1
        self.type = 'sym'

        self.alpha = 0.5
        self.eta = 0.01

    def train(self, trial):
        super().train(trial)

        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)

        agent = PPOAtariA2Agent(self)
        agent.training_loop(self.env, name, trial)


class ConfigMontezumaDPM(ConfigAtari):
    def __init__(self, num_threads, device, shift, path, steps=8, lr=1e-4, n_env=128, gamma=[0.998, 0.99]):
        super().__init__(env_name='MontezumaRevengeNoFrameskip-v4', steps=steps, lr=lr, n_env=n_env, gamma=gamma, num_threads=num_threads, device=device, shift=shift,
                         path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 0.25
        self.motivation_horizon = 16
        self.type = 'v3m2h16'

        self.learned_projection_dim = self.feature_dim
        self.forward_model_dim = self.feature_dim * 4

    def train(self, trial):
        super().train(trial)

        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)

        agent = PPOAtariDPMAgent(self)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOAtariDPMAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)


class ConfigMontezumaDPMAnalysis(ConfigMontezumaDPM):
    def __init__(self, num_threads, device, shift, path, steps=32, lr=1e-4, n_env=1, gamma=[0.998, 0.99]):
        super().__init__(num_threads, device, shift, path, steps, lr, n_env, gamma)

    def init_environment(self):
        print('Creating {0:d} environments'.format(self.n_env))
        self.env = MultiEnvParallel([WrapperMontezuma(self.env_name, render_mode='rgb_array') for _ in range(self.n_env)], self.n_env, self.num_threads)

        self.input_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

    def analysis(self, task):
        name = '{0:s}_{1:s}'.format(self.__class__.__name__, self.type)
        print(name)

        agent = PPOAtariDPMAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.analytic_loop(self.env, name, task)
