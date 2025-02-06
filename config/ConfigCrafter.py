from agents.crafter.PPOCrafterAgent import PPOCrafterAgent
from agents.crafter.PPOCrafterSNDAgent import PPOCrafterSNDAgent
from config import ConfigPPO
from utils.CrafterWrapper import WrapperCrafter
from utils.MultiEnvWrapper import MultiEnvParallel


class ConfigCrafter(ConfigPPO):
    def __init__(self, env_name, steps, lr, n_env, gamma, num_threads, device, shift, path):
        super().__init__(path='ppo_crafter.yaml', steps=steps, lr=lr, n_env=n_env, gamma=gamma, device=device)

        self.num_threads = num_threads
        self.shift = shift
        self.path = path

        self.env_name = env_name
        self.render_mode = None

        self.input_shape = None
        self.action_dim = None
        self.feature_dim = 512
        self.ppo_feature_dim = 1024
        self.env = None

        self.init_environment()

    def init_environment(self):
        print('Creating {0:d} environments'.format(self.n_env))
        self.env = MultiEnvParallel([WrapperCrafter(self.env_name) for _ in range(self.n_env)], self.n_env, self.num_threads)

        self.input_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n


class ConfigCrafterBaseline(ConfigCrafter):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='Crafter', steps=1, lr=1e-4, n_env=1, gamma=0.99, num_threads=num_threads, device=device, shift=shift, path=path)

    def train(self, trial):
        super().train(trial)

        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, '', trial)

        agent = PPOCrafterAgent(self)
        agent.training_loop(self.env, name, trial)


class ConfigCrafterSND(ConfigCrafter):
    def __init__(self, num_threads, device, shift, path):
        super().__init__(env_name='Crafter', steps=1, lr=1e-4, n_env=1, gamma=[0.998, 0.99], num_threads=num_threads, device=device, shift=shift, path=path)

        self.motivation_lr = 1e-4
        self.motivation_scale = 1
        self.type = 'vicreg'

    def train(self, trial):
        super().train(trial)

        trial += self.shift
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)

        agent = PPOCrafterSNDAgent(self)
        agent.training_loop(self.env, name, trial)

    def inference(self, trial):
        name = '{0:s}_{1:s}_{2:d}'.format(self.__class__.__name__, self.type, trial)
        print(name)

        agent = PPOCrafterSNDAgent(self)
        if len(self.path) > 0:
            agent.load(self.path)
        agent.inference_loop(self.env, name, trial)
