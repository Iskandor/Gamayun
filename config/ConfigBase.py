from pathlib import Path

import yaml


class Hyperparameters:
    def __init__(self, **kwargs):
        self.add(**kwargs)

    def add(self, **kwargs):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]


class Config:
    def __init__(self, yaml_path):
        self.type = None

        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        for key in data:
            self.__dict__[key] = data[key]

    def __str__(self):
        result = ''
        for key in self.__dict__:
            result += '{0} : {1} \n'.format(key, self.__dict__[key])
        return result

    def train(self, trial):
        filepath = './{0:s}_{1:s}.config'.format(self.__class__.__name__, self.type)
        with open(filepath, 'w') as f:
            f.write(str(self))


class ConfigPPO(Config):
    def __init__(self, steps, lr, n_env, gamma, device='cpu'):
        super().__init__(Path.cwd() / 'config' / 'yaml' / 'ppo.yaml')
        self.steps = steps
        self.n_env = n_env
        self.device = device
        self.lr = lr
        self.gamma = gamma
        # self.hyperparameters = Hyperparameters()
