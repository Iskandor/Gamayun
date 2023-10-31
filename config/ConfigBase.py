from pathlib import Path

import yaml


class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        for key in data:
            self.__dict__[key] = data[key]

    def __str__(self):
        result = ''
        for key in self.__dict__:
            result += '{0} : {1} \n'.format(key, self.__dict__[key])
        return result


class ConfigPPO(Config):
    def __init__(self, steps, lr, n_env, gamma, device='cpu'):
        super().__init__(Path.cwd() / 'config' / 'yaml' / 'ppo.yaml')
        self.steps = steps
        self.lr = lr
        self.n_env = n_env
        self.gamma = gamma
        self.device = device
