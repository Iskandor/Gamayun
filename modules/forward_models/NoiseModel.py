import torch.nn as nn
import numpy as np
from enum import Enum
from modules import init_orthogonal


class NoiseModelType(Enum):
    NoiseModel = 0
    NoiseModelSkipConnection = 1


def chooseModel(config, noise_generator_type):
    noise_generator = None
    if noise_generator_type == NoiseModelType.NoiseModel:
        noise_generator = NoiseModel(config)
    elif noise_generator_type == NoiseModelType.NoiseModelSkipConnection:
        noise_generator = NoiseModelSkipConnection(config)
    else:
        noise_generator = NoiseModel(config)

    return noise_generator


class NoiseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.action_dim = config.action_dim
        self.noise_dim = config.noise_dim

        self.noise_generator = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.feature_dim)
        )
        nn.init.uniform_(self.noise_generator[-1].weight, -0.01, 0.01)
    

    def forward(self, x):
        return self.noise_generator(x)
    

class NoiseModelSkipConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.action_dim = config.action_dim
        self.noise_dim = config.noise_dim

        self.noise_generator = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.feature_dim),
        )
        nn.init.uniform_(self.noise_generator[-1].weight, -0.01, 0.01)

        
    def forward(self, x):
        x = self.noise_generator[0](x)
        residual = x
        x = self.noise_generator[1](x)
        x = self.noise_generator[2](x)
        x = x + residual
        x = self.noise_generator[3](x)
        x = self.noise_generator[4](x)
        return x