import torch.nn as nn
import numpy as np
from enum import Enum
from modules import init_orthogonal

class ForwardModelType(Enum):
    ForwardModel = 0
    ForwardModelSkipConnection = 1

def chooseModel(config, forward_model_type):
    forward_model = None
    if forward_model_type == ForwardModelType.ForwardModel:
        forward_model = ForwardModel(config)
    elif forward_model_type == ForwardModelType.ForwardModelSkipConnection:
        forward_model = ForwardModelSkipConnection(config)
    else:
        forward_model = ForwardModel(config)

    return forward_model

class ForwardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.feature_dim)
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)

    def forward(self, x):
        x = self.forward_model[0](x)  # First Linear Layer
        x = self.forward_model[1](x)
        x = self.forward_model[2](x)  # Second Linear Layer
        x = self.forward_model[3](x)
        x = self.forward_model[4](x)  # Third Linear Layer
        return x
    

class ForwardModelSkipConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.feature_dim)
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)

    def forward(self, x):
        x = self.forward_model[0](x)  # First Linear Layer
        residual = x
        x = self.forward_model[1](x)
        x = self.forward_model[2](x)  # Second Linear Layer
        x = x + residual
        x = self.forward_model[3](x)
        x = self.forward_model[4](x)  # Third Linear Layer
        return x