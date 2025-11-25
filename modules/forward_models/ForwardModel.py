import torch.nn as nn
import numpy as np
from enum import Enum
from modules import init_orthogonal

class ForwardModelType(Enum):
    ForwardModel = 0
    ForwardModelSkipConnection = 1
    ForwardModelSkipConnectionTwice = 2
    ForwardModelSkipConnectionPyramidScheme = 3
    ForwardModelSkipConnectionBatchNorm = 4
    ForwardModelLinearResidual = 5
    ForwardModelNoiseSkipConnection= 6

def chooseModel(config, forward_model_type, action_multiplier = 1):
    forward_model = None
    if forward_model_type == ForwardModelType.ForwardModel:
        forward_model = ForwardModel(config)
    elif forward_model_type == ForwardModelType.ForwardModelSkipConnection:
        forward_model = ForwardModelSkipConnection(config, action_multiplier)
    elif forward_model_type == ForwardModelType.ForwardModelSkipConnectionTwice:
        forward_model = ForwardModelSkipConnectionTwice(config)
    elif forward_model_type == ForwardModelType.ForwardModelSkipConnectionPyramidScheme:
        forward_model = ForwardModelSkipConnectionPyramidScheme(config)
    elif forward_model_type == ForwardModelType.ForwardModelSkipConnectionBatchNorm:
        forward_model = ForwardModelSkipConnectionBatchNorm(config)
    elif forward_model_type == ForwardModelType.ForwardModelLinearResidual:
        forward_model = ForwardModelLinearResidual(config)
    elif forward_model_type == ForwardModelType.ForwardModelNoiseSkipConnection:
        forward_model = ForwardModelNoiseSkipConnection(config)
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
            nn.Linear(self.feature_dim + self.action_dim + (getattr(config, "hidden_dim", 0) or 0), self.forward_model_dim),
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
    def __init__(self, config, action_multiplier):
        super().__init__()
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim*action_multiplier + (getattr(config, "hidden_dim", 0) or 0), self.forward_model_dim),
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
    
class ForwardModelSkipConnectionBatchNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim),
            nn.BatchNorm1d(self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.BatchNorm1d(self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.feature_dim)
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[3], gain)
        init_orthogonal(self.forward_model[6], gain)

    def forward(self, x):
        x = self.forward_model[0](x)  # First Linear Layer
        x = self.forward_model[1](x)
        x = self.forward_model[2](x)
        residual = x
        x = self.forward_model[3](x)  # Second Linear Layer
        x = self.forward_model[4](x)
        x = x + residual
        x = self.forward_model[5](x)
        x = self.forward_model[6](x)  # Third Linear Layer
        return x
    
class ForwardModelSkipConnectionTwice(nn.Module):
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
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.feature_dim)
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)
        init_orthogonal(self.forward_model[6], gain)

    def forward(self, x):
        x = self.forward_model[0](x)  # First Linear Layer
        x = self.forward_model[1](x)
        residual = x
        x = self.forward_model[2](x)  # Second Linear Layer
        x = x + residual
        x = self.forward_model[3](x)
        residual = x
        x = self.forward_model[4](x)  # Third Linear Layer
        x = x + residual
        x = self.forward_model[5](x)
        x = self.forward_model[6](x)  # Fourth Linear Layer
        return x
    
    
class ForwardModelSkipConnectionPyramidScheme(nn.Module):
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
            nn.Linear(self.forward_model_dim, self.forward_model_dim/2),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim/2, self.forward_model_dim/2),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim/2, self.feature_dim)
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)
        init_orthogonal(self.forward_model[6], gain)
        init_orthogonal(self.forward_model[8], gain)

    def forward(self, x):
        x = self.forward_model[0](x)  # First Linear Layer
        x = self.forward_model[1](x)
        residual = x
        x = self.forward_model[2](x)  # Second Linear Layer
        x = x + residual
        x = self.forward_model[3](x)
        x = self.forward_model[4](x)  # Third Linear Layer
        x = self.forward_model[5](x)
        residual = x
        x = self.forward_model[6](x)  # Fourth Linear Layer
        x = x + residual
        x = self.forward_model[7](x)
        x = self.forward_model[8](x)  # Fifth Linear Layer
        return x
    

class ForwardModelSkipConnectionDupe(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim
        self.hidden_dim = hidden_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim + hidden_dim, self.forward_model_dim),
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
    

class ForwardModelLinearResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim),
            nn.ReLU(),
            nn.Linear(self.forward_model_dim, self.feature_dim)
        )

        gain = np.sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)

    def forward(self, x):
        return self.forward_model(x)
    

class ForwardModelNoiseSkipConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.noise_dim = config.noise_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim + self.feature_dim, self.forward_model_dim),
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