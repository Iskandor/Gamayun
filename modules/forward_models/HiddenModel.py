import torch.nn as nn
import numpy as np
from enum import Enum
from modules import init_orthogonal

class HiddenModelType(Enum):
    HiddenModel = 0
    HiddenModelSkipConnection = 1

def chooseModel(config, hidden_model_type):
    hidden_model = None
    if hidden_model_type == HiddenModelType.HiddenModel:
        hidden_model = HiddenModel(config)
    elif hidden_model_type == HiddenModelType.HiddenModelSkipConnection:
        hidden_model = HiddenModelSkipConnection(config)
    else:
        hidden_model = HiddenModel(config)

    return hidden_model

class HiddenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim

        self.hidden_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.hidden_model[1], gain)
        init_orthogonal(self.hidden_model[3], gain)
        init_orthogonal(self.hidden_model[5], gain)

    def forward(self, x):
        x = self.hidden_model[0](x)
        x = self.hidden_model[1](x)  # First Linear Layer
        x = self.hidden_model[2](x)  
        x = self.hidden_model[3](x)  # Second Linear Layer
        x = self.hidden_model[4](x)  
        x = self.hidden_model[5](x)  # Third Linear Layer
        return x
    

class HiddenModelSkipConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim

        self.hidden_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )

        # Ortogonálna inicializácia
        gain = np.sqrt(2)
        init_orthogonal(self.hidden_model[1], gain)
        init_orthogonal(self.hidden_model[3], gain)
        init_orthogonal(self.hidden_model[5], gain)

    def forward(self, x):
        x = self.hidden_model[0](x)  
        x = self.hidden_model[1](x)  # First Linear Layer
        residual = x
        x = self.hidden_model[2](x)  
        x = self.hidden_model[3](x)  # Second Linear Layer
        x = x + residual
        x = self.hidden_model[4](x)
        x = self.hidden_model[5](x)  # Third Linear Layer
        return x