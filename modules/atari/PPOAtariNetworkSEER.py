from math import sqrt

import torch
import torch.nn as nn
import numpy as np

from loss.SEERLoss import SEERLoss
from modules import init_orthogonal
from modules.PPO_AtariModules import PPOAtariMotivationNetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetworkSEER(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        # input_channels = 1
        # input_height = config.input_shape[1]
        # input_width = config.input_shape[2]

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        # self.input_shape = (input_channels, input_height, input_width)
        self.input_shape = config.input_shape

        self.target_model = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)

        learned_model = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=sqrt(2))

        learned_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(learned_projection[1], gain)
        init_orthogonal(learned_projection[3], gain)

        self.learned_model = nn.Sequential(
            learned_model,
            learned_projection
        )

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim + self.hidden_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        init_orthogonal(self.forward_model[0], np.sqrt(2))
        init_orthogonal(self.forward_model[2], np.sqrt(2))
        init_orthogonal(self.forward_model[4], np.sqrt(2))

        self.hidden_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        init_orthogonal(self.hidden_model[1], np.sqrt(2))
        init_orthogonal(self.hidden_model[3], np.sqrt(2))
        init_orthogonal(self.hidden_model[5], np.sqrt(2))

        self.loss = SEERLoss(config, self.target_model, self.learned_model, self.forward_model, self.hidden_model)

    def forward(self, state):
        value, action, probs = super().forward(state)
        features = self.target_model(self.preprocess(state))

        return value, action, probs, features

    def error(self, state, action, next_state):
        p_state = self.learned_model(self.preprocess(state))
        z_state = self.target_model(self.preprocess(state))

        distillation_error = (torch.abs(z_state - p_state) + 1e-8).pow(2).mean(dim=1, keepdim=True)

        z_next_state = self.target_model(next_state)
        h_next_state = self.hidden_model(z_next_state)
        p_next_state = self.forward_model(torch.cat([z_state, action, h_next_state], dim=1))

        forward_error = (torch.abs(z_next_state - p_next_state) + 1e-8).pow(2).mean(dim=1, keepdim=True)

        return distillation_error, forward_error

    def loss_function(self, state, action, next_state):
        return self.loss(self.preprocess(state), action, self.preprocess(next_state))

    @staticmethod
    def preprocess(state):
        # return state[:, 0, :, :].unsqueeze(1)
        return state
