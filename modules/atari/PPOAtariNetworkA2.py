from math import sqrt

import torch
import torch.nn as nn

from loss.A2Loss import A2Loss
from modules import init_orthogonal
from modules.PPO_AtariModules import PPOAtariMotivationNetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetworkA2(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        self.encoder_a = self.features
        self.encoder_b = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)

        self.hidden_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        hidden_gain = 0.1
        init_orthogonal(self.hidden_model[1], hidden_gain)
        init_orthogonal(self.hidden_model[3], hidden_gain)
        init_orthogonal(self.hidden_model[5], hidden_gain)

        self.associative_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim + self.hidden_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        associative_gain = 0.1
        init_orthogonal(self.associative_model[1], associative_gain)
        init_orthogonal(self.associative_model[3], associative_gain)
        init_orthogonal(self.associative_model[5], associative_gain)

        self.loss = A2Loss(config, self.encoder_a, self.encoder_b, self.hidden_model, self.associative_model)

    def forward(self, state):
        value, action, probs = super().forward(state)
        features_a = self.encoder_a(self.preprocess(state))
        features_b = self.encoder_b(self.preprocess(state))

        return value, action, probs, features_a, features_b

    def error(self, state):
        za_state = self.encoder_a(self.preprocess(state))
        zb_state = self.encoder_b(self.preprocess(state))
        ha_state = self.hidden_model(za_state)
        hb_state = self.hidden_model(zb_state)
        pa_state = self.associative_model(torch.cat([za_state, ha_state], dim=1))
        pb_state = self.associative_model(torch.cat([zb_state, hb_state], dim=1))

        associative_error = (torch.abs(pa_state - pb_state) + 1e-8).pow(2).mean(dim=1, keepdim=True)

        return associative_error

    def loss_function(self, state):
        return self.loss(self.preprocess(state))

    @staticmethod
    def preprocess(state):
        # return state[:, 0, :, :].unsqueeze(1)
        return state
