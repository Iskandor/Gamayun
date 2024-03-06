from math import sqrt

import torch
import torch.nn as nn

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork

from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetworkA2(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        self.encoder_a = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)
        self.encoder_b = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)

        self.hidden_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        hidden_gain = sqrt(2)
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

        associative_gain = sqrt(2)
        init_orthogonal(self.associative_model[1], associative_gain)
        init_orthogonal(self.associative_model[3], associative_gain)
        init_orthogonal(self.associative_model[5], associative_gain)

    def forward(self, state, loss=False):
        za_state = self.encoder_a(self.preprocess(state))
        zb_state = self.encoder_b(self.preprocess(state))

        if loss:
            ha_state = self.hidden_model(za_state)
            hb_state = self.hidden_model(zb_state)
            pa_state = self.associative_model(torch.cat([zb_state, ha_state], dim=1))
            pb_state = self.associative_model(torch.cat([za_state, hb_state], dim=1))
            return za_state, zb_state, ha_state, hb_state, pa_state, pb_state
        else:
            value, action, probs = super().forward(state)
            return value, action, probs, za_state, zb_state

    def error(self, state):
        za_state, zb_state, _, _, pa_state, pb_state = self(state, loss=True)
        associative_error = (torch.abs(pa_state - za_state) + 1e-8).pow(2).mean(dim=1, keepdim=True) + (torch.abs(pb_state - zb_state) + 1e-8).pow(2).mean(dim=1, keepdim=True)

        return associative_error * 0.5

    @staticmethod
    def preprocess(state):
        # return state[:, 0, :, :].unsqueeze(1)
        return state
