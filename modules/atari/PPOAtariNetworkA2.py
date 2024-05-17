from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import init_orthogonal, ResMLPBlock
from modules.PPO_Modules import PPOMotivationNetwork

from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetworkA2(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        input_channels = 1
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        postprocessor_input_shape = (input_channels, input_height, input_width)

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)
        self.encoder_a = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)
        self.encoder_b = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)

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

        associative_gain = sqrt(2)
        self.associative_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim + self.hidden_dim, self.feature_dim),
            ResMLPBlock(self.feature_dim, self.feature_dim, gain=associative_gain)
        )

        init_orthogonal(self.associative_model[1], associative_gain)

    def forward(self, state, loss=False):
        if loss:
            state_a, state_b = self.augmentation(self.preprocess(state))
            za_state, zb_state = self.encoder_a(state_a), self.encoder_b(state_b)
            ha_state, hb_state = self.hidden_model(za_state), self.hidden_model(zb_state)

            pa_state = self.associative_model(torch.cat([zb_state, ha_state], dim=1))
            pb_state = self.associative_model(torch.cat([za_state, hb_state], dim=1))

            return za_state, zb_state, ha_state, hb_state, pa_state, pb_state
        else:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

    @staticmethod
    def augmentation(state, ratio=0.5, patch=8):
        N = state.shape[0]
        C = state.shape[1]
        H = state.shape[2]
        W = state.shape[3]

        h_patches = int(H / patch)
        w_patches = int(W / patch)

        mask = torch.rand((N, C, h_patches, w_patches), dtype=torch.float32, device=state.device)
        mask = (F.upsample(mask, scale_factor=patch) > ratio).type(torch.int32)

        state_a = state * mask
        state_b = state * (1 - mask)

        return state_a, state_b

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state
