from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetworkSEER(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        input_channels = 1
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        postprocessor_input_shape = (input_channels, input_height, input_width)

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.learned_projection_dim = config.learned_projection_dim
        self.forward_model_dim = config.forward_model_dim

        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=sqrt(2))
        self.target_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)
        self.learned_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)

        self.learned_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.learned_projection_dim),
            nn.GELU(),
            nn.Linear(self.learned_projection_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[0], gain)
        init_orthogonal(self.learned_projection[2], gain)

        # self.action_projection = nn.Sequential(
        #     nn.Linear(self.action_dim, self.feature_dim),
        #     nn.GELU(),
        #     nn.Linear(self.feature_dim, self.feature_dim),
        # )
        #
        # gain = sqrt(2)
        # init_orthogonal(self.action_projection[0], gain)
        # init_orthogonal(self.action_projection[2], gain)

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim + self.hidden_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)

        self.hidden_model = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.hidden_model[0], gain)
        init_orthogonal(self.hidden_model[2], gain)
        init_orthogonal(self.hidden_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, zt_state=None, zl_state=None, stage=0):
        if stage == 0:
            z_state = self.ppo_encoder(state)
            zt_state = self.target_model(self.preprocess(state))
            zl_state = self.learned_model(self.preprocess(state))

            value, action, probs = super().forward(z_state)
            return value, action, probs, zt_state, zl_state

        if stage == 1:
            p_state = self.learned_projection(zl_state)
            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = torch.zeros((zl_state.shape[0], self.hidden_dim), dtype=torch.float32, device=self.config.device)
            # z_action = self.action_projection(action)
            p_next_state = self.forward_model(torch.cat([zl_state, action, h_next_state], dim=1))
            h_next_state = self.hidden_model(z_next_state)

            return p_state, z_next_state, h_next_state, p_next_state

        if stage == 2:
            zt_state = self.target_model(self.preprocess(state))
            zl_state = self.learned_model(self.preprocess(state))

            p_state = self.learned_projection(zl_state)
            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = self.hidden_model(z_next_state)
            # z_action = self.action_projection(action)
            p_next_state = self.forward_model(torch.cat([zl_state, action, h_next_state], dim=1))

            return zt_state, zl_state, p_state, z_next_state, h_next_state, p_next_state

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state
