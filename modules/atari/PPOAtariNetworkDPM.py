import torch
import torch.nn as nn
from math import sqrt

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class ForwardModelDPM(nn.Module):
    def __init__(self, config):
        super(ForwardModelDPM, self).__init__()

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim + self.action_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim + self.action_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)

    def forward(self, z_state, action):
        y = self.forward_model[0](torch.cat([z_state, action], dim=1))
        y = self.forward_model[2](torch.cat([y, action], dim=1))
        y = self.forward_model[4](torch.cat([y, action], dim=1))

        return y


class PPOAtariNetworkDPM(PPOMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkDPM, self).__init__(config)

        input_channels = 1
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        postprocessor_input_shape = (input_channels, input_height, input_width)

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=sqrt(2))
        self.target_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)
        self.learned_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=sqrt(2))

        self.learned_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], gain)

        self.forward_model_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)
        self.forward_model = ForwardModelDPM(config)

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)

    def forward(self, state=None, action=None, next_state=None, pf_state=None, stage=ActivationStage.INFERENCE):
        if stage == ActivationStage.INFERENCE:
            ppo_features = self.ppo_encoder(state)
            value, action, probs = super().forward(ppo_features)

            return value, action, probs, ppo_features

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            zf_state = self.forward_model_encoder(state)
            zf_next_state = self.forward_model_encoder(next_state)
            pzf_next_state = self.forward_model(zf_state, action)

            zt_state = self.target_model(self.preprocess(state))
            pzt_state = self.learned_projection(self.learned_model(self.preprocess(state)))

            return zt_state, pzt_state, zf_next_state, pzf_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            zf_state = self.forward_model_encoder(state)
            zf_next_state = self.forward_model_encoder(next_state)
            pzf_next_state = self.forward_model(zf_state, action)

            zt_state = self.target_model(self.preprocess(state))
            zt_next_state = self.target_model(self.preprocess(next_state))
            pzt_state = self.learned_projection(self.learned_model(self.preprocess(state)))

            return zt_state, pzt_state, zt_next_state, zf_next_state, pzf_next_state

        if stage == ActivationStage.TRAJECTORY_UNWIND:
            pz_next_state = self.forward_model(pf_state, action)

            return pz_next_state
