import torch.nn as nn
from math import sqrt

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetworkSNDv2(PPOMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkSNDv2, self).__init__(config, activation=nn.SiLU)

        input_channels = 1
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        postprocessor_input_shape = (input_channels, input_height, input_width)

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, activation=nn.SiLU, gain=0.5)
        self.target_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, activation=nn.SiLU, gain=0.5)
        self.learned_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, activation=nn.SiLU, gain=0.5)

        self.learned_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = 0.5
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], gain)

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)

    def forward(self, state=None, next_state=None, stage=ActivationStage.INFERENCE):
        zt_state = self.target_model(self.preprocess(state))
        pzt_state = self.learned_projection(self.learned_model(self.preprocess(state)))

        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))

            return value, action, probs, zt_state, pzt_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            zt_next_state = self.target_model(self.preprocess(next_state))
            return zt_state, pzt_state, zt_next_state

