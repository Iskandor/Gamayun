import torch
import torch.nn as nn

from modules.PPO_Modules import PPOMotivationNetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariFMNetwork(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == 0:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == 1:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == 2:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state
