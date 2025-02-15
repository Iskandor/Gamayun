import torch
import torch.nn as nn

from modules.PPO_Modules import PPOMotivationNetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariFMNetwork(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.ppo_encoder = AtariStateEncoderLarge(config.input_shape, config.feature_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(config.feature_dim + config.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.feature_dim)
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
