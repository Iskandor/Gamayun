import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariFMNetwork(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)


class PPOAtariSTDIMNetwork(PPOAtariFMNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init_orthogonal(self.forward_model[0], np.sqrt(2))
        init_orthogonal(self.forward_model[2], np.sqrt(2))
        init_orthogonal(self.forward_model[4], np.sqrt(2))

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            predicted_next_state = self.forward_model(torch.cat([map_state['out'], action], dim=1))
            return map_state, map_next_state, predicted_next_state


class PPOAtariIJEPANetwork(PPOAtariFMNetwork):
    def __init__(self, config):
        super().__init__(config)
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.hidden_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )

        gain = np.sqrt(2)
        init_orthogonal(self.hidden_model[1], gain)
        init_orthogonal(self.hidden_model[3], gain)
        init_orthogonal(self.hidden_model[5], gain)

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)

    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        # Tu je otázka, či chceme vraciať aj hidden encoded state a počítať reward aj pomocou neho
        if stage == ActivationStage.MOTIVATION_INFERENCE:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            hidden_next_state = self.hidden_model(encoded_next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, hidden_next_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            encoded_state = self.ppo_encoder(state)
            encoded_next_state = self.ppo_encoder(next_state)
            hidden_next_state = self.hidden_model(encoded_next_state)
            predicted_next_state = self.forward_model(torch.cat([encoded_state, hidden_next_state, action], dim=1))
            return encoded_state, encoded_next_state, predicted_next_state, hidden_next_state
