from math import sqrt
import torch
import torch.nn as nn

from agents import ActorType
from modules import init_orthogonal
from modules.PPO_Modules import DiscreteHead, Actor, Critic2Heads, PPONetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetwork(PPONetwork):
    def __init__(self, config):
        super(PPOAtariNetwork, self).__init__(config)

        self.config = config
        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim

        self.encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)

    def forward(self, state):
        z_state = self.encoder(state)
        value, action, probs = super().forward(z_state)

        return value, action, probs
