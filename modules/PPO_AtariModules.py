from math import sqrt
import torch
import torch.nn as nn

from agents import ActorType
from modules import init_orthogonal
from modules.PPO_Modules import DiscreteHead, Actor, Critic2Heads
from modules.encoders.EncoderAtari import AtariStateEncoderLarge


class PPOAtariNetwork(torch.nn.Module):
    def __init__(self, config):
        super(PPOAtariNetwork, self).__init__()

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim

        self.features = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=0.5)

        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1)
        )

        init_orthogonal(self.critic[1], 0.1)
        init_orthogonal(self.critic[3], 0.01)

        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            DiscreteHead(self.feature_dim, config.action_dim)
        )

        init_orthogonal(self.actor[1], 0.01)
        init_orthogonal(self.actor[3], 0.01)

        self.actor = Actor(self.actor, ActorType.discrete, self.action_dim)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs


class PPOAtariMotivationNetwork(PPOAtariNetwork):
    def __init__(self, config):
        super(PPOAtariMotivationNetwork, self).__init__(config)

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            Critic2Heads(self.feature_dim)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)
