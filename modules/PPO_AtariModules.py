import numpy as np
import torch
import torch.nn as nn

from agents import ActorType
from modules import init_orthogonal
from modules.PPO_Modules import DiscreteHead, Actor, Critic2Heads
from modules.encoders.EncoderAtari import VICRegEncoderAtari, VICRegEncoderAtariV2
from modules.forward_models.ForwardModelAtari import SPModelAtari, ICMModelAtari, SEERModelAtari
from modules.novelty_models.RNDModelAtari import RNDModelAtari, STDModelAtari, BarlowTwinsModelAtari, VICRegModelAtari, SNDVModelAtari, VINVModelAtari, TPModelAtari, AMIModelAtari, \
    SpacVICRegModelAtari


class PPOAtariNetwork(torch.nn.Module):
    def __init__(self, config):
        super(PPOAtariNetwork, self).__init__()

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        input_channels = self.input_shape[0]
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim)
        )

        init_orthogonal(self.features[0], np.sqrt(2))
        init_orthogonal(self.features[2], np.sqrt(2))
        init_orthogonal(self.features[4], np.sqrt(2))
        init_orthogonal(self.features[6], np.sqrt(2))
        init_orthogonal(self.features[9], np.sqrt(2))

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


class PPOAtariNetworkRND(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkRND, self).__init__(config)
        self.rnd_model = RNDModelAtari(config)


class PPOAtariNetworkSP(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkSP, self).__init__(config)
        if config.type == 'sp':
            self.forward_model = SPModelAtari(config)
        if config.type == 'seer':
            self.forward_model = SEERModelAtari(config)


class PPOAtariNetworkICM(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkICM, self).__init__(config)
        self.forward_model = ICMModelAtari(config)

    def forward(self, state):
        value, action, probs = super().forward(state)
        features = self.forward_model.encoder(self.forward_model.preprocess(state))

        return value, action, probs, features

class PPOAtariNetworkSND(PPOAtariMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkSND, self).__init__(config)
        if config.type == 'bt':
            self.cnd_model = BarlowTwinsModelAtari(config)
        if config.type == 'vicreg':
            self.cnd_model = VICRegModelAtari(config, encoder_class=VICRegEncoderAtari)
        if config.type == 'vicreg2':
            self.cnd_model = VICRegModelAtari(config, encoder_class=VICRegEncoderAtariV2)
        if config.type == 'spacvicreg':
            self.cnd_model = SpacVICRegModelAtari(config)
        if config.type == 'st-dim':
            self.cnd_model = STDModelAtari(config)
        if config.type == 'vanilla':
            self.cnd_model = SNDVModelAtari(config)
        if config.type == 'vinv':
            self.cnd_model = VINVModelAtari(config)
        if config.type == 'tp':
            self.cnd_model = TPModelAtari(config)
        if config.type == 'ami':
            self.cnd_model = AMIModelAtari(config)

    def forward(self, state):
        value, action, probs = super().forward(state)
        features = self.cnd_model.target_model(self.cnd_model.preprocess(state))

        return value, action, probs, features
