import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from modules import init_orthogonal
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge, AtariStateEncoderResNet


class PPOAtariNetworkSNDv2(PPOMotivationNetwork):
    def __init__(self, config):
        super(PPOAtariNetworkSNDv2, self).__init__(config, activation=nn.GELU)

        self.input_shape = config.input_shape
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.ppo_feature_dim = config.ppo_feature_dim

        self.ppo_encoder = AtariStateEncoderResNet(self.input_shape, self.ppo_feature_dim, activation=nn.GELU, gain=0.5)
        self.learned_encoder = AtariStateEncoderResNet(self.input_shape, self.ppo_feature_dim, activation=nn.GELU, gain=0.5)

        self.learned_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.ppo_feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = 0.5
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], 0.01)

        self.target_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.ppo_feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = 0.5
        init_orthogonal(self.target_projection[1], gain)
        init_orthogonal(self.target_projection[3], 0.01)

    # @staticmethod
    # def preprocess(state):
    #     return state[:, 0, :, :].unsqueeze(1)

    def forward(self, state=None, next_state=None, stage=ActivationStage.INFERENCE):
        z_state = self.ppo_encoder(state)
        zt_state = self.target_projection(z_state)
        pzt_state = self.learned_projection(self.learned_encoder(state))

        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(z_state)

            return value, action, probs, zt_state, pzt_state

        if stage == ActivationStage.MOTIVATION_TRAINING:
            # zt_next_state = self.target_projection(self.ppo_encoder(next_state))
            state_a, state_b = PPOAtariNetworkSNDv2.augmentation(state)
            z_state_a = self.target_projection(self.ppo_encoder(state_a))
            z_state_b = self.target_projection(self.ppo_encoder(state_b))

            return zt_state, pzt_state, z_state_a, z_state_b

    @staticmethod
    def augmentation(state, ratio=0.5, patch=8):
        N = state.shape[0]
        C = state.shape[1]
        H = state.shape[2]
        W = state.shape[3]

        h_patches = int(H / patch)
        w_patches = int(W / patch)

        mask_a = torch.rand((N, C, h_patches, w_patches), dtype=torch.float32, device=state.device)
        mask_a = (F.interpolate(mask_a, scale_factor=patch, mode="bicubic") > ratio).type(torch.int32)
        mask_b = torch.rand((N, C, h_patches, w_patches), dtype=torch.float32, device=state.device)
        mask_b = (F.interpolate(mask_b, scale_factor=patch, mode="bicubic") > ratio).type(torch.int32)

        # state_a = PPOAtariNetworkSNDv2.noise(state) * mask
        # state_b = PPOAtariNetworkSNDv2.noise(state) * (1 - mask)
        state_a = state * mask_a
        state_b = state * mask_b

        return state_a, state_b

    @staticmethod
    def noise(state, k=0.2):
        pointwise_noise = k * (2.0 * torch.rand(state.shape, device=state.device) - 1.0)
        return state + pointwise_noise
