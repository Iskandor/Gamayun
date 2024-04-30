from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from modules import init_orthogonal, ResMLPBlock
from modules.PPO_Modules import PPOMotivationNetwork
from modules.encoders.EncoderAtari import AtariStateEncoderLarge, AtariStateEncoderUniversal


class ForwardModelSEER(nn.Module):
    def __init__(self, config):
        super(ForwardModelSEER, self).__init__()

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.forward_model_dim = config.forward_model_dim

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim + self.hidden_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim + self.action_dim + self.hidden_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim + self.action_dim + self.hidden_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.forward_model[0], gain)
        init_orthogonal(self.forward_model[2], gain)
        init_orthogonal(self.forward_model[4], gain)

    def forward(self, z_state, action, h_next_state):
        y = self.forward_model[0](torch.cat([z_state, action, h_next_state], dim=1))
        y = self.forward_model[2](torch.cat([y, action, h_next_state], dim=1))
        y = self.forward_model[4](torch.cat([y, action, h_next_state], dim=1))

        return y


class PPOAtariNetworkSEER_V5M4(PPOMotivationNetwork):
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
            nn.GELU(),
            nn.Linear(self.feature_dim, self.learned_projection_dim),
            nn.GELU(),
            nn.Linear(self.learned_projection_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], gain)

        self.forward_model = ForwardModelSEER(config)

        self.hidden_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.hidden_model[1], gain)
        init_orthogonal(self.hidden_model[3], gain)
        init_orthogonal(self.hidden_model[5], gain)

    def forward(self, state=None, action=None, next_state=None, h_next_state=None, stage=0):
        if stage == 0:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == 1:
            batch = state.shape[0]

            # distillation
            zl_state = self.learned_model(self.preprocess(state))
            pz_state = self.learned_projection(zl_state)
            z_state = self.target_model(self.preprocess(state))

            # state prediction
            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = h_next_state.unsqueeze(0).expand(batch, -1)
            pz_next_state = self.forward_model(zl_state, action, h_next_state)
            h_next_state = self.hidden_model(z_next_state)

            return z_state, pz_state, z_next_state, pz_next_state, h_next_state

        if stage == 2:
            # distillation
            zl_state = self.learned_model(self.preprocess(state))
            pz_state = self.learned_projection(zl_state)
            a_state, b_state = self.augmentation(self.preprocess(state))
            za_state = self.target_model(a_state)
            zb_state = self.target_model(b_state)
            zt_state = self.target_model(self.preprocess(state))

            # state prediction
            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = self.hidden_model(z_next_state)
            pz_next_state = self.forward_model(zl_state, action, h_next_state)

            return za_state, zb_state, zt_state, pz_state, z_next_state, pz_next_state, h_next_state

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state

    @staticmethod
    def augmentation(state, ratio=0.5, patch=8):
        N = state.shape[0]
        C = state.shape[1]
        H = state.shape[2]
        W = state.shape[3]

        h_patches = int(H / patch)
        w_patches = int(W / patch)

        mask = torch.rand((N, C, h_patches, w_patches), dtype=torch.float32, device=state.device)
        mask = (F.upsample(mask, scale_factor=patch) > ratio).type(torch.int32)

        state_a = state * mask
        state_b = state * (1 - mask)

        # img_a = to_pil_image(state_a[0])
        # img_a.show()
        # img_b = to_pil_image(state_b[0])
        # img_b.show()

        return state_a, state_b


class PPOAtariNetworkSEER_V5M13(PPOMotivationNetwork):
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

        # self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim, gain=sqrt(2))
        # self.target_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)
        self.target_model = AtariStateEncoderUniversal(postprocessor_input_shape, self.feature_dim, gain=0.5)

        gain = 0.01
        self.ppo_encoder = nn.Sequential(
            nn.GELU(),
            ResMLPBlock(self.feature_dim * 4, self.feature_dim, gain=gain),
            nn.GELU(),
            ResMLPBlock(self.feature_dim, self.feature_dim, gain=gain),
        )

        self.learned_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)

        self.learned_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.learned_projection_dim),
            nn.GELU(),
            nn.Linear(self.learned_projection_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[1], gain)
        init_orthogonal(self.learned_projection[3], gain)

        self.forward_model = ForwardModelSEER(config)

        self.hidden_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.hidden_model[1], gain)
        init_orthogonal(self.hidden_model[3], gain)
        init_orthogonal(self.hidden_model[5], gain)

    def forward(self, state=None, action=None, next_state=None, h_next_state=None, stage=0):
        if stage == 0:
            z = self.ppo_encoder(self.target_model(state).detach())
            value, action, probs = super().forward(z)
            return value, action, probs

        if stage == 1:
            batch = state.shape[0]

            # distillation
            zl_state = self.learned_model(self.preprocess(state))
            pz_state = self.learned_projection(zl_state)
            z_state = self.target_model(self.preprocess(state))

            # state prediction
            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = h_next_state.unsqueeze(0).expand(batch, -1)
            pz_next_state = self.forward_model(zl_state, action, h_next_state)
            h_next_state = self.hidden_model(z_next_state)

            return z_state, pz_state, z_next_state, pz_next_state, h_next_state

        if stage == 2:
            # distillation
            zl_state = self.learned_model(self.preprocess(state))
            pz_state = self.learned_projection(zl_state)
            z_state = self.target_model(self.preprocess(state))

            # state prediction
            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = self.hidden_model(z_next_state)
            pz_next_state = self.forward_model(zl_state, action, h_next_state)

            return z_state, pz_state, z_next_state, pz_next_state, h_next_state

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state


class PPOAtariNetworkSEER_V8(PPOMotivationNetwork):
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

        self.forward_learned_projection = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.forward_learned_projection[0], gain)
        init_orthogonal(self.forward_learned_projection[2], gain)
        init_orthogonal(self.forward_learned_projection[4], gain)

        self.forward_target_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.forward_target_projection[0], gain)
        init_orthogonal(self.forward_target_projection[2], gain)
        init_orthogonal(self.forward_target_projection[4], gain)

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.hidden_dim, self.forward_model_dim),
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

    def forward(self, state, action=None, next_state=None, h_next_state=None, stage=0):
        if stage == 0:
            z_state = self.target_model(self.preprocess(state))

            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs, z_state

        if stage == 1:
            batch = state.shape[0]

            # distillation
            pz_state = self.learned_projection(self.learned_model(self.preprocess(state)))

            # state prediction
            z_next_state = self.target_model(self.preprocess(next_state))
            pi_zt_next_state = self.forward_target_projection(z_next_state)
            pi_zl_next_state = self.forward_learned_projection(torch.cat([pz_state, action], dim=1))
            h_next_state = h_next_state.unsqueeze(0).expand(batch, -1)
            pz_next_state = self.forward_model(torch.cat([pi_zl_next_state, h_next_state], dim=1))
            h_next_state = self.hidden_model(z_next_state)

            return pz_state, pi_zt_next_state, h_next_state, pz_next_state

        if stage == 2:
            # distillation
            pz_state = self.learned_projection(self.learned_model(self.preprocess(state)))
            z_state = self.target_model(self.preprocess(state))

            # state prediction
            pi_zl_next_state = self.forward_learned_projection(torch.cat([pz_state.detach(), action], dim=1))

            z_next_state = self.target_model(self.preprocess(next_state))
            h_next_state = self.hidden_model(z_next_state)
            pi_zt_next_state = self.forward_target_projection(z_next_state.detach())

            pz_next_state = self.forward_model(torch.cat([pi_zl_next_state, h_next_state], dim=1))

            return z_state, pz_state, z_next_state, pz_next_state, h_next_state, pi_zl_next_state, pi_zt_next_state

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state


class PPOAtariNetworkSEER_V9(PPOMotivationNetwork):
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

        self.projection1 = nn.Linear(self.target_model.hidden_size, self.target_model.local_layer_depth)  # x1 = global, x2=patch, n_channels = 32
        self.projection2 = nn.Linear(self.target_model.local_layer_depth, self.target_model.local_layer_depth)

        self.learned_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim, gain=0.5)

        self.learned_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.learned_projection_dim),
            nn.GELU(),
            nn.Linear(self.learned_projection_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[0], gain)
        init_orthogonal(self.learned_projection[2], gain)

        self.forward_model = ForwardModelSEER(config)

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

    def forward(self, state=None, action=None, next_state=None, h_next_state=None, stage=0):
        if stage == 0:
            zt_state = self.target_model(self.preprocess(state))

            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs, zt_state

        if stage == 1:
            batch = state.shape[0]

            # distillation
            pz_state = self.learned_projection(self.learned_model(self.preprocess(state)))

            # state prediction
            z_state = self.ppo_encoder(state)
            z_next_state = self.ppo_encoder(next_state)
            h_next_state = h_next_state.unsqueeze(0).expand(batch, -1)
            pz_next_state = self.forward_model(z_state, action, h_next_state)
            h_next_state = self.hidden_model(z_next_state)

            return pz_state, z_next_state, h_next_state, pz_next_state

        if stage == 2:
            # distillation
            pz_state = self.learned_projection(self.learned_model(self.preprocess(state)))
            zt_state = self.target_model(self.preprocess(state))
            zt_next_state = self.target_model(self.preprocess(next_state))
            # z_state, map_state = z_state['out'], z_state['f5']

            # state prediction
            z_state = self.ppo_encoder(state)
            z_next_state = self.ppo_encoder(next_state)
            # z_next_state, map_next_state = z_next_state['out'], z_next_state['f5']
            h_next_state = self.hidden_model(z_next_state)

            pz_next_state = self.forward_model(z_state, action, h_next_state)

            return zt_state, zt_next_state, pz_state, z_next_state, pz_next_state, h_next_state

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state
