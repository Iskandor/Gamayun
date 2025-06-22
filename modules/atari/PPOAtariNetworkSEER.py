from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from modules.PPO_Modules import ActivationStage

from modules import init_orthogonal, ResMLPBlock, AttentionBlock
from modules.PPO_Modules import PPOMotivationNetwork, ActivationStage
from modules.encoders.EncoderAtari import AtariStateEncoderLarge, AtariStateEncoderUniversal


class ForwardModelSEER_V1(nn.Module):
    def __init__(self, config):
        super(ForwardModelSEER_V1, self).__init__()

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
        y = self.forward_model[1](y)
        y = self.forward_model[2](torch.cat([y, action], dim=1))
        y = self.forward_model[3](y)
        y = self.forward_model[4](torch.cat([y, action], dim=1))

        return y

class ForwardModelSEER_V2(nn.Module):
    def __init__(self, config):
        super(ForwardModelSEER_V2, self).__init__()

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.forward_model_dim = config.forward_model_dim

        self.input_module = nn.Linear(self.feature_dim + self.action_dim, self.forward_model_dim)
        self.forward_model = nn.Sequential(
            nn.GELU(),
            ResMLPBlock(self.forward_model_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.input_module, gain)
        init_orthogonal(self.forward_model[3], gain)

    def forward(self, z_state, action):
        y = self.input_module(torch.cat([z_state, action], dim=1))
        y = self.forward_model(y)

        return y

class ForwardModelSEER_V3(nn.Module):
    def __init__(self, config):
        super(ForwardModelSEER_V3, self).__init__()

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.forward_model_dim = config.forward_model_dim

        self.attention_module = AttentionBlock(self.feature_dim, self.forward_model_dim)
        self.input_module = nn.Linear(self.feature_dim + self.action_dim + self.forward_model_dim, self.forward_model_dim)
        self.forward_model = nn.Sequential(
            nn.GELU(),
            ResMLPBlock(self.forward_model_dim + self.action_dim, self.forward_model_dim),
            nn.GELU(),
            nn.Linear(self.forward_model_dim, self.feature_dim),
        )

        gain = sqrt(2)
        init_orthogonal(self.input_module, gain)
        init_orthogonal(self.forward_model[3], gain)

    def forward(self, z_state, action, prev_pz_next_state):
        context = self.attention_module(z_state, prev_pz_next_state)
        y = self.input_module(torch.cat([z_state, action, context], dim=1))
        y = self.forward_model(y)

        return y

class InverseModelSEER(nn.Module):
    def __init__(self, config):
        super(InverseModelSEER, self).__init__()

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.action_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.inverse_model[0], gain)
        init_orthogonal(self.inverse_model[2], gain)
        init_orthogonal(self.inverse_model[4], gain)

    def forward(self, z_state, z_next_state):
        x = torch.cat([z_state, z_next_state], dim=1)
        return self.inverse_model(x)

class EncoderMaskClassifier(nn.Module):
    def __init__(self, encoder, config):
        super(EncoderMaskClassifier, self).__init__()
        self.feature_dim = config.feature_dim
        self.candidate_num = 4

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )

        gain = 0.5
        init_orthogonal(self.classifier[1], gain)
        init_orthogonal(self.classifier[3], gain)

    def forward(self, state):
        N = state.shape[0]
        C = state.shape[1]
        H = state.shape[2]
        W = state.shape[3]

        state_ref, inputs, targets = self.augmentation(state, candidate_num=self.candidate_num)

        z_ref = self.encoder(state_ref).view(N, 1, self.feature_dim).repeat(1, self.candidate_num, 1).view(-1, self.feature_dim)
        z_can = self.encoder(inputs.view(-1, C, H, W)).view(-1, self.feature_dim)
        z = z_ref + z_can
        logits = self.classifier(z).view(N, self.candidate_num)

        return logits, targets

    def augmentation(self, state, patch=32, candidate_num=4, noise=2):
        N = state.shape[0]
        C = state.shape[1]
        H = state.shape[2]
        W = state.shape[3]

        x = torch.randint(low=0, high=W - patch, size=(N,), device=state.device)
        y = torch.randint(low=0, high=H - patch, size=(N,), device=state.device)

        ref_indices, s = self.cut_state(state, x, y, patch)

        state_ref = state.flatten()
        state_ref[ref_indices.flatten()] = 0.
        state_ref = state_ref.view(N, C, H, W)

        result = [s]
        target = [1]

        for i in range(candidate_num - 1):
            nx = torch.randint(low=-noise, high=noise, size=(N,), device=state.device) + x
            ny = torch.randint(low=-noise, high=noise, size=(N,), device=state.device) + y

            nx.clamp_(0, W - patch - 1)
            ny.clamp_(0, H - patch - 1)

            _, s = self.cut_state(state, nx, ny, patch, ref_indices)
            result.append(s)
            target.append(0)

        result = torch.stack(result).permute(1, 0, 2, 3, 4)
        target = torch.tensor(target, device=state.device, dtype=torch.int32).unsqueeze(0).repeat(N, 1)

        indices = torch.argsort(torch.rand(*result.shape[:2], device=state.device), dim=1)
        result = torch.gather(result, dim=1, index=indices.view(N, candidate_num, 1, 1, 1).repeat(1, 1, result.shape[2], result.shape[3], result.shape[4]))
        target = torch.gather(target, dim=1, index=indices)
        target = torch.argmax(target, dim=1)

        # img_a = to_pil_image(result[0, 0])
        # img_a.show()
        # img_a = to_pil_image(result[0, 1])
        # img_a.show()
        # img_a = to_pil_image(result[0, 2])
        # img_a.show()
        # img_a = to_pil_image(result[0, 3])
        # img_a.show()
        # img_b = to_pil_image(result[0, 4])
        # img_b.show()

        return state_ref, result, target

    def cut_state(self, state, x, y, patch, ref_indices=None):
        N = state.shape[0]
        C = state.shape[1]
        H = state.shape[2]
        W = state.shape[3]

        indices_x = torch.arange(patch, device=state.device).view(1, 1, -1, 1) + x.view(-1, 1, 1, 1)
        indices_y = torch.arange(patch, device=state.device).view(1, 1, 1, -1) + y.view(-1, 1, 1, 1)

        # Calculate the linear indices for index_select
        linear_indices = (indices_y + H * indices_x) + torch.arange(N, device=state.device).view(-1, 1, 1, 1) * (C * H * W)
        if ref_indices is None:
            ref_indices = linear_indices

        # Select the 32x32 parts using index_select
        # c = torch.index_select(state.flatten(), dim=0, index=linear_indices.flatten())
        # c = c.view(N, C, H, W)

        result = torch.zeros_like(state).flatten()
        result[ref_indices.flatten()] = torch.index_select(state.flatten(), dim=0, index=linear_indices.flatten())
        result = result.view(N, C, H, W)

        return linear_indices, result


class PPOAtariNetworkSEER_V1(PPOMotivationNetwork):
    def __init__(self, config):
        super().__init__(config)

        input_channels = 1
        input_height = config.input_shape[1]
        input_width = config.input_shape[2]
        postprocessor_input_shape = (input_channels, input_height, input_width)

        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.learned_projection_dim = config.learned_projection_dim
        self.forward_model_dim = config.forward_model_dim

        self.input_shape = config.input_shape

        self.ppo_encoder = AtariStateEncoderLarge(self.input_shape, self.feature_dim)
        self.target_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim)
        self.learned_model = AtariStateEncoderLarge(postprocessor_input_shape, self.feature_dim)

        self.learned_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.learned_projection_dim),
            nn.GELU(),
            nn.Linear(self.learned_projection_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.learned_projection[0], gain)
        init_orthogonal(self.learned_projection[2], gain)

        self.forward_model = ForwardModelSEER_V2(config)
        self.inverse_model = InverseModelSEER(config)


    def forward(self, state=None, action=None, next_state=None, stage=0):
        if stage == ActivationStage.INFERENCE:
            value, action, probs = super().forward(self.ppo_encoder(state))
            return value, action, probs

        if stage == ActivationStage.MOTIVATION_INFERENCE:
            # distillation
            zt_state = self.target_model(self.preprocess(state))
            pz_state = self.learned_projection(self.learned_model(self.preprocess(state)))

            # state prediction
            z_state = self.ppo_encoder(state)
            z_next_state = self.ppo_encoder(next_state)
            pz_next_state = self.forward_model(z_state, action)

            _, _, next_probs = super().forward(z_next_state)
            _, _, p_next_probs = super().forward(pz_next_state)

            p_action = self.inverse_model(z_state, z_next_state)
            pp_action = self.inverse_model(z_state, pz_next_state)

            return zt_state, pz_state, z_next_state, pz_next_state, next_probs, p_next_probs, p_action, pp_action

        if stage == ActivationStage.MOTIVATION_TRAINING:
            # distillation
            pz_state = self.learned_projection(self.learned_model(self.preprocess(state)))
            zt_state = self.target_model(self.preprocess(state))
            zt_next_state = self.target_model(self.preprocess(next_state))

            # state prediction
            map_state = self.ppo_encoder(state, fmaps=True)
            map_next_state = self.ppo_encoder(next_state, fmaps=True)
            z_state = map_state['out']
            z_next_state = map_next_state['out']

            pz_next_state = self.forward_model(z_state, action)
            p_action = self.inverse_model(z_state, z_next_state)

            return zt_state, zt_next_state, pz_state, z_next_state, pz_next_state, p_action, map_state, map_next_state

    @staticmethod
    def preprocess(state):
        return state[:, 0, :, :].unsqueeze(1)
        # return state
