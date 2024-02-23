import random
import time
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from analytic.ResultCollector import ResultCollector
from modules import init_orthogonal, init_xavier_uniform
from utils.Augmentation import aug_random_apply, aug_pixelate, aug_mask_tiles, aug_noise


class AtariStateEncoderSmall(nn.Module):

    def __init__(self, input_shape, feature_dim, gain=sqrt(2)):
        super().__init__()
        self.feature_size = feature_dim
        self.hidden_size = self.feature_size

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 64 * (self.input_width // 8) * (self.input_height // 8)
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        # gain = nn.init.calculate_gain('relu')
        init_orthogonal(self.main[0], gain)
        init_orthogonal(self.main[2], gain)
        init_orthogonal(self.main[4], gain)
        init_orthogonal(self.main[7], gain)

    def forward(self, inputs):
        out = self.main(inputs)
        return out


class AtariStateEncoderLarge(nn.Module):

    def __init__(self, input_shape, feature_dim, gain=0.5):
        super().__init__()
        self.feature_size = feature_dim
        self.hidden_size = self.feature_size

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 128 * (self.input_width // 8) * (self.input_height // 8)
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        # gain = nn.init.calculate_gain('relu')
        init_orthogonal(self.main[0], gain)
        init_orthogonal(self.main[2], gain)
        init_orthogonal(self.main[4], gain)
        init_orthogonal(self.main[6], gain)
        init_orthogonal(self.main[9], gain)

        self.local_layer_depth = self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, gain=sqrt(2)):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        init_orthogonal(self.conv1, gain)
        init_orthogonal(self.conv2, gain)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class AtariStateEncoderResNet(nn.Module):

    def __init__(self, input_shape, feature_dim, gain=sqrt(2)):
        super().__init__()
        self.feature_size = feature_dim
        self.hidden_size = self.feature_size

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 64 * (self.input_width // 4) * (self.input_height // 4)  # 36864

        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(32, 64, gain=gain),
            ResidualBlock(64, 64, gain=gain),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        # gain = nn.init.calculate_gain('relu')
        init_orthogonal(self.main[0], gain)

    def forward(self, inputs):
        out = self.main(inputs)
        return out


class ST_DIMEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(ST_DIMEncoderAtari, self).__init__()

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.encoder = AtariStateEncoderLarge(input_shape, feature_dim)
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth)

    def forward(self, state, fmaps=False):
        return self.encoder(state, fmaps)

    def loss_function_crossentropy(self, states, next_states):
        f_t_maps, f_t_prev_maps = self.encoder(next_states, fmaps=True), self.encoder(states, fmaps=True)

        # Loss 1: Global at time t, f5 patches at time t-1
        f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
        sy = f_t_prev.size(1)
        sx = f_t_prev.size(2)
        N = f_t.size(0)

        positive = []
        for y in range(sy):
            for x in range(sx):
                positive.append(f_t_prev[:, y, x, :].T)

        predictions = self.classifier1(f_t)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.config.device).unsqueeze(0).repeat(logits.shape[0], 1)
        loss1 = nn.functional.cross_entropy(logits, target, reduction='mean')
        norm_loss1 = torch.norm(logits, p=2, dim=[1, 2]).mean()

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        f_t = f_t_maps['f5']
        predictions = []
        positive = []
        for y in range(sy):
            for x in range(sx):
                predictions.append(self.classifier2(f_t[:, y, x, :]))
                positive.append(f_t_prev[:, y, x, :].T)

        predictions = torch.stack(predictions)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.config.device).unsqueeze(0).repeat(logits.shape[0], 1)
        loss2 = nn.functional.cross_entropy(logits, target, reduction='mean')
        norm_loss2 = torch.norm(logits, p=2, dim=[1, 2]).mean()

        loss = loss1 + loss2
        norm_loss = norm_loss1 + norm_loss2

        return loss, norm_loss

    def loss_function_cdist(self, states, next_states):
        f_t_maps, f_t_prev_maps = self.encoder(next_states, fmaps=True), self.encoder(states, fmaps=True)

        # Loss 1: Global at time t, f5 patches at time t-1
        f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
        sy = f_t_prev.size(1)
        sx = f_t_prev.size(2)

        N = f_t.size(0)
        target = torch.ones((N, N), device=self.config.device) - torch.eye(N, N, device=self.config.device)
        loss1 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier1(f_t) + 1e-8
                positive = f_t_prev[:, y, x, :] + 1e-8
                logits = torch.cdist(predictions, positive, p=0.5)
                step_loss = nn.functional.mse_loss(logits, target)
                loss1 += step_loss

        loss1 = loss1 / (sx * sy)

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        f_t = f_t_maps['f5']
        loss2 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier2(f_t[:, y, x, :]) + 1e-8
                positive = f_t_prev[:, y, x, :] + 1e-8
                logits = torch.cdist(predictions, positive, p=0.5)
                step_loss = nn.functional.mse_loss(logits, target)
                loss2 += step_loss

        loss2 = loss2 / (sx * sy)

        loss = loss1 + loss2

        return loss


class SNDVEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(SNDVEncoderAtari, self).__init__()

        self.config = config
        fc_size = (input_shape[1] // 8) * (input_shape[2] // 8)

        self.layers = [
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),

            nn.Flatten(),

            nn.Linear(64 * fc_size, feature_dim)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2.0 ** 0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.encoder = nn.Sequential(*self.layers)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states_a, states_b, target):
        xa = states_a.clone()
        xb = states_b.clone()

        # normalise states
        # if normalise is not None:
        #     xa = normalise(xa)
        #     xb = normalise(xb)

        # states augmentation
        xa = self.augment(xa)
        xb = self.augment(xb)

        # obtain features from model
        za = self(xa)
        zb = self(xb)

        # predict close distance for similar, far distance for different states
        predicted = ((za - zb) ** 2).mean(dim=1)

        # similarity MSE loss
        loss_sim = ((target - predicted) ** 2).mean()

        # L2 magnitude regularisation
        magnitude = (za ** 2).mean() + (zb ** 2).mean()

        # care only when magnitude above 200
        loss_magnitude = torch.relu(magnitude - 200.0)

        loss = loss_sim + loss_magnitude

        return loss

    def augment(self, x):
        x = self.aug_random_apply(x, 0.5, self.aug_mask_tiles)
        x = self.aug_random_apply(x, 0.5, self.aug_noise)

        return x.detach()

    @staticmethod
    def aug_random_apply(x, p, aug_func):
        mask = (torch.rand(x.shape[0]) < p)
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        mask = mask.float().to(x.device)
        y = (1.0 - mask) * x + mask * aug_func(x)

        return y

    @staticmethod
    def aug_mask_tiles(x, p=0.1):

        if x.shape[2] == 96:
            tile_sizes = [1, 2, 4, 8, 12, 16]
        else:
            tile_sizes = [1, 2, 4, 8, 16]

        tile_size = tile_sizes[np.random.randint(len(tile_sizes))]

        size_h = x.shape[2] // tile_size
        size_w = x.shape[3] // tile_size

        mask = (torch.rand((x.shape[0], 1, size_h, size_w)) < (1.0 - p))

        mask = torch.kron(mask, torch.ones(tile_size, tile_size))

        return x * mask.float().to(x.device)

    # uniform aditional noise
    @staticmethod
    def aug_noise(x, k=0.2):
        pointwise_noise = k * (2.0 * torch.rand(x.shape, device=x.device) - 1.0)
        return x + pointwise_noise


class BarlowTwinsEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(BarlowTwinsEncoderAtari, self).__init__()

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = AtariStateEncoderLarge(input_shape, feature_dim)
        self.lam = 5e-3

        self.lam_mask = torch.maximum(torch.ones(self.feature_dim, self.feature_dim, device=self.config.device) * self.lam, torch.eye(self.feature_dim, self.feature_dim, device=self.config.device))

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        n = states.shape[0]
        d = self.feature_dim
        y_a = self.augment(states)
        y_b = self.augment(states)
        z_a = self.encoder(y_a)
        z_b = self.encoder(y_b)

        # z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        # z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)

        c = torch.matmul(z_a.t(), z_b) / n
        c_diff = (c - torch.eye(d, d, device=self.config.device)).pow(2) * self.lam_mask
        loss = c_diff.sum()

        return loss

    def augment(self, x):
        # ref = transforms.ToPILImage()(x[0])
        # ref.show()
        # transforms_train = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=(0.66, 1.0))])
        # transforms_train = transforms.RandomErasing(p=1)
        # print(x.max())
        ax = x + torch.randn_like(x) * 0.1
        ax = nn.functional.upsample(nn.functional.avg_pool2d(ax, kernel_size=2), scale_factor=2, mode='bilinear')
        # print(ax.max())

        # aug = transforms.ToPILImage()(ax[0])
        # aug.show()

        return ax


class VICRegEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(VICRegEncoderAtari, self).__init__()

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = AtariStateEncoderLarge(input_shape, feature_dim, gain=0.5)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        z_a = self.encoder(states)
        z_b = self.encoder(next_states)

        return self.vicreg_loss_function(z_a, z_b)

    def vicreg_loss_function(self, z_a, z_b):
        inv_loss = self.invariance(z_a, z_b)
        var_loss = self.variance(z_a) + self.variance(z_b)
        cov_loss = self.covariance(z_a) + self.covariance(z_b)

        la = 1.
        mu = 1.
        nu = 1. / 25

        return la * inv_loss + mu * var_loss + nu * cov_loss

    @staticmethod
    def variance(z, gamma=1):
        return F.relu(gamma - z.std(0)).mean()

    @staticmethod
    def invariance(z1, z2):
        return F.mse_loss(z1, z2)

    @staticmethod
    def covariance(z):
        n, d = z.shape
        mu = z.mean(0)
        cov = torch.matmul((z - mu).t(), z - mu) / (n - 1)
        cov_loss = cov.masked_select(~torch.eye(d, dtype=torch.bool, device=z.device)).pow_(2).sum() / d

        return cov_loss


class VICRegEncoderAtariV2(VICRegEncoderAtari):
    def __init__(self, input_shape, feature_dim):
        super(VICRegEncoderAtariV2, self).__init__(input_shape, feature_dim)

        self.augmentor_a = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        init_xavier_uniform(self.augmentor_a[0])
        init_xavier_uniform(self.augmentor_a[2])

        self.augmentor_b = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        init_xavier_uniform(self.augmentor_b[0])
        init_xavier_uniform(self.augmentor_b[2])

    def loss_function(self, states, next_states):
        aug_a = self.augmentor_a(states)
        aug_b = self.augmentor_b(states)

        x_a = states + aug_a
        x_b = states + aug_b

        # img_a = to_pil_image((aug_a[0] + 1.)/2)
        # img_a.show()
        # img_b = to_pil_image((aug_b[0] + 1.)/2)
        # img_b.show()

        z_a = self.encoder(x_a.detach())
        z_b = self.encoder(x_b.detach())

        return self.vicreg_loss_function(z_a, z_b) + self.augmentor_loss_function(states, x_a, aug_a, x_b, aug_b)

    def augmentor_loss_function(self, states, x_a, aug_a, x_b, aug_b):
        n = x_a.size(0)
        # var = -(F.mse_loss(aug_a, aug_b) + aug_a.reshape(n, -1).std(dim=0).mean() + aug_b.reshape(n, -1).std(dim=0).mean())
        var = self.variance(aug_a.reshape(n, -1)) + self.variance(aug_b.reshape(n, -1)) + self.covariance(aug_a.reshape(n, -1)) + self.covariance(aug_b.reshape(n, -1))
        con = F.mse_loss(states, x_a) + F.mse_loss(states, x_b)

        analytic = ResultCollector()
        analytic.update(augmentor_loss_var=var.unsqueeze(-1).detach(), augmentor_loss_con=con.unsqueeze(-1).detach())

        return var + con


class SpacVICRegEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(SpacVICRegEncoderAtari, self).__init__()

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim
        self.sample_shift = 4

        self.encoder = AtariStateEncoderLarge(input_shape, feature_dim, gain=0.5)

        self.spatial_projection = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        batch_size = states.shape[0]
        #
        # index_a = torch.randint(low=0, high=self.sample_shift, size=(batch_size,), device=self.config.device)
        index_b = torch.randint(low=0, high=self.sample_shift, size=(batch_size,), device=self.config.device)
        # index_a += torch.arange(0, batch_size, device=self.config.device, dtype=torch.int)
        index_b += torch.arange(0, batch_size, device=self.config.device, dtype=torch.int)
        # index_a = index_a.clip_(max=batch_size - 1)
        index_b = index_b.clip_(max=batch_size - 1)

        x_a = states
        x_b = next_states[index_b]

        # y_a = self.augment(x_a)
        # y_b = self.augment(x_b)
        y_a = x_a
        y_b = x_b
        z_a = self.encoder(y_a, fmaps=True)
        z_b = self.encoder(y_b, fmaps=True)

        z_a_f5, z_a = z_a['f5'], z_a['out']
        z_b_f5, z_b = z_b['f5'], z_b['out']

        # global loss
        inv_loss = self.invariance(z_a, z_b)
        var_loss = self.variance(z_a) + self.variance(z_b)
        cov_loss = self.covariance(z_a) + self.covariance(z_b)

        la = 1.
        mu = 1.
        nu = 1. / 25

        global_loss = la * inv_loss + mu * var_loss + nu * cov_loss

        # spatial_loss

        # spatial_loss = 0

        sy = z_a_f5.size(1)
        sx = z_a_f5.size(2)
        N = z_a_f5.size(0)

        p_a = self.spatial_projection(z_a).unsqueeze(1).repeat(1, sy * sx, 1)
        p_b = self.spatial_projection(z_b).unsqueeze(1).repeat(1, sy * sx, 1)
        s_a = z_a_f5.reshape(N, sy * sx, -1)
        s_b = z_b_f5.reshape(N, sy * sx, -1)

        spatial_inv_loss = self.invariance(p_a, s_a) + self.invariance(p_b, s_b)
        # spatial_var_loss = self.variance(s_a, dim=1).mean() + self.variance(s_b).mean()
        # spatial_cov_loss = self.covariance(s_a)  # + self.covariance(z_b)

        return global_loss + spatial_inv_loss

    @staticmethod
    def variance(z, gamma=1, dim=0):
        return F.relu(gamma - z.std(dim)).mean(dim)

    @staticmethod
    def invariance(z1, z2):
        return F.mse_loss(z1, z2)

    @staticmethod
    def covariance(z):
        if len(z.shape) == 3:
            n, f, d = z.shape
            mu = z.mean(dim=1, keepdim=True)
            cov = torch.bmm((z - mu).t(), z - mu) / (n - 1)
            cov_loss = cov.masked_select(~torch.eye(d, dtype=torch.bool, device=z.device)).pow_(2).sum(1) / d
        else:
            n, d = z.shape
            mu = z.mean()
            cov = torch.matmul((z - mu).t(), z - mu) / (n - 1)
            cov_loss = cov.masked_select(~torch.eye(d, dtype=torch.bool, device=z.device)).pow_(2).sum() / d

        return cov_loss

    @staticmethod
    def augment(x, p=0.5):
        ax = x
        ax = aug_random_apply(ax, p, aug_pixelate)
        ax = aug_random_apply(ax, p, aug_mask_tiles)
        ax = aug_random_apply(ax, p, aug_noise)

        return ax


class AMIEncoderAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(AMIEncoderAtari, self).__init__()

        self.config = config
        self.dim = 32
        self.window = 4
        self.action_dim = action_dim
        self.classifier1 = nn.Linear(72, self.dim)
        self.classifier2 = nn.Linear(512, self.dim)

    def forward(self, actions):
        # a = torch.nn.functional.pad(actions, (0, 0, 3, 0))
        a = torch.nn.functional.unfold(actions.unsqueeze(0).unsqueeze(0), kernel_size=(self.window, self.action_dim)).squeeze(0).transpose(0, 1)
        a = self.classifier1(a)
        return a

    def loss_function(self, representations, next_representations, actions):
        N = representations.shape[0] - (self.window - 1)
        state_representation = self.classifier2(next_representations[(self.window - 1):, :] - representations[:-(self.window - 1), :])
        action_representation = self(actions)

        logits = torch.matmul(state_representation, action_representation.T)
        target = torch.arange(N).to(self.config.device)
        loss = nn.functional.cross_entropy(logits, target, reduction='mean')

        return loss
