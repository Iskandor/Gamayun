import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    def __init__(self):
        super(VICRegLoss, self).__init__()

    def __call__(self, z_a, z_b):
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
