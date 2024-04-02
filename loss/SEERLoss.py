import torch
import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss


class SEERLoss(nn.Module):
    def __init__(self, config, model):
        super(SEERLoss, self).__init__()
        self.config = config
        self.model = model

        self.vicreg_loss = VICRegLoss()

    def __call__(self, states, action, next_states):
        zt_state, zl_state, p_state, z_next_state, h_next_state, p_next_state = self.model(states, action, next_states, stage=2)

        # loss_learned = self._vc_loss(zl_state)
        loss_target = self._vicreg_loss(zt_state, z_next_state)
        loss_distillation = self._distillation_loss(p_state, zt_state.detach())
        # loss_forward = self._forward_loss_prior(p_state.detach(), p_next_state, z_next_state.detach())
        loss_forward = self._forward_loss(p_next_state, z_next_state.detach())
        # loss_backward = self._backward_loss(b_state, zl_state)
        loss_hidden = self._hidden_loss(h_next_state)

        ResultCollector().update(loss_target=loss_target.unsqueeze(-1).detach().cpu(),
                                 loss_distillation=loss_distillation.unsqueeze(-1).detach().cpu(),
                                 loss_forward=loss_forward.unsqueeze(-1).detach().cpu(),
                                 # loss_backward=loss_backward.unsqueeze(-1).detach().cpu(),
                                 loss_hidden=loss_hidden.unsqueeze(-1).detach().cpu())

        return loss_target + loss_distillation * self.config.delta + loss_forward * self.config.pi + loss_hidden * self.config.eta

    def _vicreg_loss(self, z_state, z_next_state):
        loss = self.vicreg_loss(z_state, z_next_state)
        return loss

    @staticmethod
    def _vc_loss(z_state):
        loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1/25
        return loss

    @staticmethod
    def _forward_loss(p_next_state, z_next_state):
        # loss = F.mse_loss(p_next_state, z_next_state)
        loss = F.l1_loss(p_next_state, z_next_state)
        return loss

    # exploding feature space
    def _forward_crossentropy_loss(self, p_next_state, z_next_state):
        logits = torch.matmul(p_next_state, z_next_state.T)
        target = torch.arange(self.config.batch_size, dtype=torch.float32, device=self.config.device).unsqueeze(0).repeat(self.config.batch_size, 1)
        loss = F.cross_entropy(logits, target, reduction='mean')

        return loss

    @staticmethod
    def _forward_loss_prior(p_state, p_next_state, z_next_state):
        loss = F.mse_loss(p_next_state, z_next_state) + F.mse_loss(p_next_state, p_state)
        return loss

    @staticmethod
    def _backward_loss(b_state, z_state):
        loss = F.mse_loss(b_state, z_state)
        return loss

    @staticmethod
    def _distillation_loss(p_state, z_state):
        loss = F.mse_loss(p_state, z_state)
        return loss

    @staticmethod
    def _hidden_loss(h_next_state):
        loss = torch.abs(h_next_state).mean() + (h_next_state.std(dim=0)).mean()
        return loss
