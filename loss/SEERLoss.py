import torch
import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss


class SEERLoss(nn.Module):
    def __init__(self, config, target_model, learned_model, forward_model, hidden_model):
        super(SEERLoss, self).__init__()
        self.config = config
        self.target_model = target_model
        self.learned_model = learned_model
        self.forward_model = forward_model
        self.hidden_model = hidden_model

    def __call__(self, states, action, next_states):
        z_state = self.target_model(states)
        p_state = self.learned_model(states)

        z_next_state = self.target_model(next_states)
        h_next_state = self.hidden_model(z_next_state)
        p_next_state = self.forward_model(torch.cat([z_state, action, h_next_state], dim=1))

        loss_target = self._target_loss(z_state)
        loss_distillation = self._distillation_loss(p_state, z_state)
        loss_forward = self._forward_loss(p_next_state, z_next_state)
        loss_hidden = self._hidden_loss(h_next_state)

        ResultCollector().update(loss_target=loss_target.unsqueeze(-1).detach().cpu(),
                                 loss_distillation=loss_distillation.unsqueeze(-1).detach().cpu(),
                                 loss_forward=loss_forward.unsqueeze(-1).detach().cpu(),
                                 loss_hidden=loss_hidden.unsqueeze(-1).detach().cpu())

        return loss_target + loss_distillation + loss_forward * self.config.pi + loss_hidden * self.config.eta

    @staticmethod
    def _target_loss(z_state):
        loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1/25
        return loss

    @staticmethod
    def _forward_loss(p_next_state, z_next_state):
        loss = F.mse_loss(p_next_state, z_next_state)
        return loss

    @staticmethod
    def _distillation_loss(pz_state, z_state):
        loss = F.mse_loss(pz_state, z_state.detach())
        return loss

    @staticmethod
    def _hidden_loss(h_next_state):
        loss = torch.abs(h_next_state).mean() + (h_next_state.std(dim=0)).mean()
        return loss
