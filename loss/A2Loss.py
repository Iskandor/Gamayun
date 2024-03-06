import torch
import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss


class A2Loss(nn.Module):
    def __init__(self, config, model):
        super(A2Loss, self).__init__()
        self.config = config
        self.model = model

    def __call__(self, states):
        za_state, zb_state, ha_state, hb_state, pa_state, pb_state = self.model(states, loss=True)

        loss_encoder = self._encoder_loss(za_state) + self._encoder_loss(zb_state)
        loss_hidden = self._hidden_loss(ha_state) + self._hidden_loss(hb_state)
        loss_associative = self._associative_loss(za_state, pa_state) + self._associative_loss(zb_state, pb_state)

        ResultCollector().update(loss_encoder=loss_encoder.unsqueeze(-1).detach().cpu(),
                                 loss_hidden=loss_hidden.unsqueeze(-1).detach().cpu(),
                                 loss_associative=loss_associative.unsqueeze(-1).detach().cpu(),
                                 )

        return loss_encoder + loss_hidden * self.config.eta + loss_associative * self.config.alpha

    @staticmethod
    def _encoder_loss(z_state):
        loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1/25
        return loss

    @staticmethod
    def _associative_loss(z_state, p_state):
        loss = F.mse_loss(p_state, z_state)
        return loss

    @staticmethod
    def _hidden_loss(h_state):
        loss = torch.abs(h_state).mean() + (h_state.std(dim=0)).mean()
        return loss
