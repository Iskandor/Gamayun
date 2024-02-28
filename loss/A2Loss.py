import torch
import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss


class A2Loss(nn.Module):
    def __init__(self, config, encoder_a, encoder_b, hidden_model, associative_model):
        super(A2Loss, self).__init__()
        self.config = config
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.associative_model = associative_model
        self.hidden_model = hidden_model

    def __call__(self, states):
        za_state = self.encoder_a(states)
        zb_state = self.encoder_b(states)
        ha_state = self.hidden_model(za_state)
        hb_state = self.hidden_model(zb_state)
        pa_state = self.associative_model(torch.cat([zb_state, ha_state], dim=1))
        pb_state = self.associative_model(torch.cat([za_state, hb_state], dim=1))

        loss_encoder = self._encoder_loss(za_state) + self._encoder_loss(zb_state)
        loss_hidden = self._hidden_loss(ha_state) + self._hidden_loss(hb_state)
        loss_associative = self._associative_loss(za_state, pa_state) + self._associative_loss(zb_state.detach(), pb_state)

        ResultCollector().update(loss_encoder=loss_encoder.unsqueeze(-1).detach().cpu(),
                                 loss_hidden=loss_hidden.unsqueeze(-1).detach().cpu(),
                                 loss_associative=loss_associative.unsqueeze(-1).detach().cpu(),
                                 )

        return loss_encoder + loss_hidden * self.config.eta + loss_associative

    @staticmethod
    def _encoder_loss(z_state):
        loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1/25
        return loss

    @staticmethod
    def _associative_loss(z_state, p_state):
        loss = F.mse_loss(z_state, p_state)
        return loss

    @staticmethod
    def _hidden_loss(h_state):
        loss = torch.abs(h_state).mean() + (h_state.std(dim=0)).mean()
        return loss
