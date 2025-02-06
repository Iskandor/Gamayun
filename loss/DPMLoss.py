import torch
import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss
from modules.PPO_Modules import ActivationStage


class DPMLoss(nn.Module):
    def __init__(self, config, model):
        super(DPMLoss, self).__init__()
        self.config = config
        self.horizon = config.motivation_horizon
        self.model = model

        self.vicreg_loss = VICRegLoss()

    def __call__(self, states, actions, next_states):
        zt_state, pzt_state, zt_next_state, zf_state, zf_next_state, pzf_next_state = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        loss_encoder = self._vc_loss(zf_state)
        loss_target = self.vicreg_loss(zt_state, zt_next_state)
        loss_distillation = self._distillation_loss(pzt_state, zt_state.detach())
        loss_prediction = self._prediction_loss(pzf_next_state, zf_next_state.detach())
        loss_prediction_t0 = torch.clone(loss_prediction)

        for t in range(1, self.horizon):
            pzf_next_state = self.model(None, actions[t:], next_states[t:], pzf_next_state[:-1], stage=ActivationStage.TRAJECTORY_UNWIND)
            loss_prediction += self._prediction_loss(pzf_next_state, zf_next_state[t:].detach())

        loss_prediction_tH = torch.clone(loss_prediction)

        ResultCollector().update(loss_target=loss_target.unsqueeze(-1).detach().cpu(),
                                 loss_encoder=loss_encoder.unsqueeze(-1).detach().cpu(),
                                 loss_distillation=loss_distillation.unsqueeze(-1).detach().cpu(),
                                 loss_prediction_t0=loss_prediction_t0.unsqueeze(-1).detach().cpu(),
                                 loss_prediction_tH=loss_prediction_tH.unsqueeze(-1).detach().cpu(),
                                 )
        return loss_encoder + loss_target + loss_distillation + loss_prediction

    @staticmethod
    def _distillation_loss(p_state, z_state):
        loss = F.mse_loss(p_state, z_state)
        return loss

    @staticmethod
    def _prediction_loss(p_next_state, z_next_state):
        loss = F.mse_loss(p_next_state, z_next_state)
        return loss

    @staticmethod
    def _vc_loss(z_state):
        loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1 / 25
        return loss
