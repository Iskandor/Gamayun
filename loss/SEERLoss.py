import torch
import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.STDIMLoss import STDIMLoss
from loss.VICRegLoss import VICRegLoss
from modules.PPO_Modules import ActivationStage


class SEERLoss_V1(nn.Module):
    def __init__(self, config, model):
        super(SEERLoss_V1, self).__init__()
        self.config = config
        self.model = model

        self.vicreg_loss = VICRegLoss()
        self.stdim_loss =  STDIMLoss(self.config.feature_dim, self.model.ppo_encoder.local_layer_depth, config.device)

    def __call__(self, states, action, next_states):
        zt_state, zt_next_state, pz_state, z_next_state, pz_next_state, p_action, map_state, map_next_state = self.model(states, action, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        loss_target = self.vicreg_loss(zt_state, zt_next_state)
        loss_stdim, loss_stdim_norm = self.stdim_loss(z_next_state, map_state['f5'], map_next_state['f5'])

        loss_distillation = self._distillation_loss(pz_state, zt_state.detach())
        loss_forward = self._forward_loss(pz_next_state, z_next_state)
        loss_inverse = self._inverse_loss(p_action, action)

        ResultCollector().update(loss_target=loss_target.unsqueeze(-1).detach().cpu(),
                                 loss_stdim=loss_stdim.unsqueeze(-1).detach().cpu(),
                                 loss_stdim_norm=loss_stdim_norm.unsqueeze(-1).detach().cpu(),
                                 loss_distillation=loss_distillation.unsqueeze(-1).detach().cpu(),
                                 loss_forward=loss_forward.unsqueeze(-1).detach().cpu(),
                                 loss_inverse=loss_inverse.unsqueeze(-1).detach().cpu())

        return loss_target + loss_distillation * self.config.delta + loss_forward * self.config.pi + loss_inverse + loss_stdim + loss_stdim_norm * 1e-4

    @staticmethod
    def _vc_loss(z_state):
        loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1 / 25
        return loss

    @staticmethod
    def _forward_loss(p_next_state, z_next_state):
        loss = F.mse_loss(p_next_state, z_next_state)
        return loss

    @staticmethod
    def _inverse_loss(p_action, action):
        loss = F.cross_entropy(p_action, action)
        return loss

    @staticmethod
    def _distillation_loss(p_state, z_state):
        loss = F.mse_loss(p_state, z_state)
        return loss
