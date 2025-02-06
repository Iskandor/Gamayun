import torch.nn as nn
import torch.nn.functional as F

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss
from modules.PPO_Modules import ActivationStage


class SNDLoss(nn.Module):
    def __init__(self, config, model):
        super(SNDLoss, self).__init__()
        self.config = config
        self.model = model

        self.vicreg_loss = VICRegLoss()

    def __call__(self, states, next_states):
        zt_state, pz_state, z_state_a, z_state_b = self.model(states, next_states, stage=ActivationStage.MOTIVATION_TRAINING)
        # zt_state, pz_state, zt_next_state = self.model(states, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        loss_target = self.vicreg_loss(z_state_a, z_state_b)
        print(loss_target.item())
        loss_distillation = self._distillation_loss(pz_state, zt_state.detach())

        ResultCollector().update(loss_target=loss_target.unsqueeze(-1).detach().cpu(),
                                 loss_distillation=loss_distillation.unsqueeze(-1).detach().cpu(),
                                 )

        return loss_target + loss_distillation

    @staticmethod
    def _distillation_loss(p_state, z_state):
        loss = F.mse_loss(p_state, z_state)
        return loss
