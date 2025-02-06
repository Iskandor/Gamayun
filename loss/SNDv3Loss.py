import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch_topological.nn import VietorisRipsComplex, SignatureLoss

from analytic.ResultCollector import ResultCollector
from loss.VICRegLoss import VICRegLoss
from modules.PPO_Modules import ActivationStage


class SNDv3Loss(nn.Module):
    def __init__(self, config, model):
        super(SNDv3Loss, self).__init__()
        self.config = config
        self.model = model

        self.vicreg_loss = VICRegLoss()
        self.vr = VietorisRipsComplex(dim=1)
        self.topo_loss = SignatureLoss(p=2, dimensions=1)

    def __call__(self, states):
        zt_state, pz_state = self.model(states, stage=ActivationStage.MOTIVATION_TRAINING)

        sim_loss, vc_loss = self.metric_loss(states, zt_state)
        loss_target = sim_loss + vc_loss
        loss_distillation = self.distillation_loss(pz_state, zt_state.detach())

        ResultCollector().update(sim_loss=sim_loss.unsqueeze(-1).detach().cpu(),
                                 vc_loss=vc_loss.unsqueeze(-1).detach().cpu(),
                                 loss_distillation=loss_distillation.unsqueeze(-1).detach().cpu(),
                                 )

        return loss_target + loss_distillation

    @staticmethod
    def metric_loss(states, z_state):
        permutation = torch.randperm(states.shape[0])

        states_a = states
        states_b = torch.clone(states[permutation])

        z_state_a = z_state
        z_state_b = torch.clone(z_state[permutation])

        ncc_states = SNDv3Loss.normalized_cross_correlation(states_a, states_b, False, reduction='none')
        ncc_z_states = SNDv3Loss.normalized_cross_correlation(z_state_a, z_state_b, False, reduction='none')

        sim_loss = F.l1_loss(ncc_z_states, ncc_states) * 10
        vc_loss = VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1 / 25

        return sim_loss, vc_loss

    @staticmethod
    def distillation_loss(p_state, z_state):
        loss = F.mse_loss(p_state, z_state)
        return loss

    def topology_loss(self, state, z_state):
        batch = state.shape[0]
        s = state.reshape(batch, 1, -1)
        z = z_state.unsqueeze(1)

        pi_s = self.vr(s)
        pi_z = self.vr(z)

        loss = 0

        for i in range(batch):
            loss += self.topo_loss([s[i], pi_z[i]], [z[i], pi_s[i]])
        loss += VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1 / 25
        return loss

    @staticmethod
    def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
        """ N-dimensional normalized cross correlation (NCC)

        Args:
            x (~torch.Tensor): Input tensor.
            y (~torch.Tensor): Input tensor.
            return_map (bool): If True, also return the correlation map.
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.

        Returns:
            ~torch.Tensor: Output scalar
            ~torch.Tensor: Output tensor
        """

        shape = x.shape
        b = shape[0]

        # reshape
        x = x.view(b, -1)
        y = y.view(b, -1)

        # mean
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)

        # deviation
        x = x - x_mean
        y = y - y_mean

        dev_xy = torch.mul(x, y)
        dev_xx = torch.mul(x, x)
        dev_yy = torch.mul(y, y)

        dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
        dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

        # if (dev_xx_sum == 0).nonzero().any() or (dev_yy_sum == 0).nonzero().any():
        #     print(0)

        ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                        torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum) + eps) + eps)
        ncc_map = ncc.view(b, *shape[1:])

        # reduce
        if reduction == 'mean':
            ncc = torch.mean(torch.sum(ncc, dim=1))
        elif reduction == 'sum':
            ncc = torch.sum(ncc)
        elif reduction == 'none':
            ncc = torch.sum(ncc, dim=1)
        else:
            raise KeyError('unsupported reduction type: %s' % reduction)

        if not return_map:
            return ncc

        return ncc, ncc_map

    def ssim_loss(self, states, z_state):
        permutation = torch.randperm(states.shape[0])

        states_a = states
        states_b = torch.clone(states[permutation])
        states_a = (states_a + 1) / 2
        states_b = (states_b + 1) / 2

        states_ssim = ssim(states_a, states_b, data_range=1, size_average=False)

        z_state_a = z_state
        z_state_b = torch.clone(z_state[permutation])

        z_loss = F.mse_loss(z_state_a, z_state_b, reduction='none').mean(dim=1)

        return F.mse_loss(z_loss, (1 - states_ssim)) + VICRegLoss.variance(z_state) + VICRegLoss.covariance(z_state) * 1 / 25
