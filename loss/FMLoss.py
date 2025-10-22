import torch
import torch.nn.functional as F
import torch.nn
import math

from modules.PPO_Modules import ActivationStage
from analytic.ResultCollector import ResultCollector

# General Forward and inverse Loss
class FMLoss(torch.nn.Module):
    def __init__(self):
        super(FMLoss, self).__init__()

    @staticmethod
    def _forward_loss(predicted_state, real_state):
        fwd_loss = F.mse_loss(predicted_state, real_state)
        return fwd_loss
    
    @staticmethod
    def _inverse_loss(action_encoder, action_forward_model, actions):
        if actions.dim() == 2:
            actions = actions.argmax(dim=1).long()

        preds_encoder = action_encoder.argmax(dim=1)
        acc_encoder = (preds_encoder == actions).float().mean()
        preds_forward_model = action_forward_model.argmax(dim=1)
        acc_forward_model = (preds_forward_model == actions).float().mean()

        inverse_loss = F.cross_entropy(action_encoder, actions)
        return inverse_loss, acc_encoder, acc_forward_model


# ST-DIM specific loss + general one
class STDIMLoss(FMLoss):
    def __init__(self, model, feature_size, local_layer_depth, device):
        super(STDIMLoss, self).__init__()

        self.model = model
        self.projection1 = torch.nn.Linear(feature_size, local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.projection2 = torch.nn.Linear(local_layer_depth, local_layer_depth).to(device)
        self.device = device

    def __call__(self, states, actions, next_states):
        map_state, map_next_state, p_next_state, action_encoder, action_forward_model = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        map_state_f5 = map_state['f5']
        map_next_state_out, map_next_state_f5 = map_next_state['out'], map_next_state['f5']

        local_local_loss, local_local_norm = self.local_local_loss(map_state_f5, map_next_state_f5)
        global_local_loss, global_local_norm = self.global_local_loss(map_next_state_out, map_state_f5)

        loss = global_local_loss + local_local_loss
        norm_loss = global_local_norm + local_local_norm
        norm_loss *= 1e-4
        
        inverse_loss, acc_encoder, acc_forward_model = super()._inverse_loss(action_encoder, action_forward_model, actions)
        fwd_loss = super()._forward_loss(p_next_state, map_next_state_out)
        total_loss = loss + norm_loss + fwd_loss + inverse_loss

        ResultCollector().update(loss=loss.unsqueeze(-1).detach().cpu(),
                                 norm_loss=norm_loss.unsqueeze(-1).detach().cpu(),
                                 fwd_loss=fwd_loss.unsqueeze(-1).detach().cpu(),
                                 total_loss=total_loss.unsqueeze(-1).detach().cpu(),
                                 acc_encoder=acc_encoder.unsqueeze(-1).detach().cpu(),
                                 acc_forward_model=acc_forward_model.unsqueeze(-1).detach().cpu())
        
        return total_loss

    def global_local_loss(self, z_next_state, map_state):
        # Loss 1: Global at time t, f5 patches at time t-1
        N = z_next_state.size(0)
        sy = map_state.size(1)
        sx = map_state.size(2)
        
        positive = []
        for y in range(sy):
            for x in range(sx):
                positive.append(map_state[:, y, x, :].T)

        predictions = self.projection1(z_next_state)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.device).unsqueeze(0).repeat(logits.shape[0], 1)

        loss = F.cross_entropy(logits, target, reduction='mean')
        norm_loss = torch.norm(logits, p=2, dim=[1, 2]).mean()

        return loss, norm_loss

    def local_local_loss(self, map_state, map_next_state):
        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        N = map_next_state.size(0)
        sy = map_state.size(1)
        sx = map_state.size(2)

        predictions = []
        positive = []
        for y in range(sy):
            for x in range(sx):
                predictions.append(self.projection2(map_next_state[:, y, x, :]))
                positive.append(map_state[:, y, x, :].T)

        predictions = torch.stack(predictions)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.device).unsqueeze(0).repeat(logits.shape[0], 1)

        loss = F.cross_entropy(logits, target, reduction='mean')
        norm_loss = torch.norm(logits, p=2, dim=[1, 2]).mean()

        return loss, norm_loss


# ST-DIM specific loss + general one
class IJEPALoss(FMLoss):
    def __init__(self, model, device, delta):
        super(IJEPALoss, self).__init__()

        self.model = model
        self.device = device
        self.delta = delta

    def __call__(self, states, actions, next_states):
        # Probably need to change the output of AtariLargeEncoder as we don't need f maps
        map_state, map_next_state, p_next_state, h_next_state, action_encoder, action_forward_model = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        var_cov_loss = self._var_cov_loss(self, map_next_state)
        hidden_loss = self._hidden_loss(h_next_state)
        forward_loss = super()._forward_loss(p_next_state, map_next_state)
        inverse_loss, acc_encoder, acc_forward_model = super()._inverse_loss(action_encoder, action_forward_model, actions)

        total_loss = var_cov_loss + hidden_loss * self.delta + forward_loss + inverse_loss

        ResultCollector().update(loss=var_cov_loss.unsqueeze(-1).detach().cpu(),
                                 norm_loss=hidden_loss.unsqueeze(-1).detach().cpu(),
                                 fwd_loss=forward_loss.unsqueeze(-1).detach().cpu(),
                                 total_loss=total_loss.unsqueeze(-1).detach().cpu(),
                                 acc_encoder=acc_encoder.unsqueeze(-1).detach().cpu(),
                                 acc_forward_model=acc_forward_model.unsqueeze(-1).detach().cpu())

        return total_loss
        # return super()._forward_loss(predicted_state, real_state)

    # forward model loss + hidden loss + var/cov loss
    # hidden loss - gradually decreasing

    @staticmethod
    def _hidden_loss(h_next_state):
        loss = torch.abs(h_next_state).mean() + (h_next_state.std(dim=0)).mean()
        return loss

    @staticmethod
    def _var_cov_loss(self, z_state):
        loss = self.variance(z_state) + self.covariance(z_state) * 1 / 25
        return loss

    @staticmethod
    def variance(z, gamma=1):
        return F.relu(gamma - z.std(0)).mean()

    @staticmethod
    def covariance(z):
        n, d = z.shape
        mu = z.mean(0)
        cov = torch.matmul((z - mu).t(), z - mu) / (n - 1)
        cov_loss = cov.masked_select(~torch.eye(d, dtype=torch.bool, device=z.device)).pow_(2).sum() / d

        return cov_loss


class IJEPAHiddenHeadLoss(FMLoss):
    def __init__(self, model, device, total_steps,
                 shrink_start_frac=0.5, shrink_max=1.0,
                 w_align=1.0, w_h_varcov=1.0):
        super().__init__()
        self.model = model
        self.device = device

        # schedule + weights for hidden loss
        self.total_steps = int(total_steps)
        self.shrink_start_frac = float(shrink_start_frac)
        self.shrink_max = float(shrink_max)
        self.w_align = float(w_align)
        self.w_h_varcov = float(w_h_varcov)

        self.register_buffer("_update_idx", torch.zeros((), dtype=torch.long))

    def __call__(self, states, actions, next_states):
        map_state, map_next_state, p_next_state, h_next_state, action_encoder, action_forward_model = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        var_cov_loss = self._var_cov_loss(self, map_next_state)
        forward_loss = super()._forward_loss(p_next_state, map_next_state)
        inverse_loss, acc_encoder, acc_forward_model = super()._inverse_loss(action_encoder, action_forward_model, actions)
        hidden_loss = self._hidden_loss(self, h_next_state, map_next_state)

        total_loss = var_cov_loss + hidden_loss + forward_loss + inverse_loss

        ResultCollector().update(
            loss=var_cov_loss.unsqueeze(-1).detach().cpu(),
            norm_loss=hidden_loss.unsqueeze(-1).detach().cpu(),
            fwd_loss=forward_loss.unsqueeze(-1).detach().cpu(),
            total_loss=total_loss.unsqueeze(-1).detach().cpu(),
            acc_encoder=acc_encoder.unsqueeze(-1).detach().cpu(),
            acc_forward_model=acc_forward_model.unsqueeze(-1).detach().cpu()
        )

        self._update_idx += 1
        return total_loss

    @staticmethod
    def _hidden_loss(self, h_next_state, map_next_state):
        h_as_feat = self.model.proj_hidden_to_z(h_next_state)

        align = self._cosine_align(h_as_feat, map_next_state)
        spread = self._varcov(h_next_state)
        w_shrink = self._shrink_weight(self)
        shrink = (h_next_state ** 2).mean()

        hidden_loss = self.w_align * align + self.w_h_varcov * spread + w_shrink * shrink
        return hidden_loss

    @staticmethod
    def _var_cov_loss(self, z_state):
        loss = self.variance(z_state) + self.covariance(z_state) * 1 / 25
        return loss

    @staticmethod
    def variance(z, gamma=1):
        return F.relu(gamma - z.std(0)).mean()

    @staticmethod
    def covariance(z):
        n, d = z.shape
        mu = z.mean(0)
        cov = torch.matmul((z - mu).t(), z - mu) / max(n - 1, 1)
        cov_loss = cov.masked_select(~torch.eye(d, dtype=torch.bool, device=z.device)).pow_(2).sum() / max(d, 1)
        return cov_loss
    
    @staticmethod
    def _cosine_align(h, z_stopgrad):
        h = F.normalize(h, dim=-1)
        z = F.normalize(z_stopgrad.detach(), dim=-1)
        return 2.0 - 2.0 * (h * z).sum(dim=-1).mean()

    @staticmethod
    def _varcov(x, gamma=1.0, eps=1e-4):
        x = x - x.mean(0, keepdim=True)
        std = torch.sqrt(x.var(0) + eps)
        var_loss = F.relu(gamma - std).mean()

        n, d = x.shape
        if n <= 1:
            cov_loss = torch.zeros((), device=x.device)
        else:
            cov = (x.t() @ x) / (n - 1)
            off = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
            cov_loss = (off ** 2).mean()
        return var_loss + cov_loss

    @staticmethod
    def _shrink_weight(self):
        step = int(self._update_idx.item())
        t = step / max(1.0, float(self.total_steps))
        if t <= self.shrink_start_frac:
            return 0.0
        if t >= 1.0:
            return float(self.shrink_max)
        u = (t - self.shrink_start_frac) / (1.0 - self.shrink_start_frac)
        return float(self.shrink_max * 0.5 * (1.0 - math.cos(math.pi * u)))
    

# LOSS for our experimental architecture with additional hidden forward model
class IJEPAEmaEncoderLoss(FMLoss):
    def __init__(self, model, device):
        super(IJEPAEmaEncoderLoss, self).__init__()

        self.model = model
        self.device = device

    def __call__(self, states, actions, next_states):
        map_state, map_next_state, p_next_state, h_next_state, action_encoder, action_forward_model = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        var_cov_loss = self._var_cov_loss(self, map_next_state)
        hidden_loss = super()._forward_loss(p_next_state, h_next_state)
        forward_loss = super()._forward_loss(p_next_state, map_next_state)
        inverse_loss, acc_encoder, acc_forward_model = super()._inverse_loss(action_encoder, action_forward_model, actions)

        total_loss = var_cov_loss + hidden_loss + forward_loss + inverse_loss

        ResultCollector().update(loss=var_cov_loss.unsqueeze(-1).detach().cpu(),
                                 norm_loss=hidden_loss.unsqueeze(-1).detach().cpu(),
                                 fwd_loss=forward_loss.unsqueeze(-1).detach().cpu(),
                                 total_loss=total_loss.unsqueeze(-1).detach().cpu(),
                                 acc_encoder=acc_encoder.unsqueeze(-1).detach().cpu(),
                                 acc_forward_model=acc_forward_model.unsqueeze(-1).detach().cpu())

        return total_loss

    @staticmethod
    def _var_cov_loss(self, z_state):
        loss = self.variance(z_state) + self.covariance(z_state) * 1 / 25
        return loss

    @staticmethod
    def variance(z, gamma=1):
        return F.relu(gamma - z.std(0)).mean()

    @staticmethod
    def covariance(z):
        n, d = z.shape
        mu = z.mean(0)
        cov = torch.matmul((z - mu).t(), z - mu) / (n - 1)
        cov_loss = cov.masked_select(~torch.eye(d, dtype=torch.bool, device=z.device)).pow_(2).sum() / d

        return cov_loss