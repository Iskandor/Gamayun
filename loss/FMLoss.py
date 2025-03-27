import torch
import torch.nn.functional as F
import torch.nn

from modules.PPO_Modules import ActivationStage


# General Forward Loss
class FMLoss(torch.nn.Module):
    def __init__(self):
        super(FMLoss, self).__init__()

    @staticmethod
    def _forward_loss(predicted_state, real_state):
        fwd_loss = F.mse_loss(predicted_state, real_state)
        return fwd_loss


# ST-DIM specific loss + general one
class STDIMLoss(FMLoss):
    def __init__(self, model, feature_size, local_layer_depth, device):
        super(STDIMLoss, self).__init__()

        self.model = model
        self.projection1 = torch.nn.Linear(feature_size, local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.projection2 = torch.nn.Linear(local_layer_depth, local_layer_depth).to(device)
        self.device = device

    def __call__(self, states, actions, next_states):
        map_state, map_next_state, p_next_state = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)
        
        map_state_f5 = map_state['f5']
        map_next_state_out, map_next_state_f5 = map_next_state['out'], map_next_state['f5']

        local_local_loss, local_local_norm = self.local_local_loss(map_state_f5, map_next_state_f5)
        global_local_loss, global_local_norm = self.global_local_loss(map_next_state_out, map_state_f5)

        loss = global_local_loss + local_local_loss
        norm_loss = global_local_norm + local_local_norm
        norm_loss *= 1e-4
        total_loss = loss + norm_loss + super()._forward_loss(p_next_state, map_next_state_out)
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
    def __init__(self, model, device):
        super(IJEPALoss, self).__init__()

        self.model = model
        self.device = device

    def __call__(self, states, actions, next_states):
        # Probably need to change the output of AtariLargeEncoder as we don't need f maps
        map_state, map_next_state, p_next_state, h_next_state = self.model(states, actions, next_states, stage=ActivationStage.MOTIVATION_TRAINING)

        var_cov_loss = self._var_cov_loss(self, map_next_state)
        hidden_loss = self._hidden_loss(h_next_state)
        forward_loss = super()._forward_loss(p_next_state, map_next_state)
        total_loss = var_cov_loss + hidden_loss + forward_loss

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
