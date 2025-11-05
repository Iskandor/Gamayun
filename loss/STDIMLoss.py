import torch
import torch.nn as nn
import torch.nn.functional as F


class STDIMLoss(nn.Module):
    def __init__(self, feature_size, local_layer_depth, device):
        super(STDIMLoss, self).__init__()

        self.projection1 = torch.nn.Linear(feature_size, local_layer_depth).to(device)
        self.projection2 = torch.nn.Linear(local_layer_depth, local_layer_depth).to(device)
        self.device = device

    def __call__(self, z_next_state, map_state, map_next_state):
        local_local_loss, local_local_norm = self.local_local_loss(map_state, map_next_state)
        global_local_loss, global_local_norm = self.global_local_loss(z_next_state, map_state)

        loss = global_local_loss + local_local_loss
        norm_loss = global_local_norm + local_local_norm

        return loss, norm_loss

    def global_local_loss(self, z_next_state, map_state):
        # Loss 1: Global at time t, f5 patches at time t-1
        N = map_state.size(0)
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
        N = map_state.size(0)
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
