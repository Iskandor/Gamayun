import torch
import torch.nn.functional as F


class FMMotivation:
    def __init__(self, network, loss, lr, distillation_scale=1, forward_scale=1, forward_threshold=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._distillation_scale = distillation_scale
        self._forward_scale = forward_scale
        self._forward_threshold = forward_threshold
        self._device = device

    def train(self, memory, indices):
        if indices:
            sample, size = memory.sample_batches(indices)
            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)
                actions = sample.action[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states, actions, next_states)
                loss.backward()
                self._optimizer.step()

    def reward(self, z_next_state, p_next_state):
        # OR
        # error = torch.mean(torch.pow(p_next_state.view(p_next_state.shape[0], -1) - z_next_state.view(z_next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        error = F.mse_loss(p_next_state, z_next_state)
        reward = (error * self._eta).clip(0., 1.)
        return reward
