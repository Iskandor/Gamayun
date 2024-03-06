import time
import torch


class SEERMotivation:
    def __init__(self, network, loss, lr, eta=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
        self._device = device

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)
                actions = sample.action[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states, actions, next_states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("Forward model motivation training time {0:.2f}s".format(end - start))

    @staticmethod
    def _error(z_state, p_state, z_next_state, p_next_state):
        distillation_error = (z_state - p_state).pow(2).mean(dim=1, keepdim=True)
        forward_error = (z_next_state - p_next_state).pow(2).mean(dim=1, keepdim=True)

        return distillation_error, forward_error

    def reward(self, z_state, p_state, z_next_state, h_next_state, p_next_state):
        distillation_error, forward_error = self._error(z_state, p_state, z_next_state, p_next_state)

        # forward_scale = 1. / torch.norm(h_next_state, p=2, dim=1, keepdim=True)
        forward_scale = 1.

        reward = (distillation_error + forward_error * forward_scale) / 2.

        return (reward * self._eta).clip(0., 1.), distillation_error, forward_error * forward_scale
