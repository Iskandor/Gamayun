import time
import torch


class A2Motivation:
    def __init__(self, network, loss, lr, scale=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._scale = scale
        self._device = device

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("A2 motivation training time {0:.2f}s".format(end - start))

    @staticmethod
    def error(za_state, zb_state, pa_state, pb_state):
        associative_error = (torch.abs(pa_state - za_state) + 1e-8).pow(2).mean(dim=1, keepdim=True) + (torch.abs(pb_state - zb_state) + 1e-8).pow(2).mean(dim=1, keepdim=True)

        return associative_error * 0.5

    def reward(self, za_state, zb_state, pa_state, pb_state):
        reward = self.error(za_state, zb_state, pa_state, pb_state)
        return reward.clip(0., 1.) * self._scale
