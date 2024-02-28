import time
import torch


class A2Motivation:
    def __init__(self, network, lr, scale=1, device='cpu'):
        self._network = network
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
                loss = self._network.loss_function(states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("A2 motivation training time {0:.2f}s".format(end - start))

    def error(self, state):
        return self._network.error(state)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state.to(self._device)

        return self.reward(states)

    def reward(self, state):
        reward = self.error(state)
        return (reward * self._scale).clip(0., 1.)
