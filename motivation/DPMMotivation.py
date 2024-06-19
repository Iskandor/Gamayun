import time
import torch


class DPMMotivation:
    def __init__(self, network, loss, lr, distillation_scale=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._distillation_scale = distillation_scale
        self._device = device

    @property
    def loss(self):
        return self._loss

    def prepare(self, memory, indices):
        sample, size = memory.sample_batches(indices)

        states = sample.state
        actions = sample.action
        next_states = sample.next_state

        return states, actions, next_states

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                actions = sample.action[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states, actions, next_states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("DPM motivation training time {0:.2f}s".format(end - start))

    @staticmethod
    def _error(z_state, pz_state, z_next_state, pz_next_state):
        distillation_error = (z_state - pz_state).pow(2).mean(dim=1, keepdim=True)
        prediction_error = (z_next_state - pz_next_state).pow(2).mean(dim=1, keepdim=True)

        return distillation_error, prediction_error

    def reward(self, z_state, pz_state, z_next_state, pz_next_state):
        distillation_error, prediction_error = self._error(z_state, pz_state, z_next_state, pz_next_state)
        reward = distillation_error * self._distillation_scale

        return reward.clip(0., 1.), distillation_error, prediction_error
