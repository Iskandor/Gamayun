import time

import torch

from loss.SNDLoss import SNDLoss


class SNDMotivationFactory:
    @staticmethod
    def get_motivation(config, network):
        if config.type == 'vanilla':
            result = SNDVMotivation(network, None, config.motivation_lr, config.motivation_scale, config.device)
        elif config.type == 'vinv' or config.type == 'ami':
            result = SINVMotivation(network, None, config.motivation_lr, config.motivation_scale, config.device)
        elif config.type == 'tp':
            result = TPMotivation(network, None, config.motivation_lr, config.motivation_scale, config.device)
        else:
            result = SNDMotivation(network, SNDLoss(config, network), config.motivation_lr, config.motivation_scale, config.device)

        return result


class SNDMotivation:
    def __init__(self, network, loss, lr, distillation_scale=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._distillation_scale = distillation_scale
        self._device = device

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states, next_states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("SND motivation training time {0:.2f}s".format(end - start))

    @staticmethod
    def _error(z_state, pz_state):
        distillation_error = (z_state - pz_state).pow(2).mean(dim=1, keepdim=True)

        return distillation_error

    def reward(self, z_state, pz_state):
        distillation_error = self._error(z_state, pz_state)
        reward = distillation_error * self._distillation_scale

        return reward.clip(0., 1.), distillation_error


class SNDVMotivation(SNDMotivation):
    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample_batch, size = memory.sample_batches(indices)
            sample = memory.sample(indices)
            states = sample.state

            for i in range(size):
                state_batch = sample_batch.state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(state_batch, states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("SNDV motivation training time {0:.2f}s".format(end - start))


class SINVMotivation(SNDMotivation):
    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)
                actions = sample.action[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states, next_states, actions)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("SINV motivation training time {0:.2f}s".format(end - start))


class TPMotivation(SNDMotivation):
    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)
                dones = sample.done[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._loss(states, next_states, dones)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("TP motivation training time {0:.2f}s".format(end - start))
