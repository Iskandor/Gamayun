import torch
from torch.nn.utils import clip_grad_norm_


class FMMotivation:
    def __init__(self, network, loss, lr, eta=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
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
        error =  torch.mean(torch.pow(p_next_state.view(p_next_state.shape[0], -1) - z_next_state.view(z_next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        reward = (error * self._eta).clip(0., 1.)
        return error, reward


class FMIJEPAMotivation:
    def __init__(self, network, loss, lr, eta=1.0, device='cpu', ema_update_every=1,  grad_clip_norm=None):
        self._network = network
        self._loss = loss
        # Only optimize params that require grad
        trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam(trainable_params, lr=lr)

        self._eta = float(eta)
        self._device = device
        self._ema_update_every = int(ema_update_every)
        self._grad_clip_norm = grad_clip_norm
        self._motivation_steps = 0

    def train(self, memory, indices):
        if not indices:
            return
        sample, size = memory.sample_batches(indices)
        self._network.train()

        for i in range(size):
            states = sample.state[i].to(self._device)
            next_states = sample.next_state[i].to(self._device)
            actions = sample.action[i].to(self._device)

            self._optimizer.zero_grad(set_to_none=True)
            loss = self._loss(states, actions, next_states)
            loss.backward()

            if self._grad_clip_norm is not None:
                clip_grad_norm_(self._network.parameters(), self._grad_clip_norm)

            self._optimizer.step()

            # ---- EMA update of the target encoder ----
            self._motivation_steps += 1
            if hasattr(self._network, "ema_update") and (self._ema_update_every > 0):
                if (self._motivation_steps % self._ema_update_every) == 0:
                    self._network.ema_update()


    def reward(self, z_next_state, p_next_state):
        error =  torch.mean(torch.pow(p_next_state.view(p_next_state.shape[0], -1) - z_next_state.view(z_next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        reward = (error * self._eta).clip(0., 1.)
        return error, reward


class FMIJEPAMotivation2:
    def __init__(self, network, loss, lr, eta=1, device='cpu'):
        self._network = network
        self._loss = loss
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
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
        error =  torch.mean(torch.pow(p_next_state.view(p_next_state.shape[0], -1) - z_next_state.view(z_next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        reward = (error * self._eta).clip(0., 1.)
        return error, reward