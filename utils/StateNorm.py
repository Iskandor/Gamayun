import abc
from abc import ABC

import torch


class Norm(metaclass=abc.ABCMeta):
    def __init__(self, shape, device):
        self._device = device
        self._eps = 0.0000001
        self._mean = torch.zeros(shape, device=device)
        self._var = 0.01 * torch.ones(shape, device=device)
        self._std = (self._var ** 0.5) + self._eps

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '_update') and
                hasattr(subclass, 'process') and
                callable(subclass.process) or
                NotImplemented)

    @abc.abstractmethod
    def update(self, x: torch.Tensor):
        raise NotImplementedError

    def process(self, x: torch.Tensor):
        return (x - self._mean) / self._std

    def mean(self):
        return self._mean

    def get_state(self):
        return {
            'mean': self._mean,
            'var': self._var,
            'std': self._std
        }

    def set_state(self, state):
        self._mean = state['mean'].to(self._device)
        self._var = state['var'].to(self._device)
        self._std = state['std'].to(self._device)


class PreciseNorm(Norm, ABC):
    def __init__(self, shape, device):
        super().__init__(shape, device)
        self._count = 1

    def update(self, x):
        self._count += 1

        mean = self._mean + (x.mean(axis=0) - self._mean) / self._count
        var = self._var + ((x - self._mean) * (x - mean)).mean(axis=0)

        self._mean = mean
        self._var = var

        self._std = torch.sqrt(self._var / self._count) + self._eps


class ExponentialDecayNorm(Norm, ABC):
    def __init__(self, shape, device):
        super().__init__(shape, device)
        self._alpha = 0.99

    def update(self, x):
        mean = x.mean(axis=0)
        self._mean = self._alpha * self._mean + (1.0 - self._alpha) * mean

        var = ((x - mean) ** 2).mean(axis=0)
        self._var = self._alpha * self._var + (1.0 - self._alpha) * var
        self._std = torch.sqrt(self._var) + self._eps
