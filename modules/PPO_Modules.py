from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from agents import ActorType
from modules import init_orthogonal, init_uniform
from utils import one_hot_code


class ActivationStage(Enum):
    INFERENCE = 0
    MOTIVATION_INFERENCE = 1
    MOTIVATION_TRAINING = 2
    TRAJECTORY_UNWIND = 3


class DiscreteHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DiscreteHead, self).__init__()
        self.logits = nn.Linear(input_dim, action_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x):
        logits = self.logits(x)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        action = dist.sample().unsqueeze(1)

        return action, probs

    @staticmethod
    def log_prob(probs, actions):
        actions = torch.argmax(actions, dim=1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions).unsqueeze(1)

        return log_prob

    @staticmethod
    def entropy(probs):
        dist = Categorical(probs)
        entropy = -dist.entropy()
        return entropy.mean()

    @property
    def weight(self):
        return self.logits.weight

    @property
    def bias(self):
        return self.logits.bias


class ContinuousHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ContinuousHead, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Softplus()
        )

        init_uniform(self.mu[0], 0.03)
        init_uniform(self.var[0], 0.03)

        self.action_dim = action_dim

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)

        dist = Normal(mu, var.sqrt().clamp(min=1e-3))
        action = dist.sample()

        return action, torch.cat([mu, var], dim=1)

    @staticmethod
    def log_prob(probs, actions):
        dim = probs.shape[1]
        mu, var = probs[:, :dim // 2], probs[:, dim // 2:]

        p1 = - ((actions - mu) ** 2) / (2.0 * var.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2.0 * np.pi * var))

        log_prob = p1 + p2

        return log_prob

    @staticmethod
    def entropy(probs):
        dim = probs.shape[1]
        var = probs[:, dim // 2:]
        entropy = -(torch.log(2.0 * np.pi * var) + 1.0) / 2.0

        return entropy.mean()


class Actor(nn.Module):
    def __init__(self, model, head, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.head_type = head
        self.head = None
        if head == ActorType.discrete:
            self.head = DiscreteHead
        if head == ActorType.continuous:
            self.head = ContinuousHead
        if head == ActorType.multibinary:
            pass

        self.model = model

    def forward(self, x):
        return self.model(x)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)

    def encode_action(self, action):
        if self.head_type == ActorType.discrete:
            return one_hot_code(action, self.action_dim)
        if self.head_type == ActorType.continuous:
            return action
        if self.head_type == ActorType.multibinary:
            return None  # not implemented


class CriticHead(nn.Module):
    def __init__(self, input_dim, base, n_heads=1):
        super(CriticHead, self).__init__()
        self.base = base
        self.value = nn.Linear(input_dim, n_heads)

    def forward(self, x):
        x = self.base(x)
        value = self.value(x)
        return value

    @property
    def weight(self):
        return self.value.weight

    @property
    def bias(self):
        return self.value.bias


class Critic2Heads(nn.Module):
    def __init__(self, input_dim):
        super(Critic2Heads, self).__init__()
        self.ext = nn.Linear(input_dim, 1)
        self.int = nn.Linear(input_dim, 1)

        init_orthogonal(self.ext, 0.01)
        init_orthogonal(self.int, 0.01)

    def forward(self, x):
        ext_value = self.ext(x)
        int_value = self.int(x)
        return torch.cat([ext_value, int_value], dim=1).squeeze(-1)

    @property
    def weight(self):
        return self.ext.weight, self.int.weight

    @property
    def bias(self):
        return self.ext.bias, self.int.bias


class PPONetwork(torch.nn.Module):
    def __init__(self, config, activation=nn.ReLU):
        super(PPONetwork, self).__init__()

        self.config = config
        self.action_dim = config.action_dim
        self.feature_dim = config.feature_dim
        self.ppo_feature_dim = config.ppo_feature_dim

        self.ppo_encoder = None

        self.critic = nn.Sequential(
            activation(),
            nn.Linear(self.ppo_feature_dim, self.feature_dim),
            activation(),
            nn.Linear(self.feature_dim, 1)
        )

        init_orthogonal(self.critic[1], 0.1)
        init_orthogonal(self.critic[3], 0.01)

        self.actor = nn.Sequential(
            activation(),
            nn.Linear(self.ppo_feature_dim, self.feature_dim),
            activation(),
            DiscreteHead(self.feature_dim, config.action_dim)
        )

        init_orthogonal(self.actor[1], 0.01)
        init_orthogonal(self.actor[3], 0.01)

        self.actor = Actor(self.actor, ActorType.discrete, self.action_dim)

    def forward(self, features):
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs

    def ppo_eval(self, state):
        value, _, probs = PPONetwork.forward(self, self.ppo_encoder(state))
        return value, probs


class PPOMotivationNetwork(PPONetwork):
    def __init__(self, config, activation=nn.ReLU):
        super(PPOMotivationNetwork, self).__init__(config)

        self.critic = nn.Sequential(
            activation(),
            nn.Linear(self.ppo_feature_dim, self.feature_dim),
            activation(),
            Critic2Heads(self.feature_dim)
        )

        init_orthogonal(self.critic[1], 0.1)
        init_orthogonal(self.critic[3], 0.01)
