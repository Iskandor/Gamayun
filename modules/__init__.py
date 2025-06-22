from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride, activation, gain):
        super(ResConvBlock, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.act0 = activation()
        self.conv1 = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act1 = activation()

        torch.nn.init.orthogonal_(self.conv0.weight, gain)
        torch.nn.init.zeros_(self.conv0.bias)
        torch.nn.init.orthogonal_(self.conv1.weight, gain)
        torch.nn.init.zeros_(self.conv1.bias)

        # bypass conv, learnable skip connection
        self.conv_bp = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)

        torch.nn.init.orthogonal_(self.conv_bp.weight, gain)
        torch.nn.init.zeros_(self.conv_bp.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)
        y = self.conv1(y)

        y = self.act1(y + self.conv_bp(x))

        return y


class ResMLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = sqrt(2)):
        super(ResMLPBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.GELU(),
            nn.Linear(out_features, out_features, bias),
        )

        init_orthogonal(self.block[0], gain)
        init_orthogonal(self.block[2], gain)

        self.skip = nn.Linear(in_features, out_features, bias)
        init_orthogonal(self.skip, gain)

    def forward(self, x):
        y = self.skip(x) + self.block(x)
        return y


class AttentionBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = sqrt(2)):
        super(AttentionBlock, self).__init__()
        self.query_proj = nn.Linear(in_features, out_features, bias)
        self.key_proj = nn.Linear(in_features, out_features, bias)
        self.value_proj = nn.Linear(in_features, out_features, bias)

        init_orthogonal(self.query_proj, gain)
        init_orthogonal(self.key_proj, gain)
        init_orthogonal(self.value_proj, gain)

    def forward(self, z_state, past_predictions):
        """
        Forward pass with temporal attention.

        z_state: Current state (B, feature_dim)
        past_predictions: Past state predictions (B, T, feature_dim)
        """
        # Compute attention scores
        query = self.query_proj(z_state).unsqueeze(1)  # (B, 1, out_features)
        keys = self.key_proj(past_predictions)  # (B, T, out_features)
        values = self.value_proj(past_predictions)  # (B, T, out_features)

        # Attention weights: softmax over dot-product of query and keys
        attention_scores = torch.bmm(query, keys.transpose(1, 2))  # (B, 1, T)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, 1, T)

        # Compute context vector as weighted sum of values
        context_vector = torch.bmm(attention_weights, values).squeeze(1)  # (B, out_features)

        return context_vector


def init_custom(layer, weight_tensor):
    layer.weight = torch.nn.Parameter(torch.clone(weight_tensor))
    nn.init.zeros_(layer.bias)


def init_coupled_orthogonal(layers, gain=1.0):
    weight = torch.zeros(len(layers) * layers[0].weight.shape[0], *layers[0].weight.shape[1:])
    nn.init.orthogonal_(weight, gain)
    weight = weight.reshape(len(layers), *layers[0].weight.shape)

    for i, l in enumerate(layers):
        init_custom(l, weight[i])


def init_orthogonal(layer, gain=1.0):
    __init_general(nn.init.orthogonal_, layer, gain)


def init_xavier_uniform(layer, gain=1.0):
    __init_general(nn.init.xavier_uniform_, layer, gain)


def init_uniform(layer, gain=1.0):
    __init_general(nn.init.uniform_, layer, gain)


def __init_general(function, layer, gain):
    if type(layer.weight) is tuple:
        for w in layer.weight:
            function(w, gain)
    else:
        function(layer.weight, gain)

    if layer.bias is not None:
        if type(layer.bias) is tuple:
            for b in layer.bias:
                nn.init.zeros_(b)
        else:
            nn.init.zeros_(layer.bias)
