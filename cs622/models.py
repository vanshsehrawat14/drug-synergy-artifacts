"""Vansh's invariant dose network, ordered DeepSynergy MLP and minimal baselines."""
import numpy as np
import torch
from torch import nn


def mlp(widths, dropout=0.):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class DoseNetwork(nn.Module):
    augment = False

    def __init__(self, endpoint_dim, cell_dim, hidden=(128, 64), encoder_dim=32, dropout=.1,
                 encoder_layers=1, combine="sum_absdiff"):
        super().__init__()
        if encoder_layers < 1 or combine not in {"sum_absdiff", "concat"}:
            raise ValueError("Unknown endpoint encoder or pair combination")
        # Encode each drug together with its own dose, then combine symmetrically.
        layers, width = [], endpoint_dim
        for _ in range(encoder_layers):
            layers += [nn.Linear(width, encoder_dim), nn.ReLU()]
            width = encoder_dim
        self.encoder = nn.Sequential(*layers)
        # "concat" is an order-dependent ablation; it is not exchange invariant.
        self.combine = combine
        self.head = mlp([2 * encoder_dim + cell_dim, *hidden, 1], dropout)

    def forward(self, a, b, cell):
        a, b = self.encoder(a), self.encoder(b)
        pair = [a + b, torch.abs(a - b)] if self.combine == "sum_absdiff" else [a, b]
        return self.head(torch.cat([*pair, cell], dim=1)).squeeze(1)


class DeepSynergy(nn.Module):
    augment = True

    def __init__(self, endpoint_dim, cell_dim, hidden=(8192, 4096), input_dropout=.2, dropout=.5):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.net = mlp([endpoint_dim * 2 + cell_dim, *hidden, 1], dropout)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def ordered(self, a, b, cell):
        return self.net(self.input_dropout(torch.cat([a, b, cell], dim=1))).squeeze(1)

    def forward(self, a, b, cell):
        return (self.ordered(a, b, cell) + self.ordered(b, a, cell)) / 2


def make_network(spec):
    args = {k: v for k, v in spec.items() if k != "kind"}
    if spec["kind"] == "dose":
        return DoseNetwork(**args)
    if spec["kind"] == "deepsynergy":
        return DeepSynergy(**args)
    raise ValueError("Unknown neural model")


class MeanModel:
    def fit(self, features, target, validation=None):
        self.mean = float(np.mean(target))
        return self

    def predict(self, features):
        return np.full(len(features[0]), self.mean)


class AdditiveRidge:
    """Train-fitted additive endpoint/cell features, usable on unseen pairs."""
    def __init__(self, regularization=1.):
        self.regularization = regularization

    @staticmethod
    def design(features):
        a, b, cell = features
        return np.column_stack([np.ones(len(a)), a.astype(float) + b, cell])

    def fit(self, features, target, validation=None):
        x = self.design(features)
        penalty = np.eye(x.shape[1]) * self.regularization
        penalty[0, 0] = 0
        self.coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ target)
        return self

    def predict(self, features):
        return self.design(features) @ self.coefficients
