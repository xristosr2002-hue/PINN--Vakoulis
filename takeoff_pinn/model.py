"""State network with hard initial conditions + bounded physical scalars."""
import math

import numpy as np
import torch
import torch.nn as nn

from .config import ParamSpace, PhysicsConfig


class BoundedScalar(nn.Module):
    """value = low + (high-low) * sigmoid(raw); initialized at a given value."""

    def __init__(self, initial: float, low: float, high: float):
        super().__init__()
        self.low, self.high = float(low), float(high)
        z = float(np.clip((initial - low) / (high - low), 1e-5, 1.0 - 1e-5))
        self.raw = nn.Parameter(torch.tensor(math.log(z / (1.0 - z))))

    def forward(self) -> torch.Tensor:
        return self.low + (self.high - self.low) * torch.sigmoid(self.raw)


class StateNet(nn.Module):
    """tanh MLP on normalized time with hard ICs:
    V = V0 + t * y0,  x = X0 + V0*t + t^2 * y1  (so x(0), V(0), dx/dt(0) exact)."""

    def __init__(self, phys: PhysicsConfig, width: int = 64, depth: int = 4):
        super().__init__()
        self.phys = phys
        layers: list[nn.Module] = []
        last = 1
        for _ in range(depth):
            lin = nn.Linear(last, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            last = width
        head = nn.Linear(last, 2)
        nn.init.xavier_normal_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor):
        tau = 2.0 * t / self.phys.TF - 1.0
        y = self.net(tau)
        V = self.phys.V0 + t * y[:, 0:1]
        x = self.phys.X0 + self.phys.V0 * t + (t**2) * y[:, 1:2]
        return x, V


class TakeoffPINN(nn.Module):
    def __init__(self, phys: PhysicsConfig, space: ParamSpace, jitter: float = 0.0):
        super().__init__()
        self.phys, self.space = phys, space
        self.state_net = StateNet(phys)
        self.mu = BoundedScalar(space.prior_mu + 0.015 * jitter, *space.mu_bounds)
        self.cd_scale = BoundedScalar(space.prior_cd - 0.04 * jitter, *space.cd_bounds)
        self.eta = BoundedScalar(space.prior_eta + 0.01 * jitter, *space.eta_bounds)
        if space.estimate_cl:
            self.cl_scale = BoundedScalar(space.prior_cl, *space.cl_bounds)
        else:
            # Structurally non-identifiable from ground-roll data (audit #2):
            # fixed to the nominal value, not trained.
            self.cl_scale = None
            self.register_buffer("_cl_fixed", torch.tensor(float(space.prior_cl)))

    def forward(self, t: torch.Tensor):
        return self.state_net(t)

    def params_physical(self):
        cl = self.cl_scale() if self.cl_scale is not None else self._cl_fixed
        return self.mu(), self.cd_scale(), cl, self.eta()

    def physical_raw_parameters(self) -> list[nn.Parameter]:
        params = [self.mu.raw, self.cd_scale.raw, self.eta.raw]
        if self.cl_scale is not None:
            params.append(self.cl_scale.raw)
        return params
