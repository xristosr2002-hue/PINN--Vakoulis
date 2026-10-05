"""Experiment configuration: one dataclass tree, serialized next to every run."""
import json
import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch


@dataclass
class PhysicsConfig:
    m: float = 15.0           # kg
    g: float = 9.80665        # m/s^2
    rho: float = 1.225        # kg/m^3
    S: float = 0.55           # m^2
    CD0: float = 0.055        # nominal ground-roll drag coefficient
    CL0: float = 0.28         # nominal ground-effect lift coefficient
    T_MAX: float = 56.0       # N, uninstalled thrust at RPM_MAX
    RPM_MAX: float = 5000.0
    V0: float = 0.0           # m/s
    X0: float = 0.0           # m
    TF: float = 8.0           # s
    NORMAL_SOFTNESS: float = 5.0  # N, softplus scale for smooth max(W*cos(g)-L, 0)
    FRICTION_SMOOTH_V: float = 0.5  # m/s, tanh scale making friction oppose motion
    V_LIFTOFF: float = 11.0   # m/s, performance *proxy* threshold (not aerodynamic liftoff)

    @property
    def W(self) -> float:
        return self.m * self.g

    @property
    def a_scale(self) -> float:
        # Acceleration scale for residual normalization. Audit finding #1:
        # GreatestForm.py used RPM_MAX/60 (rev/s) here, under-weighting physics ~530x.
        return self.T_MAX / self.m


@dataclass
class TrueParams:
    """Ground truth used only for synthetic data generation and validation."""
    mu: float = 0.080
    cd_scale: float = 1.150
    cl_scale: float = 1.000
    eta: float = 0.910


@dataclass
class ParamSpace:
    """Bounds and (deliberately off-truth) priors for the discoverable parameters.

    Audit finding #2: with N > 0 the sensitivities of cd_scale and cl_scale are
    exactly anti-collinear, so cl_scale is fixed to its nominal value by default
    and only estimated for the degeneracy ablation (estimate_cl=True).
    """
    mu_bounds: tuple = (0.020, 0.160)
    cd_bounds: tuple = (0.600, 1.700)
    cl_bounds: tuple = (0.700, 1.250)
    eta_bounds: tuple = (0.750, 1.050)
    prior_mu: float = 0.090
    prior_cd: float = 1.000
    prior_cl: float = 1.000
    prior_eta: float = 0.930
    prior_sigma_mu: float = 0.035
    prior_sigma_cd: float = 0.35
    prior_sigma_cl: float = 0.25
    prior_sigma_eta: float = 0.10
    estimate_cl: bool = False


@dataclass
class DataConfig:
    n_data: int = 160
    sigma_v: float = 0.10   # m/s
    sigma_x: float = 0.08   # m
    n_dense: int = 1000     # dense truth grid


@dataclass
class TrainConfig:
    seed: int = 7
    n_restarts: int = 2
    pretrain_epochs: int = 900
    epochs_adam: int = 2200
    lr: float = 1.2e-3
    lr_phys: float = 2.0e-4
    weight_decay: float = 1.0e-6
    grad_clip: float = 20.0
    use_lbfgs: bool = True
    lbfgs_max_iter: int = 160
    n_collocation: int = 128
    resample_epochs: tuple = (600, 1300, 1900)
    n_resample_add: int = 40
    w_v: float = 1.0
    w_x: float = 1.0
    w_dyn: float = 3.0
    w_kin: float = 3.0
    # Parameter-prior weights default to ZERO: verification showed that even a
    # decayed weight of ~0.009 out-pulls the dynamics residual on cd_scale by
    # ~100x once the state fits, dragging the estimate to the prior mean (a
    # 3.4-sigma bias vs the GN covariance). Set nonzero only to encode genuine
    # external knowledge (e.g. real-data engineering priors), and report it.
    w_prior_start: float = 0.0
    w_prior_end: float = 0.0
    prior_decay_tau: float = 1400.0
    ramp_center: float = 500.0
    ramp_width: float = 180.0
    val_every: int = 25
    oe_max_nfev: int = 300


@dataclass
class ExperimentConfig:
    name: str = "takeoff_pinn"
    out_dir: str = "results"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    truth: TrueParams = field(default_factory=TrueParams)
    space: ParamSpace = field(default_factory=ParamSpace)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_csv: str | None = None

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.name)

    def save(self) -> str:
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
