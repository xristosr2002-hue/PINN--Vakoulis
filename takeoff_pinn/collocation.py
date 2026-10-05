"""Collocation sampling: uniform skeleton + Latin Hypercube + residual top-k."""
import numpy as np
import torch

from .config import PhysicsConfig, TrainConfig
from .losses import residuals

T_MIN = 0.04  # skip exactly t=0: hard ICs hold there and spool transient dominates


def latin_hypercube(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    return lo + (hi - lo) * u


def make_collocation(phys: PhysicsConfig, tc: TrainConfig, rng: np.random.Generator,
                     extra: np.ndarray | None = None) -> torch.Tensor:
    n_uniform = max(16, tc.n_collocation // 2)
    n_lhs = max(16, tc.n_collocation - n_uniform)
    pts = np.concatenate([
        np.linspace(T_MIN, phys.TF, n_uniform),
        latin_hypercube(n_lhs, T_MIN, phys.TF, rng),
    ])
    if extra is not None and len(extra):
        pts = np.concatenate([pts, np.asarray(extra, dtype=float)])
    pts = np.unique(np.clip(pts, T_MIN, phys.TF))
    t = torch.as_tensor(pts, dtype=torch.get_default_dtype()).reshape(-1, 1)
    return t.requires_grad_(True)


def topk_residual_points(model, phys: PhysicsConfig, scales: dict, n_add: int,
                         rng: np.random.Generator, n_candidates: int = 800) -> np.ndarray:
    cand = np.linspace(T_MIN, phys.TF, n_candidates)
    t = torch.as_tensor(cand, dtype=torch.get_default_dtype()).reshape(-1, 1).requires_grad_(True)
    r_dyn, r_kin = residuals(model, t, phys)
    score = (r_dyn.abs() / scales["a"] + r_kin.abs() / scales["v"]).detach().numpy().ravel()
    pts = cand[np.argsort(score)[-n_add:]]
    pts = pts + rng.normal(0.0, 0.012, size=pts.shape)
    return np.unique(np.clip(pts, T_MIN, phys.TF))
