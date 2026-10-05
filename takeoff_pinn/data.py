"""Data generation through the SAME torch physics used by the PINN residual."""
import numpy as np
import torch

from .config import DataConfig, PhysicsConfig, TrueParams
from .physics import accel_from_inputs, exogenous_inputs


def _as_batch(value) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.get_default_dtype())
    return t.reshape(-1)


def integrate_rk4(
    t_grid,
    mu, cd_scale, cl_scale, eta,
    cfg: PhysicsConfig,
    rpm_offset=0.0,
    headwind_offset=0.0,
):
    """Classic RK4 for [x' = V, V' = rhs_accel], vectorized over a batch of
    parameter sets. Returns (x, v) of shape (B, len(t_grid))."""
    t_np = np.asarray(t_grid, dtype=float)
    mu_b, cd_b, cl_b, eta_b = (_as_batch(p) for p in (mu, cd_scale, cl_scale, eta))
    rpm_b, hw_b = _as_batch(rpm_offset), _as_batch(headwind_offset)
    B = max(p.numel() for p in (mu_b, cd_b, cl_b, eta_b, rpm_b, hw_b))
    mu_b, cd_b, cl_b, eta_b, rpm_b, hw_b = (
        p.expand(B).clone() if p.numel() == 1 else p
        for p in (mu_b, cd_b, cl_b, eta_b, rpm_b, hw_b)
    )

    # Exogenous inputs depend on time only: evaluate them once, vectorized, at
    # every grid node and stage midpoint instead of inside each RK4 stage.
    t_mid = 0.5 * (t_np[:-1] + t_np[1:])
    with torch.no_grad():
        nodes = exogenous_inputs(torch.as_tensor(t_np, dtype=torch.get_default_dtype()), cfg)
        mids = exogenous_inputs(torch.as_tensor(t_mid, dtype=torch.get_default_dtype()), cfg)

    def f(inputs, k: int, vv: torch.Tensor) -> torch.Tensor:
        rpm, hw, gamma = (a[k] for a in inputs)
        return accel_from_inputs(vv, rpm + rpm_b, hw + hw_b, gamma,
                                 mu_b, cd_b, cl_b, eta_b, cfg)

    vi = torch.full((B,), float(cfg.V0))
    xi = torch.full((B,), float(cfg.X0))
    xs, vs = [xi], [vi]
    with torch.no_grad():
        for i in range(len(t_np) - 1):
            dt = float(t_np[i + 1] - t_np[i])
            k1v = f(nodes, i, vi)
            k2v = f(mids, i, vi + 0.5 * dt * k1v)
            k3v = f(mids, i, vi + 0.5 * dt * k2v)
            k4v = f(nodes, i + 1, vi + dt * k3v)
            # x' = V, so the x-stages are the V-stage inputs.
            xi = xi + (dt / 6.0) * (vi + 2 * (vi + 0.5 * dt * k1v)
                                    + 2 * (vi + 0.5 * dt * k2v) + (vi + dt * k3v))
            vi = torch.clamp(vi + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v), min=0.0)
            xs.append(xi)
            vs.append(vi)
    return torch.stack(xs, dim=1), torch.stack(vs, dim=1)


def make_synthetic_data(phys: PhysicsConfig, truth: TrueParams, dcfg: DataConfig, seed: int) -> dict:
    """Dense truth + stratified noisy samples (denser near spool-up and liftoff)."""
    rng = np.random.default_rng(seed)
    t_dense = np.linspace(0.0, phys.TF, dcfg.n_dense)
    x_t, v_t = integrate_rk4(t_dense, truth.mu, truth.cd_scale, truth.cl_scale, truth.eta, phys)
    x_true, v_true = x_t[0].numpy(), v_t[0].numpy()

    n1 = int(0.45 * dcfg.n_data)
    n2 = int(0.35 * dcfg.n_data)
    n3 = dcfg.n_data - n1 - n2
    t_data = np.unique(np.concatenate([
        rng.uniform(0.0, 3.0, n1),
        rng.uniform(3.0, 6.8, n2),
        rng.uniform(6.8, phys.TF, n3),
        [0.0, phys.TF],
    ]))
    x_data = np.interp(t_data, t_dense, x_true) + rng.normal(0.0, dcfg.sigma_x, t_data.shape)
    v_data = np.interp(t_data, t_dense, v_true) + rng.normal(0.0, dcfg.sigma_v, t_data.shape)
    x_data[0], v_data[0] = phys.X0, phys.V0  # known initial condition

    return {
        "t_dense": t_dense, "x_true": x_true, "v_true": v_true,
        "t_data": t_data, "x_data": x_data, "v_data": v_data,
    }


def train_val_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-in-5 validation split; t=0 anchor stays in training."""
    idx = np.arange(n)
    val = idx[2::5]
    train = np.array([i for i in idx if i not in set(val.tolist())])
    return train, val


def load_csv(path: str) -> dict:
    """Minimal real-data hook: CSV with header columns t, x, V."""
    raw = np.genfromtxt(path, delimiter=",", names=True)
    order = np.argsort(np.asarray(raw["t"], dtype=float))
    return {
        "t_data": np.asarray(raw["t"], dtype=float)[order],
        "x_data": np.asarray(raw["x"], dtype=float)[order],
        "v_data": np.asarray(raw["V"], dtype=float)[order],
        "t_dense": None, "x_true": None, "v_true": None,
    }
