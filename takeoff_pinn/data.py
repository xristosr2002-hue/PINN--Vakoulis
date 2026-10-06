"""Data generation through the SAME torch physics used by the PINN residual."""
import numpy as np
import torch

from .config import DataConfig, PhysicsConfig, TrueParams
from .physics import accel_from_inputs, exogenous_inputs, normal_force_margin


RK4_MAX_DT = 0.02  # s; resolves the ~17 ms rpm blend (4th-order regime, see tests)


def fine_grid(t_eval, t_end: float) -> np.ndarray:
    """Uniform grid with step <= RK4_MAX_DT on [0, t_end], merged with the
    evaluation times so outputs are read at integration nodes (no interpolation)."""
    n = int(np.ceil(t_end / RK4_MAX_DT - 1e-9)) + 1
    return np.union1d(np.asarray(t_eval, dtype=float), np.linspace(0.0, t_end, n))


def _as_batch(value) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.get_default_dtype())
    return t.reshape(-1)


def integrate_rk4(
    t_grid,
    mu, cd_scale, cl_scale, eta,
    cfg: PhysicsConfig,
    rpm_offset=0.0,
    headwind_offset=0.0,
    mu_speed_slope: float = 0.0,
):
    """Classic RK4 for [x' = V, V' = rhs_accel], vectorized over a batch of
    parameter sets. Returns (x, v) of shape (B, len(t_grid)).
    mu_speed_slope: truth-only model-form error (see accel_from_inputs)."""
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
                                 mu_b, cd_b, cl_b, eta_b, cfg, mu_speed_slope)

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


def first_event(t, x, g):
    """First time the event function g drops to <= 0 (from > 0), with linear
    interpolation of t and x; (nan, nan) if it never does on the grid."""
    t, x, g = (np.asarray(a, dtype=float) for a in (t, x, g))
    idx = np.where(g <= 0.0)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    j = int(idx[0])
    if j == 0:
        return float(t[0]), float(x[0])
    frac = float(np.clip(g[j - 1] / (g[j - 1] - g[j]), 0.0, 1.0))
    return (float(t[j - 1] + frac * (t[j] - t[j - 1])),
            float(x[j - 1] + frac * (x[j] - x[j - 1])))


def predict_liftoff(mu, cd_scale, cl_scale, eta, phys: PhysicsConfig,
                    rpm_offset=0.0, headwind_offset=0.0, horizon: float = 2.0,
                    dt: float = 0.01, mu_speed_slope: float = 0.0) -> dict:
    """Liftoff (L = W*cos(gamma)) by forward RK4 with the given parameters,
    integrated up to horizon*TF so events after the record end are still
    found. Batched over parameter sets; returns arrays of t, x, V at liftoff
    and x at TF."""
    t = np.linspace(0.0, horizon * phys.TF, int(round(horizon * phys.TF / dt)) + 1)
    batch = [torch.as_tensor(np.atleast_1d(np.asarray(a, dtype=float)))
             for a in (mu, cd_scale, cl_scale, eta, rpm_offset, headwind_offset)]
    B = max(b.numel() for b in batch)
    mu_b, cd_b, cl_b, eta_b, rpm_b, hw_b = (b.expand(B) if b.numel() == 1 else b for b in batch)
    x, v = integrate_rk4(t, mu_b, cd_b, cl_b, eta_b, phys,
                         rpm_offset=rpm_b, headwind_offset=hw_b, mu_speed_slope=mu_speed_slope)
    with torch.no_grad():
        margin = normal_force_margin(torch.as_tensor(t).reshape(1, -1), v,
                                     cl_b.reshape(-1, 1), phys,
                                     headwind_offset=hw_b.reshape(-1, 1)).numpy()
    x_np, v_np = x.numpy(), v.numpy()
    out = {"t": np.full(B, np.nan), "x": np.full(B, np.nan), "V": np.full(B, np.nan)}
    for i in range(B):
        out["t"][i], out["x"][i] = first_event(t, x_np[i], margin[i])
        if np.isfinite(out["t"][i]):
            out["V"][i] = np.interp(out["t"][i], t, v_np[i])
    out["x_TF"] = np.array([np.interp(phys.TF, t, xi) for xi in x_np])
    return out


EVENT_DT = 1e-3  # s, half-width of the finite-difference stencil for dg/dt at liftoff


def event_nodes(t_obs: float) -> np.ndarray:
    """Grid nodes needed to evaluate the liftoff residual at t_obs."""
    return np.array([t_obs - EVENT_DT, t_obs, t_obs + EVENT_DT])


def liftoff_time_offset(t_grid, v: torch.Tensor, cl_scale, t_obs: float, phys: PhysicsConfig):
    """Linearized liftoff-time error t_L(theta) - t_obs = -g(t_obs) / g'(t_obs),
    g = W*cos(gamma) - L (ground attitude), from states v (B, n) on a grid that
    contains event_nodes(t_obs). Smooth in theta (no event search), so it is
    usable inside least squares and finite-difference sensitivities.
    Returns (offset, gdot), each of shape (B,)."""
    nodes = event_nodes(t_obs)
    t_grid = np.asarray(t_grid, dtype=float)
    idx = np.searchsorted(t_grid, nodes)
    if not np.array_equal(t_grid[np.minimum(idx, len(t_grid) - 1)], nodes):
        raise ValueError("t_grid must contain event_nodes(t_obs)")
    cl = torch.as_tensor(np.atleast_1d(np.asarray(cl_scale, dtype=float))).reshape(-1, 1)
    with torch.no_grad():
        g = normal_force_margin(torch.as_tensor(nodes).reshape(1, 3), v[:, idx], cl, phys)
    gdot = (g[:, 2] - g[:, 0]) / (2.0 * EVENT_DT)
    return (-g[:, 1] / gdot).numpy(), gdot.numpy()


def make_synthetic_data(phys: PhysicsConfig, truth: TrueParams, dcfg: DataConfig, seed: int) -> dict:
    """Dense truth + stratified noisy samples (denser near spool-up and liftoff)."""
    rng = np.random.default_rng(seed)
    t_dense = np.linspace(0.0, phys.TF, dcfg.n_dense)
    k_mu = dcfg.truth_mu_speed_slope  # 0 unless a model-form error is injected (E10)
    x_t, v_t = integrate_rk4(t_dense, truth.mu, truth.cd_scale, truth.cl_scale, truth.eta, phys,
                             mu_speed_slope=k_mu)
    x_true, v_true = x_t[0].numpy(), v_t[0].numpy()

    n1 = int(dcfg.strata_shares[0] * dcfg.n_data)
    n2 = int(dcfg.strata_shares[1] * dcfg.n_data)
    n3 = dcfg.n_data - n1 - n2
    e1, e2 = (f * phys.TF for f in dcfg.strata_edges)
    t_data = np.unique(np.concatenate([
        rng.uniform(0.0, e1, n1),
        rng.uniform(e1, e2, n2),
        rng.uniform(e2, phys.TF, n3),
        [0.0, phys.TF],
    ]))
    x_data = np.interp(t_data, t_dense, x_true) + rng.normal(0.0, dcfg.sigma_x, t_data.shape)
    v_data = np.interp(t_data, t_dense, v_true) + rng.normal(0.0, dcfg.sigma_v, t_data.shape)
    x_data[0], v_data[0] = phys.X0, phys.V0  # known initial condition

    # Observed liftoff (wheels-off) time: a measurement with its own noise,
    # e.g. a weight-on-wheels switch or video. Drawn last so the V/x samples
    # are unchanged by this option.
    t_liftoff_obs = None
    if dcfg.observe_liftoff:
        ev = predict_liftoff(truth.mu, truth.cd_scale, truth.cl_scale, truth.eta, phys,
                             mu_speed_slope=k_mu)
        t_liftoff_obs = float(ev["t"][0] + rng.normal(0.0, dcfg.sigma_t_liftoff))

    return {
        "t_dense": t_dense, "x_true": x_true, "v_true": v_true,
        "t_data": t_data, "x_data": x_data, "v_data": v_data,
        "t_liftoff_obs": t_liftoff_obs, "sigma_t_liftoff": dcfg.sigma_t_liftoff,
        "truth_mu_speed_slope": k_mu,
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
        "t_liftoff_obs": None, "sigma_t_liftoff": None,  # set via the CLI if measured
    }
