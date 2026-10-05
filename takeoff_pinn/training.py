"""Full training pipeline: OE init -> supervised pretrain -> curriculum AdamW
-> optional L-BFGS polish, with restarts, validation-scored checkpointing,
identifiability report, correlated UQ, and on-disk artifacts."""
import json
import math
import os
import time

import numpy as np
import torch

from .collocation import make_collocation, topk_residual_points
from .config import ExperimentConfig, seed_everything
from .data import load_csv, make_synthetic_data, train_val_split
from .evaluate import compute_metrics, evaluate_model
from .identifiability import (identifiability_report, kappa_gradient,
                              parameter_covariance, sensitivity_matrix)
from .losses import compute_loss, huber_mean
from .model import TakeoffPINN
from .oe import output_error_fit
from .uq import propagate_uncertainty


def _to_col(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=float),
                           dtype=torch.get_default_dtype()).reshape(-1, 1)


def _inject_theta(model: TakeoffPINN, theta: dict) -> None:
    def raw_of(value, low, high):
        z = float(np.clip((value - low) / (high - low), 1e-6, 1 - 1e-6))
        return math.log(z / (1.0 - z))

    with torch.no_grad():
        sp = model.space
        model.mu.raw.fill_(raw_of(theta["mu"], *sp.mu_bounds))
        model.cd_scale.raw.fill_(raw_of(theta["cd_scale"], *sp.cd_bounds))
        model.eta.raw.fill_(raw_of(theta["eta"], *sp.eta_bounds))
        if model.cl_scale is not None:
            model.cl_scale.raw.fill_(raw_of(theta["cl_scale"], *sp.cl_bounds))


def pretrain(model, t_train, x_train, v_train, scales, tc) -> None:
    if tc.pretrain_epochs <= 0:
        return
    opt = torch.optim.AdamW(model.state_net.parameters(), lr=2.0e-3,
                            weight_decay=tc.weight_decay)
    for _ in range(tc.pretrain_epochs):
        opt.zero_grad(set_to_none=True)
        x_p, v_p = model(t_train)
        loss = (huber_mean((v_p - v_train) / scales["v"])
                + huber_mean((x_p - x_train) / scales["x"]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()


def train_restart(restart_id: int, data: dict, exp: ExperimentConfig):
    phys, space, tc = exp.physics, exp.space, exp.train
    rng = seed_everything(tc.seed + 1000 * restart_id)

    tr_idx, va_idx = train_val_split(len(data["t_data"]))
    t_train, x_train, v_train = (_to_col(data[k][tr_idx]) for k in ("t_data", "x_data", "v_data"))
    t_val, x_val, v_val = (_to_col(data[k][va_idx]) for k in ("t_data", "x_data", "v_data"))

    scales = {
        "v": max(1.0, float(np.nanmax(np.abs(data["v_data"])))),
        "x": max(1.0, float(np.nanmax(np.abs(data["x_data"])))),
        "a": phys.a_scale,
    }

    theta0, oe_info = output_error_fit(data["t_data"][tr_idx], data["x_data"][tr_idx],
                                       data["v_data"][tr_idx], phys, space, tc,
                                       seed=restart_id)
    print(f"restart {restart_id + 1}: OE init "
          f"mu={theta0['mu']:.5f} cd={theta0['cd_scale']:.4f} eta={theta0['eta']:.4f} "
          f"(loss={oe_info['loss']:.3e})")
    model = TakeoffPINN(phys, space, jitter=float(restart_id))
    _inject_theta(model, theta0)

    pretrain(model, t_train, x_train, v_train, scales, tc)

    extra = np.array([], dtype=float)
    t_phys = make_collocation(phys, tc, rng, extra)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.endswith("raw")],
         "lr": tc.lr, "weight_decay": tc.weight_decay},
        {"params": model.physical_raw_parameters(), "lr": tc.lr_phys, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.epochs_adam, eta_min=tc.lr * 0.03)

    history: list[dict] = []
    best_state, best_score = None, float("inf")
    for epoch in range(1, tc.epochs_adam + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = compute_loss(model, t_train, x_train, v_train, t_phys,
                                   epoch, phys, space, tc, scales)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch in tc.resample_epochs:
            new_pts = topk_residual_points(model, phys, scales, tc.n_resample_add, rng)
            extra = np.unique(np.concatenate([extra, new_pts]))
            t_phys = make_collocation(phys, tc, rng, extra)

        if epoch % tc.val_every == 0 or epoch == 1:
            model.eval()
            with torch.enable_grad():
                _, vparts = compute_loss(model, t_val, x_val, v_val, t_phys,
                                         None, phys, space, tc, scales)
            score = (vparts["v_data"] + vparts["x_data"]
                     + 0.10 * vparts["dyn"] + 0.10 * vparts["kin"])
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            mu, cd, cl, eta = (float(p.detach()) for p in model.params_physical())
            history.append({**parts, "epoch": epoch, "val_score": score,
                            "mu": mu, "cd_scale": cd, "cl_scale": cl, "eta": eta})

    if best_state is not None:
        model.load_state_dict(best_state)

    if tc.use_lbfgs:
        lbfgs = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.75, max_iter=tc.lbfgs_max_iter, max_eval=tc.lbfgs_max_iter + 60,
            tolerance_grad=1e-10, tolerance_change=1e-12,
            history_size=100, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss_l, _ = compute_loss(model, t_train, x_train, v_train, t_phys,
                                     None, phys, space, tc, scales)
            loss_l.backward()
            return loss_l

        try:
            lbfgs.step(closure)
        except RuntimeError as err:
            print(f"L-BFGS stopped early: {err}")

    model.eval()
    with torch.enable_grad():
        _, vparts = compute_loss(model, t_val, x_val, v_val, t_phys, None,
                                 phys, space, tc, scales)
    final_score = (vparts["v_data"] + vparts["x_data"]
                   + 0.10 * vparts["dyn"] + 0.10 * vparts["kin"])
    return model, history, final_score, scales, oe_info


def run_experiment(exp: ExperimentConfig) -> dict:
    t_start = time.time()
    torch.set_default_dtype(torch.float64)
    seed_everything(exp.train.seed)
    exp.save()

    data = (load_csv(exp.data_csv) if exp.data_csv
            else make_synthetic_data(exp.physics, exp.truth, exp.data, exp.train.seed))

    best = None
    for r in range(exp.train.n_restarts):
        model, history, score, scales, oe_info = train_restart(r, data, exp)
        print(f"restart {r + 1}/{exp.train.n_restarts}: val score = {score:.4e}")
        if best is None or score < best[2]:
            best = (model, history, score, scales, oe_info)
    model, history, score, scales, oe_info = best

    fit = evaluate_model(model, exp.physics)
    truth = exp.truth if exp.data_csv is None else None
    metrics = compute_metrics(fit, data, truth, exp.physics)

    theta_hat = {k: fit[k] for k in ("mu", "cd_scale", "cl_scale", "eta")}
    # Sensitivities at exactly the samples whose residuals estimate the noise
    # (E00 used all data times against training-only residuals: sigma 1.25x low).
    S, names = sensitivity_matrix(theta_hat, oe_info["t"], exp.physics, exp.space,
                                  oe_info["scales"])
    ident = identifiability_report(S, names)
    cov = parameter_covariance(S, np.asarray(oe_info["residual_data"]),
                               weights=oe_info.get("weights"))
    uq = propagate_uncertainty(theta_hat, cov, names, exp.space, exp.physics,
                               seed=exp.train.seed)

    metrics["identifiability_condition_number"] = ident["condition_number"]
    metrics["identifiability_max_corr"] = ident["max_corr"]
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    for j, name in enumerate(names):
        metrics[f"{name}_sigma"] = float(sigma[j])
    grad = kappa_gradient(theta_hat, names, exp.physics)
    metrics["kappa_sigma"] = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    metrics["x_liftoff_uq_mean"] = float(np.nanmean(uq["x_liftoff"]))
    metrics["x_liftoff_uq_std"] = float(np.nanstd(uq["x_liftoff"]))
    metrics["runtime_s"] = time.time() - t_start

    run_dir = exp.run_dir
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    return {"model": model, "fit": fit, "metrics": metrics, "history": history,
            "data": data, "uq": uq, "identifiability": ident, "cov": cov.tolist(),
            "run_dir": run_dir}
