"""Headless figures saved to the run directory (no blocking plt.show())."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import plotstyle as ps

_PARAM_LABELS = {"mu": "μ", "cd_scale": "c_D scale", "cl_scale": "c_L scale", "eta": "η"}


def _liftoff_marker(ax, t_liftoff: float) -> None:
    if np.isfinite(t_liftoff):
        ax.axvline(t_liftoff, color=ps.INK_2, lw=0.8, ls=(0, (2, 2)))
        ax.annotate("predicted liftoff", xy=(t_liftoff, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(-4, 0), textcoords="offset points", ha="right",
                    fontsize=7.5, color=ps.INK_2)


def save_all(fit: dict, history: list, data: dict, uq: dict, run_dir: str,
             t_liftoff: float) -> None:
    ps.apply()
    os.makedirs(run_dir, exist_ok=True)

    # --- states: one panel per quantity (no dual axes) ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, dkey, tkey, unit in ((axes[0], "V", "v_data", "v_true", "V [m/s]"),
                                      (axes[1], "x", "x_data", "x_true", "x [m]")):
        ax.scatter(data["t_data"], data[dkey], s=9, color=ps.MUTED, alpha=0.6,
                   label="measurements", zorder=1)
        if data.get(tkey) is not None:
            ax.plot(data["t_dense"], data[tkey], color=ps.TRUTH, ls="--", lw=1.2,
                    label="truth", zorder=2)
        ax.plot(fit["t"], fit[key], color=ps.SERIES[0], label="PINN", zorder=3)
        _liftoff_marker(ax, t_liftoff)
        ax.set(xlabel="t [s]", ylabel=unit)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "states.png"), dpi=140)
    plt.close(fig)

    # --- residuals: single series per panel --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title, unit in ((axes[0], "r_dyn", "Dynamic residual", "r_dyn [m/s²]"),
                                 (axes[1], "r_kin", "Kinematic residual", "r_kin [m/s]")):
        ax.axhline(0.0, color=ps.MUTED, lw=0.8)
        ax.plot(fit["t"], fit[key], color=ps.SERIES[0])
        rms = float(np.sqrt(np.mean(np.asarray(fit[key]) ** 2)))
        ax.set(xlabel="t [s]", ylabel=unit, title=f"{title} (RMS {rms:.3g})")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "residuals.png"), dpi=140)
    plt.close(fig)

    # --- parameter trajectories during training ----------------------------
    if history:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, key in zip(axes.ravel(), ("mu", "cd_scale", "cl_scale", "eta")):
            ax.plot(epochs, [h[key] for h in history], color=ps.SERIES[0])
            ax.set(xlabel="epoch", ylabel=_PARAM_LABELS[key])
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "parameters.png"), dpi=140)
        plt.close(fig)

    # --- UQ: parameter-only vs total, same bins ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pools = [(uq["x_liftoff_param"], "parameters only", ps.SERIES[0]),
             (uq["x_liftoff"], "parameters + input variability", ps.SERIES[1])]
    finite_all = np.concatenate([p[np.isfinite(p)] for p, _, _ in pools])
    if len(finite_all):
        bins = np.linspace(finite_all.min(), finite_all.max(), 30)
        for vals, label, color in pools:
            v = vals[np.isfinite(vals)]
            if len(v):
                axes[0].hist(v, bins=bins, color=color, alpha=0.55,
                             label=f"{label} (σ = {v.std():.2f} m)")
        axes[0].set(xlabel="ground roll to liftoff [m]", ylabel="count")
        axes[0].legend(loc="upper right")
    axes[1].hist(uq["x_final"], bins=25, color=ps.SERIES[0], alpha=0.8)
    axes[1].set(xlabel="x(TF) [m], parameters + input variability", ylabel="count")
    fig.suptitle("Correlated UQ of the physics rollout (sandwich covariance)", color=ps.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "uq.png"), dpi=140)
    plt.close(fig)
