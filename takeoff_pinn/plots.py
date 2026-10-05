"""Headless figures saved to the run directory (no blocking plt.show())."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_all(fit: dict, history: list, data: dict, uq: dict, run_dir: str,
             t_liftoff: float) -> None:
    os.makedirs(run_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fit["t"], fit["V"], label="PINN V(t)")
    axes[0].scatter(data["t_data"], data["v_data"], s=10, alpha=0.45, label="data")
    if data.get("v_true") is not None:
        axes[0].plot(data["t_dense"], data["v_true"], "--", label="truth")
    for ax in axes:
        if np.isfinite(t_liftoff):
            ax.axvline(t_liftoff, ls="-.", lw=0.8, color="k", label="predicted liftoff (L = W cos γ)")
    axes[0].set(xlabel="t [s]", ylabel="V [m/s]")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(fit["t"], fit["x"], label="PINN x(t)")
    axes[1].scatter(data["t_data"], data["x_data"], s=10, alpha=0.45, label="data")
    if data.get("x_true") is not None:
        axes[1].plot(data["t_dense"], data["x_true"], "--", label="truth")
    axes[1].set(xlabel="t [s]", ylabel="x [m]")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "states.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fit["t"], fit["r_dyn"])
    axes[0].axhline(0, ls="--", lw=0.8)
    axes[0].set(xlabel="t [s]", ylabel="r_dyn [m/s^2]", title="Dynamic residual")
    axes[0].grid(True)
    axes[1].plot(fit["t"], fit["r_kin"])
    axes[1].axhline(0, ls="--", lw=0.8)
    axes[1].set(xlabel="t [s]", ylabel="r_kin [m/s]", title="Kinematic residual")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "residuals.png"), dpi=140)
    plt.close(fig)

    if history:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, key in zip(axes.ravel(), ("mu", "cd_scale", "cl_scale", "eta")):
            ax.plot(epochs, [h[key] for h in history])
            ax.set(xlabel="epoch", ylabel=key)
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "parameters.png"), dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (("x_liftoff_param", "parameters only"),
                       ("x_liftoff", "parameters + input variability")):
        finite = uq[key][np.isfinite(uq[key])]
        if len(finite):
            axes[0].hist(finite, bins=25, alpha=0.6, label=label)
            axes[0].legend(fontsize=8)
        axes[0].set(xlabel="ground roll to liftoff [m]", ylabel="count")
        axes[0].grid(True)
    axes[1].hist(uq["x_final"], bins=25, alpha=0.8)
    axes[1].set(xlabel="x(TF) [m]", ylabel="count")
    axes[1].grid(True)
    fig.suptitle("Correlated UQ of the physics rollout (sandwich covariance)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "uq.png"), dpi=140)
    plt.close(fig)
