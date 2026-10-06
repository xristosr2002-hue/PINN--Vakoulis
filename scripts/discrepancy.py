"""Post-hoc model-discrepancy analysis of finished runs (experiment E10).

For every run directory of a tag: rebuild config and synthetic data, reload
the trained network, and compare
  (a) state reconstruction: PINN network vs output-error rollout (the ODE
      integrated with the identified parameters), both against truth;
  (b) discrepancy discovery: the PINN dynamic residual r(t) against the TRUE
      missing acceleration da(t) = a_truth(t, V_true) - a_model(t, V_true; theta_hat).

    python scripts/discrepancy.py --tag E10_model_error --results results
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from takeoff_pinn.config import ExperimentConfig  # noqa: E402
from takeoff_pinn.data import integrate_rk4, make_synthetic_data  # noqa: E402
from takeoff_pinn.evaluate import evaluate_model  # noqa: E402
from takeoff_pinn.model import TakeoffPINN  # noqa: E402
from takeoff_pinn.physics import accel_from_inputs, exogenous_inputs  # noqa: E402

T_SKIP = 0.5  # s; exclude spool-up, where both curves are dominated by the rpm transient


def analyse_run(run_dir: str) -> dict:
    exp = ExperimentConfig.load(os.path.join(run_dir, "config.json"))
    phys = exp.physics
    data = make_synthetic_data(phys, exp.truth, exp.data, exp.train.seed)
    model = TakeoffPINN(phys, exp.space)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))
    fit = evaluate_model(model, phys, n=exp.data.n_dense)
    theta = {k: fit[k] for k in ("mu", "cd_scale", "cl_scale", "eta")}

    t = torch.as_tensor(data["t_dense"])
    v_true = torch.as_tensor(data["v_true"])
    rpm, hw, gamma = exogenous_inputs(t, phys)
    tr = exp.truth
    a_truth = accel_from_inputs(v_true, rpm, hw, gamma, torch.tensor(tr.mu),
                                torch.tensor(tr.cd_scale), torch.tensor(tr.cl_scale),
                                torch.tensor(tr.eta), phys, exp.data.truth_mu_speed_slope)
    a_model = accel_from_inputs(v_true, rpm, hw, gamma,
                                *(torch.tensor(theta[k]) for k in ("mu", "cd_scale", "cl_scale", "eta")),
                                phys)
    da = (a_truth - a_model).numpy()

    x_oe, v_oe = integrate_rk4(data["t_dense"], *(theta[k] for k in ("mu", "cd_scale", "cl_scale", "eta")), phys)
    keep = data["t_dense"] >= T_SKIP
    r = fit["r_dyn"][keep]
    d = da[keep]
    ss_res = float(np.sum((r - d) ** 2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    return {
        "run": os.path.basename(run_dir), "seed": exp.train.seed,
        "v_rmse_pinn": rmse(fit["V"], data["v_true"]), "x_rmse_pinn": rmse(fit["x"], data["x_true"]),
        "v_rmse_oe": rmse(v_oe[0].numpy(), data["v_true"]), "x_rmse_oe": rmse(x_oe[0].numpy(), data["x_true"]),
        "da_rms": float(np.sqrt(np.mean(d**2))), "r_dyn_rms": float(np.sqrt(np.mean(r**2))),
        "residual_vs_discrepancy_corr": float(np.corrcoef(r, d)[0, 1]),
        "residual_vs_discrepancy_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "t": data["t_dense"].tolist(), "r_dyn": fit["r_dyn"].tolist(), "da": da.tolist(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    torch.set_default_dtype(torch.float64)

    runs = sorted(glob.glob(os.path.join(args.results, f"{args.tag}_s*")))
    rows = [analyse_run(r) for r in runs if os.path.exists(os.path.join(r, "model.pt"))]
    keys = ("v_rmse_pinn", "v_rmse_oe", "x_rmse_pinn", "x_rmse_oe", "da_rms", "r_dyn_rms",
            "residual_vs_discrepancy_corr", "residual_vs_discrepancy_r2")
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "std": float(np.std([r[k] for r in rows]))} for k in keys}
    summary["n_runs"] = len(rows)
    out = os.path.join("experiments", args.tag, "discrepancy.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
