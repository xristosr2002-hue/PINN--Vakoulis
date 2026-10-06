"""Run the full PINN pipeline over several seeds and summarize (Agent 4 harness).

Each seed changes BOTH the noise realization and the network initialization.
Overrides use dotted dataclass paths and are validated (an unknown field is an
error, never a silent no-op):

    python scripts/multiseed.py --tag E05 --seeds 7 17 27 37 47 \
        --set train.lbfgs_accept_on_val=true physics.NORMAL_SOFTNESS=0.5
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from takeoff_pinn.run import build_preset  # noqa: E402
from takeoff_pinn.training import run_experiment  # noqa: E402

PARAMS = ("mu", "cd_scale", "cl_scale", "eta", "kappa")
REPORT = ("v_rmse_true", "x_rmse_true", "r_dyn_rms", "r_kin_rms", "t_liftoff_pred",
          "x_liftoff_pred", "ground_roll_pct_err", "liftoff_time_abs_err",
          "x_liftoff_sigma_param", "x_liftoff_sigma_total", "identifiability_max_corr", "runtime_s", "x_liftoff_oe",
          "val_pre_lbfgs", "val_post_lbfgs", "gen_ground_roll_pct_err", "param_corr_max")


def apply_override(exp, assignment: str) -> None:
    path, raw = assignment.split("=", 1)
    *parents, field = path.split(".")
    target = exp
    for p in parents:
        target = getattr(target, p)
    if not hasattr(target, field):
        raise AttributeError(f"unknown config field: {path}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if isinstance(value, list):
        value = tuple(value)
    setattr(target, field, value)


def run_one(seed: int, preset: str, tag: str, out_dir: str, overrides: list, threads: int) -> dict:
    """One full pipeline run (also the process-pool entry point)."""
    if threads:
        torch.set_num_threads(threads)
    exp = build_preset(preset)
    exp.name, exp.out_dir = f"{tag}_s{seed}", out_dir
    exp.train.seed = seed
    for a in overrides:
        apply_override(exp, a)
    m = run_experiment(exp)["metrics"]
    m["seed"] = seed
    m["_truth"] = {"mu": exp.truth.mu, "cd_scale": exp.truth.cd_scale,
                   "cl_scale": exp.truth.cl_scale, "eta": exp.truth.eta,
                   "kappa": m["kappa_true"]}
    return m


def summarize(rows: list[dict]) -> dict:
    out = {"n_seeds": len(rows)}
    for k in REPORT:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                  "max": float(np.nanmax(vals))}
    for p in PARAMS:
        truth = rows[0]["_truth"][p]
        est = np.array([r[p] for r in rows])
        sig = np.array([r.get(f"{p}_sigma", np.nan) for r in rows])
        err = est - truth
        out[p] = {"rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean()),
                  "mean_sigma": float(np.nanmean(sig)),
                  "z_std": float(np.nanstd(err / sig)),
                  "cover68": float(np.nanmean(np.abs(err / sig) < 1.0))}
        if f"{p}_oe" in rows[0]:  # classical output-error estimate, same data
            err_oe = np.array([r[f"{p}_oe"] for r in rows]) - truth
            out[p]["rmse_oe"] = float(np.sqrt(np.mean(err_oe**2)))
    if "x_liftoff_sigma_param" in rows[0]:
        err = np.array([r["x_liftoff_pred"] - r["x_liftoff_true"] for r in rows])
        sig = np.array([r["x_liftoff_sigma_param"] for r in rows])
        out["x_liftoff_z"] = {"rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean()),
                              "mean_sigma": float(sig.mean()), "z_std": float(np.std(err / sig))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    ap.add_argument("--preset", default="full")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.jobs > 1:
        # runtime_s is then not comparable to sequential runs (shared CPU)
        threads = max(1, (os.cpu_count() or 1) // args.jobs)
        n = len(args.seeds)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(run_one, args.seeds, [args.preset] * n, [args.tag] * n,
                                 [args.out_dir] * n, [args.overrides] * n, [threads] * n))
    else:
        rows = [run_one(s, args.preset, args.tag, args.out_dir, args.overrides, 0)
                for s in args.seeds]
    print(f"[{args.tag}] {len(rows)} seeds done ({time.time() - t0:.0f} s)", flush=True)

    summary = summarize(rows)
    record = {"tag": args.tag, "preset": args.preset, "overrides": args.overrides, "jobs": args.jobs,
              "seeds": args.seeds, "summary": summary, "rows": rows}
    out = os.path.join("experiments", args.tag, "multiseed.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
