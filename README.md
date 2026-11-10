# takeoff_pinn v2 — PINN + output-error identification of UAV ground roll

Rework of the audited v1 package (`../takeoff_pinn/`, tag `baseline-v1`) for the scenario of the
project brief: 15 kg UAV, **S = 5 m²**, ρ = 1.225 kg/m³, 5000 RPM, **13.5 s ground roll**.
Every change relative to the baseline is recorded with its evidence in
[`experiments/LOG.md`](experiments/LOG.md); the baseline itself is [`experiments/E00_baseline.md`](experiments/E00_baseline.md).

## Quick start

```bash
pip install -r requirements.txt
python -m takeoff_pinn.run --preset quick        # ~1 min sanity run (under-fits; mechanics only)
python -m takeoff_pinn.run --preset full         # reference run (~2-4 min CPU)
python -m pytest                                 # test suite (~15 min; integration runs included)
```

Real data: `--csv run.csv` with columns `t,x,V` (SI units, t = 0 at brake release). Pass the
measured wheels-off time if available: `--t-liftoff 13.48 --sigma-t-liftoff 0.02`. With it, c_L is
estimated (default); without it, c_L is fixed at its nominal value with a warning, and c_D is then
conditional on that value (`--fix-cl` forces the fixed mode).
The run requires the RPM/wind/slope profiles of `physics.py` to describe that test — replace
them with the measured inputs before using real data (see *Limitations*).

## Physical model (`takeoff_pinn/physics.py`, single source of truth)

```
dx/dt = V
m dV/dt = η T(rpm, V_air) − D − μ N tanh(V/ε_V) − W sin γ
D = q S C_D0 c_D,  L = q S C_L0 c_L,  q = ½ ρ V_air²,  V_air = V + headwind
N = smooth max(W cos γ − L, 0)            (softplus, scale 0.5 N)
liftoff: first t with L ≥ W cos γ          (ground attitude, no rotation model)
```

Identifiable lumped drag coefficient while N > 0: **κ = C_D0·c_D − μ·C_L0·c_L**.

## Pipeline (`training.run_experiment`)

1. Synthetic data from the same physics (or CSV), stratified sampling, optional observed liftoff time.
2. **Output-error fit** (forward RK4 + least squares), noise-weighted by per-channel variance
   (maximum likelihood, Klein & Morelli relaxation); one extra row for the liftoff time when observed.
3. PINN (tanh MLP, hard initial conditions) initialized at the OE estimate: supervised pretrain →
   AdamW with physics curriculum and residual-guided collocation → L-BFGS polish.
4. Honest uncertainty: sandwich covariance with per-channel noise (validated by Monte-Carlo
   coverage), κ σ, correlated propagation to the liftoff prediction (parameter-only and with
   run-to-run input variability).
5. Metrics: state errors, residuals, parameters (PINN and OE side by side), liftoff
   prediction, a +2 m/s headwind generalization case.

## Configuration

All settings live in the dataclasses of `takeoff_pinn/config.py` and are saved as `config.json`
next to every run. Experiments override fields from the command line:

```bash
python scripts/multiseed.py --tag MY_EXP --seeds 7 17 27 37 47 --jobs 4 --set train.data_weighting=noise
python scripts/oe_montecarlo.py --seeds 40 --jobs 6 --out experiments/MY_EXP/oe_mc.json
```

Unknown override fields raise an error (never a silent no-op). Each seed changes both the noise
realization and the network initialization; `--jobs > 1` changes thread counts, so runtimes are
not comparable to sequential runs (results are statistically, not bitwise, identical).

## Layout

| Path | Content |
|---|---|
| `takeoff_pinn/` | package (physics, data/RK4/events, model, losses, OE, identifiability, UQ, training, CLI) |
| `tests/` | pytest suite (physics hand checks, RK4 order, invariants, covariance honesty, smoke runs) |
| `scripts/` | multi-seed harness, OE Monte-Carlo |
| `experiments/` | `LOG.md` (change log), per-experiment configs/metrics/summaries |


## Reproducibility

float64 everywhere; seeds via `TrainConfig.seed` (restart r uses seed + 1000·r); synthetic data
depends only on the seed and the config. Environment used for the logged experiments:
Python 3.12.2, torch 2.11.0 (CPU), numpy 2.3.5, scipy 1.16.3.

## Limitations

See the final section of `experiments/LOG.md`. In short: synthetic data only (inverse crime),
exogenous inputs (RPM, wind, slope) are fixed analytic profiles, liftoff uses the ground-attitude
lift coefficient (no rotation), thrust is an rpm² surrogate, and network (epistemic) uncertainty
is not quantified.
