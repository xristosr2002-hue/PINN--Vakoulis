# takeoff-pinn — PINN and output-error identification of UAV takeoff ground roll

Research code for the diploma thesis *Development of a Physics-Informed Neural Network for Enhanced
UAV Takeoff Performance Prediction and Physics Discovery* (Christos Vakoulis).

The package reconstructs the ground-roll trajectory `x(t), V(t)` of a small fixed-wing UAV from
noisy measurements, identifies the physical parameters of its force balance (rolling friction,
drag and lift corrections, thrust installation efficiency) with honest uncertainty, and predicts the
liftoff time and ground-roll distance.

> **Status: needs further development.** The method is validated on synthetic data only (see
> [Current status](#current-development-status)). In the final configuration the classical
> output-error (OE) method drives parameter identification; the PINN acts as a physics-constrained
> state estimator. This is a reported result, not a limitation hidden by tuning.

## Scientific objectives

1. Predict takeoff performance (liftoff time, ground-roll distance) of a 15 kg UAV with quantified
   uncertainty.
2. Identify physical parameters of the ground-roll dynamics from a single takeoff record
   ("physics discovery"), and state which combinations are actually identifiable.
3. Evaluate, on evidence over multiple seeds, when a PINN adds value over the classical
   output-error method — including under model-form error.

## Physical problem

Scenario of the project brief: m = 15 kg, wing area S = 5 m², ρ = 1.225 kg/m³, 5000 RPM, a 13.5 s
ground roll ending at liftoff. 1-D point-mass ground roll, time-only inputs (RPM spool-up with
excitation, headwind, runway slope) defined in `takeoff_pinn/physics.py`.

### Governing equations (as implemented in `takeoff_pinn/physics.py`)

```
dx/dt = V
m dV/dt = η·T(rpm, V_air) − D − μ·N·tanh(V / 0.5) − W·sin γ
T = T_max (rpm/rpm_max)² · clip(1 − 0.01·V_air, 0.70, 1.05)      T_max = 45.44 N
D = q S C_D0 c_D,   L = q S C_L0 c_L,   q = ½ ρ V_air²,   V_air = max(V + headwind, 0)
N = 0.5 · softplus((W cos γ − L) / 0.5)                          smooth max(W cos γ − L, 0) [N]
liftoff: first t with W cos γ − L = 0                            (ground attitude, no rotation)
```

with C_D0 = 0.055 and C_L0 = 0.28. Unknown parameters: μ (rolling friction), c_D, c_L (drag and
lift correction factors) and η (thrust installation efficiency).
While the wheels carry load only the lumped drag coefficient
**κ = C_D0·c_D − μ·C_L0·c_L** is identifiable; c_L becomes identifiable when the liftoff time is
observed (experiments C9/E08 in [`experiments/LOG.md`](experiments/LOG.md)).

## Methodology

Pipeline (`takeoff_pinn/training.py::run_experiment`):

1. **Data.** Synthetic measurements from the same physics (seeded noise σ_V = 0.10 m/s,
   σ_x = 0.08 m, 160 stratified samples, observed liftoff time σ_t = 0.02 s), or a CSV of real data.
2. **Inverse parameter identification — output error.** Forward RK4 simulation + bounded least
   squares, noise-weighted by per-channel variance re-estimated from the residuals (maximum
   likelihood), with one extra row for the observed liftoff time (`takeoff_pinn/oe.py`).
3. **PINN state estimation** initialized at the OE estimate (see below). By default the physical
   parameters stay frozen at the OE estimate (`TrainConfig.pinn_trains_params = False`; evidence E09b).
4. **Identifiability and uncertainty.** Sensitivity matrix, condition number and correlations;
   sandwich covariance with per-channel noise (validated by Monte-Carlo coverage); correlated
   propagation to the liftoff prediction, parameter-only and with run-to-run input variability
   (`identifiability.py`, `uq.py`). Estimates pinned at a bound are flagged.
5. **Metrics** (`evaluate.py`): state errors vs truth, residual RMS, parameters (PINN and OE side by
   side), liftoff prediction, and a +2 m/s headwind generalization case.

### Neural network architecture (`takeoff_pinn/model.py`)

tanh MLP 1 → 4×64 → 2 on normalized time τ = 2t/TF − 1, Xavier-normal initialization
(12 738 weights). Hard initial conditions through the output transform
`V = V0 + t·y0`, `x = X0 + V0·t + t²·y1`, so x(0), V(0) and dx/dt(0) are exact. Physical
parameters are bounded scalars `low + (high − low)·sigmoid(raw)`. float64 throughout.

### Physics-informed loss (`takeoff_pinn/losses.py`)

```
J = w_v H((V − V_d)/v_s) + w_x H((x − x_d)/x_s)
  + ρ(epoch) [ w_dyn H(r_dyn/a_s) + w_kin H(r_kin/v_s) ] + w_prior·prior + event term
r_dyn = dV/dt − f(t, V; θ),   r_kin = dx/dt − V   (automatic differentiation)
```

H is the Huber loss (δ = 1), a_s = T_max/m, w_v = w_x = 1, w_dyn = w_kin = 3, ρ is a sigmoid
curriculum (centre 500, width 180 epochs), prior weights default to 0. Collocation: half uniform,
half Latin hypercube on [t_min, TF], plus 40 top-residual points added at epochs 600/1300/1900
(`collocation.py`).

### Optimization (`takeoff_pinn/training.py`)

Supervised pretraining (900 epochs, AdamW) → AdamW with the physics curriculum (2200 epochs,
lr 1.2e-3 for the network and 2e-4 for physical parameters, cosine schedule to 3 %, weight decay
1e-6, gradient clipping 20) → L-BFGS polish (160 iterations, strong Wolfe line search).
2 restarts (restart r uses seed + 1000·r); the restart with the best validation score is kept.

## Installation

Python ≥ 3.10 (experiments were run with Python 3.12.2, CPU).

```bash
git clone https://github.com/xristosr2002-hue/PINN--Vakoulis.git
cd PINN--Vakoulis
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                  # optional: installs the `takeoff-pinn` command
```

## Running

```bash
python -m takeoff_pinn.run --preset quick        # ~1 min sanity run (under-fits; mechanics only)
python -m takeoff_pinn.run --preset full         # reference run (~2-4 min CPU)
python -m pytest                                 # test suite (~20 min; integration runs included)
```

Each run writes `config.json`, `provenance.json`, `metrics.json`, `history.json`, `model.pt` and
figures to `results/<name>/` (ignored by git).

Experiments override any config field from the command line (unknown fields raise an error):

```bash
python scripts/multiseed.py --tag MY_EXP --seeds 7 17 27 37 47 --jobs 4 --set train.data_weighting=noise
python scripts/oe_montecarlo.py --seeds 40 --jobs 6 --out experiments/MY_EXP/oe_mc.json
```

Real data: `--csv run.csv` with columns `t,x,V` (SI units, t = 0 at brake release), and the
measured wheels-off time if available: `--t-liftoff 13.48 --sigma-t-liftoff 0.02`. With it c_L is
estimated (default); without it c_L is fixed at its nominal value with a warning and c_D is
conditional on that value (`--fix-cl` forces the fixed mode). The RPM/wind/slope profiles of
`physics.py` must first be replaced by the measured inputs of that test (see *Limitations*).

## Repository structure

| Path | Content |
|---|---|
| `takeoff_pinn/` | package: `physics` (equations), `data` (RK4, events, synthetic data), `model`, `losses`, `collocation`, `oe` (output error), `identifiability`, `uq`, `training`, `evaluate`, `plots`, `provenance`, `run` (CLI) |
| `tests/` | pytest suite: physics hand checks, RK4 order, invariants, covariance honesty, smoke runs |
| `scripts/` | multi-seed harness, OE Monte-Carlo, discrepancy analysis |
| `experiments/` | `LOG.md` (every change with its evidence), per-experiment configs/metrics, `TEMPLATE.md` |
| `docs/` | `TECHNICAL_REPORT.md` (audit, revision and validation), `WORKFLOW.md` (git and experiment workflow) |

## Reproducibility

- float64 everywhere; seeds via `TrainConfig.seed`; synthetic data depends only on the seed and
  the config.
- Every run saves its full `config.json` (reload with `ExperimentConfig.load`) and a
  `provenance.json` with the git commit, a dirty-tree flag and the Python/torch/numpy/scipy
  versions. A run is reproducible from (commit, config, seed); a dirty flag means it is not.
- Logged experiments: Python 3.12.2, torch 2.11.0 (CPU), numpy 2.3.5, scipy 1.16.3.
  `--jobs > 1` changes thread counts, so parallel results are statistically, not bitwise,
  identical to sequential ones.
- Release tags: `v1.0.0` = audited v1 package (baseline E00), `v2.0.0` = code that produced
  E00–E10 and `experiments/final_reference`.

**Historical records.** The records in `experiments/` and `docs/TECHNICAL_REPORT.md` were written
in the author's working repository, where this code lived in `takeoff_pinn_v2/` and the v1
package in `takeoff_pinn/`. There, `0f73c7e` / `baseline-v1` corresponds to tag `v1.0.0` here.
The records are kept verbatim.

## Experimental validation methodology

- Synthetic truth with known parameters; every metric is computed **against truth**, never judged
  on training loss.
- One change at a time against the preserved baseline E00, logged as
  CHANGE / WHY / EXPECTED / ACTUAL / DECISION in [`experiments/LOG.md`](experiments/LOG.md).
- Decisions use 10 seeds (pipeline, `scripts/multiseed.py`) or 40 seeds (OE Monte-Carlo,
  `scripts/oe_montecarlo.py`); uncertainty honesty is checked by Monte-Carlo coverage (z-scores).
- Robustness: a model-form error injected into the truth only (E10) and a +2 m/s headwind
  generalization case.

## Current development status

Reference run (`experiments/final_reference`, full preset, seed 7): liftoff time 13.499 ± 0.019 s
(truth 13.496 s), ground roll 113.49 ± 0.25 m (truth 113.46 m), c_L 1.0009 ± 0.0016 (truth 1.000).
Assessment from [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md): software, mathematical
model, PINN training and synthetic results pass; **overall: needs further development**.

### Limitations

Synthetic data only (inverse crime); exogenous inputs are analytic profiles; liftoff uses the
ground-attitude lift coefficient (no rotation model); thrust is an rpm² surrogate with constant η;
the scenario is marginal (liftoff near terminal speed); μ, c_D and η lie in a joint, strongly
correlated valley; network (epistemic) uncertainty is not quantified. Under model-form error (E10)
the PINN neither beats the OE rollout nor detects the missing force. Details and future work:
`docs/TECHNICAL_REPORT.md` §18–19.
