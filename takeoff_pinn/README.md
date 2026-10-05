# takeoff_pinn

Modular, reproducible hybrid PINN for UAV takeoff ground-roll identification.
Replaces the legacy root scripts (kept for provenance; see `../AUDIT_REPORT.md`).

## Quick start

    python -m takeoff_pinn.run --preset quick          # ~2-5 min sanity run
    python -m takeoff_pinn.run --preset full           # thesis-quality run
    python -m takeoff_pinn.run --preset full --estimate-cl   # degeneracy ablation
    python -m pytest                                   # test suite

## Pipeline

1. Synthetic data (or `--csv` with columns `t,x,V`) from the SAME torch physics
   as the PINN residual (no numpy/torch model divergence possible).
2. Output-error (classical SID) initialization of (mu, cd_scale, eta) by pure
   maximum likelihood -- no prior pinning. cl_scale is fixed-known by default
   because it is structurally non-identifiable from ground-roll data
   (AUDIT_REPORT.md section 2.3.2; confirmed numerically: corr(cd,cl) = -1.000).
3. Supervised pretrain of the state net (hard ICs: x(0), V(0), dx/dt(0) exact).
4. AdamW + cosine with physics-weight curriculum; validation-scored checkpoints
   use final weights (no curriculum bias); residual-guided collocation refresh.
5. Optional L-BFGS polish (float64, strong Wolfe, fixed collocation).
6. Identifiability report (sensitivity SVD) + Gauss-Newton covariance
   + correlated Monte-Carlo UQ (parametric-only; documented limitation).

## Physics improvements over the legacy scripts

- Rolling friction opposes the direction of motion (smooth tanh sign); the
  legacy unconditional `-mu*N` made the model roll backwards at idle thrust,
  hidden by a `V >= -0.2` integrator clamp.
- All input profiles are C-infinity (sigmoid blends); the legacy smoothstep
  was C1-only, degrading RK4 to ~2nd order and roughening residuals.
- The acceleration loss scale is `T_MAX/m` (GreatestForm.py used `RPM_MAX/60`,
  i.e. rev/s -- a ~530x under-weighting of the physics residual).
- Forward simulations integrate on a fine grid and interpolate at data times;
  the legacy OE stage integrated on the gappy data grid, biasing estimates.
- Package-wide float64 contract (the v3 float32 regression produced an exactly
  zero finite-difference Jacobian in the OE stage).

## Known identifiability structure (single ground roll, rpm^2 thrust)

- (cd_scale, cl_scale): exactly anti-collinear -> cl fixed by default.
- (mu, eta): correlation ~ -0.9999; separated only by the spool-up/excitation
  window. Their joint uncertainty is reported through the correlated GN
  covariance instead of being hidden by tight priors (the legacy approach).
  Richer throttle excitation (steps/doublets) is the way to shrink it.

## Artifacts per run (`results/<name>/`)

`config.json`, `metrics.json`, `history.json`, `model.pt`,
`states.png`, `residuals.png`, `parameters.png`, `uq.png`
