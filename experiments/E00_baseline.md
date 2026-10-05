# E00 — Baseline (never overwrite)

| Field | Value |
|---|---|
| Experiment ID | E00 |
| Git commit | `0f73c7e` (annotated tag `baseline-v1`) |
| Date | 2026-10-05 |
| Purpose | Reproducible reference for every later change |
| Code | `PINN/takeoff_pinn/` (copied verbatim to `takeoff_pinn_v2/takeoff_pinn/` before any change) |
| Command | `python -m takeoff_pinn.run --preset full --name baseline_repro` |
| Environment | Python 3.12.2, torch 2.11.0+cpu, numpy 2.3.5, scipy 1.16.3, CPU (8 threads), float64 |
| Artifacts | `E00_baseline/{config,metrics,history}.json` |

## Configuration

- **Physics:** m = 15 kg, S = 0.55 m², ρ = 1.225, C_D0 = 0.055, C_L0 = 0.28, T_MAX = 56 N @ 5000 RPM, TF = 8 s,
  V_LIFTOFF (proxy) = 11 m/s, softplus normal-force scale 5 N, friction tanh scale 0.5 m/s.
- **Truth (synthetic):** μ = 0.080, c_D = 1.150, c_L = 1.000 (fixed, not estimated), η = 0.910.
- **Data:** 160 (+2 anchors) samples, σ_v = 0.10 m/s, σ_x = 0.08 m; 1-in-5 interleaved validation split.
- **Network:** tanh MLP 1→4×64→2, Xavier-normal, hard ICs `V = t·y0`, `x = t²·y1`; **12 738** weights
  + 3 bounded physical scalars (sigmoid box).
- **Optimizer:** OE least-squares init → 900 supervised pretrain epochs (AdamW 2e-3) → 2200 AdamW epochs
  (net lr 1.2e-3, phys lr 2e-4, cosine to 3 %) → L-BFGS (160 it., strong Wolfe). 2 restarts.
- **Loss:** Huber(δ=1) on max-normalized residuals, w_v = w_x = 1, w_dyn = w_kin = 3 with sigmoid
  curriculum (centre 500, width 180), priors weight 0.
- **Collocation:** 64 uniform + 64 LHS on [0.04, 8] s, +40 top-residual points at epochs 600/1300/1900.
- **Seeds:** global seed 7; restart r uses 7 + 1000·r.

## Results (bit-identical to the archived `results/full_v2`)

| Quantity | Value | Truth | Error |
|---|---|---|---|
| μ | 0.07591 | 0.080 | 0.00409 |
| c_D scale | 1.18366 | 1.150 | 0.03366 |
| η | 0.89861 | 0.910 | 0.01139 |
| OE-init θ (μ, c_D, η) | 0.07714, 1.1876, 0.9024 | | |
| V-RMSE vs truth | 0.01139 m/s | | noise 0.10 |
| x-RMSE vs truth | 0.02422 m | | noise 0.08 |
| r_dyn RMS / r_kin RMS | 0.00996 m/s² / 0.01243 m/s | | |
| t to V_LIFTOFF | 5.1172 s | 5.1122 s | 0.0050 s |
| x at V_LIFTOFF | 26.607 m | 26.574 m | 0.124 % |
| GN σ (μ, c_D, η) as computed by code | 0.0136, 0.0332, 0.0362 | | **under-reported, see below** |
| UQ σ(x_LOF) | 0.497 m | | |
| Sensitivity cond. number / max corr | 439 / 0.99991 (μ–η) | | |
| Runtime | 150 s | | |

Final-epoch loss components (selected restart, epoch 2200): total 5.80e-5, v_data 1.49e-5,
x_data 8.7e-7, dyn 1.15e-5 (×w=3), kin 2.5e-6 (×w=3), val score 1.94e-5.

## Known defects of the baseline (found in the 2026-10-05 audit, verified numerically)

1. **GN covariance inconsistency.** S is built at all 162 data times, but the residual (σ²) comes from
   the 130 training points, and the degrees of freedom use 2·162 − 3. So σ is under-reported by 1.25×.
   The consistent pooled σ is 0.0169 / 0.0410 / 0.0451, matching the earlier AUDIT_REPORT table.
2. **Heteroscedastic channels are pooled.** The normalized residual std is v: 0.0055 and x: 0.0011.
   The honest (sandwich) σ of the current unweighted estimator is **0.0224 / 0.0530 / 0.0596**.
   An estimator weighted by per-channel noise (ML output error) would reach 0.0152 / 0.0440 / 0.0402.
3. **κ sign error in documentation.** The identifiable lumped coefficient is
   κ = C_D0·c_D **−** μ·C_L0·c_L, not "+" as written in AUDIT_REPORT.md and the thesis. A pair holding
   the "−" form fixed reproduces the truth trajectory to 3×10⁻⁵ m, while the "+" form deviates by 0.36 m.
4. The PINN stage moves θ by less than 0.1 σ away from the OE initializer, so the parameter estimates
   are essentially the OE estimates. The PINN contributes state reconstruction and residual diagnostics.
