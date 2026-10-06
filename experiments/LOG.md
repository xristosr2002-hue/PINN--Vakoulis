# Change & experiment log (takeoff_pinn_v2)

Every change is compared with E00 (`E00_baseline.md`, tag `baseline-v1`).
Format per Rule 6: CHANGE / WHY / EXPECTED EFFECT / ACTUAL EFFECT / DECISION.

---

## C1 — Correct κ and report it as a metric

- **Requested by:** Agent 2 (physics). Verified by Agents 1 and 3. Agent 4 confirmed no numeric effect on training.
- **CHANGE:** Added `identifiability.lumped_drag_coefficient` (κ = C_D0·c_D − μ·C_L0·c_L) and the
  metrics `kappa`, `kappa_true` and `kappa_rel_err`. Added a regression test that pins the invariant direction.
- **WHY:** While N > 0, the resistance is μ·W·cosγ + q·S·(C_D0·c_D − μ·C_L0·c_L). Lift *reduces*
  friction, so the sign is "−". The v1 AUDIT_REPORT and the thesis print "+", and their
  "κ matched truth to 0.12 %" check used the wrong formula. With c_L fixed, the reported c_D is
  conditional on the c_L assumption; κ is not.
- **EXPECTED:** No change to training or estimates. Correct identifiable quantity is reported.
- **ACTUAL:** Moving (c_D, c_L) at constant κ(−) changes x(t) by < 1e-3 m (3e-5 m measured).
  The "+" combination changes it by 0.36 m. New tests pass (41/41 in the full suite at C2).
- **DECISION:** Keep. Thesis and audit text are corrected in the v2 thesis sources (see T1).

---

## C2 / E02 — Remove per-call overhead in RK4, OE Jacobian and sensitivities (Agent 1)

- **CHANGE:** Exogenous inputs (rpm, headwind, slope) are evaluated once per integration grid
  (`physics.exogenous_inputs` + `accel_from_inputs`; `rhs_accel` is unchanged for callers).
  The OE fit gets scipy's own 2-point Jacobian rule through one batched RK4 call, and
  `sensitivity_matrix` batches its 2p perturbations.
- **WHY:** Profiling showed integration cost is torch call overhead (0.77 s / 530 steps; B=8 costs the
  same as B=1). The OE stage took 28 s per restart, about 37 % of a full run.
- **EXPECTED:** Same numbers, faster.
- **ACTUAL:** RK4 is bit-identical (max |Δ| = 0) and 1.9× faster. OE takes 7.8 s instead of 28.4 s, with
  estimates within 1e-6 (2e-5 σ). Sensitivities are bit-identical, 0.4 s instead of 6.2 s.
  **Full preset: every metric bit-identical to E00, runtime 150 s → 80 s.** Test suite 169 s → 57 s.
- **DECISION:** Keep (Agent 3: the equation of motion is still written in exactly one place).

---

## C3 / E03 — Honest parameter covariance (Agents 1 + 4; challenged by 3)

- **CHANGE:** `identifiability.parameter_covariance` is the sandwich covariance with per-channel ([V; x])
  noise variance. It raises if residual and S rows are misaligned. The pipeline builds S at the
  OE training samples (`oe_info["t"]`). Metrics `*_gn_sigma` are renamed to `*_sigma` (old values
  were wrong), and `kappa_sigma` is added (linear propagation).
- **WHY:** E00 combined S at 162 data times with residuals from 130 training points (σ 1.25× low) and
  pooled two channels whose normalized noise differs about 5×.
- **EXPECTED:** z = err/σ has unit spread across noise realizations.
- **ACTUAL (Monte-Carlo, 40 seeds, `scripts/oe_montecarlo.py`, `E03_covariance/oe_mc_unweighted.json`):**

  | σ variant | z-std μ / c_D / η | 68 % cover | 95 % cover |
  |---|---|---|---|
  | E00 code | 1.68 / 1.73 / 1.68 | 0.43–0.45 | 0.725 |
  | pooled, consistent | 1.35 / 1.40 / 1.35 | 0.50–0.60 | 0.78–0.80 |
  | **sandwich (C3)** | **1.02 / 1.08 / 1.02** | 0.65–0.68 | 0.90–0.95 |

  Empirical OE RMSE: μ 0.0249, c_D 0.0628, η 0.0662 (bias −0.005 / −0.001 / −0.015, all < 0.25 RMSE).
  Full preset (seed 7): estimates unchanged. σ = 0.0225 / 0.0534 / 0.0600, κ σ = 0.0084.
  z = −0.18 / 0.63 / −0.19 and κ z = 0.36. UQ σ(x_LOF) 0.497 → 0.513 m. Tests 44/44.
- **DECISION:** Keep. The E00 "all within 1σ" claim still holds, with honest σ about 1.3× the thesis values
  and 1.7× the values the v1 code printed.

---

## C4 / E04 — Slide scenario: S = 5 m², TF = 13.5 s, physical liftoff (user decision + Agent 2)

- **CHANGE:** `PhysicsConfig` gets S = 5.0, TF = 13.5 and T_MAX = 45.5 N (recalibrated; user chose this
  option out of T_MAX / C_L0 / C_D0 / shortening the window). The arbitrary `V_LIFTOFF = 11 m/s` proxy is
  replaced by the physical event **L ≥ W·cosγ** at ground attitude (`physics.normal_force_margin`).
  Liftoff is predicted by forward RK4 with the identified parameters up to 2·TF
  (`evaluate.predict_liftoff`, also used by UQ). Data strata are fractions of TF (identical at TF = 8).
  The RK4 fine grid uses dt ≤ 0.02 s (`data.fine_grid`; 401 points at TF = 8, as before).
- **WHY:** The brief specifies S = 5 m² and a 13.5 s run. With v1's other values, L = W at 7.9 s, so the
  ground-roll ODE would integrate about 6 s of flight with L up to 1.28·W. At S = 5 the 11 m/s proxy
  has no physical meaning: V_stall(C_Lmax = 1.2) = 6.3 m/s, and at S = 0.55 the proxy lay below stall.
- **EXPECTED:** A physically consistent ground roll that ends at liftoff ≈ 13.5 s, and different
  identifiability (lift unloads the wheels, so friction now varies in time).
- **ACTUAL (seed 7, full preset; `E04_scenario/metrics.json`):**
  truth liftoff t = 13.498 s, x = 113.62 m, V = 13.13 m/s.
  V-RMSE 0.0121 m/s, x-RMSE 0.0389 m (noise 0.10 / 0.08). r_dyn RMS 0.0090 m/s², r_kin RMS 0.0089 m/s.
  μ 0.0781 ± 0.0179, c_D 1.1437 ± 0.0774, η 0.9028 ± 0.0583, **κ 0.04104 ± 0.00079 (truth 0.04085, 0.47 %)**.
  Max sensitivity correlation 0.875 (E00: 0.9999); μ–η correlation 0.936.
  Predicted liftoff 13.580 s / 114.51 m (error 0.082 s / 0.79 %). UQ σ(x_LOF) = 5.6 m, dominated by
  the near-terminal approach to liftoff (net force about 2.6 N at liftoff). Runtime 79 s.
- **Side findings → new changes:**
  (i) The κ degeneracy is broken by 0.32 m, but only through the 5 N softplus smoothing of
  max(W − L, 0), which is not physics → C5.
  (ii) The `quick` preset under-fits the longer record (x-RMSE 2.5 m), and the PINN stage then moves θ
  *away* from the OE estimate. The smoke-test threshold became a mechanics check (documented).
  Test suite runtime rose to 173 s because of the UQ rollout horizon.
- **DECISION:** Keep. E04 is the reference for all later slide-scenario changes. The v1 identifiability
  facts are pinned to an explicit legacy config in the tests.

---

## E04-MS — Multi-seed reference for the slide scenario (Agent 4)

`scripts/multiseed.py --tag E04_scenario`, seeds 7/17/27/37/47 (each seed changes both the noise and the initialization).
V-RMSE 0.0143 ± 0.0039 m/s, x-RMSE 0.048 ± 0.024 m, r_dyn RMS 0.0094 m/s².
Parameter RMSE: μ 0.0122, c_D 0.052, η 0.040, κ 0.00068 (1.7 %). Ground-roll error 0.88 ± 0.29 %.
The PINN θ stays within 0.002 of the OE θ in every seed (see log): the parameters come from the
OE stage.

---

## C5 / E05 — Normal-force smoothing 5 N → 0.5 N (Agent 2; challenged by Agents 1 and 4)

- **CHANGE:** `NORMAL_SOFTNESS` 5 → 0.5 N. `T_MAX` 45.5 → 45.44 N, so the user's calibration rule (truth
  liftoff at the end of the 13.5 s record) still holds. Adds metrics `*_oe`, `x_liftoff_oe`,
  `val_pre_lbfgs` and `val_post_lbfgs` (diagnostics only, numerics unchanged).
- **WHY (Agent 2):** max(W − L, 0) is smoothed only to keep gradients finite at the liftoff kink. Near
  liftoff the net force is about 2.6 N, but the 5 N softplus overstated N by up to 3.5 N
  (friction +0.28 N, about 10 % of the acceleration) and spuriously broke the c_D/c_L degeneracy.
  The κ-invariant trajectory deviated by 3.8e-3 m before liftoff; with 0.5 N it deviates by 2.6e-4 m.
  **Agent 1 challenge:** does a sharper kink hurt the autograd residual? The kink sits at the domain
  edge (liftoff), so inside [0, TF] the residual stays smooth. **Agent 4:** decide on measured outcomes.
- **ACTUAL (5 seeds, E04 → E05):** V-RMSE 0.0143 → 0.0131, x-RMSE 0.048 → 0.037, r_dyn 0.0094 → 0.0092.
  μ/c_D/η RMSE 0.0122/0.052/0.040 → 0.0125/0.051/0.040; κ RMSE 0.00068 → 0.00077 (within seed scatter).
  Ground-roll error 0.88 → 0.91 %. Truth liftoff 13.496 s, 113.46 m.
  Sensitivity corr(c_D, c_L) −0.99877 → −0.99999.
- **DECISION:** Keep. The physics is more correct and no metric regresses beyond seed scatter.

### Evidence gathered in E05 for other proposals

- **C6 (proposed by Agent 1 in the audit, ML-1): accept L-BFGS only if validation does not worsen.**
  In 5/5 seeds L-BFGS *improved* the validation score (×10⁵: 2.90→2.72, 3.39→3.04, 5.67→3.61,
  3.19→2.46, 3.06→2.62). **DECISION: Rejected.** No measurable benefit. The diagnostic stays, so a
  regression would show up in the metrics.
- **ML-3: does the PINN improve the parameters over its OE initializer?** RMSE PINN vs OE:
  μ 0.0125 vs 0.0127, c_D 0.0509 vs 0.0541, η 0.0399 vs 0.0411, κ 0.00077 vs 0.00064.
  x_LOF bias +0.31 m (PINN) vs +0.20 m (OE). **Conclusion:** no difference beyond seed scatter. The
  parameters are identified by the OE stage; the PINN provides state reconstruction and residual
  diagnostics.

---

## C8 / E06 + E07 — Noise-weighted (maximum-likelihood) output error (Agent 1 + Agent 2 SID practice)

- **CHANGE:** `TrainConfig.oe_weighted` (now **default True**) runs the output-error relaxation of
  Klein & Morelli: estimate the per-channel noise variance R from the residuals, refit minimizing
  rᵀR⁻¹r, and repeat (`oe_relax_iters = 2`). The row weights feed the sandwich covariance, which
  then reduces to the inverse Fisher information.
- **WHY:** The normalized V and x noise differ about 5×. Equal weighting wastes most of the information
  in the cleaner channel (x). The Fisher calculation predicted about 2.4× smaller σ.
- **ACTUAL — OE Monte-Carlo, 40 seeds, slide scenario (`E06_oe_mc_slide`):**
  RMSE μ/c_D/η 0.0191/0.0825/0.0623 → **0.0090/0.0381/0.0291** (2.1–2.2×), bias reduced 4–6×.
  Weighted σ z-std 1.15 (95 % coverage 0.90): about 15 % optimistic, borderline at this sample size.
  Reported as is, not tuned.
- **ACTUAL — full pipeline, 10 seeds (`E07a` unweighted vs `E07b` weighted, seeds 7…97, --jobs 4):**

  | | E07a | E07b |
  |---|---|---|
  | μ / c_D / η RMSE (PINN) | 0.0215 / 0.095 / 0.071 | **0.0067 / 0.024 / 0.020** |
  | κ RMSE PINN / OE | 0.00087 / 0.00097 | 0.00071 / **0.00024** |
  | x_LOF error RMSE PINN / OE | 0.91 / 0.88 m | 1.17 / **0.24 m** |
  | x_LOF bias PINN / OE | +0.46 / +0.16 m | +0.94 / −0.05 m |
  | σ(x_LOF) param-only | 9.4 ± 8.7 m (erratic) | 0.42 ± 0.04 m |
  | V / x RMSE vs truth | 0.0146 / 0.044 | 0.0150 / 0.061 |

- **DECISION:** Keep, and make it the default. The identified parameters are 3–4× more accurate, and the classical takeoff
  prediction is 3.6× more accurate. The erratic E07a takeoff UQ came from the large unweighted
  covariance pushing samples into the strongly nonlinear near-terminal regime.
- **New finding → C10:** the PINN stage *degrades* κ and the takeoff prediction relative to its own OE
  initializer (κ bias +5e-4, correlation with the x_LOF error 0.87). Its data loss weights V and x by signal
  maxima rather than noise, so its optimum is not the maximum-likelihood one. This was already present
  in v1 (E07a bias 0.46 vs 0.16 m) and is now clearly exposed.
