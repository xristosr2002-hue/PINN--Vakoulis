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

---

## C9 — Observed liftoff time as a measurement (Agent 2 proposal; Agents 1, 3 and 4 reviewed)

- **CHANGE (opt-in, `ParamSpace.estimate_cl` + an observed liftoff time):** the synthetic generator draws
  t_obs = t_L + N(0, σ_t), with σ_t = 0.02 s (wheels-off switch or video). It is drawn last, so V/x are unchanged, and
  it is a noisy measurement, not truth leakage. The weighted OE gets one row: the linearized liftoff-time
  error −g(t_obs)/ġ(t_obs) in σ_t units, with g = W·cosγ − L. The covariance takes it as a known-variance
  row (`parameter_covariance(S_known=...)`). The PINN gets a likelihood-consistent event term when it
  trains c_L. The CLI gains `--t-liftoff` / `--sigma-t-liftoff` for real data. Also adds a +2 m/s headwind
  generalization metric and `param_corr_max`.
- **WHY (Agent 4 finding):** a 5 % error in the assumed c_L shifts the predicted ground roll by
  −12.5 % / +14.4 % while the record is fitted almost equally well (κ unchanged). Every result up to
  E07 fixed c_L at its *true* value, which is optimistic and impossible with real data.
- **Evidence before implementing (linearized Fisher):** σ(c_L) 1.38 → 0.0017 (σ_t = 0.02 s) and 0.0047 (σ_t = 0.1 s).
  σ(c_D) 0.57 → 0.033, equal to the "c_L known" value. Condition number 1.3e9 → 1e6.
- **Checks:** the linearized event residual matches the exact event shift (0.182 vs 0.1825 s for +6 % μ).
  OE recovers c_L = 1.0009 from a 1.05 start (seed 7). Tests 58/58.
- **DECISION:** Accepted after E08 (below) and made the default (C11).

---

## C10 / E09 — Stop the PINN stage degrading its maximum-likelihood initializer

Same 10 seeds; E09_ref reproduces E07b bit-for-bit (C9 code inactive, so the regression check passes).

| metric (10 seeds) | E09_ref (PINN trains θ) | E09a data_weighting=noise | **E09b θ frozen at OE** |
|---|---|---|---|
| μ / c_D / η RMSE | 0.0067 / 0.024 / 0.020 | 0.0067 / 0.063 / 0.029 | **0.0058 / 0.026 / 0.019** |
| κ RMSE (bias) | 0.00071 (+0.00052) | 0.0027 (−0.0022) | **0.00024 (+0.00005)** |
| κ 68 % coverage | 0 / 10 | 1 / 10 | z-std 0.64 |
| x_LOF error RMSE (bias) | 1.17 m (+0.93) | 6.38 m (−4.61) | **0.24 m (−0.05)** |
| ground-roll error | 0.95 % | 4.06 % | **0.19 %** |
| +2 m/s headwind generalization error | 1.30 % | 5.68 % | **0.26 %** |
| liftoff-time error | 0.079 s | 0.333 s | **0.017 s** |
| V / x RMSE vs truth | 0.0150 / 0.061 | 0.063 / 0.054 | **0.0070 / 0.023** |
| r_dyn / r_kin RMS | 0.0090 / 0.0112 | 0.042 / 0.077 | 0.0094 / 0.0124 |

- **E09a (Agent 1's likelihood-consistent PINN data weights): REJECTED.** Up-weighting x by about 100× tips the
  data/physics balance: the network fits x noise against the equations, so residuals rise 5× and θ drifts.
  Fixing that would need re-tuned physics weights (more unjustified hyperparameters).
- **E09b (Agent 3's minimal option): ACCEPTED, now the default** (`pinn_trains_params = False`). It is better on every metric.
- **Agent 4 follow-up, reported plainly (Rule 10):** with θ from OE, integrating the ODE with θ_OE gives
  V/x RMSE **0.0029 / 0.0081** against the PINN's 0.0070 / 0.023. On this well-specified synthetic problem
  ("inverse crime"), the classical output-error method beats the PINN for parameters *and* states.
  The PINN's possible value (robustness to model-form error, residual diagnostics) can only show up
  when the model is wrong → E10.

---

## E08 — Realistic c_L: fixed 5 % off vs. identified from the observed liftoff time (C9 evaluation)

Same 10 seeds and v2 pipeline (weighted OE, θ frozen at OE). "c_L = truth" is E09b (optimistic, impossible with
real data). "c_L fixed +5 %" is `space.prior_cl=1.05`. "c_L from t_LOF" is `space.prior_cl=1.05 space.estimate_cl=true` (start 5 % off).

| metric (10 seeds) | c_L = truth (E09b) | **c_L fixed +5 % (E08b)** | **c_L from t_LOF (E08c)** |
|---|---|---|---|
| c_L RMSE (z-std) | 0 (assumed) | 0.050 | **0.0017 (0.94)** |
| μ / c_D / η RMSE | 0.0058 / 0.0256 / 0.0192 | 0.0057 / 0.0287 / 0.0187 | 0.0058 / 0.0255 / 0.0192 |
| κ RMSE | 0.00024 | 0.00024 | 0.00024 |
| V / x RMSE vs truth | 0.0070 / 0.023 | 0.0078 / 0.024 | 0.0111 / 0.026 |
| x_LOF error RMSE (bias) | 0.24 m (−0.05) | **14.27 m (−14.27)** | **0.30 m (+0.09)** |
| ground-roll error | 0.19 % | **12.6 %** | **0.22 %** |
| liftoff-time error | 0.017 s | 1.10 s | 0.018 s |
| +2 m/s headwind generalization error | 0.26 % | **17.6 %** | **0.29 %** |
| reported σ(x_LOF) param-only, z-std | 0.41 m, 0.58 | 0.34 m, **4.2** | 0.27 m, 1.09 |

- **Key result (Agent 4):** with a 5 % c_L error the fit is as good as with the true c_L (same κ, same state
  errors), yet the takeoff prediction is wrong by 12.6 % and the reported uncertainty does not see it
  (0.34 m σ against a 14.3 m error). An assumption error outside the model cannot show up in a parametric covariance.
- **C9 result:** the observed liftoff time recovers c_L to 0.17 % (honest σ), and μ/c_D/η are then as accurate as
  with c_L known exactly, as the Fisher analysis predicted. Takeoff and generalization errors return to the
  "c_L known" level, and the takeoff σ is honest (z-std 1.09).

---

## C11 — Default: estimate c_L whenever a liftoff time is observed; safe fallback otherwise

- **CHANGE:** `ParamSpace.estimate_cl = True` and `prior_cl = 1.05` (a nominal, CFD-like value deliberately 5 % off the
  synthetic truth, in line with the "priors ≠ truth" rule). Without an observed liftoff time, `run_experiment`
  **falls back to c_L fixed at prior_cl** with a warning and records the switch in `config.json`, instead of
  fitting the degenerate (c_D, c_L) direction. The CLI `--estimate-cl` is replaced by `--fix-cl`.
- **WHY:** E08. Agent 3: the fallback protects real-data users who have no liftoff timing.
- **Tests:** OE unit tests now state `estimate_cl=False` explicitly. A new smoke test covers the fallback.

---

## E10 — Model-form error in the truth: does the PINN help when the physics is wrong?

- **Setup:** `data.truth_mu_speed_slope=0.02`, so the synthetic truth has rolling friction μ(1 + 0.02·V) (friction growing with
  speed, as on grass). The identification model keeps a constant μ. Same 10 seeds. c_L is fixed at its true value
  (the run started before C11) to isolate the friction error. Analysis: `scripts/discrepancy.py` (control: E09b).
- **Hypotheses (Agent 1/2):** (a) the PINN, whose physics is a soft constraint, reconstructs states better than the
  misspecified ODE rollout; (b) the PINN dynamic residual r(t) reveals the missing acceleration Δa(t).
- **ACTUAL:**

  | | OE rollout | PINN |
  |---|---|---|
  | V / x RMSE vs truth | **0.0181 / 0.026** | 0.0197 / 0.037 |
  | corr(r_dyn, Δa) (R²) | — | **0.002 (−0.61)**; control 0.00 |
  | RMS Δa / RMS r_dyn | 0.0157 m/s² | 0.0112 m/s² (floor ≈ 0.009) |

  Parameters absorb the missing physics. μ +0.049 (+61 %), c_D +0.21 and η +0.14 are **pinned at the upper bound 1.05 in
  10/10 seeds**. κ is off by 5 %, and the fit stays within the noise. Takeoff error is 2.3 %, the +2 m/s headwind case 3.9 %.
  The parameter-only σ(x_LOF) is inflated (≈ 18 m) because the covariance is evaluated at a bound.
- **Conclusion (reported plainly, Rule 10):** both hypotheses are **rejected** for this discrepancy. Re-fitted constant-μ physics absorbs
  most of the error, and what remains (0.016 m/s²) is barely above the PINN's residual floor. The physics-weighted network follows the model,
  not the data. The informative symptom is **parameters pinned at their bounds** → C12.

---

## C12 — Flag estimates pinned at a bound

- **CHANGE:** `identifiability.params_at_bounds` and the metric `params_at_bounds`, with a warning when it is non-empty.
- **WHY:** In E10 the misspecified model fitted the data within noise. The only visible red flag was η on its bound, where
  the linearized covariance is meaningless, yet v2 reported a σ for it without comment.
- **DECISION:** Keep (pure diagnostic; unit-tested).

---

## C13 — OE efficiency with c_L estimated (Agent 1, found while timing the final run)

- **Finding:** with c_L free, the first (unweighted) OE pass leaves out the liftoff row, so it is a degenerate 4-parameter
  fit that wandered along the c_D–c_L valley until `max_nfev` (355 s per call). Every restart also recomputed the identical OE fit.
- **CHANGE:** c_L is held at its nominal value in the first pass and freed once the liftoff row is weighted (`solve(..., free)`).
  `run_experiment` computes the OE fit once and shares it across restarts.
- **ACTUAL:** OE 355 s → 24 s with the same estimate (to 5 digits). Full reference run 760 s → **135 s**. Parameters
  identical to 1e-7.
- **DECISION:** Keep.

---

## FINAL — Reference run (`experiments/final_reference`, full preset, final defaults, seed 7, sequential CPU)

| quantity | estimate ± σ | truth |
|---|---|---|
| μ | 0.0709 ± 0.0072 | 0.080 |
| c_D | 1.1113 ± 0.0302 | 1.150 |
| c_L | 1.0009 ± 0.0016 | 1.000 |
| η | 0.8802 ± 0.0234 | 0.910 |
| κ | 0.04126 ± 0.00036 | 0.04085 |
| t_LOF [s] | 13.499 ± 0.019 | 13.496 |
| x_LOF [m] | 113.49 ± 0.25 (± 5.73 incl. input variability) | 113.46 |
| x_LOF, +2 m/s headwind [m] | 91.03 | 90.98 |

V / x RMSE vs truth 0.0087 m/s / 0.0508 m; r_dyn / r_kin RMS 0.0114 / 0.0108.
No parameter at a bound. Runtime 135 s. μ, c_D and η are each about 1.3σ low *together*, which is the
joint thrust-vs-resistance valley (parameter correlation 0.9999); κ is 1.15σ high. Over 10 seeds (E08c) σ is honest.
