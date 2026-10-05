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
