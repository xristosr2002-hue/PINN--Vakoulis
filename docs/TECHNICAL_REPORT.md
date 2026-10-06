# Technical report — PINN for UAV takeoff ground roll: audit, revision and validation (v2)

Project: *Development of a Physics-Informed Neural Network for Enhanced UAV Takeoff Performance
Prediction and Physics Discovery*.
Brief (slide): 15 kg UAV, S = 5 m², ρ = 1.225 kg/m³, 5000 RPM, 13.5 s ground roll, $\dot V = \tfrac{g}{W}[T - D - \mu(W-L)]$.
Repository state: branch `v2-slide-scenario`; baseline tag `baseline-v1` (`0f73c7e`); v2 code in `takeoff_pinn_v2/`.
Every number below comes from a file under `takeoff_pinn_v2/experiments/`; the full change log with
CHANGE / WHY / EXPECTED / ACTUAL / DECISION entries is `experiments/LOG.md`.

---

## 1. Existing architecture (v1, baseline E00)

- **Network:** tanh MLP, 1 → 4×64 → 2 (12 738 weights). Input is normalized time; outputs are V and x with hard ICs
  (`V = t·y0`, `x = t²·y1`).
- **Physical parameters:** μ, c_D-scale and η as sigmoid-bounded scalars, *co-trained* with the network. c_L is fixed.
- **Pipeline:** an output-error (OE) least-squares fit (equal-weight, scale-normalized) → 900 supervised pretraining epochs →
  2200 AdamW epochs with a physics curriculum and residual-guided collocation → L-BFGS polish. 2 restarts.
- **Loss:** Huber on max-normalized V/x data residuals plus dynamic/kinematic ODE residuals (w = 1, 1, 3, 3). Priors have weight 0.
- **UQ:** a Gauss–Newton covariance with one pooled σ², and Monte-Carlo propagation to a speed-threshold "liftoff".

## 2. Existing physical model (v1)

ẋ = V, m V̇ = ηT(rpm, V_air) − D − μN·tanh(V/0.5) − W sinγ, D = qSC_D0c_D, L = qSC_L0c_L, N = softplus(W cosγ − L; 5 N).
This is algebraically the slide equation (W = mg) plus a slope term and directional friction, and it is dimensionally correct.
v1 values: **S = 0.55 m²**, **TF = 8 s**, T_max = 56 N, and "liftoff" = **V ≥ 11 m/s** (a proxy).

## 3. Problems discovered (Phase 1 audit + findings during the work)

| ID | Problem | Evidence |
|---|---|---|
| PH-1 | S = 0.55 m² and TF = 8 s contradict the brief (5 m², 13.5 s) | slide |
| PH-2 | Liftoff = V ≥ 11 m/s is arbitrary; at S = 0.55 it is below stall (19–21 m/s) | L = W balance |
| PH-1b | With S = 5 and v1 thrust, L = W at **7.9 s**, so the ground-roll ODE would integrate ~6 s of flight (L up to 1.28 W) | E04 pre-analysis |
| PH-7 | κ printed as C_D0c_D **+** μC_L0c_L in the audit and thesis; the identifiable combination is **−** | trajectory invariance 3e-5 m vs 0.36 m |
| CR-6 | Reported σ **1.7× too small** (S rows ≠ residual rows; pooled σ² across channels 5–10× apart) | 40-seed MC: 95 % coverage 72 % |
| ML-2 | OE weights V and x by signal maxima, not noise, which wastes the cleaner channel | Fisher: ~2.4× |
| ML-3 | PINN co-training with a non-likelihood loss biases κ (>1σ in 10/10 seeds) and degrades the takeoff prediction 5× | E09 |
| E08 | With c_L fixed 5 % off, the fit is perfect but takeoff distance is wrong by 12.6 % and the UQ is blind to it | E08b |
| PH-4 | 5 N softplus distorts N by up to 3.5 N near liftoff (net force ~2.6 N) and fakes c_L identifiability | invariant 3.8e-3 vs 2.6e-4 m |
| PH-4b | tanh friction gives a creep of 0.19 m/s at idle (real stiction would hold the vehicle) | rollout at 0.5 s |
| CR-1 | CSV path accepts only t, x, V; rpm/wind/slope stay synthetic profiles | code |
| CR-3 | Hard-coded scenario constants (strata 3.0/6.8 s, 401-pt grid, UQ input σ) | code |
| ML-4 | 0.77 s per RK4 solve, all overhead; OE 28 s per restart | profiling |
| CR-4 | Diverged duplicate repo `takeoff-pinn-sota/` | diff |
| META | v1 thesis cover names NTUA; the brief is from AUTh (LFMT/UAV-IRC) | slide logos |

## 4. Problems by severity

- **Critical / high:** PH-1, PH-1b, PH-2, PH-7, CR-6, E08 (c_L assumption silently decides the takeoff prediction), ML-3.
- **Medium:** ML-2, PH-4, CR-1, CR-3, PH-3 (slope as f(t)), PH-5 (constant η, rpm² thrust surrogate), PH-6 (c_D conditional on c_L).
- **Low:** ML-4, PH-4b, ML-1 (L-BFGS not re-validated; turned out harmless), CR-4, META.

## 5–9. Changes implemented, with justification

| Change | What | ML / numerical justification | Aerospace justification | Code-review notes | Evidence | Decision |
|---|---|---|---|---|---|---|
| C1 | κ sign fixed; κ, κσ reported | — | lift reduces friction ⇒ "−" | regression test pins direction | invariance test | ✔ |
| C2 | precomputed exogenous inputs; batched OE Jacobian/sensitivities | overhead-bound solver | — | one equation-of-motion definition kept | E02: all metrics **bit-identical**, 150 → 80 s | ✔ |
| C3 | sandwich covariance, per channel, aligned rows | heteroscedastic LS theory | — | raises on misalignment | E03/E06: z-std 1.7 → 1.0 | ✔ |
| C4 | brief scenario (S = 5, TF = 13.5), **T_max calibrated (user decision)**, liftoff L ≥ W cosγ | — | physically consistent ground roll | strata/grid as fractions | E04 | ✔ |
| C5 | softplus 0.5 N; T_max → 45.44 N | gradients finite at the domain edge only | N error ≪ net force | — | E05 no regression | ✔ |
| C6 | L-BFGS acceptance gate | proposed by Agent 1 | — | — | L-BFGS improved val in 29/30 runs (E05, E09_ref, E09b, E08c; one −1 %) | ✘ rejected |
| C7 | UQ split: parametric vs + input variability | — | separate epistemic and aleatory spread | σ_in in config | — | ✔ |
| C8 | maximum-likelihood OE (Klein & Morelli relaxation) | efficiency | standard flight-test SID | default on | E06: RMSE ÷2.1; E07: PINN θ ÷3–4 | ✔ |
| C9 | observed liftoff time as a measurement (identifies c_L) | known-variance row in cov | wheels-off time is routinely measured | drawn last, no truth leak | Fisher σ(c_L) 1.4 → 0.0017; E08 | ✔ |
| C10a | PINN data terms weighted by noise | likelihood consistency | — | — | E09a: takeoff error ×5 | ✘ rejected |
| C10b | θ kept at OE estimate (PINN = state estimator) | avoids the non-ML loss bias | — | minimal | E09b: better on every metric | ✔ |
| C11 | default: estimate c_L if liftoff observed, else fixed with warning | — | — | safe real-data default | E08c | ✔ |
| E10 | truth-only model-form error option + discrepancy analysis | — | speed-dependent friction on grass | off by default, bit-identical at 0 | E10 | tool |

## 10. Experimental evidence (selected; seeds 7…97 for the pipeline, 100…139 for OE MC)

**Covariance honesty (40 seeds, OE):** v1 z-std 1.68 / 1.73 / 1.68 (95 % coverage 0.725); sandwich 1.02 / 1.08 / 1.02 (0.90–0.95).
Slide scenario: v1 1.69 / 1.67 / 1.68 → sandwich 1.00 / 0.99 / 0.99.

**OE weighting (40 seeds, slide):** RMSE μ/c_D/η 0.0191 / 0.0825 / 0.0623 → **0.0090 / 0.0381 / 0.0291**.

**PINN stage (10 seeds, c_L known):**

| | co-train θ (v1 behaviour) | noise-weighted data (rejected) | **θ frozen at OE** |
|---|---|---|---|
| κ RMSE | 0.00071 | 0.0027 | **0.00024** |
| takeoff-distance error RMSE | 1.17 m | 6.38 m | **0.24 m** |
| +2 m/s headwind case | 1.30 % | 5.68 % | **0.26 %** |
| V / x RMSE | 0.015 / 0.061 | 0.063 / 0.054 | **0.0070 / 0.023** |

OE rollout with the same θ: V/x RMSE **0.0029 / 0.0081**. The classical ODE solution beats the network states.

**c_L (10 seeds):** see the E08 table in `LOG.md`. Fixed +5 %: 12.6 % error, z = 4.2. Identified from t_LOF: c_L RMSE 0.0017,
takeoff 0.22 % (z-std 1.09), generalization 0.29 %.

**Model-form error E10:** truth friction μ(1+0.02V), constant-μ model, c_L at truth, 10 seeds. State RMSE V/x: OE rollout 0.0181 / 0.026, PINN 0.0197 / 0.037. corr(PINN residual, true missing acceleration) = 0.002 (control 0.00). The parameters absorb the error: μ bias +0.049, c_D +0.21, η +0.14 (η **pinned at its bound in 10/10 seeds**), and the fit stays within noise. Takeoff error 2.3 %, generalization 3.9 %. **Both pro-PINN hypotheses are rejected** for this discrepancy, which led to the `params_at_bounds` diagnostic (C12).

## 11. Baseline vs final (slide scenario, same 10 seeds)

| metric | v1 method (c_L **known exactly**, its best case) | final v2 (c_L **unknown**, start 5 % off) |
|---|---|---|
| μ / c_D / η RMSE | 0.0215 / 0.095 / 0.071 | **0.0058 / 0.0255 / 0.0192** |
| c_L RMSE | not identifiable | **0.0017** |
| κ RMSE | 0.00087 | **0.00024** |
| takeoff-distance error RMSE | 0.91 m | **0.30 m** |
| reported σ honesty (z-std, μ) | 1.06 *(v1's own code reports 1.7× smaller σ)* | 0.73 (slightly conservative) |
| runtime per full run (sequential, CPU) | 150 s (E00, S = 0.55) | 135 s (S = 5 scenario, 4-parameter OE + liftoff row) |

With a realistic, uncertain c_L, the v1 method would carry the E08b failure mode (about 12 % takeoff error per 5 % c_L error).

## 12. Final architecture

Unchanged network: tanh MLP 1→4×64→2 with hard ICs (12 738 weights). Pipeline: **weighted OE (+ liftoff row) →
θ̂_ML** → PINN state estimation with θ frozen (900 pretrain + 2200 AdamW + L-BFGS) → sandwich covariance →
correlated UQ of the ODE rollout. 2 restarts. float64 throughout.

## 13. Final hyperparameters

Same as E00 except: OE weighted (2 relaxation passes); `pinn_trains_params = False`; NORMAL_SOFTNESS 0.5 N;
RK4 dt ≤ 0.02 s; UQ 250 samples (σ_rpm 40, σ_wind 0.08 m/s for the *total* interval only); estimate_cl = True with
prior_cl = 1.05; liftoff timing σ_t = 0.02 s. Seeds: `TrainConfig.seed` (restart r: seed + 1000 r).

## 14. Final loss formulation

OE (identification): min Σ (V_sim − V)²/σ̂_V² + (x_sim − x)²/σ̂_x² + ((t_LOF(θ) − t_LOF^obs)/σ_t)², with σ̂ re-estimated
from residuals (relaxation). PINN (state estimation, θ fixed):
J = H((V−V_d)/v_s) + H((x−x_d)/x_s) + ρ(e)[3 H(r_dyn/a_s) + 3 H(r_kin/v_s)], H = Huber(δ = 1), a_s = T_max/m.

## 15. Final physical equations

ẋ = V; m V̇ = ηT − D − μN tanh(V/0.5) − W sinγ; T = T_max (n/n_max)² clip(1 − 0.01 V_air; 0.7, 1.05), T_max = 45.44 N;
D = qS C_D0 c_D, L = qS C_L0 c_L, S = 5 m², C_D0 = 0.055, C_L0 = 0.28; N = 0.5·softplus((W cosγ − L)/0.5);
liftoff g(t) = W cosγ − L = 0. Identifiable while N > 0: κ = C_D0c_D − μC_L0c_L.

## 16. Final discovered parameters (reference run, seed 7)

| μ | c_D | c_L | η | κ |
|---|---|---|---|---|
| 0.0709 ± 0.0072 (truth 0.080) | 1.1113 ± 0.0302 (1.150) | 1.0009 ± 0.0016 (1.000) | 0.8802 ± 0.0234 (0.910) | 0.04126 ± 0.00036 (0.04085) |

μ, c_D and η are each about 1.3σ low together (joint valley, parameter correlation 0.9999). Over 10 seeds (E08c) the σ are honest.

## 17. Final takeoff predictions (reference run, seed 7)

Liftoff time **13.499 ± 0.019 s** (truth 13.496; observed 13.499). Ground roll **113.49 ± 0.25 m** (truth 113.46; error 0.025 %). With run-to-run input variability: ± 5.7 m. Headwind +2 m/s case: 91.03 m (truth 90.98; 0.053 %). States: V/x RMSE 0.0087 m/s / 0.051 m (noise 0.10 / 0.08). Residual RMS r_dyn 0.0114 m/s², r_kin 0.0108 m/s.

## 18. Remaining limitations

1. **Synthetic data only.** All accuracy claims are conditional on the model describing the real vehicle (inverse crime). E10 shows what happens when it does not.
2. **Exogenous inputs are analytic profiles.** Real data needs measured rpm/wind/slope; the CSV hook takes t, x, V (+ liftoff time) only.
3. **Liftoff uses ground-attitude C_L.** There is no rotation model and no C_L,max.
4. **Thrust surrogate.** T ∝ rpm², constant η (blockage is J-dependent in reality), T_max calibrated, not measured.
5. **Marginal scenario.** Liftoff is near terminal speed, so it is extremely sensitive: +0.5 kg → +14 % ground roll; +2 kg → no liftoff.
6. **Joint μ–c_D–η valley.** Parameter-estimate correlation ≈ 0.998; individual values are limited by excitation.
7. **Weighted OE σ about 15 % optimistic** (z-std 1.15, 40 seeds).
8. **Network epistemic uncertainty is not quantified** (it does not enter the ODE-rollout prediction).
9. **Minor physics simplifications:** slope as f(t); tanh friction creep at idle.
10. **Test suite takes ~15–20 min** (integration runs included).

## 19. Recommended future work

Real flight tests with logged rpm and wind, plus a wheels-off sensor; throttle doublets/steps (D-optimal excitation)
to break the μ–c_D–η valley; C_T(J) propeller maps; a 2-D rotation/climb model with C_L,max; hybrid
"universal differential equation" terms for model-form discrepancies (the E10 residual is the diagnostic);
ensembles/B-PINN for the state-estimation uncertainty.

---

## FINAL STATUS

| Layer | Status | Basis |
|---|---|---|
| SOFTWARE | PASS | full suite 64/64 passed (22 min incl. integration runs); bit-identical baseline reproduction; config save/load; README; LOG |
| MATHEMATICAL MODEL | PASS | dimensional check; κ invariant test; RK4 order 4; liftoff residual vs exact event; covariance MC-validated |
| AEROSPACE PHYSICS | PASS (within the stated 1-D ground-roll assumptions) | brief scenario; physical liftoff; force balance (Fig. forces); limitations 3–5, 9 |
| PINN TRAINING | PASS | converges in every seed; residual RMS ~0.009 m/s²; L-BFGS improved validation in 29/30 logged runs (one −1 %) |
| RESULTS | PASS (synthetic) | c_L 0.17 %, takeoff 0.22 %, generalization 0.29 % with honest σ; negative results reported |
| **OVERALL** | **NEEDS FURTHER DEVELOPMENT** | method validated on synthetic data only. Before research use on a real UAV it needs real flight data with measured inputs and liftoff timing. In this well-specified setting the classical OE method, not the PINN, drives identification. |
