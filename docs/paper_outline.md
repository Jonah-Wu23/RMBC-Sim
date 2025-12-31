# Paper Outline: IEEE SMC 2026 (6 Pages)

**Title**
**Robust Calibration of Mobility Digital Twins via Observation Operator Audit: A Bayesian-Assimilation Loop with Regime Separation for Corridor Reliability**
*(中文：基于观测算子审计的移动数字孪生鲁棒校准：以体制分离支撑走廊可靠性的贝叶斯-同化闭环)*

**Keywords**
Mobility Digital Twins; Bus Operations; Schedule Adherence; Observation Operator Audit; Regime Separation; Bayesian Optimization; Data Assimilation; Distributional Robustness.

---

## 0. Abstract (200-250 words)

*   **Context:** Urban bus operations are stochastic Cyber-Physical Systems (CPS). The “reality gap” is driven by **operational semantics** (e.g., schedule adherence, holding/layover) that can contaminate door-to-door measurements with non-transport regimes.
*   **Problem:** Calibration based on mean moving speed or un-audited travel-time labels becomes **unidentifiable** (equifinality) and brittle under regime transfer, because it conflates transport dynamics with operator-induced artifacts.
*   **Methodology (RCMDT):** We propose the **Robust Calibration of Mobility Digital Twins (RCMDT)**, a hierarchical framework featuring a cybernetic **Bayesian-Assimilation Loop**:
    1.  **Outer Loop (Micro-Level):** Utilizes Kriging-based **Bayesian Optimization (BO)** with Uncertainty Quantification (UQ) to invert behavioral parameters (stop logic).
    2.  **Inner Loop (Macro-Level):** Integrates a **Constraint-Aware Iterative Ensemble Smoother (IES)** to assimilate corridor reliability states (flow logic).
*   **Observation Operator Audit ("Caliber Audit"):** We formalize an auditable **observation-operator family** (Op-L2-v0/v1/v1.1) and a physically motivated **regime-separation rule** (Rule C: $T^*=325s$, $v^*=5$ km/h) to decontaminate non-transport “ghost jams”.
*   **Calibration vs Validation:** Calibration optimizes a combined loss (RMSE-style) for parameter/state updates, while robustness is validated using distributional evidence (K-S + worst-window stress) under frozen parameters.
*   **Results:** On two Hong Kong bus corridors (68X, 960), raw off-peak transfer has K-S $\approx$ 0.54. After auditable decontamination, K-S $\approx$ 0.26, with the worst 15-min window still borderline-pass (K-S $\approx$ 0.33) under the strictest valid operator. Sparse real trajectories are used only as diagnostics; claims rely on distribution-level evidence and kinematic regime separation.

---

## 1. Introduction

### 1.1 Digital Twins as Probabilistic CPS Mirrors
*   Define the Mobility Digital Twin as a probabilistic mirror of a complex CPS.
*   Highlight the **Inverse Problem**: Estimating system parameters $\theta$ from noisy output $y$ is ill-posed, non-convex, and computationally expensive.

### 1.2 The Micro-Macro Coupling Challenge
*   **Stop-Level Dynamics (Micro):** Passenger processes and operational control at stops (boarding/alighting, holding/layover), inducing long-tail stop/idle times.
*   **Corridor Reliability (Macro):** Emergent corridor properties defined by the stability of travel time distributions (e.g., 90th percentile lateness).
*   **The Gap:** Existing methods (GA/PSO) decouple these layers, leading to "overfitting to the mean" while failing to capture the **Reliability** (tail distribution).

### 1.3 Contributions (SMC Focus)
1.  **Operator-Aware Observability:** We introduce an **Observation Operator Audit** (Op-L2 family) and auditable **regime separation** (Rule C) to decontaminate non-transport artifacts and restore identifiability in human-centric CPS.
2.  **Cybernetic Mechanism:** We propose the **Bayesian-Assimilation Loop**, a closed-loop architecture where a Bayesian surrogate guides micro-parameter search while an IES inner loop stabilizes macro-state assimilation.
3.  **Reliability-based Robust Validation:** We establish a distribution-level validation protocol (K-S + worst-window stress test) and demonstrate regime-transfer robustness under frozen parameters.

---

## 2. Related Work

### 2.1 Simulation-based Optimization in CPS
*   Evolution from Black-box (GA) to **Surrogate-Assisted Methods** (Bayesian Optimization).
*   *Gap:* Most focus on deterministic error minimization, neglecting **Distributional Robustness** and uncertainty quantification.

### 2.2 Data Assimilation for Dynamic Systems
*   Kalman Filters (EnKF) and Ensemble Smoothers (IES).
*   *Gap:* Typically used for online state correction; rarely coupled with **behavioral parameter inversion** in a unified hierarchical loop.

### 2.3 Robustness & Uncertainty in System Identification
*   Distributional metrics (Wasserstein, K-S test).
*   *Gap:* Lack of frameworks addressing identifiability issues caused by **observation operator misalignment** in human-centric systems.

---

## 3. Problem Formulation

### 3.1 System Dynamics & State Space
*   Stochastic Mapping $\mathcal{M}$: $Y_{reliability} = \mathcal{M}(\theta_{stop}, \mathbf{X}_{corridor}, \xi)$.
    *   $\theta_{stop}$: Micro-behavioral parameters (e.g., boarding/alighting coefficients).
    *   $\mathbf{X}_{corridor}$: Macro-state background traffic flows.
    *   $\xi$: Stochastic noise.

### 3.2 Objectives & Metrics: Calibration vs Validation
*   Defined over a 1-hour sliding window.
*   **Reliability Vector** includes: Mean Travel Time, 90th Percentile (P90), and bounded worst-case error.
*   **Calibration objective (optimization):** Minimize a combined loss on reliability summaries (e.g., weighted RMSE on Mean/P90) used by BO (L1) and IES (L2).
*   **Validation/robustness (evidence):** Report distributional alignment via K-S on travel-time CDFs and a worst-window stress test (15-min), without re-calibration.

### 3.3 Observation Operator Audit (Observability Analysis)
*   **Proposition:** Reliance on *Moving Speed* ($\mathcal{O}_{move}$) leads to equifinality.
*   **Layered Operators:**
    *   **Op-L2-v0 (Speed):** Naive moving average (Unidentifiable).
    *   **Op-L2-v1 (D2D):** Door-to-Door reliability including dwell dynamics.
    *   **Op-L2-v1.1 (Audited):** D2D with auditable **regime separation** (Rule C: $T^*=325s$, $v^*=5$ km/h) to exclude non-transport schedule-hold artifacts (“ghost jams”) in robustness verification.

---

## 4. Methodology: The RCMDT Framework

### 4.1 Architecture: The Bayesian-Assimilation Loop (Fig. 1)
*   **A "Control System" View:**
    *   **Outer Loop (Controller L1):** Bayesian Optimizer (BO) using Kriging Surrogates to propose $\theta_{stop}$.
    *   **Inner Loop (Controller L2):** IES Assimilator adjusting $\mathbf{X}_{corridor}$ to match reliability constraints.
    *   **Feedback (for optimization):** Calibration loss on reliability summaries (RMSE-style, e.g., mean/P90) and constraint violations; K-S is reserved for validation/robustness reporting.

### 4.2 L1: Outer Loop - Surrogate-Driven Behavioral Inversion
*   **Gaussian Process (Kriging):** Models the objective landscape $f(\theta)$ with Mean (prediction) and Variance (uncertainty).
*   **Acquisition Strategy:** Expected Improvement (EI) guides the search to robust global optima.

### 4.3 L2: Inner Loop - Constraint-Aware Assimilation
*   **Observation Operator:** Maps simulator state to the **Reliability Vector** centered on distributional reliability metrics.
*   **Constraint-Aware Update:**
    *   Standard IES update: $\mathbf{X}^{a} = \mathbf{X}^{f} + \mathbf{K}(\mathbf{d} - \mathbf{H}\mathbf{X}^{f})$.
    *   **Safety Rails:** explicit bounds (e.g., non-negative flow, max density) injected to ensure physical consistency.

### 4.4 Operator-Aware Observation & Diagnostic Decomposition
*   **(M1) Auditable Rule C definition:** A sample is flagged as non-transport if $(T > T^*) \wedge (v_{eff} < v^*)$, with $v^*=5$ km/h (traffic-only / moving-regime threshold) and $T^*=325s$ (chosen from the audit plot’s interpretable knee/critical boundary and kept fixed in the worst-window stress test; sensitivity can be summarized in the appendix). The goal is to separate operator artifacts for a reproducible robustness test.
*   **(M2) Holding proxy vs simulated dwell:** The holding proxy is constructed from observable real signals and is a different physical quantity than simulated dwell. Dwell/holding plots are used to localize mechanism gaps and avoid use as strict calibration targets.
*   **(M3) Traffic-only definition:** Traffic-only segments satisfy $v \ge 5$ km/h and exclude stop/idle phases. When plotting trajectories, Real/Sim axes can be traffic-only accumulated time, so full-time vs traffic-only comparisons are diagnostic and avoid equivalence claims.
*   **(M4) External corroboration (traffic-only):** We cross-check ghost-labeled intervals against the Transport Department’s processed IRN segment speeds (`segment_id`=ROUTE_ID, `speed` in km/h), using only `valid=Y` records. This external traffic-only data is used strictly as an independent sanity check, separate from Rule C definition or tuning [2], [3].

---

## 5. Experimental Setup

### 5.1 Testbed & Protocol
*   **Simulation Platform:** SUMO (Simulation of Urban MObility).
*   **Data Source:** High-frequency traces (Hong Kong), two corridors (68X, 960).
*   **External traffic-only data (corroboration only):** TD processed IRN segment speeds (`segment_id`=ROUTE_ID, `speed` in km/h, `valid` flag; use only `valid=Y`) [2], [3].
*   **Peak/off-peak context:** Use Annual Traffic Census 2024 as institutional background for peak-hour framing; avoid hour-level congestion claims on a given day [1].
*   **Protocol:**
    *   *Unit:* 1-hour time windows.
    *   *Sets:* Training (Day 1 AM), Testing (Day 1 PM), Robustness Check (Day 2).

### 5.2 Objectives & Metrics
*   **Calibration objective (optimization):** combined loss (RMSE-style) for BO/IES updates (e.g., reliability-vector summaries such as mean/P90).
*   **Validation/robustness (evidence):** K-S distance on D2D travel-time CDFs (raw vs audited-clean vs simulation) + worst-window (15-min) K-S (and/or P90 error) under a fixed operator (Rule C).
*   **Kinematic plausibility (diagnostic):** regime separation in $(T, v_{eff})$ and traffic-only trajectory decomposition; exclude “trajectory smoothness”.

---

## 6. Results and Discussion

### 6.1 Measurement Audit & Regime Separation (Op-L2-v1.1 / Rule C)
*   **Figure:** `plots/P14_ghost_audit.png`
*   **One-line conclusion:** Rule C systematically isolates long-duration, low-speed samples (the raw long tail) from the transport regime.
*   **Reviewer-guard:** Rule C is an auditable threshold + decision boundary ($T^*=325s$, $v^*=5$ km/h): $v^*$ matches the traffic-only definition and $T^*$ is chosen from the audit structure and then held fixed for the borderline worst-window stress test, excluding post-hoc tuning.
*   **External corroboration (traffic-only):** Strategic/major-road segment speeds remain normal during the same off-peak window where D2D shows a $v_{eff}<5$ km/h long tail (e.g., 15:00–16:00: min 8.4 km/h; median 47.2 km/h), supporting non-transport operational semantics rather than network-wide congestion [2], [3].

### 6.2 Robustness Verification (Distribution-level, P14 Stress Test)
*   **Figure:** `plots/P14_robustness_cdf.png`
*   **One-line conclusion:** After auditable decontamination, K-S $\approx$ 0.26; worst 15-min window remains borderline-pass (K-S $\approx$ 0.33) under the strictest valid operator. Raw labels yield K-S $\approx$ 0.54.
*   **Reviewer-guard:** This is the primary validation evidence; trajectory/spacetime plots are used only as diagnostics.

### 6.3 Supplementary: Kinematic Evidence for Ghost Artifacts (Trip/Segment Level)
*   **Figures:** `plots/ghost_physical_evidence_68X.png`, `plots/ghost_physical_evidence_960.png`
*   **One-line conclusion:** Ghost samples cluster in the $(T, v_{eff})$ region below $(T^*, v^*)$, showing consistent physical separation across corridors.
*   **Reviewer-guard:** Each plot reports `N_clean/N_ghost` to prevent “selected points” criticism.

### 6.4 Diagnostics Under Sparse Real Observations (Diagnostic Use Only)
*   **Figures:** `plots/trajectory_stepped_68X.png`, `plots/trajectory_stepped_960.png`
*   **One-line conclusion:** Full-time trajectory steps are dominated by stop/holding; traffic-only decomposition reveals the moving-physics “skeleton”.
*   **Scope note:** Real trajectories are sparse (Trips=1, Points=3/4), so these plots explain *where mechanisms differ* and avoid serving as validation evidence.

### 6.5 Mechanism Gap Localization: Holding Proxy vs Simulated Dwell (Diagnostic)
*   **Figures:** `plots/dwell_distribution_68X.png`, `plots/dwell_distribution_960.png`
*   **One-line conclusion:** The holding proxy exhibits systematic shifts vs simulated dwell, indicating schedule adherence/layover as a major unmodeled mechanism.
*   **Reviewer-guard:** Proxy $\neq$ dwell; used to answer *where mechanisms are missing* and avoid asserting whether the calibrated model is “correct”.

### 6.6 Calibration Improvement (B2): BO vs LHS
*   **Figure:** `plots/B2_phase_comparison.png`
*   **One-line conclusion:** BO reduces combined loss; RMSE improves clearly on 68X while 960 shows modest change (trade-off acknowledged).
*   **Reviewer-guard:** Multi-corridor objectives exhibit trade-offs; we avoid claiming uniform improvement across all routes/metrics.

### 6.7 Supplementary: Surrogate Uncertainty Dynamics (Non-monotonic)
*   **Figure:** `plots/B2_gp_variance.png`
*   **Interpretation:** Variance is non-monotonic (exploration jumps / noisy objective); included as a diagnostic and avoid using as a convergence proof.

### 6.8 Limitations / Threats to Validity
*   Sparse real trajectory observations $\rightarrow$ trajectory/spacetime used only as illustrative diagnostics.
*   Holding proxy vs dwell non-comparability $\rightarrow$ dwell figures are mechanism-gap signals; avoid strict validation.
*   Rule C threshold choice $\rightarrow$ defended via audit plot + K-S before/after + worst-window stress test with borderline operator.
*   External IRN speeds cover strategic/major roads only $\rightarrow$ treat as corroboration; avoid primary evidence [2], [3].
*   Multi-corridor trade-offs $\rightarrow$ report phase-wise results; future work: multi-objective weighting / Pareto view.
*   Offline calibration $\rightarrow$ online adaptation and explicit holding control remain future work.

---

## 7. Conclusion

*   **Summary:** RCMDT bridges the reality gap by coupling stop-level dynamics and corridor reliability via a Bayesian-Assimilation loop, grounded in an auditable observation operator.
*   **Impact:** A blueprint for calibrating stochastic, human-centric CPS.
*   **Future Work:** Real-time online DT adaptation and explicit modeling of schedule adherence/holding control.

---

## Figure & Table Checklist (SMC Style)

*   **Fig 1. System Block Diagram:** Control-theoretic view. Inputs (Data) $\rightarrow$ Controllers (BO & IES) $\rightarrow$ Plant (DT) $\rightarrow$ Feedback (Reliability Metrics).

### Main paper (recommend <=6 figures)
*   **Fig 2. Measurement Audit + Rule C Logic:** `plots/P14_ghost_audit.png` (raw vs clean + auditable boundary).
*   **Fig 3. Robustness CDF (P14):** `plots/P14_robustness_cdf.png` (raw fail $\rightarrow$ clean pass; worst-window stress metric).
*   **Fig 4. Phase-wise Improvement (B2):** `plots/B2_phase_comparison.png` (BO vs LHS; trade-off-aware wording).
*   **Fig 5. Trajectory Decomposition (Diagnostic):** choose one of `plots/trajectory_stepped_68X.png` / `plots/trajectory_stepped_960.png` (explicitly labeled “illustrative; avoid validation use”).

### Supplementary / Appendix (remaining plots)
*   `plots/ghost_physical_evidence_68X.png`, `plots/ghost_physical_evidence_960.png` (kinematic regime separation evidence)
*   `plots/dwell_distribution_68X.png`, `plots/dwell_distribution_960.png`
*   the other `plots/trajectory_stepped_*.png`
*   `plots/B2_gp_variance.png` (uncertainty dynamics; non-monotonic)

---

## References

[1] Transport Department, The Government of the Hong Kong Special Administrative Region, Annual Traffic Census 2024. Hong Kong, China: Transport Department, 2024.

[2] Transport Department, “Traffic data of strategic / major roads – traffic speeds of road network segments (processed data),” Data.Gov.HK, Hong Kong, China. [Online]. Available: https://resource.data.one.gov.hk/td/traffic-detectors/irnAvgSpeed-all.xml [Accessed: Dec. 31, 2025].

[3] Transport Department, “Traffic speed, volume and occupancy – data dictionary,” Data.Gov.HK, Hong Kong, China, Apr. 18, 2024. [Online]. Available: https://app.data.gov.hk/v1/historical-archive/get-data-dictionary?url=https%3A%2F%2Fstatic.data.gov.hk%2Ftd%2Ftraffic-data-strategic-major-roads%2Finfo%2Ftraffic_speed_volume_occ_info.csv&date=20240418 [Accessed: Dec. 31, 2025].
