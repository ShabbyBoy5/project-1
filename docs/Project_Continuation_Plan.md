# Project Continuation Plan: Causal Counterfactual Explanations for Time-Series Anomaly Detection

## Overview

This plan continues the development of a system that generates causality-aware counterfactual explanations for anomalies in time-series data, with ECG as the primary case study (MIT-BIH Arrhythmia Database, ECG5000). The end-to-end pipeline will:
- Discover causal structure (PCMCI+ and Granger/VAR-LASSO) and construct an SCM over clinically-meaningful ECG features and anomaly scores.
- Generate counterfactual trajectories consistent with the causal graph using SCM-based interventions and constraint-aware generative models (VAE/diffusion).
- Evaluate counterfactuals on validity, proximity, plausibility, and causal constraint satisfaction.
- Visualize graphs, explanations, and trajectory overlays; provide dashboards for qualitative analysis.

Current repo state (as of 2025-10-27):
- Implemented:
  - `src/feature_extraction.py`: beat segmentation for MIT-BIH; feature sets (time-domain, Welch PSD, simple morphology, ACF); per-record processing + CSV export.
  - `src/wavelet_based_delineation.py`: wavelet delineation (P/QRS/T) + HR/HRV, added clinical intervals (PR, QRS, QT, QTc) and amplitudes; augmentation of feature CSV.
  - Dependencies pinned in `requirements.txt` (includes PyWavelets, tigramite, sklearn, wfdb, neurokit2, scipy, pandas, matplotlib). Notebook `Causality_py.ipynb` mirrors feature pipeline and has tigramite install cells.
- Missing/Planned:
  - Causal discovery integration (PCMCI+, Granger with VAR-LASSO) and graph fusion.
  - SCM definition/learning and do-intervention utilities.
  - Counterfactual synthesis (classifier- and anomaly-model–conditioned); constraint projection to satisfy causal graph.
  - Baselines (ProCE, CALIME, non-causal CFs) and anomaly detectors.
  - Evaluation metrics and dashboards; formal experiment scripts.

## Objectives

1) Build a modular pipeline to produce causality-consistent counterfactual explanations for detected anomalies.
2) Integrate causal discovery (PCMCI+, VAR-LASSO/Granger) and fit an SCM over ECG features + anomaly predictions.
3) Synthesize counterfactual trajectories that flip predictions with minimal, plausible, and causally valid changes.
4) Deliver quantitative evaluation and qualitative visualization/dashboards; compare to ProCE, CALIME, and non-causal baselines.

## Architecture

Planned package layout (building on existing `src/`):

- `src/data/`
  - `mitbih.py`: loaders, annotation handling (wraps existing functions), dataset splits.
  - `ecg5000.py`: UCR ECG5000 loader; standardize to beat-level windows and features.
- `src/features/`
  - `feature_extraction.py`: existing module (kept, moved into a package namespace).
  - `wavelet_delineation.py`: existing module (renamed) with clinical intervals; optional neurokit2 comparators.
- `src/models/`
  - `anomaly.py`: IsolationForest/LOF/OC-SVM; optional deep AE/LSTM-AE.
  - `predictor.py`: supervised classifiers for validity checking (RF, XGBoost/sklearn baseline).
- `src/causality/`
  - `pcmci_pipeline.py`: tigramite PCMCI+ routines.
  - `granger_var_lasso.py`: VAR-LASSO with lag search + stability selection.
  - `graph_fusion.py`: combine PCMCI+ and Granger edges with confidence aggregation.
- `src/scm/`
  - `scm.py`: structural equations (parametric and nonparametric), do() interventions, sampling.
  - `learn_scm.py`: fit functions from observational data + graph.
- `src/counterfactual/`
  - `scm_cf.py`: counterfactual via do-interventions and structural simulation.
  - `gen_cf.py`: constraint-aware VAE/diffusion CFs with causal projection.
- `src/eval/`
  - `metrics.py`: validity, proximity (L1/L2/DTW), plausibility (log-likelihood, clinical priors), causal satisfaction.
  - `benchmark.py`: experiment harness and comparisons.
- `src/viz/`
  - `graphs.py`: causal graph plots (networkx/pygraphviz, Plotly).
  - `timeseries.py`: overlays of original vs CF beats/trajectories; attribution heatmaps.
- `scripts/`
  - `run_mitbih_pipeline.py`, `run_ecg5000_pipeline.py`, `run_causality.py`, `run_counterfactuals.py`, `run_benchmark.py`.
- `docs/`
  - This plan, plus experiment sheets.

## Methodology

- Causal discovery
  - PCMCI+: time-lagged graph over feature vectors per beat or over multivariate per-record streams (aggregate by sliding windows). Use partial correlation or momentary conditional independence tests from tigramite; select max lag (e.g., 1–5 beats).
  - Granger via VAR-LASSO: fit sparse VAR with BIC/AIC-driven lag selection, lasso penalty via CV; infer directed edges when coefficients significant/stable.
  - Graph fusion: union with weights = f(p-values, stability), or intersection for high precision; prune with domain priors (e.g., PR → QRS → QT temporal order).
- SCM construction
  - Nodes: clinically-meaningful ECG features and derived intervals (PR, QRS, QTc, ST), plus anomaly score ŷ and classification label.
  - Structural equations: linear-Gaussian for first pass; nonparametric (GP/GRF) for robustness; noise terms estimated from residuals.
  - Interventions: do(X=x_cf) on parent variables; simulator samples descendants to generate consistent counterfactual feature vectors; decoder maps features back to waveform space when needed.
- Counterfactual synthesis
  - Classifier-conditional target (flip anomaly): min Δx s.t. f(x+Δx) flips; project onto SCM-consistent manifold via proximal step (solve: min ||Δx|| + λ C_scm(x+Δx)).
  - Generative: conditional VAE or diffusion model over beat windows with conditioning on target label and causal constraints via penalties or guided denoising.
- Evaluation
  - Validity: model prediction flips.
  - Proximity: L1/L2 on features; DTW on waveforms.
  - Plausibility: log p(x_cf) under generative model; clinical priors (e.g., QTc in normal bounds); HRV sanity.
  - Causal satisfaction: no violations of graph directions; counterfactuals respect do-interventions (descendant changes consistent with SCM); edge-consistency score.

## Implementation Plan (actionable)

Leverage existing code and extend incrementally. Items marked [P0] are critical; [P1] important; [P2] nice-to-have.

1) Data & features [P0]
   - Refactor `src/feature_extraction.py` into `src/features/feature_extraction.py`; expose:
     - `process_record(record, data_dir|pn_dir) -> (beats: np.ndarray, feat_df: pd.DataFrame)`
     - `run_mitbih_pipeline(record_list, out_dir, ...) -> Dict[str, np.ndarray], pd.DataFrame`
   - Keep/rename `src/wavelet_based_delineation.py` → `src/features/wavelet_delineation.py`, exposing:
     - `augment_with_wavelet_clinical(features_csv, pb_dir) -> pd.DataFrame`
   - Add ECG5000 loader in `src/data/ecg5000.py` with consistent interfaces.

2) Anomaly detection & predictors [P0]
   - Implement `src/models/anomaly.py` with sklearn baselines (IsolationForest, LOF, OneClassSVM). API:
     - `fit(X_train)`, `score(X) -> anomaly_score`, `predict(X) -> {0,1}`.
   - Implement `src/models/predictor.py` classifiers (RF/GBM) for validity checks.

3) Causal discovery [P0]
   - `src/causality/pcmci_pipeline.py`
     - Inputs: multivariate time series per record (e.g., clinical+beat features in temporal order), lags, cond-indep test.
     - Output: adjacency with lagged edges and strengths.
   - `src/causality/granger_var_lasso.py`
     - Fit sparse VAR with lasso; infer directed edges; return adjacency and lag weights.
   - `src/causality/graph_fusion.py`
     - Combine edges; thresholds; export DAG (resolve cycles with ordering priors: P → QRS → QT → T).

4) SCM [P1]
   - `src/scm/scm.py`: structural equations per node; simulate descendants; sample noise.
   - `src/scm/learn_scm.py`: fit equations from data + graph (linear first).

5) Counterfactuals [P1]
   - `src/counterfactual/scm_cf.py`: local search on parent sets (do-interventions) to flip model; project to SCM manifold.
   - `src/counterfactual/gen_cf.py`: cVAE baseline over beats with conditioning on label; optional diffusion model later [P2].

6) Evaluation [P0]
   - `src/eval/metrics.py`: implement metrics:
     - Validity (flip %), Proximity (L1/L2/DTW), Plausibility (log-likelihood, clinical constraint violations), Causal score.
   - `src/eval/benchmark.py`: run baselines (ProCE, CALIME via existing packages or re-implement outlines) vs ours.

7) Visualization [P1]
   - `src/viz/graphs.py`: networkx plot + Plotly interactivity.
   - `src/viz/timeseries.py`: waveform overlays, feature trajectories; CF vs original side-by-side.
   - Optional Streamlit dashboard.

8) Scripts [P0]
   - `run_mitbih_pipeline.py`: end-to-end feature extraction + augmentation (already functionally in place).
   - `run_causality.py`: build temporal datasets, run PCMCI+ and VAR-LASSO, fuse graph, export.
   - `run_counterfactuals.py`: train anomaly model, synthesize CFs, evaluate, save artifacts.
   - `run_benchmark.py`: aggregate metrics across methods and datasets; generate tables.

### Pseudocode hooks (using existing code)

- Feature pipeline (exists)
```python
records = ['100','101','103','105']
beats_by_rec, feat_all = run_mitbih_pipeline(records, out_dir="artifacts/mitbih", data_dir="data/mitdb")
feat_aug = augment_with_wavelet_clinical("artifacts/mitbih/all_records_features.csv", pb_dir='mitdb/')
```

- PCMCI+ (to implement)
```python
from tigramite import data_processing as pp
from tigramite import independence_tests, plotting
from tigramite.pcmci import PCMCI

# X: T x D multivariate series (stack records or per-record analysis)
X = build_series_from_features(feat_aug, keys=["PR_s","QRS_s","QTc_Bazett_s","HR_bpm", ...])
data = pp.DataFrame(X)
ci_test = independence_tests.ParCorr()  # or GPDC
pcmci = PCMCI(dataframe=data, cond_ind_test=ci_test, verbosity=1)
results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)
G_pcmci = threshold_results(results)
```

- VAR-LASSO Granger (to implement)
```python
G_granger = fit_var_lasso_granger(X, max_lag=5, alpha_grid=[1e-3,1e-2,1e-1])
G = fuse_graphs(G_pcmci, G_granger, mode="union", conf_threshold=0.6)
```

- SCM + CF (to implement)
```python
scm = learn_linear_scm(G, X)
clf = train_anomaly_classifier(feat_aug)
x = select_anomalous_instance(feat_aug)
x_cf = counterfactual_via_scm(scm, clf, x, target_label=1)
```

## Evaluation

- Metrics
  - Validity: fraction of CFs flipping the model prediction.
  - Proximity: feature-space L1/L2; waveform DTW/SSIM; percent change of clinical intervals.
  - Plausibility: log p(x_cf) under generative model; clinical constraints (QTc ∈ [350, 450] ms; HR ∈ [50, 120] bpm) violation rate.
  - Causal satisfaction: edges respected; counterfactual effects localized to descendants of intervened nodes; structural residual consistency.

- Baselines and Comparisons
  - ProCE, CALIME (implementations or re-implement baselines if packages unavailable) and non-causal CFs (Vanilla gradient/proximal, DiCE-like).
  - Report across MIT-BIH and ECG5000; per-class breakdown for AAMI types.

- Experimental Protocol
  - Train/val/test splits per record; avoid leakage by record and by patient.
  - Evaluate on anomalies flagged by anomaly detectors and on supervised misclassifications.
  - Ablations: (a) PCMCI-only vs VAR-LASSO-only vs fused; (b) linear-SCM vs nonparametric; (c) with vs without waveform generator.

## Deliverables

- Code modules as listed in Architecture and Implementation Plan.
- Reproducible scripts in `scripts/` producing:
  - Causal graph visualizations (.png/.html) and exported graphs (.json/.gml).
  - Counterfactual waveforms (.npy) and overlays (.html).
  - Metrics tables (.csv) and plots (.png).
- Documentation:
  - README updates and a short user guide.
  - This plan and experiment sheets.

## Timeline (8–10 weeks)

- Week 1: Refactor features; add ECG5000 loader; dataset curation; unit tests.
- Week 2: Anomaly detection baselines + predictor for validity; artifacts saved.
- Week 3: PCMCI+ pipeline; produce initial graphs; sanity-check with clinical priors.
- Week 4: VAR-LASSO Granger; graph fusion; export DAGs.
- Week 5: Linear SCM learning + do-intervention simulator; first SCM-based CFs.
- Week 6: Generative CFs (cVAE); projection onto SCM constraints; evaluation v1.
- Week 7: Baseline implementations (ProCE, CALIME); comprehensive benchmarks.
- Week 8: Visualization + dashboard; write-up of results; polishing and ablations.
- Weeks 9–10 [optional]: Diffusion-based CFs; domain adaptation; cross-dataset generalization.

## Extensions and Publication-Facing Enhancements

- Diffusion models for ECG beat generation with causal guidance (classifier-free guidance with SCM-based penalty term).
- Domain adaptation and transfer (ECG lead variation, patient-specific adapters).
- Semi-supervised anomaly detection with causal regularization.
- Robustness tests under label noise and missing features; uncertainty-aware SCMs.
- Human-in-the-loop adjustments of causal graph with active querying.

## Dataset and Preprocessing Notes

- MIT-BIH: Use `wfdb` to load records; prefer MLII lead, fallback to V5; sampling rate FS=360 Hz.
- Annotations: keep AAMI-consistent mapping (already in code), binary normal (N) vs anomaly (non-N) label present.
- Beat windows: default 200 ms pre, 400 ms post; adjustable.
- ECG5000: Convert to the same schema (beats, features, clinical intervals if possible). Normalize per record to stabilize features.

## Risks and Edge Cases

- Short records or sparse anomalies → unstable causal discovery (mitigate via aggregation and regularization).
- Lead differences (MLII vs V5) → morphology variability (mitigate via per-record scaling and robust features).
- Cycles in fused graph → resolve via temporal ordering and clinical priors.
- Counterfactuals leaving physiologic ranges → enforce clinical constraints and plausibility penalties.

## Ready-for-Parallelization Checklist

- Each module has clear inputs/outputs and file artifacts.
- Teams can work independently on: causality, SCM, generators, evaluation, and visualization.
- Scripts orchestrate end-to-end flows and dump artifacts into `artifacts/`.
