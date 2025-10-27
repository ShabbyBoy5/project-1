#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline runner using uv (https://github.com/astral-sh/uv)
# Overrides via env vars:
#   DATA_DIR, OUT_BASE, TAU_MAX_CAUSAL, TAU_MAX_SCM, ANOM_MODEL, MAX_K, BOUNDS_LO, BOUNDS_HI, MIN_SCORE_DROP, MAX_MAHA, MAX_RESID_Z, BEAM_WIDTH, LIMIT

DATA_DIR=${DATA_DIR:-dataset}
OUT_BASE=${OUT_BASE:-artifacts}
FEATURES_CSV=${FEATURES_CSV:-$OUT_BASE/ecg5000/features.csv}
CAUSAL_DIR=${CAUSAL_DIR:-$OUT_BASE/causality_ecg5000}
SCM_PATH=${SCM_PATH:-$OUT_BASE/scm/ecg5000_linear_lagged.json}
ANOM_MODEL=${ANOM_MODEL:-isoforest}
TAU_MAX_CAUSAL=${TAU_MAX_CAUSAL:-5}
TAU_MAX_SCM=${TAU_MAX_SCM:-3}
MAX_K=${MAX_K:-2}
BOUNDS_LO=${BOUNDS_LO:-5}
BOUNDS_HI=${BOUNDS_HI:-95}
MIN_SCORE_DROP=${MIN_SCORE_DROP:-0.1}
MAX_MAHA=${MAX_MAHA:-6.0}
MAX_RESID_Z=${MAX_RESID_Z:-3.0}
BEAM_WIDTH=${BEAM_WIDTH:-10}
LIMIT=${LIMIT:-200}

# Ensure uv and venv
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required. See https://github.com/astral-sh/uv" >&2
  exit 1
fi

# Create venv (idempotent) and install deps
uv venv || true
uv pip install -r requirements_ecg.txt

# 1) Feature extraction
echo "[1/6] Extracting features -> $FEATURES_CSV"
uv run python scripts/run_ecg5000_pipeline.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_BASE/ecg5000"

# 2) Causal discovery (PCMCI+ and VAR-LASSO) and fusion
echo "[2/6] Running causality -> $CAUSAL_DIR"
uv run python scripts/run_causality.py \
  --csv "$FEATURES_CSV" \
  --fallback_csv "$FEATURES_CSV" \
  --tau_max "$TAU_MAX_CAUSAL" --max_lag "$TAU_MAX_CAUSAL" \
  --pc_alpha 0.05 --fdr fdr_bh \
  --fuse union \
  --outdir "$CAUSAL_DIR"

# 3) Fit SCM from fused graph
echo "[3/6] Fitting SCM -> $SCM_PATH"
uv run python scripts/run_scm_fit.py \
  --csv "$FEATURES_CSV" \
  --edges "$CAUSAL_DIR/graph_fused.json" \
  --tau_max "$TAU_MAX_SCM" \
  --method ridge --alpha 1.0 \
  --out "$SCM_PATH"

# 4) Train anomaly model and score full dataset
ANOM_DIR="$OUT_BASE/anomaly_ecg5000"
echo "[4/6] Training anomaly model ($ANOM_MODEL) -> $ANOM_DIR"
uv run python scripts/run_anomaly.py \
  --csv "$FEATURES_CSV" \
  --model "$ANOM_MODEL" \
  --outdir "$ANOM_DIR" \
  --calibrate

# 5) Generate counterfactuals with strict settings
ANOM_CSV="$ANOM_DIR/features_with_${ANOM_MODEL}.csv"
CF_DIR="$OUT_BASE/counterfactuals_ecg5000/strict_${BOUNDS_LO}_${BOUNDS_HI}_min${MIN_SCORE_DROP}_maha${MAX_MAHA}"
echo "[5/6] Generating counterfactuals -> $CF_DIR"
uv run python scripts/run_counterfactuals.py \
  --csv "$ANOM_CSV" \
  --scm "$SCM_PATH" \
  --model "$ANOM_MODEL" \
  --max_k "$MAX_K" \
  --bounds_pct "$BOUNDS_LO" "$BOUNDS_HI" \
  --prefer_mahalanobis \
  --cost_mode best --lambda_maha 0.1 \
  --priority spec_centroid spec_bw \
  --min_score_drop "$MIN_SCORE_DROP" \
  --max_maha "$MAX_MAHA" \
  --max_resid_z "$MAX_RESID_Z" \
  --beam_width "$BEAM_WIDTH" \
  --limit "$LIMIT" \
  --outdir "$CF_DIR"

# 6) Summarize CF quality and SCM diagnostics
echo "[6/6] Summaries"
uv run python scripts/evaluate_cfs.py \
  --cf_json "$CF_DIR/cf_results_${ANOM_MODEL}.json"

uv run python scripts/evaluate_scm_forecast.py \
  --csv "$FEATURES_CSV" \
  --edges "$CAUSAL_DIR/graph_fused.json" \
  --tau_max "$TAU_MAX_SCM" --method ridge --alpha 1.0 \
  --out "$OUT_BASE/scm/eval_report_reg.json"

echo "Done. Artifacts in: $OUT_BASE"
