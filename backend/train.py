"""
Sahl UFC Predictor - Training Pipeline
==========================================
- XGBoost binary classifier on the 24 engineered features
- Walk-forward CV (no random shuffling - this is time-series!)
- Isotonic calibration (Platt is also implemented as fallback)
- Outputs:
    models/xgb_model.json     XGBoost booster
    models/calibrator.pkl     fitted isotonic regressor
    models/feature_list.json  feature column order
    output/reliability.png    reliability curve
    output/shap_summary.png   SHAP feature importance
    output/metrics.json       accuracy, log-loss, Brier, AUC
"""

from __future__ import annotations
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train")


FEATURE_COLS = [
    "elo_diff", "elo_trend_diff", "win_streak_diff", "win_pct_diff",
    "recent_form_diff", "slpm_diff", "sapm_diff", "str_acc_diff",
    "str_def_diff", "kd_per15_diff", "td_per15_diff", "td_acc_diff",
    "td_def_diff", "ctrl_pct_diff", "sub_att_per15_diff", "getup_speed_diff",
    "age_diff", "height_diff_cm", "reach_diff_cm", "weight_class_encoded",
    "stance_match", "layoff_diff_days", "ufc_exp_diff", "title_bout",
]


def walk_forward_split(df: pd.DataFrame, n_splits: int = 5):
    """Yields (train_idx, val_idx) growing-window splits sorted by date."""
    df = df.sort_values("event_date").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        val_end = min(fold_size * (k + 1), n)
        yield np.arange(0, train_end), np.arange(train_end, val_end)


def fit_xgb(X_train, y_train, X_val=None, y_val=None) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=7,
        early_stopping_rounds=40 if X_val is not None else None,
    )
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../data/features.csv")
    ap.add_argument("--out-dir", default="../models")
    ap.add_argument("--report-dir", default="../output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(args.report_dir); rep_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df = df[df["is_decided"] == 1].copy()
    df = df.sort_values("event_date").reset_index(drop=True)
    log.info("Loaded %d decided fights", len(df))

    X = df[FEATURE_COLS].values
    y = df["a_won"].values

    # ---------- Walk-forward CV ----------
    cv_metrics = []
    for fold, (tr, va) in enumerate(walk_forward_split(df, n_splits=5), 1):
        m = fit_xgb(X[tr], y[tr], X[va], y[va])
        p = m.predict_proba(X[va])[:, 1]
        cv_metrics.append({
            "fold": fold,
            "n_train": len(tr), "n_val": len(va),
            "acc": float(accuracy_score(y[va], (p > 0.5).astype(int))),
            "logloss": float(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y[va], p)),
            "auc": float(roc_auc_score(y[va], p)) if len(set(y[va])) > 1 else float("nan"),
        })
        log.info("Fold %d -> %s", fold, cv_metrics[-1])

    # ---------- Final train on all but last 15% (for calibration & test) ----------
    cut = int(len(df) * 0.85)
    X_tr, y_tr = X[:cut], y[:cut]
    X_te, y_te = X[cut:], y[cut:]
    # split off a validation tail for early stopping
    val_cut = int(len(X_tr) * 0.9)
    model = fit_xgb(X_tr[:val_cut], y_tr[:val_cut], X_tr[val_cut:], y_tr[val_cut:])

    # ---------- Calibration on val set ----------
    p_val_raw = model.predict_proba(X_tr[val_cut:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_tr[val_cut:])

    p_te_raw = model.predict_proba(X_te)[:, 1]
    p_te_cal = iso.transform(p_te_raw)

    final_metrics = {
        "test_n": int(len(X_te)),
        "raw": {
            "acc": float(accuracy_score(y_te, (p_te_raw > 0.5).astype(int))),
            "logloss": float(log_loss(y_te, np.clip(p_te_raw, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y_te, p_te_raw)),
            "auc": float(roc_auc_score(y_te, p_te_raw)) if len(set(y_te)) > 1 else float("nan"),
        },
        "calibrated": {
            "acc": float(accuracy_score(y_te, (p_te_cal > 0.5).astype(int))),
            "logloss": float(log_loss(y_te, np.clip(p_te_cal, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y_te, p_te_cal)),
            "auc": float(roc_auc_score(y_te, p_te_cal)) if len(set(y_te)) > 1 else float("nan"),
        },
        "cv_folds": cv_metrics,
    }
    log.info("Final test metrics: %s", json.dumps(final_metrics, indent=2))

    # ---------- Reliability curve ----------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    for label, p in [("Raw XGBoost", p_te_raw), ("Isotonic-calibrated", p_te_cal)]:
        frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Predicted P(A wins)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Curve - Sahl UFC Predictor")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rep_dir / "reliability.png")
    plt.close(fig)
    log.info("Saved reliability curve")

    # ---------- SHAP summary ----------
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te[: min(800, len(X_te))])
        plt.figure(figsize=(9, 7), dpi=120)
        shap.summary_plot(
            shap_vals, X_te[: min(800, len(X_te))],
            feature_names=FEATURE_COLS, show=False, max_display=24,
        )
        plt.tight_layout()
        plt.savefig(rep_dir / "shap_summary.png")
        plt.close("all")
        log.info("Saved SHAP summary")
    except Exception as e:
        log.warning("SHAP step skipped: %s", e)

    # ---------- Persist ----------
    model.get_booster().save_model(str(out_dir / "xgb_model.json"))
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(rep_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    log.info("Done. Model + artifacts saved.")


if __name__ == "__main__":
    main()
