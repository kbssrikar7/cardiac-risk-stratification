"""Produces the final production model: a single calibrated XGBoost classifier
on the corrected radiomics features, replacing the stacked ensemble.

Why: permutation importance (check_dl_branch_value.py) confirmed the
Attention-MLP branch contributes exactly 0 signal on the corrected features,
same as it did on the buggy ones - the ensemble was always just XGBoost with
extra steps. Dropping it removes a stale-distribution DL model, a meta-learner,
and their associated failure modes, with no accuracy cost.

Also addresses:
- Tier 1 #3 (evaluation methodology): reports repeated stratified k-fold CV
  metrics (mean +/- std) as the headline numbers, not a single lucky/unlucky
  80/20 split.
- Tier 5 #16 (calibration): wraps the final model in CalibratedClassifierCV
  and actually plots+checks the calibration curve, rather than assuming
  Platt scaling worked.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                              classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from common import risk_score_label_mapping
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

RANDOM_STATE = 42
CSV_PATH = "combined_radiomics_features_FIXED.csv"
OUT_MODEL = "training/best_prognostic_model_FINAL.pkl"  # staged; not swapped into production until asked
OUT_CALIBRATION_PLOT = "training/calibration_curve.png"


def dynamic_resample(X, y):
    unique, counts = np.unique(y, return_counts=True)
    maj_class, maj_count = unique[np.argmax(counts)], counts.max()
    under_strategy = {int(maj_class): int(maj_count * 0.5)}
    over_strategy = {int(c): int(int(maj_count * 0.5) * 0.75) for c in unique if c != maj_class}
    k_neighbors = max(1, counts.min() - 1)
    pipeline = ImbPipeline([
        ("u", RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_STATE)),
        ("o", SMOTE(sampling_strategy=over_strategy, random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
    ])
    return pipeline.fit_resample(X, y)


def main():
    df = pd.read_csv(CSV_PATH)
    drop_cols = [c for c in ["PatientID", "Risk_Category", "Reasoning"] if c in df.columns]
    X_full = df.drop(columns=drop_cols + ["Risk_Score"], errors="ignore")
    y_full = df["Risk_Score"].astype(int).to_numpy()
    label_mapping = risk_score_label_mapping(df)
    all_classes = sorted(np.unique(y_full))
    print("Classes:", label_mapping, " n =", len(y_full))

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, stratify=y_full, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_full.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_full.columns)

    def objective(trial):
        param = {
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 500, 1000]),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5]),
            "gamma": trial.suggest_categorical("gamma", [0.0, 0.1, 0.3]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [1.0, 5.0, 10.0]),
            "objective": "multi:softprob", "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE, "n_jobs": 1,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        X_arr, y_arr = X_train_s.values, np.array(y_train)
        aucs = []
        for tr_idx, val_idx in skf.split(X_arr, y_arr):
            X_tr_res, y_tr_res = dynamic_resample(X_arr[tr_idx], y_arr[tr_idx])
            clf = xgb.XGBClassifier(**param)
            clf.fit(X_tr_res, y_tr_res, verbose=False)
            proba = clf.predict_proba(X_arr[val_idx])
            y_val_bin = label_binarize(y_arr[val_idx], classes=all_classes)
            aucs.append(roc_auc_score(y_val_bin, proba, average="macro", multi_class="ovr"))
        return float(np.mean(aucs))

    print("Optuna tuning (60 trials, 5-fold CV per trial)...")
    study = optuna.create_study(direction="maximize", study_name="xgb_final_calibrated")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    print("Best trial:", study.best_trial.params, " mean CV AUC:", study.best_value)

    best = study.best_trial.params
    best_params = {
        "max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]), "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]), "min_child_weight": int(best["min_child_weight"]),
        "gamma": float(best["gamma"]), "reg_lambda": float(best["reg_lambda"]),
        "objective": "multi:softprob", "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE, "n_jobs": -1,
    }

    # ---- Repeated stratified k-fold evaluation on the FULL corrected dataset (Tier 1 #3) ----
    print("\nRepeated 5-fold x 5-repeat CV on the full corrected dataset (the trustworthy headline numbers):")
    X_full_s = pd.DataFrame(scaler.transform(X_full), columns=X_full.columns).values
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    accs, f1s, baccs, aucs = [], [], [], []
    for tr_idx, val_idx in rskf.split(X_full_s, y_full):
        X_tr_res, y_tr_res = dynamic_resample(X_full_s[tr_idx], y_full[tr_idx])
        clf = xgb.XGBClassifier(**best_params)
        clf.fit(X_tr_res, y_tr_res, verbose=False)
        proba = clf.predict_proba(X_full_s[val_idx])
        preds = proba.argmax(1)
        y_val = y_full[val_idx]
        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro", labels=all_classes, zero_division=0))
        baccs.append(balanced_accuracy_score(y_val, preds))
        try:
            aucs.append(roc_auc_score(label_binarize(y_val, classes=all_classes), proba, average="macro", multi_class="ovr"))
        except ValueError:
            pass  # a fold's val split can be missing a class at n=150; skip AUC for that fold only
    print(f"Accuracy:          {np.mean(accs):.3f} +/- {np.std(accs):.3f}  (n={len(accs)} folds)")
    print(f"F1 (macro):        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"Balanced accuracy: {np.mean(baccs):.3f} +/- {np.std(baccs):.3f}")
    print(f"AUC (macro OVR):   {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (n={len(aucs)} folds with all classes present)")

    # ---- Final model: train on all resampled training data, calibrate, hold-out check ----
    X_train_res, y_train_res = dynamic_resample(X_train_s.values, np.array(y_train))
    base_xgb = xgb.XGBClassifier(**best_params)
    calibrated = CalibratedClassifierCV(base_xgb, cv=3, method="sigmoid")
    calibrated.fit(X_train_res, y_train_res)

    probs_test = calibrated.predict_proba(X_test_s.values)
    preds_test = probs_test.argmax(1)
    print("\n=== Calibrated final model, held-out 20% test (illustrative; the CV numbers above are the real estimate) ===")
    print(classification_report(y_test, preds_test, target_names=le.classes_, zero_division=0))

    # ---- Calibration check (Tier 5 #16) ----
    print("\nCalibration check (predicted-prob vs observed-frequency, one-vs-rest per class):")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for cls_idx, cls_name in enumerate(le.classes_):
        y_true_bin = (y_test == cls_idx).astype(int)
        prob_pos = probs_test[:, cls_idx]
        brier = brier_score_loss(y_true_bin, prob_pos)
        print(f"  class {cls_name}: Brier score = {brier:.4f} (lower is better; 0.25 = uninformative for a 2-way split)")
        if y_true_bin.sum() >= 2 and y_true_bin.sum() <= len(y_true_bin) - 2:
            frac_pos, mean_pred = calibration_curve(y_true_bin, prob_pos, n_bins=5, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", label=f"class {cls_name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve (final calibrated XGBoost, corrected features)")
    plt.legend()
    plt.savefig(OUT_CALIBRATION_PLOT, dpi=120, bbox_inches="tight")
    print(f"Saved calibration plot: {OUT_CALIBRATION_PLOT}")

    # ---- Save as the production model file ----
    joblib.dump({"model": calibrated, "scaler": scaler, "label_mapping": label_mapping}, OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
