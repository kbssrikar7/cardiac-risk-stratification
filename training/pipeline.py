"""Consolidated training pipeline: correlation pruning -> Optuna search over
both hyperparameters AND model family (XGBoost / LightGBM / logistic
regression) -> repeated stratified 5x5 CV evaluation -> calibration check ->
a structured, git-tracked run record.

Replaces the ad hoc pattern of running feature_selection.py,
finalize_calibrated_xgb.py, ablate_infarct_features.py and
finalize_infarct_model.py by hand in some remembered order, then manually
copy-pasting one script's best_params into another's hardcoded dict. Every
run here produces one comparable, versioned JSON record - see
training/runs/README.md.

This script does NOT promote its output model to production - see
training/promote_model.py for that explicit, separate step.
"""
import argparse
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import lightgbm as lgb
import xgboost as xgb

from common import RANDOM_STATE, brier_calibration_report, dynamic_resample, greedy_correlation_prune, repeated_cv_eval, summarize

TRAINING_DIR = Path(__file__).parent
REPO_ROOT = TRAINING_DIR.parent
BASE_CSV = REPO_ROOT / "combined_radiomics_features_FIXED.csv"
INFARCT_TRAIN_CSV = TRAINING_DIR / "infarct_features_train.csv"
INFARCT_TEST_CSV = TRAINING_DIR / "infarct_features_test.csv"
RUNS_DIR = TRAINING_DIR / "runs"
STUDY_DB = f"sqlite:///{TRAINING_DIR / 'optuna_studies.db'}"
INFARCT_FEATURE_COLS = ["infarct_volume_cm3", "infarct_pct_of_myocardium", "no_reflow_volume_cm3", "no_reflow_pct_of_infarct"]


def git_commit_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_dataset(use_infarct_features: bool, corr_threshold: float):
    base_df = pd.read_csv(BASE_CSV, dtype={"PatientID": str})
    base_df["PatientID"] = base_df["PatientID"].str.zfill(3)
    drop_cols = [c for c in ["PatientID", "Risk_Category", "Reasoning"] if c in base_df.columns]

    if use_infarct_features:
        infarct_train = pd.read_csv(INFARCT_TRAIN_CSV, dtype={"PatientID": str})
        infarct_test = pd.read_csv(INFARCT_TEST_CSV, dtype={"PatientID": str})
        infarct_all = pd.concat([infarct_train, infarct_test], ignore_index=True)
        infarct_all["PatientID"] = infarct_all["PatientID"].str.zfill(3)
        df = base_df.merge(infarct_all, on="PatientID", how="inner")
    else:
        df = base_df

    X_full = df.drop(columns=drop_cols + ["Risk_Score"], errors="ignore")
    radiomic_cols = [c for c in X_full.columns if c.startswith("original_")]
    non_radiomic_cols = [c for c in X_full.columns if c not in radiomic_cols]

    kept_radiomic, dropped = greedy_correlation_prune(X_full[radiomic_cols], corr_threshold)
    feature_columns = non_radiomic_cols + kept_radiomic
    print(f"Radiomics: {len(radiomic_cols)} -> {len(kept_radiomic)} after pruning |corr|>{corr_threshold} (dropped {len(dropped)})")
    print(f"Final feature set: {len(feature_columns)} features ({len(non_radiomic_cols)} non-radiomic + {len(kept_radiomic)} radiomic)")

    le = LabelEncoder()
    y = le.fit_transform(df["Risk_Score"].astype(str))
    label_mapping = dict(zip(range(len(le.classes_)), le.classes_))
    return df, X_full[feature_columns], y, label_mapping, feature_columns


def build_classifier(family: str, params: dict):
    if family == "xgboost":
        return xgb.XGBClassifier(**params)
    if family == "lightgbm":
        return lgb.LGBMClassifier(**params, verbosity=-1)
    if family == "logreg":
        return LogisticRegression(**params)
    raise ValueError(f"unknown model family: {family}")


def suggest_params(trial, family: str):
    if family == "xgboost":
        return {
            "max_depth": trial.suggest_categorical("xgb_max_depth", [3, 5, 7]),
            "learning_rate": trial.suggest_categorical("xgb_learning_rate", [0.01, 0.05, 0.1]),
            "n_estimators": trial.suggest_categorical("xgb_n_estimators", [200, 500, 1000]),
            "subsample": trial.suggest_categorical("xgb_subsample", [0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("xgb_colsample_bytree", [0.6, 0.8, 1.0]),
            "min_child_weight": trial.suggest_categorical("xgb_min_child_weight", [1, 3, 5]),
            "gamma": trial.suggest_categorical("xgb_gamma", [0.0, 0.1, 0.3]),
            "reg_lambda": trial.suggest_categorical("xgb_reg_lambda", [1.0, 5.0, 10.0]),
            "objective": "multi:softprob", "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
    if family == "lightgbm":
        return {
            "num_leaves": trial.suggest_categorical("lgb_num_leaves", [15, 31, 63]),
            "learning_rate": trial.suggest_categorical("lgb_learning_rate", [0.01, 0.05, 0.1]),
            "n_estimators": trial.suggest_categorical("lgb_n_estimators", [200, 500, 1000]),
            "subsample": trial.suggest_categorical("lgb_subsample", [0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("lgb_colsample_bytree", [0.6, 0.8, 1.0]),
            "min_child_samples": trial.suggest_categorical("lgb_min_child_samples", [5, 10, 20]),
            "reg_lambda": trial.suggest_categorical("lgb_reg_lambda", [1.0, 5.0, 10.0]),
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
    if family == "logreg":
        return {
            "C": trial.suggest_categorical("logreg_C", [0.01, 0.1, 1.0, 10.0]),
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
        }
    raise ValueError(f"unknown model family: {family}")


def optuna_search(X_df, y, n_trials, families, study_name):
    X_train, _, y_train, _ = train_test_split(X_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_df.columns)
    all_classes = sorted(np.unique(y))

    def objective(trial):
        family = trial.suggest_categorical("model_family", families)
        params = suggest_params(trial, family)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        X_arr, y_arr = X_train_s.values, np.array(y_train)
        aucs = []
        for tr_idx, val_idx in skf.split(X_arr, y_arr):
            X_tr_res, y_tr_res = dynamic_resample(X_arr[tr_idx], y_arr[tr_idx])
            clf = build_classifier(family, params)
            clf.fit(X_tr_res, y_tr_res)
            proba = clf.predict_proba(X_arr[val_idx])
            y_val_bin = label_binarize(y_arr[val_idx], classes=all_classes)
            aucs.append(roc_auc_score(y_val_bin, proba, average="macro", multi_class="ovr"))
        return float(np.mean(aucs))

    RUNS_DIR.mkdir(exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=STUDY_DB, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    family = best["model_family"]
    # suggest_params() returns unprefixed keys ready for build_classifier() directly
    # (the xgb_/lgb_/logreg_ prefixes only exist in the trial.suggest_* call names,
    # for Optuna's own bookkeeping); replaying the winning trial through a
    # FixedTrial reconstructs the exact params dict, including the fixed
    # (non-searched) fields like objective/eval_metric that never appear in
    # study.best_trial.params at all since they're plain literals, not suggest_* calls.
    params = suggest_params(optuna.trial.FixedTrial(best), family)
    return family, params, study.best_value


def promoted_baseline_metric(metric="f1_macro"):
    if not RUNS_DIR.exists():
        return None
    records = [json.loads(p.read_text()) for p in RUNS_DIR.glob("*.json")]
    promoted = [r for r in records if isinstance(r, dict) and r.get("promoted")]
    if not promoted:
        return None
    latest = max(promoted, key=lambda r: r["timestamp"])
    return latest["cv_metrics"][metric]["mean"], latest["run_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-infarct-features", action="store_true", default=True)
    parser.add_argument("--no-infarct-features", dest="use_infarct_features", action="store_false")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--families", nargs="+", default=["xgboost", "lightgbm", "logreg"], choices=["xgboost", "lightgbm", "logreg"])
    parser.add_argument("--run-name", default=None, help="label for this run, e.g. 'infarct-features'")
    args = parser.parse_args()

    df, X, y, label_mapping, feature_columns = load_dataset(args.use_infarct_features, args.corr_threshold)
    print(f"Dataset: {X.shape}, classes: {label_mapping}")

    study_name = args.run_name or ("with_infarct" if args.use_infarct_features else "baseline")
    print(f"\n--- Optuna search over {args.families} ({args.n_trials} trials, study='{study_name}') ---")
    family, params, inner_auc = optuna_search(X, y, args.n_trials, args.families, study_name)
    print(f"Selected model family: {family}")
    print(f"Best params: {params}")
    print(f"Inner CV AUC: {inner_auc:.4f}")

    print("\n--- Repeated stratified 5x5 CV (headline metric) ---")
    result = repeated_cv_eval(lambda X_tr, y_tr: build_classifier(family, params).fit(X_tr, y_tr), X, y, label=study_name)
    cv_metrics = summarize(result)

    # ---- Held-out split for an illustrative report + calibration check ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_res, y_train_res = dynamic_resample(X_train_s, np.array(y_train))

    base_clf = build_classifier(family, params)
    calibrated = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    calibrated.fit(X_train_res, y_train_res)
    probs_test = calibrated.predict_proba(X_test_s)
    preds_test = probs_test.argmax(1)
    class_names = [label_mapping[k] for k in sorted(label_mapping)]
    print("\n=== Calibrated final model, held-out 20% test (illustrative only; CV numbers above are the real estimate) ===")
    print(classification_report(y_test, preds_test, target_names=class_names, zero_division=0))

    brier = brier_calibration_report(probs_test, np.array(y_test), class_names)
    print("Per-class Brier scores:", brier)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit_hash()
    run_id = f"{timestamp}_{commit}"

    # ---- Final model trained on ALL resampled data (this is the candidate artifact) ----
    X_all_s = scaler.fit_transform(X)
    X_all_res, y_all_res = dynamic_resample(X_all_s, y)
    final_base = build_classifier(family, params)
    final_calibrated = CalibratedClassifierCV(final_base, cv=3, method="sigmoid")
    final_calibrated.fit(X_all_res, y_all_res)

    model_path = RUNS_DIR / f"{run_id}_model.pkl"
    joblib.dump({
        "model": final_calibrated, "scaler": scaler, "label_mapping": label_mapping,
        "feature_columns": feature_columns, "model_family": family,
    }, model_path)

    plot_path = RUNS_DIR / f"{run_id}_calibration.png"
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for cls_idx, cls_name in enumerate(class_names):
        y_true_bin = (np.array(y_test) == cls_idx).astype(int)
        if 2 <= y_true_bin.sum() <= len(y_true_bin) - 2:
            frac_pos, mean_pred = calibration_curve(y_true_bin, probs_test[:, cls_idx], n_bins=5, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", label=f"class {cls_name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve ({family}, {study_name})")
    plt.legend()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": commit,
        "run_name": study_name,
        "use_infarct_features": args.use_infarct_features,
        "corr_threshold": args.corr_threshold,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "model_family": family,
        "hyperparameters": params,
        "inner_search_auc": inner_auc,
        "cv_metrics": cv_metrics,
        "brier_scores": brier,
        "model_artifact": str(model_path.relative_to(REPO_ROOT)),
        "calibration_plot": str(plot_path.relative_to(REPO_ROOT)),
        "promoted": False,
    }
    record_path = RUNS_DIR / f"{run_id}.json"
    record_path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved run record: {record_path}")

    baseline = promoted_baseline_metric("f1_macro")
    print("\n=== Evaluation gate ===")
    if baseline is None:
        print("No currently-promoted model recorded yet - nothing to compare against.")
    else:
        baseline_f1, baseline_run_id = baseline
        new_f1 = cv_metrics["f1_macro"]["mean"]
        verdict = "BEATS" if new_f1 > baseline_f1 else ("MATCHES" if abs(new_f1 - baseline_f1) < 1e-9 else "REGRESSES vs")
        print(f"New run F1-macro={new_f1:.3f} {verdict} currently-promoted run {baseline_run_id} (F1-macro={baseline_f1:.3f})")
    print(f"\nThis run was NOT auto-promoted. To promote it: python training/promote_model.py {run_id}")


if __name__ == "__main__":
    main()
