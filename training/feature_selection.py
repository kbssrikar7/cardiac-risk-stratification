"""Tier 3: addresses the high feature-to-sample ratio (107 radiomics features vs
~120 training samples) via correlation pruning, then compares the pruned
feature set against the full one under the identical repeated stratified CV
protocol used in finalize_calibrated_xgb.py - same hyperparameters, so any
difference is attributable to the feature set, not a hyperparameter search
lottery.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

RANDOM_STATE = 42
CSV_PATH = "combined_radiomics_features_FIXED.csv"
CORR_THRESHOLD = 0.95


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


def greedy_correlation_prune(df_features, threshold):
    corr = df_features.corr().abs()
    mean_corr = corr.mean()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > threshold].tolist()
        for other in correlated_with:
            if col in to_drop or other in to_drop:
                continue
            # drop whichever of the pair has higher average correlation to everything else
            to_drop.add(col if mean_corr[col] >= mean_corr[other] else other)
    return [c for c in df_features.columns if c not in to_drop], sorted(to_drop)


def evaluate(X_df, y, best_params, label):
    X_s = StandardScaler().fit_transform(X_df)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    accs, f1s, baccs, aucs = [], [], [], []
    all_classes = sorted(np.unique(y))
    for tr_idx, val_idx in rskf.split(X_s, y):
        X_tr_res, y_tr_res = dynamic_resample(X_s[tr_idx], y[tr_idx])
        clf = xgb.XGBClassifier(**best_params)
        clf.fit(X_tr_res, y_tr_res, verbose=False)
        proba = clf.predict_proba(X_s[val_idx])
        preds = proba.argmax(1)
        y_val = y[val_idx]
        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro", labels=all_classes, zero_division=0))
        baccs.append(balanced_accuracy_score(y_val, preds))
        try:
            aucs.append(roc_auc_score(label_binarize(y_val, classes=all_classes), proba, average="macro", multi_class="ovr"))
        except ValueError:
            pass
    print(f"\n=== {label} (n_features={X_df.shape[1]}) ===")
    print(f"Accuracy:          {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"F1 (macro):        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"Balanced accuracy: {np.mean(baccs):.3f} +/- {np.std(baccs):.3f}")
    print(f"AUC (macro OVR):   {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (n={len(aucs)} folds)")


def main():
    df = pd.read_csv(CSV_PATH)
    drop_cols = [c for c in ["PatientID", "Risk_Category", "Reasoning"] if c in df.columns]
    X_full = df.drop(columns=drop_cols + ["Risk_Score"], errors="ignore")
    radiomic_cols = [c for c in X_full.columns if c.startswith("original_")]
    clinical_cols = [c for c in X_full.columns if c not in radiomic_cols]

    le = LabelEncoder()
    y = le.fit_transform(df["Risk_Score"].astype(str))

    kept_radiomic, dropped = greedy_correlation_prune(X_full[radiomic_cols], CORR_THRESHOLD)
    print(f"Radiomics features: {len(radiomic_cols)} -> {len(kept_radiomic)} after pruning |corr|>{CORR_THRESHOLD}")
    print(f"Dropped {len(dropped)}: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")

    X_pruned = X_full[clinical_cols + kept_radiomic]

    final = joblib.load("training/best_prognostic_model_FINAL.pkl")
    base_est = final["model"].calibrated_classifiers_[0].estimator
    best_params = base_est.get_params()
    # get_params() includes some sklearn-wrapper-only args; keep just what XGBClassifier(**parm) accepts cleanly
    keep_keys = ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree",
                 "min_child_weight", "gamma", "reg_lambda", "objective", "eval_metric", "random_state", "n_jobs"]
    best_params = {k: best_params[k] for k in keep_keys if k in best_params}
    print("\nReusing the already-tuned hyperparameters for both runs (fair, apples-to-apples comparison):")
    print(best_params)

    evaluate(X_full, y, best_params, "FULL feature set (baseline)")
    evaluate(X_pruned, y, best_params, "PRUNED feature set")


if __name__ == "__main__":
    main()
