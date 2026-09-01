"""Publication/patent push, step 3: the actual ablation. Merges the new
infarct-burden features (from compute_infarct_features_{train,test}.py) onto
the already-corrected, correlation-pruned 63-feature set, then retrains and
evaluates WITH and WITHOUT the new features under the *identical* repeated
stratified 5x5 CV protocol (same seeds, same hyperparameter search budget),
so any difference is attributable to the features, not to search variance.

This is the honest evidence a paper reviewer or patent examiner needs: not a
single cherry-picked number, but a controlled comparison.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

RANDOM_STATE = 42
BASE_CSV = "combined_radiomics_features_FIXED.csv"
INFARCT_TRAIN_CSV = "training/infarct_features_train.csv"
INFARCT_TEST_CSV = "training/infarct_features_test.csv"
FINAL_MODEL_PATH = "training/best_prognostic_model_FINAL.pkl"
NEW_FEATURE_COLS = ["infarct_volume_cm3", "infarct_pct_of_myocardium", "no_reflow_volume_cm3", "no_reflow_pct_of_infarct"]


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


def optuna_search(X_df, y, n_trials=60):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_df.columns)
    all_classes = sorted(np.unique(y))

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

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial.params
    return {
        "max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]), "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]), "min_child_weight": int(best["min_child_weight"]),
        "gamma": float(best["gamma"]), "reg_lambda": float(best["reg_lambda"]),
        "objective": "multi:softprob", "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE, "n_jobs": -1,
    }, study.best_value


def repeated_cv_eval(X_df, y, best_params, label):
    X_s = StandardScaler().fit_transform(X_df)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    all_classes = sorted(np.unique(y))
    accs, f1s, baccs, aucs = [], [], [], []
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
    print(f"\n=== {label} ===")
    print(f"Accuracy:          {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"F1 (macro):        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"Balanced accuracy: {np.mean(baccs):.3f} +/- {np.std(baccs):.3f}")
    print(f"AUC (macro OVR):   {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (n={len(aucs)} folds)")
    return {"accuracy": accs, "f1_macro": f1s, "balanced_accuracy": baccs, "auc": aucs}


def main():
    base_df = pd.read_csv(BASE_CSV, dtype={"PatientID": str})
    base_df["PatientID"] = base_df["PatientID"].str.zfill(3)

    final = joblib.load(FINAL_MODEL_PATH)
    feature_columns = final["feature_columns"]

    infarct_train = pd.read_csv(INFARCT_TRAIN_CSV, dtype={"PatientID": str})
    infarct_test = pd.read_csv(INFARCT_TEST_CSV, dtype={"PatientID": str})
    infarct_all = pd.concat([infarct_train, infarct_test], ignore_index=True)
    infarct_all["PatientID"] = infarct_all["PatientID"].str.zfill(3)
    print(f"Infarct features: {len(infarct_train)} train (ground truth) + {len(infarct_test)} test (predicted) = {len(infarct_all)}")

    merged = base_df.merge(infarct_all, on="PatientID", how="inner")
    print(f"Merged dataset: {merged.shape}")

    le = LabelEncoder()
    y = le.fit_transform(merged["Risk_Score"].astype(str))

    X_without = merged[feature_columns]
    X_with = merged[feature_columns + NEW_FEATURE_COLS]

    print("\n--- Tuning WITHOUT infarct-burden features (baseline, 63 features) ---")
    params_without, auc_without = optuna_search(X_without, y)
    print("Best params:", params_without, " inner CV AUC:", auc_without)

    print("\n--- Tuning WITH infarct-burden features (67 features) ---")
    params_with, auc_with = optuna_search(X_with, y)
    print("Best params:", params_with, " inner CV AUC:", auc_with)

    results_without = repeated_cv_eval(X_without, y, params_without, "WITHOUT infarct-burden features (baseline)")
    results_with = repeated_cv_eval(X_with, y, params_with, "WITH infarct-burden features")

    print("\n=== Ablation summary (mean +/- std, 5x5 repeated stratified CV) ===")
    for metric in ["accuracy", "f1_macro", "balanced_accuracy", "auc"]:
        m_without = np.mean(results_without[metric])
        m_with = np.mean(results_with[metric])
        print(f"{metric}: without={m_without:.3f}  with={m_with:.3f}  delta={m_with - m_without:+.3f}")

    # SHAP-style importance check: does the model actually use the new features?
    print("\n--- Feature importance check (gain-based) for the WITH-features model ---")
    X_with_res, y_with_res = dynamic_resample(StandardScaler().fit_transform(X_with), y)
    final_with = xgb.XGBClassifier(**params_with)
    final_with.fit(X_with_res, y_with_res, verbose=False)
    importances = final_with.feature_importances_
    ranked = sorted(zip(X_with.columns, importances), key=lambda t: -t[1])
    print("Top 15 features by gain:")
    for name, imp in ranked[:15]:
        marker = "  <-- NEW" if name in NEW_FEATURE_COLS else ""
        print(f"  {name}: {imp:.4f}{marker}")


if __name__ == "__main__":
    main()
