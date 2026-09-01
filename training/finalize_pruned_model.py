"""Retrains the final production model on the correlation-pruned feature set
(feature_selection.py showed 63 features match 111-feature performance
exactly), reusing the already-tuned hyperparameters. Saves the kept feature
list alongside the model so inference code knows which columns to select.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

RANDOM_STATE = 42
CSV_PATH = "combined_radiomics_features_FIXED.csv"
OUT_MODEL = "training/best_prognostic_model_FINAL.pkl"


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


def greedy_correlation_prune(df_features, threshold=0.95):
    corr = df_features.corr().abs()
    mean_corr = corr.mean()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        for other in upper.index[upper[col] > threshold].tolist():
            if col in to_drop or other in to_drop:
                continue
            to_drop.add(col if mean_corr[col] >= mean_corr[other] else other)
    return [c for c in df_features.columns if c not in to_drop]


def main():
    df = pd.read_csv(CSV_PATH)
    drop_cols = [c for c in ["PatientID", "Risk_Category", "Reasoning"] if c in df.columns]
    X_full = df.drop(columns=drop_cols + ["Risk_Score"], errors="ignore")
    radiomic_cols = [c for c in X_full.columns if c.startswith("original_")]
    clinical_cols = [c for c in X_full.columns if c not in radiomic_cols]

    kept_radiomic = greedy_correlation_prune(X_full[radiomic_cols])
    feature_columns = clinical_cols + kept_radiomic
    X = X_full[feature_columns]
    print(f"Using {len(feature_columns)} features ({len(clinical_cols)} clinical + {len(kept_radiomic)} radiomics)")

    le = LabelEncoder()
    y = le.fit_transform(df["Risk_Score"].astype(str))
    label_mapping = dict(zip(range(len(le.classes_)), le.classes_))

    best_params = {
        "max_depth": 7, "learning_rate": 0.01, "n_estimators": 200, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0.0, "reg_lambda": 1.0,
        "objective": "multi:softprob", "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE, "n_jobs": -1,
    }

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_res, y_res = dynamic_resample(X_s, y)

    base_xgb = xgb.XGBClassifier(**best_params)
    calibrated = CalibratedClassifierCV(base_xgb, cv=3, method="sigmoid")
    calibrated.fit(X_res, y_res)

    joblib.dump({
        "model": calibrated,
        "scaler": scaler,
        "label_mapping": label_mapping,
        "feature_columns": feature_columns,
    }, OUT_MODEL)
    print(f"Saved: {OUT_MODEL} (with feature_columns recorded for inference-time column selection)")


if __name__ == "__main__":
    main()
