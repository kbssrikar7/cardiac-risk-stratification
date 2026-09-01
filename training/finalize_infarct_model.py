"""Publication/patent push, final step: saves the infarct-burden-augmented
model as a staged candidate artifact, using the hyperparameters Optuna found
during ablate_infarct_features.py's WITH-features search (inner CV AUC 0.983).

This is a candidate, not a production promotion - it lives in training/ until
the user reviews the ablation results and explicitly decides whether to
replace best_prognostic_model.pkl with it.
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
BASE_CSV = "combined_radiomics_features_FIXED.csv"
INFARCT_TRAIN_CSV = "training/infarct_features_train.csv"
INFARCT_TEST_CSV = "training/infarct_features_test.csv"
FINAL_MODEL_PATH = "training/best_prognostic_model_FINAL.pkl"
OUT_MODEL = "training/best_prognostic_model_INFARCT.pkl"
NEW_FEATURE_COLS = ["infarct_volume_cm3", "infarct_pct_of_myocardium", "no_reflow_volume_cm3", "no_reflow_pct_of_infarct"]

BEST_PARAMS = {
    "max_depth": 5, "learning_rate": 0.01, "n_estimators": 500, "subsample": 0.8,
    "colsample_bytree": 1.0, "min_child_weight": 1, "gamma": 0.1, "reg_lambda": 1.0,
    "objective": "multi:softprob", "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE, "n_jobs": -1,
}


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
    base_df = pd.read_csv(BASE_CSV, dtype={"PatientID": str})
    base_df["PatientID"] = base_df["PatientID"].str.zfill(3)

    final = joblib.load(FINAL_MODEL_PATH)
    feature_columns = final["feature_columns"] + NEW_FEATURE_COLS

    infarct_train = pd.read_csv(INFARCT_TRAIN_CSV, dtype={"PatientID": str})
    infarct_test = pd.read_csv(INFARCT_TEST_CSV, dtype={"PatientID": str})
    infarct_all = pd.concat([infarct_train, infarct_test], ignore_index=True)
    infarct_all["PatientID"] = infarct_all["PatientID"].str.zfill(3)

    merged = base_df.merge(infarct_all, on="PatientID", how="inner")
    print(f"Merged dataset: {merged.shape}, using {len(feature_columns)} features")

    le = LabelEncoder()
    y = le.fit_transform(merged["Risk_Score"].astype(str))
    label_mapping = dict(zip(range(len(le.classes_)), le.classes_))

    X = merged[feature_columns]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_res, y_res = dynamic_resample(X_s, y)

    base_xgb = xgb.XGBClassifier(**BEST_PARAMS)
    calibrated = CalibratedClassifierCV(base_xgb, cv=3, method="sigmoid")
    calibrated.fit(X_res, y_res)

    joblib.dump({
        "model": calibrated,
        "scaler": scaler,
        "label_mapping": label_mapping,
        "feature_columns": feature_columns,
    }, OUT_MODEL)
    print(f"Saved candidate: {OUT_MODEL}")
    print("Not promoted to production - staged in training/ pending review.")


if __name__ == "__main__":
    main()
