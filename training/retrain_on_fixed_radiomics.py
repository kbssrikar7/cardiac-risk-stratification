"""Retrains the XGBoost prognostic model and the stacked ensemble on the
myocardium/LV-cavity label bug fix (combined_radiomics_features_FIXED.csv).

Adapts the Optuna XGBoost search and the out-of-fold stacking approach drafted
in heartriskstratificationq4.py's uncommitted working-tree changes, as a clean
self-contained script rather than executing that file's full linear notebook
export (which has several sections coupled by implicit shared global state,
including a stale-variable bug in its own OOF helper call).

The Attention-MLP branch is NOT retrained here: it was already shown (via
permutation importance on the pre-fix ensemble) to contribute ~0 signal to the
final prediction, so it's reused as-is from attention_best.pt. Its predictions
on the corrected radiomics features are therefore somewhat out-of-distribution
relative to what it was trained on - a known, accepted limitation until the
Attention-MLP itself is revisited.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                              f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

from common import risk_score_label_mapping

RANDOM_STATE = 42
CSV_PATH = "combined_radiomics_features_FIXED.csv"
CLINICAL_COLS = ["Age", "LVEF", "Troponin", "NTProBNP"]
OUT_XGB_MODEL = "training/best_prognostic_model_FIXED.pkl"
OUT_ENSEMBLE = "training/stacked_fusion_smote_FIXED.pkl"


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
    print("Loaded corrected radiomics CSV:", df.shape)

    drop_cols = [c for c in ["PatientID", "Risk_Category", "Reasoning"] if c in df.columns]
    X_full = df.drop(columns=drop_cols + ["Risk_Score"], errors="ignore")
    y_full = df["Risk_Score"].astype(int).to_numpy()
    label_mapping = risk_score_label_mapping(df)
    all_classes = sorted(np.unique(y_full))
    print("Classes:", label_mapping)

    # ---------------- XGBoost + Optuna (same search space/CV as the existing pipeline) ----------------
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

    print("Starting Optuna tuning (60 trials, 5-fold CV per trial)...")
    study = optuna.create_study(direction="maximize", study_name="xgb_multi_opt_fixed")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    print("Best trial params:", study.best_trial.params, "  mean CV AUC:", study.best_value)

    best = study.best_trial.params
    best_xgb_params = {
        "max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]), "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]), "min_child_weight": int(best["min_child_weight"]),
        "gamma": float(best["gamma"]), "reg_lambda": float(best["reg_lambda"]),
        "objective": "multi:softprob", "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE, "n_jobs": -1,
    }
    X_train_res, y_train_res = dynamic_resample(X_train_s.values, np.array(y_train))
    final_xgb = xgb.XGBClassifier(**best_xgb_params)
    final_xgb.fit(X_train_res, y_train_res, verbose=False)

    joblib.dump({"model": final_xgb, "scaler": scaler, "label_mapping": label_mapping}, OUT_XGB_MODEL)
    print(f"Saved: {OUT_XGB_MODEL}")

    probs_test = final_xgb.predict_proba(X_test_s.values)
    preds_test = probs_test.argmax(1)
    print("\n=== XGBoost hold-out test performance (single 80/20 split - still not the Tier 1 CV fix, just this stage) ===")
    print(classification_report(y_test, preds_test, target_names=le.classes_, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, preds_test))

    # ---------------- Ensemble: proper out-of-fold XGBoost meta-features + existing DL branch ----------------
    print("\nBuilding stacked ensemble with out-of-fold XGBoost meta-features...")
    X_full_scaled = scaler.transform(X_full)  # use the SAME fitted scaler consistently for all 150 rows

    def get_oof_predictions(X, y, base_model, skf):
        oof = np.zeros((X.shape[0], len(np.unique(y))))
        for tr_idx, val_idx in skf.split(X, y):
            X_tr_res, y_tr_res = dynamic_resample(X[tr_idx], y[tr_idx])
            m = clone(base_model)
            m.fit(X_tr_res, y_tr_res)
            oof[val_idx] = m.predict_proba(X[val_idx])
        return oof

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = get_oof_predictions(X_full_scaled, y_full, xgb.XGBClassifier(**best_xgb_params), skf)

    # DL branch: reuse the existing pretrained Attention-MLP as-is (see module docstring for why).
    def mlp_layers(in_dim, sizes, drop):
        seq, prev = [], in_dim
        for h in sizes:
            seq += [nn.Linear(prev, h), nn.InstanceNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        return nn.Sequential(*seq), prev

    class AttentionFusion(nn.Module):
        def __init__(self, dims):
            super().__init__()
            total = sum(dims)
            hidden = max(64, total // 2)
            self.net = nn.Sequential(nn.Linear(total, hidden), nn.ReLU(), nn.Linear(hidden, len(dims)))

        def forward(self, embs):
            concat = torch.cat(embs, dim=1)
            w = torch.softmax(self.net(concat), dim=1)
            return torch.cat([embs[i] * w[:, i].unsqueeze(1) for i in range(len(embs))], dim=1), w

    class AttentionMLP(nn.Module):
        def __init__(self, in_c, in_r, n_cls, drop):
            super().__init__()
            self.clin, co = mlp_layers(in_c, (64, 32), drop)
            self.rad, ro = mlp_layers(in_r, (256, 128, 64), drop)
            self.fusion = AttentionFusion([co, ro])
            self.cls = nn.Sequential(nn.Linear(co + ro, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, n_cls))

        def forward(self, xc, xr):
            fused, w = self.fusion([self.clin(xc), self.rad(xr)])
            return self.cls(fused), w

    Xc = df[CLINICAL_COLS].to_numpy(dtype=float)
    Xr = df[[c for c in df.columns if c.startswith("original_")]].to_numpy(dtype=float)
    sc_c, sc_r = StandardScaler(), StandardScaler()
    Xc_s, Xr_s = sc_c.fit_transform(Xc), sc_r.fit_transform(Xr)

    dl_model = AttentionMLP(Xc_s.shape[1], Xr_s.shape[1], len(le.classes_), 0.3)
    dl_model.load_state_dict(torch.load("attention_best.pt", map_location="cpu"))
    dl_model.eval()
    with torch.no_grad():
        logits, _ = dl_model(torch.tensor(Xc_s).float(), torch.tensor(Xr_s).float())
        probs_dl = torch.softmax(logits, 1).numpy()

    X_meta = np.hstack([oof_xgb, probs_dl])
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        X_meta, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
    )
    X_meta_bal, y_meta_bal = SMOTE(random_state=RANDOM_STATE, k_neighbors=2).fit_resample(X_meta_train, y_meta_train)

    meta = LogisticRegression(max_iter=300, solver="lbfgs")
    cal_meta = CalibratedClassifierCV(meta, cv=3, method="sigmoid")
    cal_meta.fit(X_meta_bal, y_meta_bal)

    probs_ens = cal_meta.predict_proba(X_meta_test)
    preds_ens = probs_ens.argmax(1)
    print("\n=== Stacked ensemble (OOF meta-features, SMOTE-balanced) hold-out performance ===")
    print(classification_report(y_meta_test, preds_ens, target_names=le.classes_, zero_division=0))
    f1 = f1_score(y_meta_test, preds_ens, average="macro")
    bacc = balanced_accuracy_score(y_meta_test, preds_ens)
    auc_v = roc_auc_score(label_binarize(y_meta_test, classes=all_classes), probs_ens, average="macro", multi_class="ovr")
    print(f"Macro F1={f1:.4f} | BalancedAcc={bacc:.4f} | ROC-AUC={auc_v:.4f}")

    print("\n=== 5-fold cross-validated ensemble summary (on the balanced meta-features) ===")
    cv_scores = cross_validate(
        cal_meta, X_meta_bal, y_meta_bal, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro",
                 "balanced_accuracy": "balanced_accuracy", "roc_auc_ovr_macro": "roc_auc_ovr"},
    )
    for k in ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc_ovr_macro"]:
        print(f"{k}: {cv_scores['test_' + k].mean():.3f} ± {cv_scores['test_' + k].std():.3f}")

    joblib.dump({
        "xgb_model": final_xgb,
        "dl_model_state_dict": dl_model.state_dict(),
        "meta_learner": cal_meta,
        "scalers": {"clinical": sc_c, "radiomic": sc_r},
        "label_encoder": le,
    }, OUT_ENSEMBLE)
    print(f"\nSaved: {OUT_ENSEMBLE}")


if __name__ == "__main__":
    main()
