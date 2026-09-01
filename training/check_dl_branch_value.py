"""Checks whether the Attention-MLP (DL) branch still contributes anything to the
retrained ensemble's predictions on the corrected radiomics features, via
permutation importance on the meta-learner's 8 input features (4 XGB probs + 4
DL probs). Answers Tier 2 #8's open question without retraining anything."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

CSV_PATH = "combined_radiomics_features_FIXED.csv"
CLINICAL_COLS = ["Age", "LVEF", "Troponin", "NTProBNP"]
RANDOM_STATE = 42

ens = joblib.load("training/stacked_fusion_smote_FIXED.pkl")
xgb_model = ens["xgb_model"]
meta = ens["meta_learner"]
le = ens["label_encoder"]
sc_c, sc_r = ens["scalers"]["clinical"], ens["scalers"]["radiomic"]

xgb_assets = joblib.load("training/best_prognostic_model_FIXED.pkl")
xgb_scaler = xgb_assets["scaler"]

df = pd.read_csv(CSV_PATH)
drop_cols = [c for c in ["PatientID", "Risk_Category", "Reasoning"] if c in df.columns]
X_full = df.drop(columns=drop_cols + ["Risk_Score"], errors="ignore")
y_full = le.transform(df["Risk_Score"].astype(str))


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


Xc = sc_c.transform(df[CLINICAL_COLS].to_numpy(dtype=float))
Xr = sc_r.transform(df[[c for c in df.columns if c.startswith("original_")]].to_numpy(dtype=float))
dl_model = AttentionMLP(Xc.shape[1], Xr.shape[1], len(le.classes_), 0.3)
dl_model.load_state_dict(ens["dl_model_state_dict"])
dl_model.eval()
with torch.no_grad():
    logits, _ = dl_model(torch.tensor(Xc).float(), torch.tensor(Xr).float())
    probs_dl = torch.softmax(logits, 1).numpy()

probs_xgb = xgb_model.predict_proba(xgb_scaler.transform(X_full))
X_meta = np.hstack([probs_xgb, probs_dl])

_, X_meta_test, _, y_meta_test = train_test_split(
    X_meta, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
)

feature_names = [f"xgb_p({c})" for c in le.classes_] + [f"dl_p({c})" for c in le.classes_]
result = permutation_importance(meta, X_meta_test, y_meta_test, n_repeats=30, random_state=RANDOM_STATE, scoring="accuracy")

print("Permutation importance (accuracy drop when shuffled), corrected features:")
for name, mean, std in sorted(zip(feature_names, result.importances_mean, result.importances_std), key=lambda t: -t[1]):
    print(f"  {name}: {mean:+.4f} ± {std:.4f}")
