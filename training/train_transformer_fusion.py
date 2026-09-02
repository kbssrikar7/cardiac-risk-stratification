"""Publication/patent push, transformer-fusion ablation, step 2: a hybrid
CNN-transformer fusion model, evaluated under the exact same repeated
stratified 5x5 CV protocol as the calibrated XGBoost baseline
(training/pipeline.py), so the comparison in TECHNICAL_REPORT.md is
apples-to-apples.

Design follows the small-medical-imaging-dataset literature pattern found
while researching this ablation (TransMed-style: frozen/pretrained CNN
feature extraction, transformer only for fusing modalities on top of that -
not a pure vision transformer trained from scratch, which the same
literature says needs far more data than this project's 150 patients):

- "Imaging" modality: the 128-dim frozen bottleneck embedding from the
  already-trained 5-class U-Net (compute_cnn_embeddings.py), NOT fine-tuned
  here - keeps each of the 25 CV folds cheap, and matches the transfer-
  learning recommendation for small datasets.
- "Tabular" modality: the same clinical + correlation-pruned radiomics (+
  optionally infarct-burden) features used everywhere else in this project.
- Fusion: each modality projected to the same width, treated as a 2-token
  sequence, one transformer encoder block (multi-head self-attention + FFN),
  mean-pooled, softmax classification head.

Expected result, stated up front rather than discovered after the fact: at
n=150, this is very likely to match or underperform the calibrated XGBoost
baseline - the project's own Attention-MLP ablation already measured zero
added signal from a much simpler deep branch at this same sample size. A
negative/marginal result here is still a legitimate, citable finding.
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

from common import RANDOM_STATE, greedy_correlation_prune, repeated_cv_eval, summarize

TRAINING_DIR = Path(__file__).parent
REPO_ROOT = TRAINING_DIR.parent
BASE_CSV = REPO_ROOT / "combined_radiomics_features_FIXED.csv"
INFARCT_TRAIN_CSV = TRAINING_DIR / "infarct_features_train.csv"
INFARCT_TEST_CSV = TRAINING_DIR / "infarct_features_test.csv"
CNN_EMBEDDINGS_CSV = TRAINING_DIR / "cnn_embeddings.csv"
OUT_REPORT = TRAINING_DIR / "transformer_fusion_ablation.json"  # not under runs/ - not a pipeline.py run record

tf.random.set_seed(RANDOM_STATE)


def build_fusion_model(n_tabular: int, n_cnn: int, n_classes: int, d_model: int = 32, num_heads: int = 4):
    inputs = layers.Input(shape=(n_tabular + n_cnn,))
    tabular = layers.Lambda(lambda x: x[:, :n_tabular])(inputs)
    imaging = layers.Lambda(lambda x: x[:, n_tabular:])(inputs)

    tab_token = layers.Dense(d_model, activation="relu")(tabular)
    img_token = layers.Dense(d_model, activation="relu")(imaging)
    tokens = layers.Lambda(lambda t: tf.stack(t, axis=1))([tab_token, img_token])  # (batch, 2, d_model)

    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(tokens, tokens)
    x = layers.LayerNormalization()(tokens + attn_out)
    ffn = layers.Dense(d_model * 2, activation="relu")(x)
    ffn = layers.Dense(d_model)(ffn)
    x = layers.LayerNormalization()(x + ffn)

    pooled = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation="softmax")(pooled)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


class TransformerFusionClassifier:
    """Thin sklearn-style wrapper (.fit/.predict_proba) so this plugs directly
    into common.repeated_cv_eval, exactly like the XGBoost/LightGBM models."""

    def __init__(self, n_tabular: int, n_cnn: int, n_classes: int, epochs: int = 80, batch_size: int = 16):
        self.n_tabular = n_tabular
        self.n_cnn = n_cnn
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = build_fusion_model(self.n_tabular, self.n_cnn, self.n_classes)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)


def load_dataset(use_infarct_features: bool, corr_threshold: float = 0.95):
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

    cnn_df = pd.read_csv(CNN_EMBEDDINGS_CSV, dtype={"PatientID": str})
    cnn_df["PatientID"] = cnn_df["PatientID"].str.zfill(3)
    df = df.merge(cnn_df, on="PatientID", how="inner")

    cnn_cols = [c for c in df.columns if c.startswith("cnn_emb_")]
    tabular_full = df.drop(columns=drop_cols + ["Risk_Score"] + cnn_cols, errors="ignore")
    radiomic_cols = [c for c in tabular_full.columns if c.startswith("original_")]
    non_radiomic_cols = [c for c in tabular_full.columns if c not in radiomic_cols]
    kept_radiomic, _ = greedy_correlation_prune(tabular_full[radiomic_cols], corr_threshold)
    tabular_cols = non_radiomic_cols + kept_radiomic

    le = LabelEncoder()
    y = le.fit_transform(df["Risk_Score"].astype(str))
    X = df[tabular_cols + cnn_cols]
    print(f"Dataset: {len(df)} patients, {len(tabular_cols)} tabular + {len(cnn_cols)} CNN-embedding features")
    return X, y, len(tabular_cols), len(cnn_cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-infarct-features", action="store_true", default=True)
    parser.add_argument("--no-infarct-features", dest="use_infarct_features", action="store_false")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    X, y, n_tabular, n_cnn = load_dataset(args.use_infarct_features)
    n_classes = len(np.unique(y))

    def build_and_fit(X_tr, y_tr):
        clf = TransformerFusionClassifier(n_tabular, n_cnn, n_classes, epochs=args.epochs)
        return clf.fit(X_tr, y_tr)

    label = f"Hybrid CNN-transformer fusion ({'with' if args.use_infarct_features else 'without'} infarct features)"
    result = repeated_cv_eval(build_and_fit, X, y, label=label)
    metrics = summarize(result)

    report = {
        "model": "hybrid_cnn_transformer_fusion",
        "use_infarct_features": args.use_infarct_features,
        "n_tabular_features": n_tabular,
        "n_cnn_features": n_cnn,
        "cv_metrics": metrics,
    }
    OUT_REPORT.parent.mkdir(exist_ok=True)
    existing = json.loads(OUT_REPORT.read_text()) if OUT_REPORT.exists() else []
    existing.append(report)
    OUT_REPORT.write_text(json.dumps(existing, indent=2))
    print(f"\nAppended result to {OUT_REPORT}")


if __name__ == "__main__":
    main()
