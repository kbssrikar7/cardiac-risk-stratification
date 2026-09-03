"""
Model-loading and inference logic extracted from the original Streamlit app.py.

This module intentionally reuses the exact algorithms from app.py (rule-based
reasoning, clinical LogisticRegression model, calibrated XGBoost, U-Net
Grad-CAM) so the API surface behaves identically to the old UI. The original
XGBoost + Attention-MLP stacked ensemble was retired in favor of a single
calibrated XGBoost model - see predict_with_calibrated_xgb for why. Heavy,
endpoint-specific dependencies (tensorflow, nibabel, SimpleITK, shap) are
imported lazily inside the functions that need them, so importing this module
for clinical-only prediction does not require the imaging/explainability stack
to be installed.
"""
import os

import numpy as np
import pandas as pd
import joblib

# Imported eagerly (unlike tensorflow/nibabel/SimpleITK/shap below, which stay
# lazy since they're genuinely optional depending on which endpoint is hit).
# xgboost is needed by the primary /predict and /shap paths, and both
# unpickling a stored XGBClassifier (via joblib.load) and importing xgboost
# directly trigger the same circular submodule imports (xgboost.callback,
# xgboost.sklearn, xgboost.training) - safe from one thread at module load
# time, but two request threads racing to trigger it for the first time
# concurrently (as /predict and /shap do, fired in parallel by the frontend)
# can hit Python's import system mid-initialization and raise ImportError/
# KeyError, silently downgrading /predict to the clinical-only fallback with
# no logged error. Found via a real browser E2E test against a freshly
# started backend.
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Constants and file paths
# -----------------------------
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

_joblib_cache: dict = {}
_csv_cache: dict = {}


def _load_joblib_cached(path: str):
    """joblib.load() is a multi-MB deserialization; cache by path so repeated
    /predict and /shap calls don't re-read+unpickle the same file from disk
    on every request."""
    if path not in _joblib_cache:
        _joblib_cache[path] = joblib.load(path)
    return _joblib_cache[path]


def _load_csv_cached(path: str):
    if path not in _csv_cache:
        _csv_cache[path] = pd.read_csv(path)
    return _csv_cache[path]


CLINICAL_COLS = ["Age", "LVEF", "Troponin", "NTProBNP"]
CSV_PATH = os.path.join(MODEL_DIR, "combined_radiomics_features.csv")
CLINICAL_MODEL_PATH = os.path.join(MODEL_DIR, "clinical_model.pkl")
UNET_MODEL_PATH = os.path.join(MODEL_DIR, "unet_multiclass.h5")
BEST_XGB_PATH = os.path.join(MODEL_DIR, "best_prognostic_model.pkl")


# -----------------------------
# Utility: Rule-based reasoning
# -----------------------------
def classify_patient_risk_rule_based(age: float, lvef: float, troponin: float, ntprobnp: float):
    troponin_ng_L = troponin * 1000 if troponin < 10 else troponin
    if troponin_ng_L > 50:
        return 'Very High Risk (Acute Cardiac Event)', f"Acutely elevated Troponin ({troponin}) indicates myocardial injury."
    if lvef is not None and lvef <= 40:
        return 'High Risk (Chronic Heart Failure)', f"LVEF is {lvef} (<= 40%), indicating severely reduced function."
    is_high_ntprobnp = False
    if age > 75 and ntprobnp > 1800: is_high_ntprobnp = True
    elif age >= 50 and ntprobnp > 900: is_high_ntprobnp = True
    elif age < 50 and ntprobnp > 450: is_high_ntprobnp = True
    if is_high_ntprobnp:
        return 'High Risk (Chronic Heart Failure)', f"NT-proBNP of {ntprobnp} is very high for age {age}, indicating severe heart stress."
    if lvef is not None and 41 <= lvef <= 54:
        return 'Moderate Risk', f"LVEF is {lvef} (41-54%), indicating mildly reduced function."
    is_moderate_ntprobnp = False
    if age > 75 and 125 < ntprobnp <= 1800: is_moderate_ntprobnp = True
    elif age >= 50 and 125 < ntprobnp <= 900: is_moderate_ntprobnp = True
    elif age < 50 and 125 < ntprobnp <= 450: is_moderate_ntprobnp = True
    if is_moderate_ntprobnp:
        return 'Moderate Risk', f"NT-proBNP of {ntprobnp} is moderately elevated for age {age}."
    return 'Low Risk', f"LVEF ({lvef}), Troponin ({troponin}), and NT-proBNP ({ntprobnp}) are within normal limits for age {age}."


# -----------------------------
# Clinical-only model (train/load)
# -----------------------------
def load_or_train_clinical_model(csv_path: str = CSV_PATH, model_path: str = CLINICAL_MODEL_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if 'Risk_Score' in df.columns and 'Risk_Category' in df.columns:
        # Risk_Score is the clean 0..N ordinal target; Risk_Category holds the
        # human-readable name for each score. Fitting a LabelEncoder on
        # Risk_Score.astype(str) instead (as this used to) makes classes_ the
        # sorted *Risk_Score digit strings* ('0','1',...), not the category
        # names - `classes[pred_idx]` then returns e.g. '2' instead of 'High
        # Risk (Chronic Heart Failure)'. Found via E2E testing on held-out
        # patients: predict_clinical_model returned bare digit strings.
        y = df['Risk_Score'].astype(int).to_numpy()
        name_by_score = df[['Risk_Score', 'Risk_Category']].drop_duplicates().set_index('Risk_Score')['Risk_Category']
        classes = [name_by_score[i] for i in sorted(name_by_score.index)]
        le = None
    elif 'Risk_Category' in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df['Risk_Category'])
        classes = list(le.classes_)
    elif 'Risk_Score' in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df['Risk_Score'].astype(str))
        classes = list(le.classes_)
    else:
        raise ValueError("CSV must contain 'Risk_Score' or 'Risk_Category'")

    X = df[CLINICAL_COLS].copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, multi_class='auto')
    model.fit(X_scaled, y)

    assets = {
        "model": model,
        "scaler": scaler,
        "classes": classes,
        "label_encoder": le
    }
    joblib.dump(assets, model_path)
    return assets


def predict_clinical_model(assets, age: float, lvef: float, troponin: float, ntprobnp: float):
    X = np.array([[age, lvef, troponin, ntprobnp]], dtype=float)
    Xs = assets["scaler"].transform(X)
    probs = assets["model"].predict_proba(Xs)[0]
    idx = int(np.argmax(probs))
    return assets["classes"][idx], probs, idx


def shap_explain_instance(assets, age: float, lvef: float, troponin: float, ntprobnp: float):
    import shap

    X = np.array([[age, lvef, troponin, ntprobnp]], dtype=float)
    Xs = assets["scaler"].transform(X)
    model = assets["model"]

    bg_orig = np.array([
        [60, 55, 10, 300],
        [70, 45, 20, 900],
        [50, 60, 5, 150],
        [65, 50, 15, 600]
    ], dtype=float)
    bg_scaled = assets["scaler"].transform(bg_orig)

    explainer = shap.Explainer(model.predict_proba, bg_scaled)
    exp = explainer(Xs)

    probs = model.predict_proba(Xs)[0]
    pred_idx = int(np.argmax(probs))
    vals = np.array(exp.values)
    if vals.ndim == 3 and vals.shape[0] >= 1:
        sv = vals[0, :, pred_idx]
    elif vals.ndim == 2:
        sv = vals[0]
    else:
        sv = vals.squeeze()

    feature_names = CLINICAL_COLS
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "value": [age, lvef, troponin, ntprobnp]
    }).sort_values("shap_value", key=np.abs, ascending=False)

    return sv, feature_names, contributions


def try_shap_from_best_model(clinical_assets, age: float, lvef: float, troponin: float, ntprobnp: float):
    """SHAP explanation using best_prognostic_model.pkl, matching app.py's primary SHAP path."""
    if not os.path.exists(BEST_XGB_PATH) or not os.path.exists(CSV_PATH):
        raise FileNotFoundError("best_prognostic_model.pkl or CSV not found")
    best_assets = _load_joblib_cached(BEST_XGB_PATH)
    model_best = best_assets.get("model", None)
    scaler_best = best_assets.get("scaler", None)
    label_map = best_assets.get("label_mapping", None)
    if model_best is None or label_map is None:
        raise ValueError("best_prognostic_model.pkl missing required keys")
    classes_best = [label_map[k] if isinstance(label_map.get(k, k), str) else k for k in sorted(label_map.keys())] if isinstance(label_map, dict) else clinical_assets["classes"]

    df_all = _load_csv_cached(CSV_PATH)
    X_row = df_all.iloc[:1].copy()
    X_row.iloc[0, :] = np.nan
    for col, val in zip(CLINICAL_COLS, [age, lvef, troponin, ntprobnp]):
        if col in X_row.columns:
            X_row.loc[X_row.index[0], col] = float(val)

    bg = df_all.drop(columns=[c for c in ["PatientID", "Risk_Score", "Risk_Category", "Reasoning"] if c in df_all.columns], errors='ignore')
    X_input = X_row.drop(columns=[c for c in ["PatientID", "Risk_Score", "Risk_Category", "Reasoning"] if c in X_row.columns], errors='ignore')

    # Correlation-pruned models record which columns they were actually trained on
    # (107 radiomics features were pruned to 59); restrict to exactly those, in order,
    # so the column count matches what the model/scaler expect.
    feature_columns = best_assets.get("feature_columns")
    if feature_columns:
        bg = bg[feature_columns]
        X_input = X_input[feature_columns]

    bg = bg.sample(min(50, len(bg)), random_state=42)

    # model_best is an XGBClassifier (tree ensemble). Get exact SHAP values straight from
    # XGBoost's own pred_contribs, rather than going through the `shap` package: this model's
    # base_score is serialized in a multiclass JSON-array format that shap<0.50's XGBoost
    # loader cannot parse ("could not convert string to float"), while shap>=0.50 fixes that
    # but hard-requires numpy>=2, which conflicts with tensorflow-cpu==2.15.0's numpy<2.0 pin
    # elsewhere in this same service. XGBoost's native path sidesteps both problems, needs no
    # background sample, and is exact rather than an approximation - and, incidentally, is
    # what closed the actual bug this was written to fix: the old generic
    # shap.Explainer(model.predict_proba, background) permutation approximation needed
    # O(features) black-box model calls per explained instance, which over this model's 111
    # features took 90+ seconds on constrained CPU (e.g. Render's free tier), timing out the
    # request entirely.
    if scaler_best is not None:
        bg_filled = bg.fillna(bg.mean(numeric_only=True))
        x_filled = X_input.fillna(bg.mean(numeric_only=True))
        X_scaled = scaler_best.transform(x_filled.values)
    else:
        X_scaled = X_input.values

    # The production model is a CalibratedClassifierCV wrapping the actual XGBClassifier
    # (for well-calibrated probabilities); pred_contribs needs the raw booster underneath.
    xgb_estimator = model_best
    if hasattr(model_best, "calibrated_classifiers_"):
        xgb_estimator = model_best.calibrated_classifiers_[0].estimator

    feat_names = bg.columns.tolist()
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feat_names)
    contribs = xgb_estimator.get_booster().predict(dmatrix, pred_contribs=True)

    probs_best = model_best.predict_proba((X_scaled if scaler_best is not None else X_input.values))[0]
    pred_idx_best = int(np.nanargmax(probs_best))
    cls_names = classes_best if classes_best else clinical_assets["classes"]

    # contribs shape is (n_samples, n_classes, n_features + 1); the last column per class is
    # the bias/base-value term, not a feature contribution, so it's dropped here.
    vals = np.array(contribs)
    if vals.ndim == 3 and vals.shape[0] >= 1:
        sv = vals[0, pred_idx_best, :-1]
    else:
        sv = vals[0, :-1]

    disp_idx = [feat_names.index(c) for c in CLINICAL_COLS if c in feat_names]
    if len(disp_idx) > 0:
        feat_names_disp = [feat_names[i] for i in disp_idx]
        sv_disp = sv[disp_idx]
        vals_disp = [age, lvef, troponin, ntprobnp][:len(disp_idx)]
    else:
        order = np.argsort(np.abs(sv))[::-1][:10]
        feat_names_disp = [feat_names[i] for i in order]
        sv_disp = sv[order]
        vals_disp = [np.nan] * len(order)

    contrib_df = pd.DataFrame({
        "feature": feat_names_disp,
        "shap_value": sv_disp,
        "value": vals_disp
    }).sort_values("shap_value", key=np.abs, ascending=False)

    return sv_disp, feat_names_disp, contrib_df, probs_best, cls_names, pred_idx_best


# -----------------------------
# Primary prediction: calibrated XGBoost on the corrected, correlation-pruned
# radiomics features
# -----------------------------
def predict_with_calibrated_xgb(age: float, lvef: float, troponin: float, ntprobnp: float):
    """Replaces the former XGB+AttentionMLP stacked ensemble. Permutation
    importance showed the Attention-MLP branch contributed exactly 0 signal to
    the ensemble's predictions (on both the original and the corrected
    radiomics features), so the ensemble/meta-learner/DL branch were retired
    in favor of a single calibrated XGBoost model - same accuracy, far less
    complexity and fewer ways to silently break."""
    if not os.path.exists(BEST_XGB_PATH) or not os.path.exists(CSV_PATH):
        raise FileNotFoundError("best_prognostic_model.pkl or CSV not found")

    best_assets = _load_joblib_cached(BEST_XGB_PATH)
    model = best_assets.get("model")
    scaler = best_assets.get("scaler")
    label_mapping = best_assets.get("label_mapping")
    feature_columns = best_assets.get("feature_columns")
    if model is None or scaler is None or label_mapping is None:
        raise ValueError("best_prognostic_model.pkl missing required keys")

    df_all = _load_csv_cached(CSV_PATH)
    drop_cols = [c for c in ["PatientID", "Risk_Score", "Risk_Category", "Reasoning"] if c in df_all.columns]
    features_df = df_all.drop(columns=drop_cols, errors='ignore')
    if feature_columns:
        features_df = features_df[feature_columns]

    row = features_df.median(numeric_only=True)
    for col, val in zip(CLINICAL_COLS, [age, lvef, troponin, ntprobnp]):
        if col in features_df.columns:
            row[col] = float(val)
    X = row.to_frame().T

    X_scaled = scaler.transform(X.values)
    probs = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(probs))
    classes = [label_mapping[k] for k in sorted(label_mapping.keys())]
    return classes[pred_idx], probs, pred_idx, classes


# -----------------------------
# MRI + U-Net Grad-CAM utilities
# -----------------------------
def load_nii_volume(path: str):
    import SimpleITK as sitk
    itk_img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(itk_img)  # Z, Y, X
    vol = np.transpose(vol, (2, 1, 0))     # X, Y, Z
    return vol


def preprocess_volume_for_unet(vol: np.ndarray, target_size=(128, 128)):
    import tensorflow as tf
    slices = []
    vol = vol.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    for i in range(vol.shape[2]):
        sl = vol[:, :, i]
        sl = tf.image.resize(sl[..., None], target_size, method="bilinear").numpy().squeeze()
        slices.append(sl)
    X = np.array(slices)[..., None]
    return X


# Center-crop fractions searched by localize_and_preprocess_volume, widest first.
LOCALIZATION_CROP_FRACTIONS = (1.0, 0.8, 0.65, 0.5, 0.35)


def _center_crop_box(height: int, width: int, frac: float):
    if frac >= 1.0:
        return (0, height, 0, width)
    ch, cw = max(1, int(round(height * frac))), max(1, int(round(width * frac)))
    y0 = (height - ch) // 2
    x0 = (width - cw) // 2
    return (y0, y0 + ch, x0, x0 + cw)


def localize_and_preprocess_volume(unet_model, vol: np.ndarray, target_size=(128, 128),
                                    crop_fractions=LOCALIZATION_CROP_FRACTIONS):
    """NOT used by generate_gradcam_overlay - kept only as a documented,
    tested, and REJECTED attempt (see TECHNICAL_REPORT.md Section 11). Do not
    wire this into the live pipeline without re-validating against ground
    truth first.

    Attempted test-time fix for a real generalization failure: the model,
    trained only on EMIDEC (whose acquisition protocol crops tightly on the
    heart), predicted 100% background at 1.0 confidence on every pixel of a
    real external LGE-MRI scan whose heart was a smaller, off-center feature
    in a wider field. The idea: search a small set of center-crop fractions
    per slice and keep whichever crop's resize gives the highest total
    cavity+myocardium probability mass, instead of always the full-frame
    resize.

    This looked like it worked on the failing external case (0 detected
    pixels -> a real, correctly-located partial detection) and even
    increased predicted pixel *counts* on several EMIDEC cases - but pixel
    count is not accuracy. Validated against real ground-truth Dice on 15
    EMIDEC training patients and found to regress it badly: mean myocardium
    Dice 0.833 -> 0.528, cavity Dice 0.884 -> 0.781. The crop search picks
    tighter crops that make the network predict larger, less accurate blobs,
    not more correct ones - confirmed via real Dice, not assumed from pixel
    counts. The framing-mismatch limitation this was meant to fix remains
    open; a model trained on more acquisition protocols and explicit crop
    augmentation is the durable fix, not a test-time crop search.

    Returns (X, crop_boxes): X has the same shape as preprocess_volume_for_unet's
    output; crop_boxes[i] = (y0, y1, x0, x1) is the pixel box (in the original
    slice's coordinates) chosen for slice i, needed to crop the display image
    to match what was actually fed to the model.
    """
    import tensorflow as tf

    vol = vol.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    n_slices = vol.shape[2]
    n_crops = len(crop_fractions)
    height, width = vol.shape[0], vol.shape[1]

    boxes_by_slice = [_center_crop_box(height, width, f) for f in crop_fractions]

    batch = np.zeros((n_slices * n_crops,) + target_size + (1,), dtype=np.float32)
    idx = 0
    for i in range(n_slices):
        sl = vol[:, :, i]
        for (y0, y1, x0, x1) in boxes_by_slice:
            cropped = sl[y0:y1, x0:x1]
            resized = tf.image.resize(cropped[..., None], target_size, method="bilinear").numpy()
            batch[idx] = resized
            idx += 1

    preds = unet_model.predict(batch, verbose=0)  # (n_slices*n_crops, h, w, C)
    foreground_mass = preds[..., 1:].sum(axis=(1, 2, 3))

    X = np.zeros((n_slices,) + target_size + (1,), dtype=np.float32)
    chosen_boxes = []
    for i in range(n_slices):
        scores = foreground_mass[i * n_crops:(i + 1) * n_crops]
        best_j = int(np.argmax(scores))
        X[i] = batch[i * n_crops + best_j]
        chosen_boxes.append(boxes_by_slice[best_j])

    return X, chosen_boxes


def generate_gradcam_overlay(unet_model, vol: np.ndarray, target_class_idx: int = 2):
    import tensorflow as tf
    from PIL import Image
    import matplotlib.pyplot as plt

    X = preprocess_volume_for_unet(vol)
    preds = unet_model.predict(X, verbose=0)  # (n, h, w, C)
    masks = np.argmax(preds, axis=-1)

    slice_scores = np.sum(masks == target_class_idx, axis=(1, 2))
    best_idx = int(np.argmax(slice_scores))
    proc_slice = X[best_idx:best_idx + 1]  # (1, h, w, 1)
    orig_slice = vol[:, :, best_idx]

    target_layer_name = None
    preferred_names = [
        'c6_conv2_gradcam_target',
        'conv2d_5', 'conv2d_6'
    ]
    for nm in preferred_names:
        if nm in [l.name for l in unet_model.layers]:
            target_layer_name = nm
            break
    if target_layer_name is None:
        for layer in reversed(unet_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer_name = layer.name
                break
    if target_layer_name is None:
        raise RuntimeError("No Conv2D layer found for Grad-CAM")

    grad_model = tf.keras.models.Model(
        [unet_model.inputs],
        [unet_model.get_layer(target_layer_name).output, unet_model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(proc_slice)
        loss = tf.reduce_mean(predictions[:, :, :, target_class_idx])
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    fmap = conv_out[0].numpy()  # (h, w, channels)
    heatmap = np.tensordot(fmap, pooled_grads, axes=([2], [0]))
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    heatmap_resized = tf.image.resize(heatmap[..., None], orig_slice.shape, method="bilinear").numpy().squeeze()

    cmap = plt.get_cmap('jet')
    heat_rgba = (cmap(heatmap_resized) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert('RGBA')

    base = (255 * (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)).astype(np.uint8)
    base_img = Image.fromarray(base).convert('L').convert('RGBA')

    alpha = 0.35
    blended = Image.blend(base_img, heat_img, alpha)
    return blended, best_idx


def build_unet_multiclass(input_shape=(128, 128, 1), num_classes=3):
    from tensorflow.keras import models, layers

    inputs = layers.Input(input_shape, name='input_layer_2')
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c1_conv1')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c1_conv2')(c1)
    p1 = layers.MaxPooling2D((2, 2), name='p1')(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2_conv1')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2_conv2')(c2)
    p2 = layers.MaxPooling2D((2, 2), name='p2')(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c3_conv1')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c3_conv2')(c3)
    p3 = layers.MaxPooling2D((2, 2), name='p3')(c3)

    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='bn_conv1')(p3)
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='bn_conv2')(bn)

    u1 = layers.UpSampling2D((2, 2), name='u1')(bn)
    u1 = layers.concatenate([u1, c3], name='u1_concat')
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c4_conv1')(u1)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c4_conv2')(c4)

    u2 = layers.UpSampling2D((2, 2), name='u2')(c4)
    u2 = layers.concatenate([u2, c2], name='u2_concat')
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c5_conv1')(u2)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c5_conv2')(c5)

    u3 = layers.UpSampling2D((2, 2), name='u3')(c5)
    u3 = layers.concatenate([u3, c1], name='u3_concat')
    c6 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c6_conv1')(u3)
    c6 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c6_conv2_gradcam_target')(c6)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='final_output_layer')(c6)
    return models.Model(inputs, outputs)


def load_unet_model(model_path: str = UNET_MODEL_PATH):
    from tensorflow.keras import models

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"U-Net model '{model_path}' not found.")
    try:
        return models.load_model(model_path, compile=False)
    except Exception:
        unet_model = build_unet_multiclass(input_shape=(128, 128, 1), num_classes=3)
        unet_model.load_weights(model_path)
        return unet_model
