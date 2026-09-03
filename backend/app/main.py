import base64
import io
import os
import tempfile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import ml

app = FastAPI(title="Cardiac Risk Stratification API")

_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

_clinical_assets = None
_unet_model = None


def get_clinical_assets():
    global _clinical_assets
    if _clinical_assets is None:
        _clinical_assets = ml.load_or_train_clinical_model()
    return _clinical_assets


def get_unet_model():
    global _unet_model
    if _unet_model is None:
        _unet_model = ml.load_unet_model()
    return _unet_model


class ClinicalInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    lvef: float = Field(..., ge=0, le=80)
    troponin: float = Field(..., ge=0, le=50000)
    ntprobnp: float = Field(..., ge=0, le=100000)


class PredictResponse(BaseModel):
    risk_class: str
    probabilities: dict[str, float]
    model_used: str
    rule_based_risk: str
    rule_based_reasoning: str


class ShapContribution(BaseModel):
    feature: str
    shap_value: float
    value: float | None = None


class ShapResponse(BaseModel):
    risk_class: str
    model_used: str
    contributions: list[ShapContribution]


class GradcamResponse(BaseModel):
    overlay_png_base64: str
    slice_index: int
    num_slices: int


MAX_GRADCAM_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ClinicalInput):
    rule_risk, rule_reason = ml.classify_patient_risk_rule_based(
        payload.age, payload.lvef, payload.troponin, payload.ntprobnp
    )

    try:
        label, probs, pred_idx, classes = ml.predict_with_calibrated_xgb(
            payload.age, payload.lvef, payload.troponin, payload.ntprobnp
        )
        model_used = "calibrated_xgboost"
    except Exception:
        try:
            clinical_assets = get_clinical_assets()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No prediction model available: {e}")
        label, probs, pred_idx = ml.predict_clinical_model(
            clinical_assets, payload.age, payload.lvef, payload.troponin, payload.ntprobnp
        )
        classes = clinical_assets["classes"]
        model_used = "clinical_only"

    return PredictResponse(
        risk_class=label,
        probabilities={cls: float(probs[i]) for i, cls in enumerate(classes)},
        model_used=model_used,
        rule_based_risk=rule_risk,
        rule_based_reasoning=rule_reason,
    )


@app.get("/shap", response_model=ShapResponse)
def shap_explain(
    age: float = Query(..., ge=0, le=120),
    lvef: float = Query(..., ge=0, le=80),
    troponin: float = Query(..., ge=0, le=50000),
    ntprobnp: float = Query(..., ge=0, le=100000),
):
    try:
        clinical_assets = get_clinical_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No clinical model available: {e}")

    try:
        _, _, contrib_df, probs_best, cls_best, pred_idx_best = ml.try_shap_from_best_model(
            clinical_assets, age, lvef, troponin, ntprobnp
        )
        risk_class = cls_best[pred_idx_best] if cls_best else str(pred_idx_best)
        model_used = "best_xgb"
    except Exception:
        try:
            _, _, contrib_df = ml.shap_explain_instance(clinical_assets, age, lvef, troponin, ntprobnp)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SHAP explanation unavailable: {e}")
        label, _, _ = ml.predict_clinical_model(clinical_assets, age, lvef, troponin, ntprobnp)
        risk_class = label
        model_used = "clinical_only"

    contributions = [
        ShapContribution(feature=row.feature, shap_value=float(row.shap_value), value=(float(row.value) if row.value is not None and not pd.isna(row.value) else None))
        for row in contrib_df.itertuples(index=False)
    ]

    return ShapResponse(risk_class=str(risk_class), model_used=model_used, contributions=contributions)


@app.post("/gradcam", response_model=GradcamResponse)
def gradcam(file: UploadFile = File(...)):
    if not (file.filename or "").endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Expected a .nii or .nii.gz MRI file")

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    chunk_size = 1024 * 1024
    written = 0

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    try:
        with tmp:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_GRADCAM_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"MRI file exceeds the {MAX_GRADCAM_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
                    )
                tmp.write(chunk)

        try:
            unet_model = get_unet_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"U-Net model unavailable: {e}")

        try:
            vol, spacing_xy = ml.load_nii_volume(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read MRI volume: {e}")

        try:
            overlay_img, best_idx = ml.generate_gradcam_overlay(unet_model, vol, spacing_xy, target_class_idx=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    return GradcamResponse(
        overlay_png_base64=encoded,
        slice_index=best_idx,
        num_slices=int(vol.shape[2]),
    )
