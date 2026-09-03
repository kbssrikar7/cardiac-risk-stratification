"""
Automated tests for the FastAPI backend, pinning that the extraction from
app.py stayed behaviorally faithful and that the API contract doesn't
regress silently.

Everything here runs against the real committed model artifacts (no mocking
of ml.py's inference logic) except where noted, since the whole point of
this suite is to catch a divergence from app.py's original behavior.
"""
import base64
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import ml
from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# -----------------------------
# Rule-based reasoning (pure function, no model load)
# -----------------------------
def test_rule_based_acute_troponin_ng_l():
    risk, reason = ml.classify_patient_risk_rule_based(age=60, lvef=55, troponin=100, ntprobnp=100)
    assert risk == "Very High Risk (Acute Cardiac Event)"
    assert "Troponin" in reason


def test_rule_based_acute_troponin_ng_ml_unit_conversion():
    # troponin < 10 is interpreted as ng/mL and converted to ng/L (*1000)
    risk, _ = ml.classify_patient_risk_rule_based(age=60, lvef=55, troponin=5, ntprobnp=100)
    assert risk == "Very High Risk (Acute Cardiac Event)"


def test_rule_based_low_lvef_is_high_risk():
    risk, reason = ml.classify_patient_risk_rule_based(age=60, lvef=35, troponin=0, ntprobnp=100)
    assert risk == "High Risk (Chronic Heart Failure)"
    assert "LVEF" in reason


@pytest.mark.parametrize(
    "age,ntprobnp,expected",
    [
        (80, 1900, "High Risk (Chronic Heart Failure)"),   # age > 75, > 1800
        # NOTE: for age > 75, the original app.py elif-chain falls through to
        # the "age >= 50" branch whenever 900 < ntprobnp <= 1800, so this
        # band is classified as high risk, not moderate - a pre-existing
        # quirk in the extracted logic, pinned here rather than "fixed".
        (80, 1500, "High Risk (Chronic Heart Failure)"),    # age > 75, 900-1800 (falls into age>=50 branch)
        (80, 800, "Moderate Risk"),                         # age > 75, 125-900 (true moderate band)
        (60, 950, "High Risk (Chronic Heart Failure)"),     # age >= 50, > 900
        (60, 500, "Moderate Risk"),                         # age >= 50, 125-900
        (40, 500, "High Risk (Chronic Heart Failure)"),     # age < 50, > 450
        (40, 200, "Moderate Risk"),                         # age < 50, 125-450
    ],
)
def test_rule_based_age_banded_ntprobnp_thresholds(age, ntprobnp, expected):
    risk, _ = ml.classify_patient_risk_rule_based(age=age, lvef=60, troponin=0, ntprobnp=ntprobnp)
    assert risk == expected


def test_rule_based_low_risk_when_all_normal():
    risk, _ = ml.classify_patient_risk_rule_based(age=45, lvef=60, troponin=0, ntprobnp=50)
    assert risk == "Low Risk"


# -----------------------------
# /health
# -----------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# -----------------------------
# /predict
# -----------------------------
def test_predict_happy_path(client):
    resp = client.post(
        "/predict",
        json={"age": 70, "lvef": 30, "troponin": 100, "ntprobnp": 2000},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_used"] in ("calibrated_xgboost", "clinical_only")
    assert body["risk_class"] in body["probabilities"]
    # Guards against label_mapping regressing to bare ordinal codes ('0'..'3')
    # instead of risk-category names - happened once, see training/common.py's
    # risk_score_label_mapping().
    assert not body["risk_class"].isdigit()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["rule_based_risk"]
    assert body["rule_based_reasoning"]


@pytest.mark.parametrize(
    "payload",
    [
        {"age": 200, "lvef": 55, "troponin": 10, "ntprobnp": 100},   # age out of bounds
        {"age": 60, "lvef": 55, "troponin": -1, "ntprobnp": 100},    # negative troponin
        {"age": 60, "lvef": 90, "troponin": 10, "ntprobnp": 100},    # lvef out of bounds
    ],
)
def test_predict_validation_rejects_out_of_bounds(client, payload):
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_and_shap_concurrently_on_a_cold_process_both_use_the_real_model():
    """Regression test for a real bug found via browser E2E testing: the
    frontend fires /predict and /shap in parallel on every submission
    (Promise.allSettled). On a freshly started backend, both handlers race to
    trigger the first `import xgboost` - one via unpickling a stored
    XGBClassifier inside joblib.load, the other via try_shap_from_best_model's
    own import - and xgboost's circular submodule imports
    (callback/sklearn/training) aren't safe against two threads racing to
    import them for the first time simultaneously. This silently downgraded
    /predict to the clinical_only fallback and /shap to a different, less
    accurate model with no logged error. Fixed by importing xgboost eagerly
    at ml.py's module load time instead of lazily inside request handlers.
    Runs in a real fresh subprocess since the bug only reproduces on a truly
    cold import - the test process itself already has xgboost imported.
    """
    script = """
import sys, threading
sys.path.insert(0, "backend")
from app import ml

errors = []

def call_predict():
    try:
        ml.predict_with_calibrated_xgb(74, 35, 6.7, 5627)
    except Exception as e:
        errors.append(("predict", repr(e)))

def call_shap():
    try:
        assets = ml.load_or_train_clinical_model()
        ml.try_shap_from_best_model(assets, 74, 35, 6.7, 5627)
    except Exception as e:
        errors.append(("shap", repr(e)))

t1 = threading.Thread(target=call_predict)
t2 = threading.Thread(target=call_shap)
t1.start(); t2.start()
t1.join(); t2.join()
if errors:
    print("ERRORS:", errors)
    sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


# -----------------------------
# /shap
# -----------------------------
def test_shap_happy_path(client):
    resp = client.get("/shap", params={"age": 70, "lvef": 30, "troponin": 100, "ntprobnp": 2000})
    assert resp.status_code == 200
    body = resp.json()
    assert body["contributions"], "expected at least one SHAP contribution"
    assert not body["risk_class"].isdigit()
    abs_vals = [abs(c["shap_value"]) for c in body["contributions"]]
    assert abs_vals == sorted(abs_vals, reverse=True), "contributions must be sorted by |shap_value| desc"


# -----------------------------
# /gradcam
# -----------------------------
def test_gradcam_rejects_non_mri_filename(client):
    resp = client.post(
        "/gradcam",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 400


def test_gradcam_happy_path_returns_valid_overlay(client):
    # Exercises the full production pipeline (streaming write -> SimpleITK
    # volume load -> U-Net inference -> Grad-CAM overlay -> PNG encoding)
    # against a real synthesized .nii.gz, closing the gap where this
    # endpoint's success path had only ever been checked manually (curl /
    # live smoke tests in prior iterations), never by an automated test.
    sitk = pytest.importorskip("SimpleITK")
    import numpy as np

    rng = np.random.default_rng(42)
    # SimpleITK's array convention is (Z, Y, X); ml.load_nii_volume
    # transposes to (X, Y, Z), so num_slices in the response should equal
    # this array's Z dimension (8).
    z_slices = 8
    vol = rng.integers(0, 255, size=(z_slices, 16, 16), dtype=np.uint8)
    itk_img = sitk.GetImageFromArray(vol)

    fd, nii_path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        sitk.WriteImage(itk_img, nii_path)
        with open(nii_path, "rb") as fh:
            resp = client.post(
                "/gradcam",
                files={"file": ("scan.nii.gz", fh, "application/octet-stream")},
            )
    finally:
        os.remove(nii_path)

    assert resp.status_code == 200
    body = resp.json()
    assert body["num_slices"] == z_slices
    assert 0 <= body["slice_index"] < body["num_slices"]

    png_bytes = base64.b64decode(body["overlay_png_base64"])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "response must decode to a valid PNG"


def test_gradcam_enforces_upload_size_limit(client, monkeypatch):
    # Shrink the cap so the oversized upload is rejected during the streaming
    # write, before the U-Net model is ever loaded (the size check runs
    # first in the handler, so no model stub is needed here).
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_GRADCAM_UPLOAD_BYTES", 1024)

    oversized = b"0" * (2 * 1024)
    resp = client.post(
        "/gradcam",
        files={"file": ("scan.nii.gz", io.BytesIO(oversized), "application/octet-stream")},
    )
    assert resp.status_code == 413


def test_gradcam_cleans_up_temp_file_on_write_failure(client, monkeypatch):
    # Simulate an OSError (e.g. disk full) partway through the streaming
    # write, which happens *before* the size-cap check can catch anything.
    # The handler must still remove the temp file it created rather than
    # leaking it on the container's ephemeral disk.
    import app.main as main_module

    created_paths = []
    real_named_temp_file = tempfile.NamedTemporaryFile

    class FailingTempFile:
        def __init__(self, *args, **kwargs):
            self._real = real_named_temp_file(*args, **kwargs)
            self.name = self._real.name
            created_paths.append(self.name)
            self._writes = 0

        def write(self, data):
            self._writes += 1
            if self._writes > 1:
                raise OSError("simulated write failure")
            return self._real.write(data)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._real.close()
            return False

    monkeypatch.setattr(main_module.tempfile, "NamedTemporaryFile", FailingTempFile)

    chunk = b"0" * (1024 * 1024)
    # TestClient re-raises unhandled server exceptions by default (the real
    # ASGI server would turn this into a 500 response instead) - either way,
    # the handler's finally block must still run and remove the temp file.
    with pytest.raises(OSError):
        client.post(
            "/gradcam",
            files={"file": ("scan.nii.gz", io.BytesIO(chunk + chunk), "application/octet-stream")},
        )

    assert created_paths, "expected a temp file to have been created"
    assert not os.path.exists(created_paths[0]), "temp file must be cleaned up after a write failure"
