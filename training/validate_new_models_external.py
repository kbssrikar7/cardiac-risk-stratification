"""Post-training validation against the same two external artifacts used in
TECHNICAL_REPORT.md Section 11 (the external LGE scan, the MSD Task02_Heart
volume), for whichever new model(s) from this session's Tier-1/Tier-2 attempt
are ready: the isotropic-resampling retrain (unet_5class_isotropic.h5) and/or
the cascaded architecture (unet_cascaded_stage{1,2,3}.h5).

Run from repo root: .venv/bin/python training/validate_new_models_external.py
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, "training")
from imaging_common import preprocess_volume_for_unet as preprocess_isotropic
import retrain_unet_cascaded as cascaded

EXTERNAL_LGE = "/tmp/claude-1000/e2e/internet_test/rawimage.nii"
EXTERNAL_MSD = "/tmp/claude-1000/e2e/internet_test/msd_heart_sample.nii.gz"
CLASS_NAMES = ["background", "LV cavity", "myocardium", "infarction", "no-reflow"]


def report_single_shot(model_path, label):
    if not os.path.exists(model_path):
        print(f"[{label}] model not found at {model_path}, skipping")
        return
    model = tf.keras.models.load_model(model_path, compile=False)
    num_classes = model.output_shape[-1]
    for name, path in [("external LGE scan", EXTERNAL_LGE), ("MSD Task02_Heart", EXTERNAL_MSD)]:
        if not os.path.exists(path):
            print(f"  [{label}] {name}: file not found at {path}, skipping")
            continue
        X = preprocess_isotropic(path)
        preds = model.predict(X, verbose=0)
        masks = np.argmax(preds, axis=-1)
        counts = [int((masks == c).sum()) for c in range(num_classes)]
        print(f"  [{label}] {name}: " + ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(num_classes)))


def report_cascade():
    paths = [cascaded.STAGE1_MODEL_PATH, cascaded.STAGE2_MODEL_PATH, cascaded.STAGE3_MODEL_PATH]
    if not all(os.path.exists(p) for p in paths):
        print("[cascade] not all 3 stage models found, skipping")
        return
    stage1 = tf.keras.models.load_model(cascaded.STAGE1_MODEL_PATH, compile=False)
    stage2 = tf.keras.models.load_model(cascaded.STAGE2_MODEL_PATH, compile=False)
    stage3 = tf.keras.models.load_model(cascaded.STAGE3_MODEL_PATH, compile=False)
    for name, path in [("external LGE scan", EXTERNAL_LGE), ("MSD Task02_Heart", EXTERNAL_MSD)]:
        if not os.path.exists(path):
            print(f"  [cascade] {name}: file not found at {path}, skipping")
            continue
        X = preprocess_isotropic(path)
        final = cascaded.predict_cascade(stage1, stage2, stage3, X)
        counts = [int(final[..., c].sum()) for c in range(5)]
        print(f"  [cascade] {name}: " + ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(5)))


if __name__ == "__main__":
    print("=== Baseline (original, already-deployed single-shot model, naive preprocessing) ===")
    print("  (already measured earlier this session: LGE scan -> 100% background/0 detection; "
          "MSD volume -> real cavity+myocardium detection, ~294/173 pixels at best slice)")

    print("\n=== Isotropic-resampling retrain (Tier 1) ===")
    report_single_shot("training/unet_5class_isotropic.h5", "isotropic")

    print("\n=== Cascaded architecture (Tier 2) ===")
    report_cascade()
