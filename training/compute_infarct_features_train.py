"""Publication/patent push, step 2 (training-patient half): computes
infarct-burden features directly from the 100 training patients' ground-truth
masks - no model inference needed here since these patients' masks are real
annotations, not predictions (that half only applies to the 50 test patients,
handled separately once the 5-class model from retrain_unet_5class.py exists).

Features, matching standard clinical cardiac-MRI infarct quantification:
- infarct_volume_cm3: label-3 (infarction) voxel volume
- infarct_pct_of_myocardium: infarct / (myocardium + infarction + no-reflow) -
  the standard "infarct size as % of LV myocardium" clinical measure (labels
  2/3/4 are all myocardial tissue at different pathology stages in EMIDEC's
  scheme; label 2 specifically means *non-infarcted* myocardium)
- no_reflow_volume_cm3: label-4 (no-reflow / microvascular obstruction) voxel volume
- no_reflow_pct_of_infarct: no-reflow / infarction - the standard "MVO extent"
  clinical marker
"""
import glob
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATA_DIR = "training/data"
TRAIN_ROOT = os.path.join(DATA_DIR, "emidec-dataset-1.0.1")
ARTIFACT_DIR = "training"
OUT_CSV = os.path.join(ARTIFACT_DIR, "infarct_features_train.csv")


def find_nii_file(case_folder, subdir):
    for pattern in [
        os.path.join(case_folder, subdir, "*.nii", "*"),
        os.path.join(case_folder, subdir, "*.nii.gz"),
        os.path.join(case_folder, subdir, "*.nii"),
    ]:
        matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]
        if matches:
            return matches[0]
    return None


def compute_features(mask_path):
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)  # Z,Y,X
    spacing = img.GetSpacing()  # (X,Y,Z) in mm
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    myo = int((arr == 2).sum())
    infarct = int((arr == 3).sum())
    no_reflow = int((arr == 4).sum())

    infarct_vol_cm3 = infarct * voxel_vol_mm3 / 1000.0
    no_reflow_vol_cm3 = no_reflow * voxel_vol_mm3 / 1000.0
    total_myo_tissue = myo + infarct + no_reflow
    infarct_pct = 100.0 * infarct / total_myo_tissue if total_myo_tissue > 0 else 0.0
    no_reflow_pct = 100.0 * no_reflow / infarct if infarct > 0 else 0.0

    return {
        "infarct_volume_cm3": infarct_vol_cm3,
        "infarct_pct_of_myocardium": infarct_pct,
        "no_reflow_volume_cm3": no_reflow_vol_cm3,
        "no_reflow_pct_of_infarct": no_reflow_pct,
    }


def main():
    rows = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        pid = os.path.basename(folder).split("_")[-1].lstrip("PN").zfill(3)
        mask_path = find_nii_file(folder, "Contours")
        if not mask_path:
            print(f"  skip {folder}: no mask found")
            continue
        feats = compute_features(mask_path)
        feats["PatientID"] = pid
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"Computed infarct-burden features for {len(df)} training patients")
    print(df[["infarct_volume_cm3", "no_reflow_volume_cm3"]].describe())

    # Sanity check against the Kaggle dataset's own published stats:
    # infarction median volume 20.26 cm^3, no-reflow median 2.35 cm^3 (100 training patients only)
    nonzero_infarct = df[df["infarct_volume_cm3"] > 0]["infarct_volume_cm3"]
    nonzero_no_reflow = df[df["no_reflow_volume_cm3"] > 0]["no_reflow_volume_cm3"]
    print(f"\nMedian infarct volume (nonzero cases): {nonzero_infarct.median():.2f} cm^3 (dataset reports 20.26 cm^3)")
    print(f"Median no-reflow volume (nonzero cases): {nonzero_no_reflow.median():.2f} cm^3 (dataset reports 2.35 cm^3)")
    print(f"Patients with any infarct: {(df['infarct_volume_cm3'] > 0).sum()}/{len(df)} (dataset reports 67%)")
    print(f"Patients with any no-reflow: {(df['no_reflow_volume_cm3'] > 0).sum()}/{len(df)} (dataset reports 40%)")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
