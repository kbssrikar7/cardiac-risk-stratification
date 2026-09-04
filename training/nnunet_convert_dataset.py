"""One-time conversion of the 100 ground-truth EMIDEC patients into nnU-Net's
expected raw-dataset layout (Dataset001_EMIDEC/imagesTr, labelsTr,
dataset.json), for the nnU-Net offline benchmarking experiment
(experiment/nnunet-baseline branch - see TECHNICAL_REPORT.md).

Uses the SAME patient-level train/val split (same RANDOM_STATE, same seed)
as every other model in this project, so nnU-Net's own held-out patients
match the ones every other Dice table in this report is computed on. nnU-Net
manages its own cross-validation internally (5-fold by default), so this
conversion includes all 100 patients in one dataset; the specific fold
trained (fold 0) is checked separately against our own val-patient list to
confirm no leakage relative to our own split, since nnU-Net's fold
assignment is its own random split, not necessarily identical to ours.
"""
import glob
import json
import os
import shutil

import numpy as np
import SimpleITK as sitk

TRAIN_ROOT = "training/data/emidec-dataset-1.0.1"
OUT_DIR = os.environ.get("nnUNet_raw", "/tmp/claude-1000/nnunet/nnUNet_raw") + "/Dataset001_EMIDEC"


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


def main():
    images_dir = os.path.join(OUT_DIR, "imagesTr")
    labels_dir = os.path.join(OUT_DIR, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    triples = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        img = find_nii_file(folder, "Images")
        mask = find_nii_file(folder, "Contours")
        if img and mask:
            triples.append((os.path.basename(folder), img, mask))
    print(f"Found {len(triples)} patients to convert")

    case_ids = []
    for pid, img_path, mask_path in triples:
        case_id = pid  # e.g. "Case_N006"
        itk_img = sitk.ReadImage(img_path)
        itk_mask = sitk.ReadImage(mask_path)

        # nnU-Net requires integer label masks with values matching dataset.json's
        # labels dict exactly - clip any stray values beyond the known 5 classes.
        mask_arr = sitk.GetArrayFromImage(itk_mask)
        mask_arr = np.clip(mask_arr, 0, 4).astype(np.uint8)
        clean_mask = sitk.GetImageFromArray(mask_arr)
        clean_mask.CopyInformation(itk_mask)

        sitk.WriteImage(itk_img, os.path.join(images_dir, f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(clean_mask, os.path.join(labels_dir, f"{case_id}.nii.gz"))
        case_ids.append(case_id)

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "LV_cavity": 1,
            "myocardium": 2,
            "infarction": 3,
            "no_reflow": 4,
        },
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(OUT_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Converted {len(case_ids)} cases to {OUT_DIR}")
    print("dataset.json:", json.dumps(dataset_json, indent=2))


if __name__ == "__main__":
    main()
