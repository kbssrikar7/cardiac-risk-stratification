"""Publication/patent push, step 2 (test-patient half): computes infarct-burden
features for the 50 test patients using the trained 5-class model's PREDICTED
masks, since no ground truth exists for them - the same train/test asymmetry
already handled for the myocardium radiomics fix earlier.
"""
import glob
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

DATA_DIR = "training/data"
TEST_ROOT = os.path.join(DATA_DIR, "emidec-segmentation-testset-1.0.0")
ARTIFACT_DIR = "training"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "unet_5class.h5")
OUT_CSV = os.path.join(ARTIFACT_DIR, "infarct_features_test.csv")
NUM_CLASSES = 5


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


def load_nii_volume(path):
    itk_img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(itk_img)
    return np.transpose(vol, (2, 1, 0)), itk_img  # X,Y,Z


def preprocess_volume_for_unet(vol, target_size=(128, 128)):
    vol = vol.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    slices = [tf.image.resize(vol[:, :, i][..., None], target_size, method="bilinear").numpy().squeeze()
              for i in range(vol.shape[2])]
    return np.array(slices)[..., None]


def predict_5class_mask(image_path, model):
    vol, itk_img = load_nii_volume(image_path)
    X = preprocess_volume_for_unet(vol)
    preds = model.predict(X, verbose=0)  # (n, 128, 128, 5)
    pred_classes = np.argmax(preds, axis=-1)  # (n, 128, 128)

    orig_xy = (vol.shape[0], vol.shape[1])
    mask_full = np.zeros_like(vol, dtype=np.int32)
    for i in range(pred_classes.shape[0]):
        resized = tf.image.resize(pred_classes[i][..., None].astype(np.float32), orig_xy, method="nearest").numpy().squeeze()
        mask_full[:, :, i] = resized.astype(np.int32)
    return mask_full, itk_img


def compute_features(mask_arr, itk_img):
    spacing = itk_img.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    myo = int((mask_arr == 2).sum())
    infarct = int((mask_arr == 3).sum())
    no_reflow = int((mask_arr == 4).sum())

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
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    rows = []
    for folder in sorted(glob.glob(os.path.join(TEST_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        pid = os.path.basename(folder).split("_")[-1].zfill(3)
        img_path = find_nii_file(folder, "Images")
        if not img_path:
            print(f"  skip {folder}: no image found")
            continue
        mask_arr, itk_img = predict_5class_mask(img_path, model)
        feats = compute_features(mask_arr, itk_img)
        feats["PatientID"] = pid
        rows.append(feats)
        print(f"  {pid}: infarct={feats['infarct_volume_cm3']:.2f}cm3  no_reflow={feats['no_reflow_volume_cm3']:.2f}cm3")

    df = pd.DataFrame(rows)
    print(f"\nComputed infarct-burden features for {len(df)} test patients (from predicted masks)")
    print(f"Patients with predicted infarct: {(df['infarct_volume_cm3'] > 0).sum()}/{len(df)}")
    print(f"Patients with predicted no-reflow: {(df['no_reflow_volume_cm3'] > 0).sum()}/{len(df)}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
