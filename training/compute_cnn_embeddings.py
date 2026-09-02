"""Publication/patent push, transformer-fusion ablation, step 1: extracts a
fixed-size pooled feature vector per patient from the frozen 5-class U-Net's
bottleneck layer (training/unet_5class.h5), for use as the "imaging modality"
input to the hybrid CNN-transformer in train_transformer_fusion.py.

Frozen/transfer-learning, not fine-tuned: literature on small medical-imaging
datasets (see training/TECHNICAL_REPORT.md) recommends exactly this pattern
for a dataset this size - a pretrained CNN as a fixed feature extractor, with
only a small fusion head trained per CV fold. This also makes the fold-level
retraining in the repeated-CV evaluation cheap: the CNN forward pass happens
once here, not once per fold.

Unlike the infarct-burden features, this needs only the input MRI image, not
a ground-truth or predicted mask, so there's no train/test asymmetry here -
all 150 patients get an embedding from the same code path.
"""
import glob
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras import models

DATA_DIR = "data"
TRAIN_ROOT = os.path.join(DATA_DIR, "emidec-dataset-1.0.1")
TEST_ROOT = os.path.join(DATA_DIR, "emidec-segmentation-testset-1.0.0")
MODEL_PATH = "unet_5class.h5"
OUT_CSV = "cnn_embeddings.csv"
TARGET_SIZE = (128, 128)


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
    return np.transpose(sitk.GetArrayFromImage(itk_img), (2, 1, 0))  # X,Y,Z


def build_bottleneck_extractor(model_path):
    """Same architecture as build_unet_multiclass_plain in retrain_unet_5class.py;
    load the trained weights, then expose the bottleneck ('bn_conv2') activations
    as the model's output instead of the segmentation head."""
    try:
        full_model = tf.keras.models.load_model(model_path, compile=False)
    except TypeError:
        from retrain_unet_5class import build_unet_multiclass_plain
        full_model = build_unet_multiclass_plain(num_classes=5)
        full_model.load_weights(model_path)
    bottleneck = full_model.get_layer("bn_conv2").output
    return models.Model(inputs=full_model.input, outputs=bottleneck)


def embed_patient(extractor, image_path):
    vol = load_nii_volume(image_path).astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    slices = np.stack([
        tf.image.resize(vol[:, :, z][..., None], TARGET_SIZE, method="bilinear").numpy().squeeze()
        for z in range(vol.shape[2])
    ])[..., None].astype(np.float32)
    bottleneck_maps = extractor.predict(slices, verbose=0)  # (n_slices, H, W, 128)
    per_slice_pooled = bottleneck_maps.mean(axis=(1, 2))  # (n_slices, 128) - spatial pooling
    return per_slice_pooled.mean(axis=0)  # (128,) - pool across slices too


def main():
    extractor = build_bottleneck_extractor(MODEL_PATH)
    emb_dim = extractor.output_shape[-1]
    print(f"Bottleneck embedding dim: {emb_dim}")

    rows = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        pid = os.path.basename(folder).split("_")[-1].lstrip("PN").zfill(3)
        img_path = find_nii_file(folder, "Images")
        if not img_path:
            continue
        emb = embed_patient(extractor, img_path)
        rows.append({"PatientID": pid, **{f"cnn_emb_{i}": float(v) for i, v in enumerate(emb)}})
        print(f"  train {pid}: embedded {len(emb)}-dim vector from {img_path}")

    for folder in sorted(glob.glob(os.path.join(TEST_ROOT, "Case_*"))):
        pid = os.path.basename(folder).split("_")[-1].zfill(3)
        img_path = find_nii_file(folder, "Images")
        if not img_path:
            continue
        emb = embed_patient(extractor, img_path)
        rows.append({"PatientID": pid, **{f"cnn_emb_{i}": float(v) for i, v in enumerate(emb)}})
        print(f"  test  {pid}: embedded {len(emb)}-dim vector from {img_path}")

    df = pd.DataFrame(rows)
    print(f"\nComputed embeddings for {len(df)} patients ({emb_dim} dims each)")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
