"""Promotion candidate for the LIVE 3-class model (`unet_multiclass.h5`, the
one actually served by backend/app/ml.py's /gradcam endpoint) - retrains
retrain_unet_patient_split.py's model using the isotropic in-plane resampling
+ percentile normalization from training/imaging_common.py, validated as a
real improvement on the 5-class model in TECHNICAL_REPORT.md Section 12
(no internal Dice regression, +0.054 infarction Dice, and it fixed a real
external generalization failure: the external LGE scan went from 100%
background/zero detection to real detection).

This script exists specifically to check whether that same fix helps the
model that's actually live, not just its offline 5-class sibling. Reuses
retrain_unet_patient_split.py's split logic/seed (identical train/val
patients), architecture, and augmentation - only the preprocessing changes,
matching Section 12's own methodology (isolate one variable at a time).

Must be validated against real held-out Dice (compared to the CURRENT live
unet_multiclass.h5, not the already-superseded unet_multiclass_v2.h5) AND
re-tested against the same external artifacts from Section 11 before being
promoted - promotion is a separate, explicit step (see promote step at the
bottom of this file's usage), not automatic just because this script ran.
"""
import glob
import os

import numpy as np
import tensorflow as tf

from retrain_unet_patient_split import (
    RANDOM_STATE, TRAIN_ROOT, build_unet_multiclass_plain, dice_iou_per_class,
    find_nii_file, make_augmented_dataset,
)
from imaging_common import load_paired_slices_for_training

ARTIFACT_DIR = "training"
OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_multiclass_isotropic.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_multiclass_isotropic_checkpoint.h5")
CLASS_NAMES = ["background", "LV cavity", "myocardium"]


def load_slices_for_patients_isotropic(triples_subset, target_size=(128, 128), num_classes=3,
                                        percentile_low=10.0, percentile_high=90.0):
    Xs, Ys = [], []
    for pid, img_path, mask_path in triples_subset:
        xs, ys = load_paired_slices_for_training(img_path, mask_path, target_size, num_classes,
                                                  percentile_low=percentile_low, percentile_high=percentile_high)
        Xs.extend(xs)
        Ys.extend(ys)
    X = np.array(Xs)[..., None].astype(np.float32)
    Y = tf.keras.utils.to_categorical(np.array(Ys), num_classes=num_classes)
    return X, Y


def main(epochs=120, patience=8, percentile_low=10.0, percentile_high=90.0, out_model=None, checkpoint_path=None):
    out_model = out_model or OUT_MODEL
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    triples = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        img = find_nii_file(folder, "Images")
        mask = find_nii_file(folder, "Contours")
        if img and mask:
            triples.append((os.path.basename(folder), img, mask))
    print(f"Found {len(triples)} training patients with both image+mask")

    # Identical split logic/seed to retrain_unet_patient_split.py -> same train/val patients
    patient_keys = [t[0] for t in triples]
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = patient_keys.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(0.2 * len(shuffled)))
    val_patients = set(shuffled[:n_val])
    train_triples = [t for t in triples if t[0] not in val_patients]
    val_triples = [t for t in triples if t[0] in val_patients]
    print(f"Train patients: {len(train_triples)}, Val patients: {len(val_triples)}")

    print(f"Loading train slices (isotropic in-plane resample + percentile normalize [{percentile_low}/{percentile_high}])...")
    X_train, Y_train = load_slices_for_patients_isotropic(train_triples, percentile_low=percentile_low, percentile_high=percentile_high)
    print("Loading val slices...")
    X_val, Y_val = load_slices_for_patients_isotropic(val_triples, percentile_low=percentile_low, percentile_high=percentile_high)
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    # Baseline: the CURRENT live unet_multiclass.h5, on this SAME (isotropically
    # preprocessed) val split - fair comparison of the model that's actually deployed.
    try:
        old_model = tf.keras.models.load_model("unet_multiclass.h5", compile=False)
    except TypeError:
        old_model = build_unet_multiclass_plain()
        old_model.load_weights("unet_multiclass.h5")
    old_preds = old_model.predict(X_val, verbose=0)
    old_pred_onehot = tf.keras.utils.to_categorical(np.argmax(old_preds, axis=-1), num_classes=3)
    dice_old, iou_old = dice_iou_per_class(Y_val, old_pred_onehot)
    print("\n=== BASELINE (current live unet_multiclass.h5) on isotropically-preprocessed val set ===")
    for name, d, i in zip(CLASS_NAMES, dice_old, iou_old):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    new_model = build_unet_multiclass_plain()
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    train_ds = make_augmented_dataset(X_train, Y_train, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False, save_freq="epoch", verbose=0)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    new_model.fit(
        train_ds, validation_data=(X_val, Y_val),
        epochs=epochs, verbose=2, callbacks=[checkpoint_cb, early_stop_cb],
    )

    new_preds = new_model.predict(X_val, verbose=0)
    new_pred_onehot = tf.keras.utils.to_categorical(np.argmax(new_preds, axis=-1), num_classes=3)
    dice_new, iou_new = dice_iou_per_class(Y_val, new_pred_onehot)
    print("\n=== NEW model (isotropic resample + percentile normalize) on the SAME held-out val set ===")
    for name, d, i in zip(CLASS_NAMES, dice_new, iou_new):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    print("\n=== Comparison (myocardium is the class that matters clinically) ===")
    for name, do, io_, dn, in_ in zip(CLASS_NAMES, dice_old, iou_old, dice_new, iou_new):
        print(f"  {name}: Dice baseline={do:.4f} new={dn:.4f} delta={dn-do:+.4f}  |  IoU baseline={io_:.4f} new={in_:.4f} delta={in_-io_:+.4f}")

    new_model.save(out_model)
    print(f"\nSaved: {out_model}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--percentile-low", type=float, default=10.0)
    parser.add_argument("--percentile-high", type=float, default=90.0)
    parser.add_argument("--out-model", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        main(epochs=3, patience=2, percentile_low=args.percentile_low, percentile_high=args.percentile_high,
             out_model=args.out_model, checkpoint_path=args.checkpoint_path)
    else:
        main(percentile_low=args.percentile_low, percentile_high=args.percentile_high,
             out_model=args.out_model, checkpoint_path=args.checkpoint_path)
