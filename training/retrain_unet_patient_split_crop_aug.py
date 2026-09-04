"""Tests the OTHER durable fix TECHNICAL_REPORT.md stated for the Section 11
framing-generalization failure - training-time crop/scale augmentation -
as an alternative to Section 12/14's isotropic-resampling fix, not stacked
on top of it. Section 14 found isotropic resampling fixes the framing gap
but at a real myocardium Dice cost (0.810 -> 0.723 at the promoted 1/99
setting). This script asks: can augmentation alone teach the same
robustness without that preprocessing-driven accuracy cost?

Uses the ORIGINAL naive preprocessing (retrain_unet_patient_split.py's
global min-max, no isotropic resampling) so any result is attributable to
augmentation alone, not confounded with the preprocessing change. Adds
RandomZoom(height_factor=(0, 0.6), width_factor=(0, 0.6)) - positive factors
only, verified empirically to zoom OUT (shrink content within the same
canvas, not crop into it) - directly simulating the external scan's failure
mode: a heart that's a smaller feature within a wider field, rather than
filling the frame the way EMIDEC's acquisition convention does. Same
reflect fill_mode as the existing flip/rotate/translate augmentation, for
the same reason documented in retrain_unet_patient_split.py's
make_augmented_dataset docstring (zero-padding a one-hot mask produces an
invalid, non-one-hot target at the border).

Must be validated against real held-out Dice AND the same two external
artifacts from Section 11 before being trusted - a partial improvement
should be reported as partial, not oversold, per this project's practice.
"""
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from retrain_unet_patient_split import (
    RANDOM_STATE, TRAIN_ROOT, build_unet_multiclass_plain, dice_iou_per_class,
    find_nii_file, load_slices_for_patients,
)

ARTIFACT_DIR = "training"
OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_multiclass_crop_aug.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_multiclass_crop_aug_checkpoint.h5")
CLASS_NAMES = ["background", "LV cavity", "myocardium"]


def make_crop_augmented_dataset(X, Y, batch_size=16, seed=RANDOM_STATE):
    flip = layers.RandomFlip("horizontal", seed=seed)
    rotate = layers.RandomRotation(0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    translate = layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", interpolation="nearest", seed=seed)
    zoom_out = layers.RandomZoom(height_factor=(0.0, 0.6), width_factor=(0.0, 0.6),
                                  fill_mode="reflect", interpolation="nearest", seed=seed)
    contrast = layers.RandomContrast(0.1, seed=seed)

    def augment(image, mask):
        combined = tf.concat([image, mask], axis=-1)
        combined = flip(combined, training=True)
        combined = rotate(combined, training=True)
        combined = translate(combined, training=True)
        combined = zoom_out(combined, training=True)
        image_aug, mask_aug = combined[..., :1], combined[..., 1:]
        image_aug = tf.clip_by_value(contrast(image_aug, training=True), 0.0, 1.0)
        return image_aug, mask_aug

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(len(X), seed=seed).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(epochs=120, patience=8):
    triples = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        img = find_nii_file(folder, "Images")
        mask = find_nii_file(folder, "Contours")
        if img and mask:
            triples.append((os.path.basename(folder), img, mask))
    print(f"Found {len(triples)} training patients with both image+mask")

    patient_keys = [t[0] for t in triples]
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = patient_keys.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(0.2 * len(shuffled)))
    val_patients = set(shuffled[:n_val])
    train_triples = [t for t in triples if t[0] not in val_patients]
    val_triples = [t for t in triples if t[0] in val_patients]
    print(f"Train patients: {len(train_triples)}, Val patients: {len(val_triples)}")

    print("Loading train slices (naive preprocessing, matching the original baseline)...")
    X_train, Y_train = load_slices_for_patients(train_triples)
    print("Loading val slices...")
    X_val, Y_val = load_slices_for_patients(val_triples)
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    try:
        old_model = tf.keras.models.load_model("training/unet_multiclass_original_naive_preprocessing_backup.h5", compile=False)
    except TypeError:
        old_model = build_unet_multiclass_plain()
        old_model.load_weights("training/unet_multiclass_original_naive_preprocessing_backup.h5")
    old_preds = old_model.predict(X_val, verbose=0)
    old_pred_onehot = tf.keras.utils.to_categorical(np.argmax(old_preds, axis=-1), num_classes=3)
    dice_old, iou_old = dice_iou_per_class(Y_val, old_pred_onehot)
    print("\n=== BASELINE (original naive-preprocessing model) on this val set ===")
    for name, d, i in zip(CLASS_NAMES, dice_old, iou_old):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    new_model = build_unet_multiclass_plain()
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    train_ds = make_crop_augmented_dataset(X_train, Y_train, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, save_freq="epoch", verbose=0)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    new_model.fit(
        train_ds, validation_data=(X_val, Y_val),
        epochs=epochs, verbose=2, callbacks=[checkpoint_cb, early_stop_cb],
    )

    new_preds = new_model.predict(X_val, verbose=0)
    new_pred_onehot = tf.keras.utils.to_categorical(np.argmax(new_preds, axis=-1), num_classes=3)
    dice_new, iou_new = dice_iou_per_class(Y_val, new_pred_onehot)
    print("\n=== NEW model (naive preprocessing + zoom-out crop augmentation) on the SAME held-out val set ===")
    for name, d, i in zip(CLASS_NAMES, dice_new, iou_new):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    print("\n=== Comparison (myocardium is the class that matters clinically) ===")
    for name, do, io_, dn, in_ in zip(CLASS_NAMES, dice_old, iou_old, dice_new, iou_new):
        print(f"  {name}: Dice baseline={do:.4f} new={dn:.4f} delta={dn-do:+.4f}  |  IoU baseline={io_:.4f} new={in_:.4f} delta={in_-io_:+.4f}")

    new_model.save(OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    if smoke:
        main(epochs=3, patience=2)
    else:
        main()
