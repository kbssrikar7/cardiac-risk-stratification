"""Isolates ONE variable the cascade experiment (Section 13) could not: does
CaRe-CNN's paired rare-class batch sampling help no-reflow on its own, or was
Section 13's non-zero no-reflow Dice (0.033) actually coming from the cascade
structure (a decision narrowed to an already-localized region) rather than
the sampling itself?

Section 13 confounded two changes at once - a 3-stage cascade AND paired
sampling for the no-reflow stage - so it couldn't tell which one, if either,
was responsible for the (partial) no-reflow improvement, or which one was
responsible for the regression on every other class. This script applies
ONLY the paired-sampling change to the SAME single-shot, isotropic-preprocessed
5-class model already validated in Section 12 (retrain_unet_5class_isotropic.py) -
same architecture, same full 5-class softmax over the whole image, same
isotropic preprocessing, same patient split. The only difference from Section
12's model is batch composition: every batch is guaranteed at least one
no-reflow-positive slice, instead of uniform random batching (which, at
0.018% prevalence, usually contains zero positive examples).

Reported honestly in TECHNICAL_REPORT.md regardless of outcome, per this
project's established practice.
"""
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from retrain_unet_5class import (
    ARTIFACT_DIR, CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, TRAIN_ROOT,
    build_unet_multiclass_plain, collapse_to_3class, dice_iou_per_class,
    find_nii_file,
)
from retrain_unet_5class_isotropic import MeanForegroundDiceLogger, load_slices_for_patients_isotropic

OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_5class_isotropic_paired.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_5class_isotropic_paired_checkpoint.h5")


def make_paired_augmented_dataset(X, Y, no_reflow_positive_mask, batch_size=16, seed=RANDOM_STATE):
    """Same augmentation (flip/rotate/translate on a stacked image+mask
    tensor, contrast jitter on the image only, clamped) as
    retrain_unet_5class.py's make_augmented_dataset - only the batch
    composition differs: half the batch drawn only from no-reflow-positive
    slices, half from negative slices, so every batch sees the rare class
    instead of usually seeing zero examples of it."""
    flip = layers.RandomFlip("horizontal", seed=seed)
    rotate = layers.RandomRotation(0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    translate = layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    contrast = layers.RandomContrast(0.1, seed=seed)

    def augment(image, mask):
        combined = tf.concat([image, mask], axis=-1)
        combined = flip(combined, training=True)
        combined = rotate(combined, training=True)
        combined = translate(combined, training=True)
        image_aug, mask_aug = combined[..., :1], combined[..., 1:]
        image_aug = tf.clip_by_value(contrast(image_aug, training=True), 0.0, 1.0)
        return image_aug, mask_aug

    pos_idx = np.where(no_reflow_positive_mask)[0]
    neg_idx = np.where(~no_reflow_positive_mask)[0]
    print(f"  Paired sampling: {len(pos_idx)} no-reflow-positive slices, {len(neg_idx)} negative slices")

    half_batch = max(1, batch_size // 2)
    pos_ds = tf.data.Dataset.from_tensor_slices((X[pos_idx], Y[pos_idx]))
    pos_ds = pos_ds.repeat().shuffle(max(len(pos_idx), 1), seed=seed).batch(half_batch)
    neg_ds = tf.data.Dataset.from_tensor_slices((X[neg_idx], Y[neg_idx]))
    neg_ds = neg_ds.repeat().shuffle(max(len(neg_idx), 1), seed=seed).batch(batch_size - half_batch)

    steps_per_epoch = max(1, len(neg_idx) // (batch_size - half_batch))

    def merge(pos_batch, neg_batch):
        image = tf.concat([pos_batch[0], neg_batch[0]], axis=0)
        mask = tf.concat([pos_batch[1], neg_batch[1]], axis=0)
        return augment(image, mask)

    ds = tf.data.Dataset.zip((pos_ds, neg_ds)).map(merge, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.take(steps_per_epoch).prefetch(tf.data.AUTOTUNE), steps_per_epoch


def main(epochs=150, patience=8):
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

    print("Loading train slices (isotropic in-plane resample + percentile normalize)...")
    X_train, Y_train = load_slices_for_patients_isotropic(train_triples)
    print("Loading val slices...")
    X_val, Y_val = load_slices_for_patients_isotropic(val_triples)
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    no_reflow_idx = CLASS_NAMES.index("no-reflow")
    no_reflow_positive = (Y_train[..., no_reflow_idx].sum(axis=(1, 2)) > 0)

    model = build_unet_multiclass_plain(num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    train_ds, steps_per_epoch = make_paired_augmented_dataset(X_train, Y_train, no_reflow_positive, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, save_freq="epoch", verbose=0)
    dice_logger_cb = MeanForegroundDiceLogger(X_val, Y_val, print_every=5)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mean_fg_dice", mode="max", patience=patience, restore_best_weights=True)

    model.fit(
        train_ds, steps_per_epoch=steps_per_epoch, validation_data=(X_val, Y_val),
        epochs=epochs, verbose=2, callbacks=[checkpoint_cb, dice_logger_cb, early_stop_cb],
    )

    preds = model.predict(X_val, verbose=0)
    pred_onehot = tf.keras.utils.to_categorical(np.argmax(preds, axis=-1), num_classes=NUM_CLASSES)
    dice, iou = dice_iou_per_class(Y_val, pred_onehot, num_classes=NUM_CLASSES)
    print("\n=== Isotropic + paired-sampling 5-class model, full val labels ===")
    for name, d, i in zip(CLASS_NAMES, dice, iou):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    pred_3class = collapse_to_3class(pred_onehot)
    Y_val_3class = collapse_to_3class(Y_val)
    dice_3c, iou_3c = dice_iou_per_class(Y_val_3class, pred_3class, num_classes=3)
    print("\n=== Collapsed to 3-class (for comparison against retrain_unet_5class_isotropic.py's own numbers) ===")
    for name, d, i in zip(["background", "LV cavity", "myocardium"], dice_3c, iou_3c):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    model.save(OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    if smoke:
        main(epochs=3, patience=2)
    else:
        main()
