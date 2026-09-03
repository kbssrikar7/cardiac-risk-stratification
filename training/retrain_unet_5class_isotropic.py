"""Tier-1 improvement from the roadmap in TECHNICAL_REPORT.md Section 11:
retrains the 5-class U-Net with isotropic in-plane resampling + percentile
intensity normalization (training/imaging_common.py), replacing the naive
pixel resize + global min-max normalization retrain_unet_5class.py uses.

Motivation: Section 11 found the deployed model implicitly learned "the heart
fills the frame" from EMIDEC's consistent tight-cropping acquisition, and
completely failed (100% background at 1.0 confidence) on a real external
LGE-MRI scan whose heart was a smaller, off-center feature. nnU-Net's
standard fix for exactly this class of cross-protocol generalization failure
is resampling to a common physical voxel spacing before any pixel-grid
resize - see training/imaging_common.py's docstring for why only the
in-plane axes are resampled here, not full 3D isotropic resampling.

This is a hypothesis being tested, not a proven fix - it must be judged
against the same held-out Dice table as every other model in this project
(dice_iou_per_class on the same val patients) AND against the same two
external artifacts used in Section 11 (the external LGE scan, the MSD
Task02_Heart volume) before being trusted or promoted. Reported honestly in
TECHNICAL_REPORT.md regardless of outcome, per this project's established
practice.

Reuses retrain_unet_5class.py's split logic/seed (same train/val patients),
architecture, and augmentation unchanged - only the preprocessing changes,
so any Dice difference is attributable to that and not a confound. Also
carries forward the corrected val_mean_fg_dice EarlyStopping criterion from
retrain_unet_5class_weighted.py (val_loss was already shown to be a poor
signal for this problem - see that script's docstring), since training a new
model with the same known-bad monitor would undermine this run's own
validity for no added reason.
"""
import glob
import os

import numpy as np
import tensorflow as tf

from retrain_unet_5class import (
    ARTIFACT_DIR, CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, TRAIN_ROOT,
    build_unet_multiclass_plain, collapse_to_3class, dice_iou_per_class,
    find_nii_file, make_augmented_dataset,
)
from imaging_common import load_paired_slices_for_training

OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_5class_isotropic.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_5class_isotropic_checkpoint.h5")


class MeanForegroundDiceLogger(tf.keras.callbacks.Callback):
    """Identical to retrain_unet_5class_weighted.py's callback of the same
    name - duplicated rather than imported since that script's version is
    tied to its own module-level NUM_CLASSES/CLASS_NAMES import pattern; see
    that file's docstring for why val_loss is a bad EarlyStopping signal here."""

    def __init__(self, X_val, Y_val, print_every=5):
        super().__init__()
        self.X_val = X_val
        self.Y_val = Y_val
        self.print_every = print_every
        self.no_reflow_idx = CLASS_NAMES.index("no-reflow")
        self.fg_idx = [i for i, n in enumerate(CLASS_NAMES) if n != "background"]

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        pred_onehot = tf.keras.utils.to_categorical(np.argmax(preds, axis=-1), num_classes=NUM_CLASSES)
        dice, _ = dice_iou_per_class(self.Y_val, pred_onehot, num_classes=NUM_CLASSES)
        mean_fg_dice = float(np.mean([dice[i] for i in self.fg_idx]))
        if logs is not None:
            logs["val_mean_fg_dice"] = mean_fg_dice
        if (epoch + 1) % self.print_every == 0:
            print(f"  [epoch {epoch+1}] mean_fg_dice={mean_fg_dice:.4f}  no-reflow Dice={dice[self.no_reflow_idx]:.4f}  "
                  f"(all classes: {['%.3f' % d for d in dice]})")


def load_slices_for_patients_isotropic(triples_subset, target_size=(128, 128), num_classes=NUM_CLASSES):
    Xs, Ys = [], []
    for pid, img_path, mask_path in triples_subset:
        xs, ys = load_paired_slices_for_training(img_path, mask_path, target_size, num_classes)
        Xs.extend(xs)
        Ys.extend(ys)
    X = np.array(Xs)[..., None].astype(np.float32)
    Y = tf.keras.utils.to_categorical(np.array(Ys), num_classes=num_classes)
    return X, Y


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

    # Identical split logic/seed to retrain_unet_5class.py -> same train/val patients
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

    pixel_counts = Y_train.sum(axis=(0, 1, 2))
    total = pixel_counts.sum()
    print("\nTraining-set pixel share per class:")
    for name, count in zip(CLASS_NAMES, pixel_counts):
        print(f"  {name}: {100*count/total:.3f}%")

    val_pixel_counts = Y_val.sum(axis=(0, 1, 2))
    print("\nVal-set pixel counts:")
    for name, count in zip(CLASS_NAMES, val_pixel_counts):
        print(f"  {name}: {int(count)} pixels")

    model = build_unet_multiclass_plain(num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    train_ds = make_augmented_dataset(X_train, Y_train, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, save_freq="epoch", verbose=0)
    dice_logger_cb = MeanForegroundDiceLogger(X_val, Y_val, print_every=5)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mean_fg_dice", mode="max", patience=patience, restore_best_weights=True)

    model.fit(
        train_ds, validation_data=(X_val, Y_val),
        epochs=epochs, verbose=2, callbacks=[checkpoint_cb, dice_logger_cb, early_stop_cb],
    )

    preds = model.predict(X_val, verbose=0)
    pred_onehot = tf.keras.utils.to_categorical(np.argmax(preds, axis=-1), num_classes=NUM_CLASSES)
    dice, iou = dice_iou_per_class(Y_val, pred_onehot, num_classes=NUM_CLASSES)
    print("\n=== Isotropic-resample + percentile-normalize 5-class model, full val labels ===")
    for name, d, i in zip(CLASS_NAMES, dice, iou):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    pred_3class = collapse_to_3class(pred_onehot)
    Y_val_3class = collapse_to_3class(Y_val)
    dice_3c, iou_3c = dice_iou_per_class(Y_val_3class, pred_3class, num_classes=3)
    print("\n=== Collapsed to 3-class (for comparison against retrain_unet_5class.py's own numbers) ===")
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
