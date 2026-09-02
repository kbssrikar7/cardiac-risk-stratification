"""Attempts to fix the one open technical gap left in the 5-class
segmentation work: the no-reflow class (0.017% of training pixels, 10x
rarer than infarction) scored exactly 0.0 Dice/IoU in
retrain_unet_5class.py, because plain categorical cross-entropy barely
penalizes a model for never predicting a class that rare.

Fix: median-frequency class-weighted categorical cross-entropy (Badrinarayanan
et al., SegNet) - weight_c = median(class frequencies) / freq_c. Deliberately
NOT naive inverse-frequency weighting (weight_c = 1/freq_c): with no-reflow at
0.017% of pixels, that would give it a weight near background's frequency
ratio (~5700x), which risks the same gradient-explosion failure mode already
hit twice earlier in this project's U-Net retrain attempts (see
retrain_unet_patient_split.py's history). Median-frequency balancing keeps
weights in a much narrower, empirically safer range while still making rare
classes matter far more than they did unweighted.

Reuses everything from retrain_unet_5class.py except the loss function and
the model output path - same data loading, same patient-level split (same
RANDOM_STATE, so the exact same train/val patients), same architecture, same
augmentation - so the only variable between this run and that one is the
loss, making the comparison a real ablation, not a confound.

Result reported honestly either way in TECHNICAL_REPORT.md Section 9,
following this project's established practice for every retrain attempt.
"""
import numpy as np
import tensorflow as tf

from retrain_unet_5class import (
    ARTIFACT_DIR, CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, TRAIN_ROOT,
    build_unet_multiclass_plain, collapse_to_3class, dice_iou_per_class,
    find_nii_file, load_slices_for_patients, make_augmented_dataset,
)
import glob
import os

OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_5class_weighted.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_5class_weighted_checkpoint.h5")


def median_frequency_weights(pixel_counts):
    freqs = np.array(pixel_counts, dtype=np.float64) / np.sum(pixel_counts)
    median_freq = np.median(freqs)
    return median_freq / freqs


def weighted_categorical_crossentropy(class_weights):
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        per_pixel_weight = tf.reduce_sum(weights * y_true, axis=-1)
        unweighted_ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return unweighted_ce * per_pixel_weight

    return loss


class MeanForegroundDiceLogger(tf.keras.callbacks.Callback):
    """First run of this ablation used val_loss for EarlyStopping and found it
    a useless signal: val_loss sat flat (0.106-0.109) for 10+ epochs while
    per-class Dice swung wildly (myocardium 0.007 -> 0.220 -> 0.037 between
    consecutive logged epochs). restore_best_weights then landed on whichever
    epoch happened to have the lowest val_loss, which was NOT representative
    of that epoch's actual segmentation quality - a lottery, not a measurement.

    Fix: compute per-class Dice every epoch and write the mean over the
    foreground classes (cavity, myocardium, infarction, no-reflow - excluding
    the trivial, always-easy background class) into `logs` as
    'val_mean_fg_dice', so EarlyStopping can monitor a metric that actually
    reflects segmentation quality instead of a loss value dominated by a
    79x-weighted rare class over a handful of pixels.
    """

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


def main():
    triples = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        img = find_nii_file(folder, "Images")
        mask = find_nii_file(folder, "Contours")
        if img and mask:
            triples.append((os.path.basename(folder), img, mask))

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

    print("Loading train slices (5-class)...")
    X_train, Y_train = load_slices_for_patients(train_triples)
    print("Loading val slices (5-class)...")
    X_val, Y_val = load_slices_for_patients(val_triples)
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    pixel_counts = Y_train.sum(axis=(0, 1, 2))
    total = pixel_counts.sum()
    class_weights = median_frequency_weights(pixel_counts)
    print("\nTraining-set pixel share and median-frequency class weight:")
    for name, count, w in zip(CLASS_NAMES, pixel_counts, class_weights):
        print(f"  {name}: {100*count/total:.3f}%  ->  weight={w:.4f}")

    val_pixel_counts = Y_val.sum(axis=(0, 1, 2))
    print("\nVal-set pixel counts (sanity check: a 0.0 Dice on a class with 0 val "
          "pixels means the metric is undefined, not that the model failed):")
    for name, count in zip(CLASS_NAMES, val_pixel_counts):
        print(f"  {name}: {int(count)} pixels")
    if val_pixel_counts[CLASS_NAMES.index("no-reflow")] == 0:
        print("  WARNING: no-reflow has zero pixels in the val split - any Dice "
              "result below is uninformative regardless of what the model learned.")

    model = build_unet_multiclass_plain(num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(class_weights), metrics=["accuracy"])

    train_ds = make_augmented_dataset(X_train, Y_train, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, save_freq="epoch", verbose=0)
    dice_logger_cb = MeanForegroundDiceLogger(X_val, Y_val, print_every=5)
    # dice_logger_cb must run before early_stop_cb so 'val_mean_fg_dice' is in
    # `logs` by the time EarlyStopping inspects it in the same epoch-end pass.
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mean_fg_dice", mode="max", patience=8, restore_best_weights=True)

    model.fit(
        train_ds, validation_data=(X_val, Y_val),
        epochs=150, verbose=2, callbacks=[checkpoint_cb, dice_logger_cb, early_stop_cb],
    )

    preds = model.predict(X_val, verbose=0)
    pred_onehot = tf.keras.utils.to_categorical(np.argmax(preds, axis=-1), num_classes=NUM_CLASSES)
    dice, iou = dice_iou_per_class(Y_val, pred_onehot, num_classes=NUM_CLASSES)
    predicted_pixel_counts = pred_onehot.sum(axis=(0, 1, 2))
    print("\n=== Class-weighted 5-class model, full val labels ===")
    for name, d, i, pred_count in zip(CLASS_NAMES, dice, iou, predicted_pixel_counts):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}  predicted_pixels={int(pred_count)}")
    no_reflow_pred_count = predicted_pixel_counts[CLASS_NAMES.index("no-reflow")]
    if no_reflow_pred_count == 0:
        print("  -> The model never predicts no-reflow anywhere in the val set "
              "(not just 'predicts it in the wrong place').")
    else:
        print(f"  -> The model predicts no-reflow at {int(no_reflow_pred_count)} pixels "
              "somewhere, but with zero overlap with the true no-reflow pixels.")

    print("\n=== Comparison against the unweighted 5-class model (same val patients) ===")
    print("  (unweighted numbers are hardcoded literals from the prior recorded "
          "run in TECHNICAL_REPORT.md, not recomputed in this session - the "
          "identical RANDOM_STATE/split makes them directly comparable)")
    print("  class            unweighted Dice   weighted Dice   delta")
    unweighted_dice = {"background": 0.9974, "LV cavity": 0.8844, "myocardium": 0.7519, "infarction": 0.2751, "no-reflow": 0.0000}
    for name, d in zip(CLASS_NAMES, dice):
        prior = unweighted_dice[name]
        print(f"  {name:16s} {prior:.4f}            {d:.4f}          {d - prior:+.4f}")

    model.save(OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
