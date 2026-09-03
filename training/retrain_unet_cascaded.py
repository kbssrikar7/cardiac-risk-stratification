"""Tier-2 improvement from the roadmap in TECHNICAL_REPORT.md Section 11:
a 3-stage cascaded segmentation architecture, following CaRe-CNN (Popescu et
al., MYOSAIQ challenge, arxiv 2312.11315) and the official EMIDEC challenge's
own best-result methodology - both replace a single-shot multi-class softmax
with a cascade that progressively narrows the region of interest:

  Stage 1: background vs LV-cavity vs myocardium (background, 1, {2,3,4} merged)
  Stage 2: within the myocardial region, healthy myocardium vs infarct ({2} vs {3,4})
  Stage 3: within the infarct region, infarct-only vs no-reflow/MVO ({3} vs {4})

Each stage is fed the original image concatenated with the previous stage's
prediction, so later stages only have to make an easier, more balanced
decision within an already-localized region - directly targeting the
class-imbalance-within-one-softmax structure a single 5-class U-Net is
structurally bad at (background is 96.98% of all pixels; no-reflow is
0.017%). retrain_unet_5class.py's single-shot model scored exactly 0.0 Dice
on no-reflow (Section 4.1); CaRe-CNN's cascade + paired-sampling combination
scored 72.0% Dice on the equivalent class on a larger (439-volume,
multi-center) cohort.

Design choices, stated explicitly since they are judgment calls, not the
paper's exact recipe:
- Training uses GROUND-TRUTH one-hot masks as the "previous stage's
  prediction" fed into each subsequent stage (teacher forcing), not that
  stage's own noisy predictions - standard practice for training cascades
  stably; the true, error-compounding cascade is only assembled at INFERENCE
  time (see `predict_cascade()`), which is what full-pipeline Dice is scored
  against.
- Stage 2 and 3 losses are masked to only the pixels within their stage's
  region of interest per ground truth (a pixel outside the myocardium
  contributes nothing to Stage 2's loss) - otherwise both stages would be
  trivially dominated by "predict nothing" on the vast majority of pixels
  that aren't even in their region, defeating the point of cascading.
- Stage 3 uses PAIRED per-batch sampling (at least one no-reflow-positive
  slice and one without, every batch) - CaRe-CNN's own technique for this
  exact class, explicitly NOT the class-weighted loss already tried and
  rejected in Section 9 (a mechanistically different intervention: this
  changes what the model sees per step, not how the loss is weighted).
- Uses the isotropic in-plane resampling + percentile normalization from
  training/imaging_common.py (Tier-1 fix), so any Dice difference from
  retrain_unet_5class.py is attributable to the architecture, and any
  difference from retrain_unet_5class_isotropic.py is attributable to the
  cascade - not a confound of both changing at once.

This is a hypothesis under test, not a proven fix - Section 9's own
conclusion was that no-reflow's failure is a data-scarcity problem (2,517
positive pixels, 40/100 patients) on THIS 100-patient cohort, versus
CaRe-CNN's 439-volume cohort. There is a real chance this reproduces
Section 9's negative outcome for no-reflow specifically even if it improves
infarction Dice. Reported honestly regardless of outcome, per this
project's established practice.
"""
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from retrain_unet_5class import ARTIFACT_DIR, RANDOM_STATE, TRAIN_ROOT, find_nii_file, make_augmented_dataset
from imaging_common import load_paired_slices_for_training

CLASS_NAMES_5 = ["background", "LV cavity", "myocardium", "infarction", "no-reflow"]
OUT_DIR = ARTIFACT_DIR
STAGE1_MODEL_PATH = os.path.join(OUT_DIR, "unet_cascaded_stage1.h5")
STAGE2_MODEL_PATH = os.path.join(OUT_DIR, "unet_cascaded_stage2.h5")
STAGE3_MODEL_PATH = os.path.join(OUT_DIR, "unet_cascaded_stage3.h5")


# -----------------------------
# Architecture: same encoder-decoder shape as build_unet_multiclass_plain,
# parameterized over input channel count (stages 2/3 take extra channels for
# the concatenated previous-stage prediction) and output class count.
# -----------------------------
def build_stage_unet(input_channels: int, num_classes: int, name_prefix: str):
    inputs = layers.Input((128, 128, input_channels), name=f"{name_prefix}_input")
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c1_conv1")(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c1_conv2")(c1)
    p1 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_p1")(c1)

    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c2_conv1")(p1)
    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c2_conv2")(c2)
    p2 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_p2")(c2)

    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c3_conv1")(p2)
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c3_conv2")(c3)
    p3 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_p3")(c3)

    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_bn_conv1")(p3)
    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_bn_conv2")(bn)

    u1 = layers.UpSampling2D((2, 2), name=f"{name_prefix}_u1")(bn)
    u1 = layers.concatenate([u1, c3], name=f"{name_prefix}_u1_concat")
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c4_conv1")(u1)
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c4_conv2")(c4)

    u2 = layers.UpSampling2D((2, 2), name=f"{name_prefix}_u2")(c4)
    u2 = layers.concatenate([u2, c2], name=f"{name_prefix}_u2_concat")
    c5 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c5_conv1")(u2)
    c5 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c5_conv2")(c5)

    u3 = layers.UpSampling2D((2, 2), name=f"{name_prefix}_u3")(c5)
    u3 = layers.concatenate([u3, c1], name=f"{name_prefix}_u3_concat")
    c6 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c6_conv1")(u3)
    c6 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name=f"{name_prefix}_c6_conv2")(c6)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax", name=f"{name_prefix}_output")(c6)
    return models.Model(inputs, outputs, name=name_prefix)


def masked_categorical_crossentropy(y_true_and_mask, y_pred):
    """y_true_and_mask has one extra trailing channel beyond the class
    one-hot: 1.0 where this pixel is inside the stage's region of interest
    (per ground truth), 0.0 elsewhere. Loss is averaged only over in-region
    pixels, so a stage's loss isn't dominated by the vast majority of pixels
    that aren't even relevant to its decision."""
    y_true = y_true_and_mask[..., :-1]
    mask = y_true_and_mask[..., -1]
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    ce = ce * mask
    return tf.reduce_sum(ce) / (tf.reduce_sum(mask) + 1e-7)


def masked_dice(y_true_onehot, y_pred_onehot, mask, class_idx, eps=1e-7):
    yt = (y_true_onehot[..., class_idx] * mask).astype(bool)
    yp = (y_pred_onehot[..., class_idx] * mask).astype(bool)
    intersection = np.logical_and(yt, yp).sum()
    return (2 * intersection + eps) / (yt.sum() + yp.sum() + eps)


# -----------------------------
# Label derivation from the raw 5-class ground truth (0=bg,1=cavity,2=myo,3=infarct,4=no-reflow)
# -----------------------------
def stage1_targets(mask5):
    """3-class: background, LV cavity, myocardium-total (myo+infarct+no-reflow merged,
    since infarct/no-reflow are labels WITHIN the myocardial wall, not separate anatomy)."""
    out = np.zeros_like(mask5)
    out[mask5 == 1] = 1
    out[(mask5 == 2) | (mask5 == 3) | (mask5 == 4)] = 2
    return out


def stage2_targets_and_mask(mask5):
    """2-class within the myocardial region: 0=healthy myocardium, 1=infarct(+no-reflow).
    Mask is 1 only where mask5 in {2,3,4}."""
    region = (mask5 == 2) | (mask5 == 3) | (mask5 == 4)
    target = np.where((mask5 == 3) | (mask5 == 4), 1, 0)
    return target, region.astype(np.float32)


def stage3_targets_and_mask(mask5):
    """2-class within the infarct region: 0=infarct-only, 1=no-reflow.
    Mask is 1 only where mask5 in {3,4}."""
    region = (mask5 == 3) | (mask5 == 4)
    target = np.where(mask5 == 4, 1, 0)
    return target, region.astype(np.float32)


def load_all_patients(triples):
    """Returns per-slice arrays: image (H,W), and the raw 5-class integer
    mask (H,W) - kept un-one-hotted here since each stage derives its own
    target/mask from it."""
    all_X, all_M = [], []
    for pid, img_path, mask_path in triples:
        xs, ys = load_paired_slices_for_training(img_path, mask_path, num_classes=5)
        all_X.extend(xs)
        all_M.extend(ys)
    return np.array(all_X, dtype=np.float32)[..., None], np.array(all_M, dtype=np.int32)


def make_stage_dataset(X, prev_onehot_inputs, Y_onehot_and_mask, batch_size=16, seed=RANDOM_STATE):
    """Same paired-augmentation shape as make_augmented_dataset, extended to
    carry the concatenated previous-stage-prediction channels through the
    same geometric transforms as the image and mask (so an augmented sample
    stays internally consistent - a flipped image must see a flipped
    previous-stage mask too)."""
    flip = layers.RandomFlip("horizontal", seed=seed)
    rotate = layers.RandomRotation(0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    translate = layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", interpolation="nearest", seed=seed)

    n_img_ch = X.shape[-1]
    n_prev_ch = prev_onehot_inputs.shape[-1] if prev_onehot_inputs is not None else 0
    n_target_ch = Y_onehot_and_mask.shape[-1]

    if prev_onehot_inputs is not None:
        combined_input = np.concatenate([X, prev_onehot_inputs], axis=-1)
    else:
        combined_input = X

    def augment(inp, target):
        combined = tf.concat([inp, target], axis=-1)
        combined = flip(combined, training=True)
        combined = rotate(combined, training=True)
        combined = translate(combined, training=True)
        inp_aug = combined[..., :n_img_ch + n_prev_ch]
        target_aug = combined[..., n_img_ch + n_prev_ch:]
        return inp_aug, target_aug

    ds = tf.data.Dataset.from_tensor_slices((combined_input, Y_onehot_and_mask))
    ds = ds.shuffle(len(X), seed=seed).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_stage3_paired_dataset(X, prev_onehot_inputs, Y_onehot_and_mask, no_reflow_positive_mask,
                                batch_size=16, seed=RANDOM_STATE):
    """CaRe-CNN's paired-sampling technique: every batch is built from two
    interleaved streams, one drawn only from slices WITH at least one
    no-reflow pixel and one drawn only from slices WITHOUT, so Stage 3 always
    gets balanced supervision instead of relying on whatever a uniform random
    batch happens to contain (which, at 0.018% prevalence, is usually zero
    positive examples per batch)."""
    combined_input = np.concatenate([X, prev_onehot_inputs], axis=-1) if prev_onehot_inputs is not None else X

    pos_idx = np.where(no_reflow_positive_mask)[0]
    neg_idx = np.where(~no_reflow_positive_mask)[0]
    print(f"  Stage 3 paired sampling: {len(pos_idx)} no-reflow-positive slices, {len(neg_idx)} negative slices")
    if len(pos_idx) == 0:
        print("  WARNING: zero no-reflow-positive slices in training set - paired sampling degrades to uniform random.")
        return make_stage_dataset(X, prev_onehot_inputs, Y_onehot_and_mask, batch_size, seed)

    half_batch = max(1, batch_size // 2)
    flip = layers.RandomFlip("horizontal", seed=seed)
    rotate = layers.RandomRotation(0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    translate = layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    n_ch_in = combined_input.shape[-1]

    def augment(inp, target):
        combined = tf.concat([inp, target], axis=-1)
        combined = flip(combined, training=True)
        combined = rotate(combined, training=True)
        combined = translate(combined, training=True)
        return combined[..., :n_ch_in], combined[..., n_ch_in:]

    pos_ds = tf.data.Dataset.from_tensor_slices((combined_input[pos_idx], Y_onehot_and_mask[pos_idx]))
    pos_ds = pos_ds.repeat().shuffle(max(len(pos_idx), 1), seed=seed).batch(half_batch)
    neg_ds = tf.data.Dataset.from_tensor_slices((combined_input[neg_idx], Y_onehot_and_mask[neg_idx]))
    neg_ds = neg_ds.repeat().shuffle(max(len(neg_idx), 1), seed=seed).batch(batch_size - half_batch)

    steps_per_epoch = max(1, len(neg_idx) // (batch_size - half_batch))

    def merge(pos_batch, neg_batch):
        inp = tf.concat([pos_batch[0], neg_batch[0]], axis=0)
        target = tf.concat([pos_batch[1], neg_batch[1]], axis=0)
        return augment(inp, target)

    ds = tf.data.Dataset.zip((pos_ds, neg_ds)).map(merge, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.take(steps_per_epoch).prefetch(tf.data.AUTOTUNE), steps_per_epoch


def predict_cascade(stage1, stage2, stage3, X):
    """Real (error-compounding) inference-time cascade: each stage's ACTUAL
    prediction (not ground truth) feeds the next, unlike training's teacher
    forcing. Returns a composed 5-class one-hot array for direct Dice
    comparison against every other model in the report."""
    n = X.shape[0]
    s1_pred = stage1.predict(X, verbose=0)
    s1_onehot = tf.keras.utils.to_categorical(np.argmax(s1_pred, axis=-1), num_classes=3)

    s2_input = np.concatenate([X, s1_onehot], axis=-1)
    s2_pred = stage2.predict(s2_input, verbose=0)
    s2_onehot = tf.keras.utils.to_categorical(np.argmax(s2_pred, axis=-1), num_classes=2)

    s3_input = np.concatenate([X, s1_onehot, s2_onehot], axis=-1)
    s3_pred = stage3.predict(s3_input, verbose=0)
    s3_onehot = tf.keras.utils.to_categorical(np.argmax(s3_pred, axis=-1), num_classes=2)

    final = np.zeros((n, 128, 128, 5), dtype=np.float32)
    s1_argmax = np.argmax(s1_pred, axis=-1)
    s2_argmax = np.argmax(s2_pred, axis=-1)
    s3_argmax = np.argmax(s3_pred, axis=-1)

    final[..., 0] = (s1_argmax == 0).astype(np.float32)
    final[..., 1] = (s1_argmax == 1).astype(np.float32)
    in_myo = s1_argmax == 2
    healthy_myo = in_myo & (s2_argmax == 0)
    infarct_region = in_myo & (s2_argmax == 1)
    final[..., 2] = healthy_myo.astype(np.float32)
    is_no_reflow = infarct_region & (s3_argmax == 1)
    is_infarct_only = infarct_region & (s3_argmax == 0)
    final[..., 3] = is_infarct_only.astype(np.float32)
    final[..., 4] = is_no_reflow.astype(np.float32)
    return final


def dice_iou_per_class(y_true_onehot, y_pred_onehot, num_classes, eps=1e-7):
    dice_scores, iou_scores = [], []
    for c in range(num_classes):
        yt = y_true_onehot[..., c].astype(bool)
        yp = y_pred_onehot[..., c].astype(bool)
        intersection = np.logical_and(yt, yp).sum()
        union = np.logical_or(yt, yp).sum()
        dice_scores.append((2 * intersection + eps) / (yt.sum() + yp.sum() + eps))
        iou_scores.append((intersection + eps) / (union + eps))
    return dice_scores, iou_scores


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

    print("Loading train slices...")
    X_train, M_train = load_all_patients(train_triples)
    print("Loading val slices...")
    X_val, M_val = load_all_patients(val_triples)
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    Y5_val_onehot = tf.keras.utils.to_categorical(M_val, num_classes=5)

    # ---- Stage 1: background / cavity / myocardium-total ----
    print("\n=== Training Stage 1 (background / cavity / myocardium-total) ===")
    s1_train = stage1_targets(M_train)
    s1_val = stage1_targets(M_val)
    Y1_train = tf.keras.utils.to_categorical(s1_train, num_classes=3)
    Y1_val = tf.keras.utils.to_categorical(s1_val, num_classes=3)

    stage1 = build_stage_unet(1, 3, "stage1")
    stage1.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="categorical_crossentropy", metrics=["accuracy"])
    train_ds1 = make_augmented_dataset(X_train, Y1_train, batch_size=16)
    early_stop1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    stage1.fit(train_ds1, validation_data=(X_val, Y1_val), epochs=epochs, verbose=2, callbacks=[early_stop1])
    stage1.save(STAGE1_MODEL_PATH)

    s1_pred_val = stage1.predict(X_val, verbose=0)
    s1_pred_val_onehot = tf.keras.utils.to_categorical(np.argmax(s1_pred_val, axis=-1), num_classes=3)
    dice1, _ = dice_iou_per_class(Y1_val, s1_pred_val_onehot, num_classes=3)
    print(f"Stage 1 val Dice: background={dice1[0]:.4f} cavity={dice1[1]:.4f} myocardium-total={dice1[2]:.4f}")

    # ---- Stage 2: healthy myocardium vs infarct(+no-reflow), within myocardium ----
    print("\n=== Training Stage 2 (healthy vs infarct, within myocardium) ===")
    s2_train, mask2_train = stage2_targets_and_mask(M_train)
    s2_val, mask2_val = stage2_targets_and_mask(M_val)
    Y2_train_onehot = tf.keras.utils.to_categorical(s2_train, num_classes=2)
    Y2_val_onehot = tf.keras.utils.to_categorical(s2_val, num_classes=2)
    Y2_train_and_mask = np.concatenate([Y2_train_onehot, mask2_train[..., None]], axis=-1)
    Y2_val_and_mask = np.concatenate([Y2_val_onehot, mask2_val[..., None]], axis=-1)

    stage2 = build_stage_unet(1 + 3, 2, "stage2")
    stage2.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=masked_categorical_crossentropy)
    train_ds2 = make_stage_dataset(X_train, Y1_train, Y2_train_and_mask, batch_size=16)
    val_input2 = np.concatenate([X_val, Y1_val], axis=-1)
    early_stop2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    stage2.fit(train_ds2, validation_data=(val_input2, Y2_val_and_mask), epochs=epochs, verbose=2, callbacks=[early_stop2])
    stage2.save(STAGE2_MODEL_PATH)

    s2_pred_val = stage2.predict(val_input2, verbose=0)
    s2_pred_val_onehot = tf.keras.utils.to_categorical(np.argmax(s2_pred_val, axis=-1), num_classes=2)
    dice2_healthy = masked_dice(Y2_val_onehot, s2_pred_val_onehot, mask2_val, 0)
    dice2_infarct = masked_dice(Y2_val_onehot, s2_pred_val_onehot, mask2_val, 1)
    print(f"Stage 2 val Dice (within myocardium): healthy={dice2_healthy:.4f} infarct(+no-reflow)={dice2_infarct:.4f}")

    # ---- Stage 3: infarct-only vs no-reflow, within infarct region, paired sampling ----
    print("\n=== Training Stage 3 (infarct-only vs no-reflow, within infarct, paired sampling) ===")
    s3_train, mask3_train = stage3_targets_and_mask(M_train)
    s3_val, mask3_val = stage3_targets_and_mask(M_val)
    Y3_train_onehot = tf.keras.utils.to_categorical(s3_train, num_classes=2)
    Y3_val_onehot = tf.keras.utils.to_categorical(s3_val, num_classes=2)
    Y3_train_and_mask = np.concatenate([Y3_train_onehot, mask3_train[..., None]], axis=-1)
    Y3_val_and_mask = np.concatenate([Y3_val_onehot, mask3_val[..., None]], axis=-1)

    no_reflow_positive = (M_train == 4).any(axis=(1, 2))

    stage3 = build_stage_unet(1 + 3 + 2, 2, "stage3")
    stage3.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=masked_categorical_crossentropy)
    prev_train_stage3 = np.concatenate([Y1_train, Y2_train_onehot], axis=-1)
    train_ds3, steps3 = make_stage3_paired_dataset(X_train, prev_train_stage3, Y3_train_and_mask, no_reflow_positive, batch_size=16)
    val_input3 = np.concatenate([X_val, Y1_val, Y2_val_onehot], axis=-1)
    early_stop3 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    stage3.fit(train_ds3, steps_per_epoch=steps3, validation_data=(val_input3, Y3_val_and_mask), epochs=epochs, verbose=2, callbacks=[early_stop3])
    stage3.save(STAGE3_MODEL_PATH)

    s3_pred_val = stage3.predict(val_input3, verbose=0)
    s3_pred_val_onehot = tf.keras.utils.to_categorical(np.argmax(s3_pred_val, axis=-1), num_classes=2)
    dice3_infarct_only = masked_dice(Y3_val_onehot, s3_pred_val_onehot, mask3_val, 0)
    dice3_no_reflow = masked_dice(Y3_val_onehot, s3_pred_val_onehot, mask3_val, 1)
    print(f"Stage 3 val Dice (within infarct region): infarct-only={dice3_infarct_only:.4f} no-reflow={dice3_no_reflow:.4f}")

    # ---- Full-pipeline cascade evaluation: real cascade (no teacher forcing), against true 5-class labels ----
    print("\n=== Full-pipeline cascade (real, error-compounding inference) vs true 5-class ground truth ===")
    final_pred = predict_cascade(stage1, stage2, stage3, X_val)
    dice_final, iou_final = dice_iou_per_class(Y5_val_onehot, final_pred, num_classes=5)
    for name, d, i in zip(CLASS_NAMES_5, dice_final, iou_final):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    print("\nDone. Compare this table directly against retrain_unet_5class.py's and "
          "retrain_unet_5class_isotropic.py's 'full val labels' tables - same val patients, same metric.")


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    if smoke:
        main(epochs=3, patience=2)
    else:
        main()
