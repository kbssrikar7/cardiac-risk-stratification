"""Publication/patent push, step 1: extends segmentation from the original
3-class (background/cavity/myocardium) problem to the full 5-class EMIDEC
label set (+ infarction, + no-reflow / microvascular obstruction), which the
original pipeline discarded entirely (preprocess collapsed labels 3/4 to
background before training).

Infarct size and no-reflow extent are established real cardiac risk markers in
the literature - this is the highest-novelty, no-new-data-needed extension
available on this dataset, and it directly supersedes the earlier 3-class
retrain attempt (training/retrain_unet_patient_split.py), which only tried to
match the existing 3-class baseline. This one gives the retrain an actual
reason to exist: two new classes the original model cannot represent at all.

Reuses the patient-level split and the *fixed* paired image/mask augmentation
from retrain_unet_patient_split.py unchanged (both are class-count-agnostic).
Run locally, not Colab - see memory feedback_colab_long_running_jobs.
"""
import glob
import os

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "training/data"
TRAIN_ROOT = os.path.join(DATA_DIR, "emidec-dataset-1.0.1")
ARTIFACT_DIR = "training"
OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_5class.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_5class_checkpoint.h5")
RANDOM_STATE = 42
NUM_CLASSES = 5
CLASS_NAMES = ["background", "LV cavity", "myocardium", "infarction", "no-reflow"]


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
    return np.transpose(vol, (2, 1, 0))  # X,Y,Z


def load_slices_for_patients(triples_subset, target_size=(128, 128), num_classes=NUM_CLASSES):
    """Unlike retrain_unet_patient_split.py, does NOT collapse labels 3/4
    (infarction, no-reflow) to background - the whole point of this script."""
    Xs, Ys = [], []
    for pid, img_path, mask_path in triples_subset:
        vol = load_nii_volume(img_path).astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        mask_vol = load_nii_volume(mask_path).astype(np.int32)
        mask_vol = np.clip(mask_vol, 0, num_classes - 1)  # guard against any stray label values only
        for z in range(vol.shape[2]):
            sl = tf.image.resize(vol[:, :, z][..., None], target_size, method="bilinear").numpy().squeeze()
            m = tf.image.resize(mask_vol[:, :, z][..., None], target_size, method="nearest").numpy().squeeze().astype(np.int32)
            Xs.append(sl)
            Ys.append(m)
    X = np.array(Xs)[..., None].astype(np.float32)
    Y = tf.keras.utils.to_categorical(np.array(Ys), num_classes=num_classes)
    return X, Y


def build_unet_multiclass_plain(input_shape=(128, 128, 1), num_classes=NUM_CLASSES):
    """Same architecture as backend/app/ml.py::build_unet_multiclass, only the
    final layer's class count changes - everything upstream of it (including
    the c6_conv2_gradcam_target layer Grad-CAM keys off) is untouched, so
    Grad-CAM keeps working against this model without any changes there."""
    inputs = layers.Input(input_shape, name="input_layer_2")
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="c1_conv1")(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="c1_conv2")(c1)
    p1 = layers.MaxPooling2D((2, 2), name="p1")(c1)

    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="c2_conv1")(p1)
    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="c2_conv2")(c2)
    p2 = layers.MaxPooling2D((2, 2), name="p2")(c2)

    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="c3_conv1")(p2)
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="c3_conv2")(c3)
    p3 = layers.MaxPooling2D((2, 2), name="p3")(c3)

    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="bn_conv1")(p3)
    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="bn_conv2")(bn)

    u1 = layers.UpSampling2D((2, 2), name="u1")(bn)
    u1 = layers.concatenate([u1, c3], name="u1_concat")
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="c4_conv1")(u1)
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="c4_conv2")(c4)

    u2 = layers.UpSampling2D((2, 2), name="u2")(c4)
    u2 = layers.concatenate([u2, c2], name="u2_concat")
    c5 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="c5_conv1")(u2)
    c5 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="c5_conv2")(c5)

    u3 = layers.UpSampling2D((2, 2), name="u3")(c5)
    u3 = layers.concatenate([u3, c1], name="u3_concat")
    c6 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="c6_conv1")(u3)
    c6 = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="c6_conv2_gradcam_target")(c6)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax", name="final_output_layer")(c6)
    return models.Model(inputs, outputs)


def make_augmented_dataset(X, Y, batch_size=16, seed=RANDOM_STATE):
    """Identical to retrain_unet_patient_split.py's version (class-count
    agnostic - operates on however many one-hot channels Y has)."""
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

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(len(X), seed=seed).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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


def collapse_to_3class(onehot_5):
    """Collapses a 5-class one-hot array down to the original 3-class scheme
    (background, cavity, myocardium) by merging infarction+no-reflow into
    background, matching the ORIGINAL preprocessing exactly - this is only
    used to get an apples-to-apples number against the old 3-class baseline
    model, which is structurally incapable of predicting the other 2 classes."""
    bg = onehot_5[..., 0] + onehot_5[..., 3] + onehot_5[..., 4]
    cavity = onehot_5[..., 1]
    myo = onehot_5[..., 2]
    return np.stack([bg, cavity, myo], axis=-1)


def main():
    triples = []
    for folder in sorted(glob.glob(os.path.join(TRAIN_ROOT, "Case_*"))):
        if not os.path.isdir(folder):
            continue
        img = find_nii_file(folder, "Images")
        mask = find_nii_file(folder, "Contours")
        if img and mask:
            triples.append((os.path.basename(folder), img, mask))
    print(f"Found {len(triples)} training patients with both image+mask")

    # Same seed/logic as retrain_unet_patient_split.py -> identical train/val patient split
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

    # Class balance check - infarction/no-reflow are rare, worth knowing exact prevalence
    pixel_counts = Y_train.sum(axis=(0, 1, 2))
    total = pixel_counts.sum()
    print("\nTraining-set pixel share per class:")
    for name, count in zip(CLASS_NAMES, pixel_counts):
        print(f"  {name}: {100*count/total:.3f}%")

    # Baseline (3-class original model) on the SAME val patients, collapsed for fair comparison
    try:
        old_model = tf.keras.models.load_model("unet_multiclass.h5", compile=False)
    except TypeError:
        old_model = build_unet_multiclass_plain(num_classes=3)
        old_model.load_weights("unet_multiclass.h5")
    old_preds = old_model.predict(X_val, verbose=0)
    old_pred_onehot = tf.keras.utils.to_categorical(np.argmax(old_preds, axis=-1), num_classes=3)
    Y_val_3class = collapse_to_3class(Y_val)
    dice_old, iou_old = dice_iou_per_class(Y_val_3class, old_pred_onehot, num_classes=3)
    print("\n=== BASELINE (original 3-class model), collapsed val labels for fair comparison ===")
    for name, d, i in zip(["background", "LV cavity", "myocardium"], dice_old, iou_old):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    # New 5-class model
    new_model = build_unet_multiclass_plain(num_classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    train_ds = make_augmented_dataset(X_train, Y_train, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, save_freq="epoch", verbose=0)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    new_model.fit(
        train_ds, validation_data=(X_val, Y_val),
        epochs=150, verbose=2, callbacks=[checkpoint_cb, early_stop_cb],
    )

    new_preds = new_model.predict(X_val, verbose=0)
    new_pred_onehot = tf.keras.utils.to_categorical(np.argmax(new_preds, axis=-1), num_classes=NUM_CLASSES)
    dice_new, iou_new = dice_iou_per_class(Y_val, new_pred_onehot, num_classes=NUM_CLASSES)
    print("\n=== NEW 5-class model, full val labels ===")
    for name, d, i in zip(CLASS_NAMES, dice_new, iou_new):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    # Also collapse the new model's predictions to 3-class for the apples-to-apples comparison
    new_pred_3class = collapse_to_3class(new_pred_onehot)
    dice_new_3c, iou_new_3c = dice_iou_per_class(Y_val_3class, new_pred_3class, num_classes=3)
    print("\n=== Comparison against baseline (both collapsed to 3-class) ===")
    for name, do, io_, dn, in_ in zip(["background", "LV cavity", "myocardium"], dice_old, iou_old, dice_new_3c, iou_new_3c):
        print(f"  {name}: baseline Dice={do:.4f} IoU={io_:.4f}  |  new(collapsed) Dice={dn:.4f} IoU={in_:.4f}")

    print("\nNote: infarction/no-reflow have no baseline - the original model cannot represent those classes at all.")

    new_model.save(OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
