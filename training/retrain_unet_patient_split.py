"""Tier 1 #5/#6 + Tier 4 #11: retrains the U-Net segmentation model with a
patient-level train/val split (the original split slices instead, leaking
adjacent slices of the same patient across train/val) and data augmentation
(the original had none), then reports Dice/IoU per class (not just pixel
accuracy, which the original relied on and which masks myocardium
segmentation quality under background-pixel class imbalance).

Run locally rather than in Colab specifically because a Colab run of this same
job was lost mid-training to an idle-disconnect (see memory:
feedback_colab_long_running_jobs) - running locally in the background gets a
reliable completion signal with no such risk. Checkpoints every 5 epochs
regardless, as cheap extra insurance.

Infarction/no-reflow class preservation (Tier 4 #12) is intentionally deferred
- this script keeps the original 3-class target (background/cavity/myocardium).
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
OUT_MODEL = os.path.join(ARTIFACT_DIR, "unet_multiclass_v2.h5")
CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "unet_v2_checkpoint.h5")
RANDOM_STATE = 42


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


def load_slices_for_patients(triples_subset, target_size=(128, 128), num_classes=3):
    Xs, Ys = [], []
    for pid, img_path, mask_path in triples_subset:
        vol = load_nii_volume(img_path).astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        mask_vol = load_nii_volume(mask_path).astype(np.int32)
        mask_vol[mask_vol > 2] = 0
        for z in range(vol.shape[2]):
            sl = tf.image.resize(vol[:, :, z][..., None], target_size, method="bilinear").numpy().squeeze()
            m = tf.image.resize(mask_vol[:, :, z][..., None], target_size, method="nearest").numpy().squeeze().astype(np.int32)
            Xs.append(sl)
            Ys.append(m)
    X = np.array(Xs)[..., None].astype(np.float32)
    Y = tf.keras.utils.to_categorical(np.array(Ys), num_classes=num_classes)
    return X, Y


def build_unet_multiclass_plain(input_shape=(128, 128, 1), num_classes=3):
    """Matches backend/app/ml.py::build_unet_multiclass exactly (no augmentation
    layers) - used only to load the original unet_multiclass.h5's weights when
    load_model() can't deserialize its InputLayer config directly."""
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
    """Applies IDENTICAL random flip/rotation/translation to each (image, mask)
    pair by stacking them into one tensor before the geometric transform (so
    both see the exact same random draw), then splits them back apart.
    Contrast jitter is applied to the image only, after splitting, since it's
    a photometric change that must not touch the one-hot mask.

    This fixes a real bug from the first attempt at this: augmentation layers
    were applied only to the image inside the model, while the unmodified
    original mask was still used as the training target - training the model
    on flipped/rotated/translated images against *unflipped* masks. That
    collapsed myocardium Dice from 0.81 to ~0.0001 while pixel accuracy stayed
    high (98.4%), since myocardium is the thinnest, most spatially precise
    class and so the least tolerant of an image/mask misalignment - a very
    concrete demonstration of why pixel accuracy alone is misleading here.

    Second attempt's bug: `fill_mode="constant"` on the *combined* image+mask
    tensor fills rotated/translated border pixels with 0 across all 4
    channels - for the mask's 3 one-hot channels that produces an invalid
    all-zero target (no class at all, not a real one-hot label) at those
    border pixels, and unclamped RandomContrast could push image values
    outside a safe range on top of that. Together these caused the loss to
    diverge to the billions within a few epochs. Fixed by reflecting at
    borders (always copies real, valid one-hot values from elsewhere in the
    same image, never a degenerate label) and clamping the image to [0, 1]
    after contrast jitter.
    """
    flip = layers.RandomFlip("horizontal", seed=seed)
    rotate = layers.RandomRotation(0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    translate = layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", interpolation="nearest", seed=seed)
    contrast = layers.RandomContrast(0.1, seed=seed)

    def augment(image, mask):
        combined = tf.concat([image, mask], axis=-1)  # (128,128,4): 1 image channel + 3 one-hot mask channels
        combined = flip(combined, training=True)
        combined = rotate(combined, training=True)
        combined = translate(combined, training=True)
        image_aug, mask_aug = combined[..., :1], combined[..., 1:]
        image_aug = contrast(image_aug, training=True)
        image_aug = tf.clip_by_value(image_aug, 0.0, 1.0)
        return image_aug, mask_aug

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(len(X), seed=seed).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def dice_iou_per_class(y_true_onehot, y_pred_onehot, num_classes=3, eps=1e-7):
    dice_scores, iou_scores = [], []
    for c in range(num_classes):
        yt = y_true_onehot[..., c].astype(bool)
        yp = y_pred_onehot[..., c].astype(bool)
        intersection = np.logical_and(yt, yp).sum()
        union = np.logical_or(yt, yp).sum()
        dice_scores.append((2 * intersection + eps) / (yt.sum() + yp.sum() + eps))
        iou_scores.append((intersection + eps) / (union + eps))
    return dice_scores, iou_scores


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
    X_train, Y_train = load_slices_for_patients(train_triples)
    print("Loading val slices...")
    X_val, Y_val = load_slices_for_patients(val_triples)
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    # Baseline: existing unet_multiclass.h5 on this SAME patient-level val split (fair comparison).
    # Same fallback used elsewhere in this codebase (backend/app/ml.py::load_unet_model): the H5's
    # InputLayer config uses a newer Keras serialization ('batch_shape') this environment's Keras
    # can't deserialize directly, so rebuild the architecture and load weights instead.
    try:
        old_model = tf.keras.models.load_model("unet_multiclass.h5", compile=False)
    except TypeError:
        old_model = build_unet_multiclass_plain()
        old_model.load_weights("unet_multiclass.h5")
    old_preds = old_model.predict(X_val, verbose=0)
    old_pred_onehot = tf.keras.utils.to_categorical(np.argmax(old_preds, axis=-1), num_classes=3)
    dice_old, iou_old = dice_iou_per_class(Y_val, old_pred_onehot)
    class_names = ["background", "LV cavity", "myocardium"]
    print("\n=== BASELINE (existing unet_multiclass.h5) on patient-level held-out val set ===")
    for name, d, i in zip(class_names, dice_old, iou_old):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    new_model = build_unet_multiclass_plain()
    # clipnorm as cheap insurance against any remaining numerical instability from augmentation
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    train_ds = make_augmented_dataset(X_train, Y_train, batch_size=16)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH, save_best_only=False, save_freq="epoch", verbose=0
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    new_model.fit(
        train_ds, validation_data=(X_val, Y_val),
        epochs=120, verbose=2, callbacks=[checkpoint_cb, early_stop_cb],
    )

    new_preds = new_model.predict(X_val, verbose=0)
    new_pred_onehot = tf.keras.utils.to_categorical(np.argmax(new_preds, axis=-1), num_classes=3)
    dice_new, iou_new = dice_iou_per_class(Y_val, new_pred_onehot)
    print("\n=== NEW model (patient-split + augmentation) on the SAME held-out val set ===")
    for name, d, i in zip(class_names, dice_new, iou_new):
        print(f"  {name}: Dice={d:.4f}  IoU={i:.4f}")

    print("\n=== Comparison (myocardium is the class that matters clinically) ===")
    print(f"Myocardium Dice: baseline={dice_old[2]:.4f}  new={dice_new[2]:.4f}  delta={dice_new[2]-dice_old[2]:+.4f}")
    print(f"Myocardium IoU:  baseline={iou_old[2]:.4f}  new={iou_new[2]:.4f}  delta={iou_new[2]-iou_old[2]:+.4f}")

    new_model.save(OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
