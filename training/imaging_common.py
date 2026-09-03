"""Shared MRI preprocessing for the segmentation training scripts, extracted
because the naive version of this logic (global min-max normalize + raw
pixel resize to 128x128, ignoring physical voxel spacing) was duplicated
across training/retrain_unet_5class.py, training/retrain_unet_patient_split.py,
training/compute_infarct_features_test.py, and backend/app/ml.py (the last of
which keeps its own copy since Docker only ships backend/app/, not training/).

This module exists specifically to fix a real, evidenced generalization
failure (TECHNICAL_REPORT.md Section 11): naive pixel resize makes the
network implicitly assume "the heart fills the frame", which fails
completely (100% background at 1.0 confidence) on a real external scan whose
heart is a smaller, off-center feature in a wider field. nnU-Net's answer to
this class of problem is resampling to a common isotropic voxel spacing
before any pixel-grid resizing, since a CNN has no native notion of physical
scale otherwise.

Deviation from nnU-Net's default (documented deliberately, not an oversight):
this project's models consume independent 2D slices, not 3D volumes. EMIDEC's
acquisition is highly anisotropic (~1.46mm in-plane, 10mm through-plane) -
resampling the through-plane (Z) axis to match would synthesize slices
between real acquired ones via interpolation, which helps nothing for a
per-slice 2D pipeline and risks a batch of interpolation artifacts for no
benefit. So only the IN-PLANE (X, Y) spacing is resampled to a common value;
Z is left untouched and each slice is still handled independently, exactly as
before. The target in-plane spacing (1.4583mm) is the median across the
100-patient training cohort, matching nnU-Net's "resample to the dataset's
median spacing" convention - chosen so the primary training distribution
needs minimal interpolation, while any other input (different native
resolution/FOV) gets correctly rescaled to the same physical scale before it
reaches the network.
"""
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

TARGET_INPLANE_SPACING_MM = 1.4583333730697632  # median across the 100-patient training cohort


def resample_inplane(image: sitk.Image, target_spacing_xy: float = TARGET_INPLANE_SPACING_MM,
                      interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resamples only the X/Y axes to target_spacing_xy; Z spacing and slice
    count are left exactly as acquired. Use sitk.sitkLinear for image
    intensities, sitk.sitkNearestNeighbor for integer label masks."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_spacing = (target_spacing_xy, target_spacing_xy, orig_spacing[2])
    new_size = [
        max(1, int(round(orig_size[0] * orig_spacing[0] / target_spacing_xy))),
        max(1, int(round(orig_size[1] * orig_spacing[1] / target_spacing_xy))),
        orig_size[2],
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def percentile_normalize(arr: np.ndarray, low: float = 10.0, high: float = 90.0) -> np.ndarray:
    """Clips to the [low, high] percentile range then rescales to [0, 1].
    Replaces global min-max normalization, which a single outlier-intensity
    voxel can dominate (observed directly on a real external scan, Section 10
    "Case_148" - the segmentation ring still localized correctly there, but
    the normalized background was visibly noisier than it should have been).
    """
    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def load_and_resample_volume(path: str, target_spacing_xy: float = TARGET_INPLANE_SPACING_MM,
                              interpolator=sitk.sitkLinear) -> np.ndarray:
    """Reads a NIfTI file, resamples in-plane to target_spacing_xy, and
    returns an (X, Y, Z) numpy array - the same axis order load_nii_volume()
    returns elsewhere in this project, so downstream per-slice code needs no
    changes beyond calling this instead of the naive loader."""
    itk_img = sitk.ReadImage(path)
    resampled = resample_inplane(itk_img, target_spacing_xy, interpolator)
    vol = sitk.GetArrayFromImage(resampled)  # Z, Y, X
    return np.transpose(vol, (2, 1, 0))  # X, Y, Z


def preprocess_volume_for_unet(path: str, target_size=(128, 128),
                                target_spacing_xy: float = TARGET_INPLANE_SPACING_MM,
                                percentile_low: float = 10.0, percentile_high: float = 90.0) -> np.ndarray:
    """Inference-time preprocessing: isotropic in-plane resample -> percentile
    normalize -> per-slice resize to target_size. Returns (n_slices, H, W, 1),
    matching preprocess_volume_for_unet's existing return shape elsewhere in
    this project (backend/app/ml.py, training/compute_infarct_features_test.py).
    percentile_low/high default to 10/90 (Section 12's validated 5-class
    result); pass tighter bounds (e.g. 1/99) to preserve more dynamic range -
    Section 12's 3-class candidate found 10/90 cost real myocardium Dice."""
    vol = load_and_resample_volume(path, target_spacing_xy, interpolator=sitk.sitkLinear).astype(np.float32)
    vol = percentile_normalize(vol, percentile_low, percentile_high)
    slices = []
    for i in range(vol.shape[2]):
        sl = tf.image.resize(vol[:, :, i][..., None], target_size, method="bilinear").numpy().squeeze()
        slices.append(sl)
    return np.array(slices)[..., None]


def load_paired_slices_for_training(image_path: str, mask_path: str, target_size=(128, 128),
                                     num_classes: int = 5,
                                     target_spacing_xy: float = TARGET_INPLANE_SPACING_MM,
                                     percentile_low: float = 10.0, percentile_high: float = 90.0):
    """Training-time preprocessing for one patient: resamples image AND mask
    to the identical in-plane grid (linear for the image, nearest-neighbor
    for the integer mask, so label values stay exact), then percentile-
    normalizes the image and per-slice resizes both to target_size. Returns
    (X, Y) as per-slice numpy arrays - X is (n_slices, H, W) float32, Y is
    (n_slices, H, W) int32 label maps (not yet one-hot, matching this
    project's existing load_slices_for_patients() call sites which one-hot
    encode after concatenating across patients)."""
    itk_img = sitk.ReadImage(image_path)
    itk_mask = sitk.ReadImage(mask_path)

    resampled_img = resample_inplane(itk_img, target_spacing_xy, interpolator=sitk.sitkLinear)
    resampled_mask = resample_inplane(itk_mask, target_spacing_xy, interpolator=sitk.sitkNearestNeighbor)

    vol = np.transpose(sitk.GetArrayFromImage(resampled_img), (2, 1, 0)).astype(np.float32)
    mask_vol = np.transpose(sitk.GetArrayFromImage(resampled_mask), (2, 1, 0)).astype(np.int32)
    mask_vol = np.clip(mask_vol, 0, num_classes - 1)

    vol = percentile_normalize(vol, percentile_low, percentile_high)

    Xs, Ys = [], []
    for z in range(vol.shape[2]):
        sl = tf.image.resize(vol[:, :, z][..., None], target_size, method="bilinear").numpy().squeeze()
        m = tf.image.resize(mask_vol[:, :, z][..., None], target_size, method="nearest").numpy().squeeze().astype(np.int32)
        Xs.append(sl)
        Ys.append(m)
    return Xs, Ys
