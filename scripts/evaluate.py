#!/usr/bin/env python3
"""
Standalone CLI evaluation script for segmentation predictions against a
ground truth mask. Computes the standard metric set used in segmentation
challenges such as ISLES/BraTS: Dice, IoU, sensitivity, specificity,
precision, Hausdorff95, and volumetric similarity.

Example
-------
    python evaluate.py \\
        --prediction path/to/prediction.nii.gz \\
        --ground-truth path/to/ground_truth.nii.gz

Requirements
------------
    pip install nibabel numpy medpy
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from medpy import metric


def load_binary_mask(path: Path) -> tuple[np.ndarray, tuple]:
    """Load a NIfTI file as a binary (0/1) array, along with its voxel spacing in mm."""
    img = nib.load(str(path))
    arr = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    spacing = img.header.get_zooms()[:3]
    return arr, spacing


def compute_metrics(pred: np.ndarray, gt: np.ndarray, voxel_spacing: tuple) -> dict:
    """Compute the full segmentation metric set for a single case."""
    if gt.sum() == 0:
        raise ValueError(
            "Ground truth mask is empty (no lesion voxels) — Dice/Hausdorff95 are "
            "undefined for lesion-free cases. Evaluate only on lesion-positive cases."
        )

    result = {
        "dice": metric.binary.dc(pred, gt),
        "iou_jaccard": metric.binary.jc(pred, gt),
        "sensitivity_recall": metric.binary.sensitivity(pred, gt),
        "precision": metric.binary.precision(pred, gt) if pred.sum() > 0 else 0.0,
    }

    tn = np.sum((~pred.astype(bool)) & (~gt.astype(bool)))
    fp = np.sum(pred.astype(bool) & (~gt.astype(bool)))
    result["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    if pred.sum() > 0:
        try:
            result["hausdorff95_mm"] = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        except Exception:
            result["hausdorff95_mm"] = float("nan")
    else:
        result["hausdorff95_mm"] = float("nan")

    gt_vol, pred_vol = gt.sum(), pred.sum()
    result["volumetric_similarity"] = (
        1 - abs(pred_vol - gt_vol) / (pred_vol + gt_vol) if (pred_vol + gt_vol) > 0 else float("nan")
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation prediction against ground truth.")
    parser.add_argument("--prediction", required=True, type=Path, help="Path to the predicted mask (NIfTI).")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Path to the ground truth mask (NIfTI).")
    args = parser.parse_args()

    for path, label in [(args.prediction, "Prediction"), (args.ground_truth, "Ground truth")]:
        if not path.exists():
            sys.exit(f"Error: {label} file not found: {path}")

    pred, pred_spacing = load_binary_mask(args.prediction)
    gt, gt_spacing = load_binary_mask(args.ground_truth)

    if pred.shape != gt.shape:
        sys.exit(f"Error: shape mismatch — prediction {pred.shape} vs. ground truth {gt.shape}")

    try:
        metrics = compute_metrics(pred, gt, gt_spacing)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print(f"Prediction:   {args.prediction}")
    print(f"Ground truth: {args.ground_truth}")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name:24s}: {value:.4f}")


if __name__ == "__main__":
    main()
