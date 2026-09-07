#!/usr/bin/env python3
"""
Standalone CLI inference script for the ISLES 2022 ischemic stroke lesion
segmentation model (nnU-Net v2, 3d_fullres, DWI+ADC+FLAIR).

This wraps `nnUNetv2_predict` so a new case can be segmented without opening
the training notebook — useful for integrating the model into an external
pipeline or simply running a quick prediction from the command line.

Example
-------
    python predict.py \\
        --dwi path/to/case_dwi.nii.gz \\
        --adc path/to/case_adc.nii.gz \\
        --flair path/to/case_flair.nii.gz \\
        --output path/to/output_dir \\
        --checkpoint path/to/nnUNet_results

Requirements
------------
    pip install nnunetv2 SimpleITK

The `--checkpoint` directory must contain the trained model in nnU-Net's
standard results layout, e.g.:
    nnUNet_results/Dataset003_ISLES22/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk


DATASET_ID = 3
CONFIGURATION = "3d_fullres"
FOLD = 0
TRAINER = "nnUNetTrainer_250epochs"
CHECKPOINT_NAME = "checkpoint_best.pth"


def resample_to_reference(moving_path: Path, reference_img: sitk.Image) -> sitk.Image:
    """Resample a moving image onto the grid of a reference image (linear interpolation)."""
    moving_img = sitk.ReadImage(str(moving_path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_img)


def prepare_case(dwi_path: Path, adc_path: Path, flair_path: Path, staging_dir: Path, case_name: str = "CASE") -> None:
    """
    Assemble a single case into nnU-Net's expected multi-channel input layout.
    Channel order matches training: 0000=DWI, 0001=ADC, 0002=FLAIR.
    ADC and FLAIR are resampled onto the DWI grid, matching the preprocessing
    used at training time (see notebooks/isles22_eda_train_inference.ipynb).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    dwi_img = sitk.ReadImage(str(dwi_path))
    sitk.WriteImage(dwi_img, str(staging_dir / f"{case_name}_0000.nii.gz"))

    adc_img = resample_to_reference(adc_path, dwi_img)
    sitk.WriteImage(adc_img, str(staging_dir / f"{case_name}_0001.nii.gz"))

    flair_img = resample_to_reference(flair_path, dwi_img)
    sitk.WriteImage(flair_img, str(staging_dir / f"{case_name}_0002.nii.gz"))


def run_inference(input_dir: Path, output_dir: Path, checkpoint_dir: Path) -> None:
    """Invoke nnUNetv2_predict as a subprocess against the prepared case."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_results"] = str(checkpoint_dir)
    # nnUNet_raw / nnUNet_preprocessed are not required for inference-only runs,
    # but nnU-Net expects the variables to exist.
    env.setdefault("nnUNet_raw", str(checkpoint_dir))
    env.setdefault("nnUNet_preprocessed", str(checkpoint_dir))

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(DATASET_ID),
        "-c", CONFIGURATION,
        "-f", str(FOLD),
        "-tr", TRAINER,
        "-chk", CHECKPOINT_NAME,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"nnUNetv2_predict failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run ISLES 2022 stroke lesion segmentation on a single case.")
    parser.add_argument("--dwi", required=True, type=Path, help="Path to the DWI NIfTI file.")
    parser.add_argument("--adc", required=True, type=Path, help="Path to the ADC NIfTI file.")
    parser.add_argument("--flair", required=True, type=Path, help="Path to the FLAIR NIfTI file.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write the predicted mask to.")
    parser.add_argument("--checkpoint", required=True, type=Path,
                         help="Path to the nnUNet_results directory containing the trained model.")
    parser.add_argument("--case-name", default="CASE", help="Case identifier used for the output filename.")
    parser.add_argument("--keep-staging", action="store_true",
                         help="Keep the intermediate resampled input files instead of deleting them after inference.")
    args = parser.parse_args()

    for path, label in [(args.dwi, "DWI"), (args.adc, "ADC"), (args.flair, "FLAIR")]:
        if not path.exists():
            sys.exit(f"Error: {label} file not found: {path}")
    if not args.checkpoint.exists():
        sys.exit(f"Error: checkpoint directory not found: {args.checkpoint}")

    staging_dir = Path(tempfile.mkdtemp(prefix="isles22_predict_"))
    try:
        print("Preparing multi-channel input (resampling ADC/FLAIR onto DWI grid)...")
        prepare_case(args.dwi, args.adc, args.flair, staging_dir, case_name=args.case_name)

        print("Running inference...")
        run_inference(staging_dir, args.output, args.checkpoint)

        predicted_mask = args.output / f"{args.case_name}.nii.gz"
        if predicted_mask.exists():
            print(f"\nDone. Predicted lesion mask saved to: {predicted_mask}")
        else:
            print("\nWarning: expected output file not found — check nnUNetv2_predict logs above.")
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            print(f"Staging files kept at: {staging_dir}")


if __name__ == "__main__":
    main()
