#!/usr/bin/env python

"""
SynthBA Command-Line Interface
Run brain age prediction from the terminal.

Example:
    python -m synthba.cli -i /path/to/scans -o /path/to/output --device cuda
"""

import os
import argparse
from typing import List

import torch

# --- Constants ---
AVAILABLE_TEMPLATES = ['t1', 't2']
AVAILABLE_MODELS = ['u', 'g']
AVAILABLE_DEVICES = ['cuda', 'cpu']
NIFTI_EXTENSIONS = ('.nii', '.nii.gz')


def print_citation():
    """Prints the citation information for SynthBA."""
    italic = "\033[3m"
    reset = "\033[0m"

    print(f"""
=============================================================================================
If you use SynthBA for your research, please cite:
=============================================================================================

{italic}Puglisi, Lemuel, et al. "SynthBA: Reliable Brain Age Estimation Across Multiple MRI  
Sequences and Resolutions." 2024 IEEE International Conference on Metrology for eXtended  
Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE). IEEE, 2024.{reset}

=============================================================================================
Thanks for using SynthBA.
""")


def get_input_paths(input_path: str) -> List[str]:
    """
    Validates and collects NIfTI file paths from the input argument.
    
    The input can be a single NIfTI file, a directory of NIfTI files,
    or a .csv file listing paths to NIfTI files.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Case 1: Input is a directory
    if os.path.isdir(input_path):
        is_nii = lambda f: f.endswith(NIFTI_EXTENSIONS)
        files = os.listdir(input_path)
        paths = [os.path.join(input_path, f) for f in files if is_nii(f)]
        if not paths:
            raise FileNotFoundError(f"No NIfTI files ({NIFTI_EXTENSIONS}) found in directory: {input_path}")
        return paths

    # Case 2: Input is a .csv file
    if input_path.endswith('.csv'):
        with open(input_path, 'r') as f:
            paths = [p.strip() for p in f.readlines() if p.strip()]
        if not paths:
            raise ValueError(f"No paths found in CSV file: {input_path}")
        # Validate paths from CSV
        for p in paths:
            if not os.path.exists(p) or not p.endswith(NIFTI_EXTENSIONS):
                raise FileNotFoundError(f"Invalid or non-NIfTI path in CSV: {p}")
        return paths

    # Case 3: Input is a single file
    if input_path.endswith(NIFTI_EXTENSIONS):
        return [input_path]

    # If none of the above
    raise ValueError(f"Input must be a NIfTI file ({NIFTI_EXTENSIONS}), a directory, or a .csv file.")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SynthBA: Reliable Brain Age Estimation Across Multiple MRI Sequences and Resolutions",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required Arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Path to one of the following:\n"
             "  - A single NIfTI scan (.nii, .nii.gz)\n"
             "  - A directory containing NIfTI scans\n"
             "  - A .csv file listing paths to NIfTI scans"
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help="Path to the output directory where results and\n"
             "(optionally) preprocessed files will be stored."
    )

    # Model & Preprocessing Arguments
    parser.add_argument(
        '-m', '--model_type',
        type=str,
        default='g',
        choices=AVAILABLE_MODELS,
        help=f"SynthBA model type to use. (default: 'g', available: {AVAILABLE_MODELS})"
    )
    parser.add_argument(
        '-t', '--template',
        type=str,
        default='t1',
        choices=AVAILABLE_TEMPLATES,
        help=f"Template for alignment based on MR weighting. (default: 't1', available: {AVAILABLE_TEMPLATES})"
    )
    parser.add_argument(
        '--skip-prep',
        action='store_true',
        help="Skip all preprocessing steps (skull stripping and alignment).\n"
             "Use this only if your inputs are already skull-stripped and\n"
             "aligned to the MNI template."
    )
    parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        default=None,
        help="Optional path to a local model checkpoint (.pth file).\n"
             "If not provided, the model is downloaded from Hugging Face Hub."
    )

    # Compute Arguments
    parser.add_argument(
        '-d', '--device',
        type=str,
        default=None,
        choices=AVAILABLE_DEVICES,
        help="Device to run on. (e.g., 'cuda' or 'cpu').\n"
             "If not provided, automatically detects 'cuda' if available, else 'cpu'."
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=1,
        help="Batch size for processing multiple scans. (default: 1)"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the CLI."""
    args = parse_args()

    # --- 1. Configure Logging & Device ---
    print("--- Starting SynthBA ---")

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: --device='cuda' was requested, but CUDA is not available. Falling back to 'cpu'.")
        device = 'cpu'

    # --- 2. Validate Inputs & Outputs ---
    try:
        input_paths = get_input_paths(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 3. Set up Preprocessing ---
    preprocess_outdir = None
    if not args.skip_prep:
        preprocess_outdir = os.path.join(args.output_dir, 'preprocessed')
        os.makedirs(preprocess_outdir, exist_ok=True)
        print(f"Preprocessed files will be saved to: {preprocess_outdir}")
    
    # --- 4. Print Summary ---
    print("\n==============================================================")
    print(f"  Configuration:")
    print(f"    - Input:             {args.input}")
    print(f"    - Output Directory:  {args.output_dir}")
    print(f"    - Total Scans:       {len(input_paths)}")
    print(f"    - Device:            {device.upper()}")
    print(f"    - Model Type:        synthba-{args.model_type}")
    print(f"    - Preprocessing:     {'SKIPPED' if args.skip_prep else 'ENABLED'}")
    if not args.skip_prep:
        print(f"    - Alignment Template: {args.template.upper()}")
    print(f"    - Batch Size:        {args.batch_size}")
    print("==============================================================\n")

    # --- 5. Load Model & Run ---
    try:
        # Import dynamically to ensure argparse is fast
        from .synthba import SynthBA
    except ImportError:
        print("\nError: Could not import SynthBA. Make sure you are running this as a module, e.g.:")
        print("python -m synthba.cli -i ... -o ...\n")
        exit(1)

    print("Initializing SynthBA model (this may take a moment for first download)...")
    sba = SynthBA(
        device=device,
        checkpoint=args.checkpoint,
        model_type=args.model_type
    )
    print("Model loaded successfully.")

    print(f"Running prediction on {len(input_paths)} scan(s)...")
    results_df = sba.run_multiple(
        input_list=input_paths,
        batch_size=args.batch_size,
        preprocess=not args.skip_prep,
        preprocess_outdir=preprocess_outdir,
        mr_weighting=args.template
    )

    # --- 6. Save Results & Cite ---
    output_csv_path = os.path.join(args.output_dir, 'synthba_predictions.csv')
    results_df.to_csv(output_csv_path, index=False)

    print(f"\n--- Prediction Complete ---")
    print(f"Results saved to: {output_csv_path}")
    
    print_citation()


if __name__ == '__main__':
    main()