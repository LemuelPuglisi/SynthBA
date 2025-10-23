<p align="center">
<img src="assets/synthba-readme.png" alt="synthba" width="700"/>
</p>

<h1 align="center">SynthBA: Reliable Brain Age Estimation</h1>
<h4 align="center">Brain Age Estimation Across Multiple MRI Sequences and Resolutions</h4>

<p align="center">
<a href="https://arxiv.org/abs/2406.00365">Arxiv Paper</a> •
<a href="https://huggingface.co/spaces/lemuelpuglisi/synthba">Hugging Face Demo</a> •
<a href="https://huggingface.co/lemuelpuglisi/synthba">Hugging Face Models</a>
</p>

> [!NOTE]
> SynthBA has been peer-reviewed and accepted at IEEE MetroXRAINE 2024.

**SynthBA** (Synthetic Brain Age) is a deep learning model that predicts the biological age of the brain (i.e., **brain age**) from MRI scans. Thanks to domain randomization, it works reliably on scans of arbitrary contrast and resolution without needing retraining or harmonization.

This repository contains the official `pip`-installable package for SynthBA, which automatically handles preprocessing (skull stripping and alignment) and model inference.

## 🚀 Using `synthba` (Command-Line)

### 1. Installation

You can install SynthBA directly from PyPI:

```bash
pip install synthba
```

For the latest development version, you can install directly from this repository:

```bash
pip install git+https://github.com/lemuelpuglisi/synthba.git
```

### 2. Running Prediction

Once installed, you can use the `synthba` command in your terminal. All models and alignment templates are downloaded automatically from Hugging Face Hub on the first run.

The tool can accept a single scan, a folder of scans, or a CSV file as input.

#### Example 1: Running on a single scan

```bash
synthba -i /path/to/my_scan.nii.gz -o /path/to/output --device cuda
```

#### Example 2: Running on a directory of scans

This will process all `.nii` and `.nii.gz` files in the folder. We recommend using a larger batch size for faster processing if you have enough VRAM/RAM.

```bash
synthba -i /path/to/scans_folder -o /path/to/output --device cuda --batch_size 4
```

#### Example 3: Running on a CSV file

The CSV file should contain one absolute path to a NIfTI scan per line.

```bash
synthba -i /path/to/my_scan_list.csv -o /path/to/output --device cuda --batch_size 4
```

### 3. Output

The tool will generate two main outputs in your specified output directory:

1.  `synthba_predictions.csv`: A CSV file containing the paths to the processed scans and their predicted brain age.
2.  `preprocessed/`: A new folder containing the preprocessed (skull-stripped and aligned) scans. This folder is *only* created if preprocessing is enabled (the default).

### 4. Command-Line Options

Here is a full list of available options:

| Argument | Flag | Description |
| :--- | :--- | :--- |
| **Input Path** | `-i`, `--input` | **(Required)** Path to a single NIfTI scan, a directory of scans, or a `.csv` file listing scan paths. |
| **Output Directory** | `-o`, `--output_dir` | **(Required)** Path to a directory where results (`synthba_predictions.csv`) and preprocessed scans will be saved. |
| **Model Type** | `-m`, `--model_type` | The model variant to use. `g` (gaussian prior, best model) or `u` (uniform prior). `g` is recommended. |
| **Template** | `-t`, `--template` | The template for alignment, based on your scan's MR weighting. `t1` (default) or `t2`. |
| **Device** | `-d`, `--device` | The device to run on. `cuda` or `cpu`. If not set, it auto-detects CUDA. |
| **Batch Size** | `-b`, `--batch_size` | Number of scans to process in a single batch (default: 1). |
| **Skip Preprocessing** | `--skip-prep` | Use this flag if your scans are *already* skull-stripped and registered to MNI152 1mm space. |
| **Checkpoint** | `-c`, `--checkpoint` | Path to a custom, local model checkpoint (`.pth`) file. If not provided, the correct model is downloaded from Hugging Face. |

## 🐍 Using the Python Interface

You can also import and use the `SynthBA` class directly in your own Python scripts for more complex workflows.

### Initialization

First, import and initialize the `SynthBA` class.

```python
from synthba import SynthBA

# Initialize the model
# This will download the model checkpoint and templates if not already cached
sba = SynthBA(
    device='cuda',  # or 'cpu'
    model_type='g'  # 'g' (gaussian) or 'u' (uniform)
)
```

### `run()`: Predicting a single, loaded scan

The `run()` method is for a single `nibabel` image object.

```python
import nibabel as nib

# Load your scan as a nibabel object
scan_nii = nib.load('path/to/T1w_scan.nii.gz')

# Run prediction
# preprocess=True will apply skull-stripping and alignment
# mr_weighting='t1' tells the aligner to use the T1 template
brain_age = sba.run(
    scan=scan_nii, 
    preprocess=True, 
    mr_weighting='t1'
)

print(f"Predicted Brain Age: {brain_age:.2f} years")
```

### `run_multiple()`: Predicting a list of scan paths

The `run_multiple()` method is designed for batch processing a list of *file paths*. It returns a `pandas.DataFrame`.

**Note:** When `preprocess=True`, you **must** provide `preprocess_outdir` to specify where the processed scans should be saved.

```python
scan_paths = [
    '/data/sub-001/anat.nii.gz',
    '/data/sub-002/anat.nii.gz',
    '/data/sub-003/anat.nii.gz'
]

# Run prediction on the list of paths
results_df = sba.run_multiple(
    input_list=scan_paths,
    batch_size=2,
    preprocess=True,
    preprocess_outdir='/path/to/save/preprocessed_files/',
    mr_weighting='t1'
)

print(results_df)
#
#                          path       pred
# 0  /data/sub-001/anat.nii.gz  34.567890
# 1  /data/sub-002/anat.nii.gz  45.123456
# 2  /data/sub-003/anat.nii.gz  28.987654
```

-----

## 🐳 Legacy Docker Usage (Old Version)

For users who prefer the original Docker-based version, you can still pull and run the Docker image from [DockerHub](https://hub.docker.com/repository/docker/lemuelpuglisi/synthba/general).

```bash
docker run --rm --gpus all \
    -v /path/to/inputs:/home/inputs \
    -v /path/to/outputs:/home/outputs \
    lemuelpuglisi/synthba:latest
```

> [\!WARNING]
> The Docker version and the `pip` package are **not** the same. The Docker image may use older dependencies and does not support all features of the `pip` package (like CSV input or automatic device detection).

-----

## ❓ FAQ & Troubleshooting

1.  **Error: `Killed` (when using Docker)**

      * **Cause:** The Docker container ran out of memory. The preprocessing (SynthStrip) and inference steps can be RAM-intensive.
      * **Solution:** If using Docker Desktop (Mac or Windows), go to **Settings \> Resources** and increase the **Memory** allocated to Docker (e.g., to 16GB or more).

2.  **Error: Process crashes or is very slow during preprocessing.**

      * **Cause:** The skull-stripping model (`SynthStrip`) conforms the image to a 1mm space, which can consume a large amount of RAM (16GB+ is recommended).
      * **Solution:** If you are processing many subjects, use a smaller `--batch_size` (e.g., `-b 1`). If it still fails, you may need to run on a machine with more RAM.

3.  **Error (Python): `Exception: Please specify where to store the preprocessing output with preprocess_outdir`**

      * **Cause:** You called `sba.run_multiple(..., preprocess=True)` without specifying the `preprocess_outdir` argument.
      * **Solution:** You must provide a path to a directory where the processed files can be saved, e.g., `sba.run_multiple(..., preprocess_outdir='./preprocessed')`. The command-line tool handles this automatically.

## 🙏 Credits

SynthBA is built on top of several outstanding open-source projects:

  * **[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)**: For robust skull stripping.
  * **[ANTsPy](https://github.com/ANTsX/ANTsPy)**: For affine registration to the MNI template.
  * **[MONAI](https://monai.io/)**: For data transformations and the `DenseNet` architecture.
  * **[PyTorch](https://pytorch.org/)**: As the core deep learning framework.
  * **[Hugging Face Hub](https://huggingface.co/)**: For model and template hosting.


## 📜 Citing

If you use SynthBA for your research, please cite our paper:

```bibtex
@inproceedings{puglisi2024synthba,
  title={SynthBA: Reliable Brain Age Estimation Across Multiple MRI Sequences and Resolutions},
  author={Puglisi, Lemuel and Rondinella, Alessia and De Meo, Linda and Guarnera, Francesco and Battiato, Sebastiano and Rav{\`\i}, Daniele},
  booktitle={2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)},
  pages={555--560},
  year={2024},
  organization={IEEE}
}
```
