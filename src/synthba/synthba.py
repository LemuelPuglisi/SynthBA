import os
import hashlib
import re
from typing import List, Optional, Union, Callable

import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset
from monai.networks.nets.densenet import DenseNet201
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from huggingface_hub import hf_hub_download

from .registration import align
from .skull_stripping import SkullStripping


def _split_nii_name(filename: str) -> tuple:
    if filename.endswith('.nii.gz'):
        return filename[:-7], '.nii.gz'
    return os.path.splitext(filename)


def _sanitize_path_component(component: str) -> str:
    component = re.sub(r'[^A-Za-z0-9._-]+', '_', component).strip('._-')
    return component or 'path'


def _preprocessed_output_paths(input_list: List[str], preprocess_outdir: str) -> List[str]:
    """
    Build readable, unique output paths from the shortest distinguishing input suffix.

    For example, subject_a/visit_a/t1.nii.gz and subject_b/visit_a/t1.nii.gz
    become subject_a-visit_a-t1.nii.gz and subject_b-visit_a-t1.nii.gz.
    """
    if not input_list:
        return []

    path_parts = []
    for input_path in input_list:
        parts = [p for p in os.path.normpath(input_path).split(os.sep) if p]
        path_parts.append(parts or [os.path.basename(input_path)])

    max_depth = max(len(parts) for parts in path_parts)
    output_names = None
    for depth in range(1, max_depth + 1):
        candidate_names = []
        for parts in path_parts:
            suffix_parts = parts[-depth:]
            stem, suffix = _split_nii_name(suffix_parts[-1])
            name_parts = suffix_parts[:-1] + [stem]
            candidate_name = '-'.join(_sanitize_path_component(p) for p in name_parts) + suffix
            candidate_names.append(candidate_name)

        comparable_names = [name.lower() for name in candidate_names]
        if len(comparable_names) == len(set(comparable_names)):
            output_names = candidate_names
            break

    if output_names is None:
        output_names = []
        for index, (input_path, parts) in enumerate(zip(input_list, path_parts)):
            stem, suffix = _split_nii_name(parts[-1])
            digest = hashlib.sha1(f'{index}:{os.path.abspath(input_path)}'.encode()).hexdigest()[:8]
            output_names.append(f'{_sanitize_path_component(stem)}-{digest}{suffix}')

    return [os.path.join(preprocess_outdir, name) for name in output_names]


class SynthBA:
    """
    Predict the brain age from MRI scans of any contrast and resolution using the SynthBA model.
    """

    def __init__(
        self,
        device: str,
        checkpoint: Optional[str] = None,
        model_type: str = 'g',
        skull_stripping: Optional[SkullStripping] = None,
        align_fn: Optional[Callable] = None,
    ):
        """
        Initialize the SynthBA brain age prediction model.

        Args:
            device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
            checkpoint (Optional[str]): Path to a model checkpoint. If None, downloads from Hugging Face.
            model_type (str): Model variant to use ('u' or 'g').
            skull_stripping (Optional[SkullStripping]): Custom skull stripping method, if provided.
            align_fn (Optional[Callable]): Custom image alignment function, if provided.
        """
        if checkpoint is None and model_type not in ['u', 'g']:
            raise Exception('SynthBA `model_type` should be either `u` or `g`')
        
        self.device = device
        self.hf_fname = f'synthba-{model_type}.pth'

        if checkpoint is None:
            checkpoint = hf_hub_download(repo_id='lemuelpuglisi/synthba', filename=self.hf_fname)

        checkpoint = torch.load(checkpoint, map_location=device)
        self.model = DenseNet201(3, 1, 1, dropout_prob=0)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device).eval()

        # Preprocessing components
        self.skull_stripping = SkullStripping(self.device) if skull_stripping is None else skull_stripping
        self.align_fn = align if align_fn is None else align_fn

        # Download alignment templates for different MR weightings
        self.templates = {
            't1': nib.load(hf_hub_download(repo_id='lemuelpuglisi/synthba', filename='MNI152_T1_1mm_Brain.nii.gz')),
            't2': nib.load(hf_hub_download(repo_id='lemuelpuglisi/synthba', filename='MNI152_T2_1mm_Brain.nii.gz'))
        }

        # Image transformation pipelines
        self.transforms_fn_direct = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.Spacing(pixdim=1.4),
            transforms.ResizeWithPadOrCrop(spatial_size=(130, 130, 130), mode='minimum'),
            transforms.ScaleIntensity(minv=0, maxv=1),
        ])

        self.transforms_fn_multiple = transforms.Compose([
            transforms.CopyItemsD(keys={'scan_path'}, names=['image']),
            transforms.LoadImageD(keys='image'),
            transforms.EnsureChannelFirstD(keys='image', channel_dim='no_channel'),
            transforms.SpacingD(keys='image', pixdim=1.4),
            transforms.ResizeWithPadOrCropD(keys='image', spatial_size=(130, 130, 130), mode='minimum'),
            transforms.ScaleIntensityD(keys='image', minv=0, maxv=1),
            transforms.Lambda(lambda d: d['image']),
        ])


    @torch.inference_mode()
    def run(
        self, 
        scan: Union[Nifti1Image, Nifti2Image],
        preprocess: bool = True,
        mr_weighting: str = 't1'
    ) -> float:
        """
        Predict brain age for a single MRI scan.

        Args:
            scan (Nifti1Image | Nifti2Image): Input MRI scan.
            preprocess (bool): Whether to apply skull stripping and alignment.
            mr_weighting (str): MRI contrast type ('t1' or 't2') for template alignment.

        Returns:
            float: Predicted brain age in years.
        """
        if preprocess:
            scan = self._preprocess(scan, mr_weighting)
        scan_tensor = self.transforms_fn_direct(scan.get_fdata())
        scan_tensor = scan_tensor.unsqueeze(0).to(self.device).float()
        return self.model(scan_tensor).view(-1).item() * 100.
        

    @torch.inference_mode()
    def run_multiple(
        self, 
        input_list: List[str], 
        batch_size: int = 1,
        preprocess: bool = True, 
        preprocess_outdir: Optional[str] = None, 
        mr_weighting: str = 't1',
    ) -> pd.DataFrame:
        """
        Predict brain age for multiple MRI scans in batch.

        Args:
            input_list (List[str]): List of paths to input MRI scans.
            batch_size (int): Number of images processed per batch.
            preprocess (bool): Whether to apply skull stripping and alignment.
            preprocess_outdir (Optional[str]): Directory to save preprocessed images.
            mr_weighting (str): MRI contrast type ('t1' or 't2') for template alignment.

        Returns:
            pd.DataFrame: DataFrame with columns ['path', 'pred'] containing predicted brain ages.
        """
        if preprocess and preprocess_outdir is None:
            raise Exception('Please specify where to store the preprocessing output with preprocess_outdir')

        if preprocess:
            prep_input_list = [] 
            output_paths = _preprocessed_output_paths(input_list, preprocess_outdir)
            for inp_path, out_path in zip(input_list, output_paths):
                self._preprocess(nib.load(inp_path), mr_weighting).to_filename(out_path)
                prep_input_list.append(out_path)
            input_list = prep_input_list

        data = [{'scan_path': p} for p in input_list]
        dataset = Dataset(data=data, transform=self.transforms_fn_multiple)    
        loader = DataLoader(dataset=dataset, batch_size=batch_size)

        brain_age_list = []
        for _, images in enumerate(loader):
            brain_ages = self.model(images.to(self.device))
            brain_age_list += list(brain_ages.view(-1).cpu().numpy() * 100.)

        data = [{'path': p, 'pred': y} for p, y in zip(input_list, brain_age_list)]
        return pd.DataFrame(data)


    def _preprocess(self, scan: Union[Nifti1Image, Nifti2Image], mr_weighting: str):
        """
        Apply skull stripping and spatial alignment to an MRI scan.

        Args:
            scan (Nifti1Image | Nifti2Image): Input MRI scan.
            mr_weighting (str): MRI contrast type ('t1' or 't2') for selecting the reference template.

        Returns:
            Nifti1Image | Nifti2Image: Preprocessed and aligned MRI scan.
        """
        if mr_weighting not in self.templates.keys():
            raise Exception('Unknown weighting. Please select between: ' + ', '.join(self.templates.keys()))
        scan = self.skull_stripping.run(scan)[0]
        scan = self.align_fn(scan, self.templates[mr_weighting])
        return scan
