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
from synthba import skull_stripping


class SynthBA:
    """
    Predict the brain age from any contrast and resolution using the SynthBA model.
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
        """
        if checkpoint is None and model_type not in ['u', 'g']:
            raise Exception('SynthBA `model_type` should be either `u` or `g`')
        
        self.device = device
        self.hf_fname = f'synthba-{model_type}.pth'

        if  checkpoint is None:
            checkpoint = hf_hub_download(repo_id='lemuelpuglisi/synthba', filename=self.hf_fname)

        checkpoint = torch.load(checkpoint, map_location=device)
        self.model = DenseNet201(3, 1, 1, dropout_prob=0)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device).eval()

        # Prepare the preprocessing methods. We allow users to provide different functions using composition.
        self.skull_stripping = SkullStripping(self.device) if skull_stripping is None else skull_stripping
        self.align_fn = align if align_fn is None else align_fn

        # download the templates. Input scans will be aligned to one of these depeding on their weighting
        self.templates = {
            't1': nib.load(hf_hub_download(repo_id='lemuelpuglisi/synthba', filename='MNI152_T1_1mm_Brain.nii.gz')),
            't2': nib.load(hf_hub_download(repo_id='lemuelpuglisi/synthba', filename='MNI152_T2_1mm_Brain.nii.gz'))
        }

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
        scan: Union[Nifti1Image,Nifti2Image],
        preprocess: bool = True,
        mr_weighting: str = 't1'
    ) -> float:
        """
        """
        if mr_weighting not in self.templates.keys():
            raise Exception('Uknown weighting. Please select between:' + ','.join(self.supported_weightings))

        if preprocess:
            mri = self.skull_stripping.run(mri)[0]
            mri = self.align_fn(mri, self.templates[mr_weighting])

        scan_tensor = self.transforms_fn_direct(scan.get_fdata())
        scan_tensor = scan_tensor.unsqueeze(0).to(self.device).float()  
        return self.model(scan_tensor).view(-1).item() * 100.
        

    @torch.inference_mode()
    def run_multiple(self, input_list: List[str], batch_size: int = 1) -> pd.DataFrame:
        """
        """
        data    = [{ 'scan_path': p } for p in input_list]
        dataset = Dataset(data=data, transform=self.transforms_fn_multiple)    
        loader  = DataLoader(dataset=dataset, batch_size=batch_size)

        brain_age_list = []
        for _, images in enumerate(loader):
            brain_ages = self.model(images.to(self.device))
            brain_age_list += list(brain_ages.view(-1).cpu().numpy() * 100.)

        data = [{'path': p, 'pred': y} for p,y in zip(input_list, brain_age_list)]
        return pd.DataFrame(data)
