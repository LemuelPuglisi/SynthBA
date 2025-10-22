from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset
from monai.networks.nets.densenet import DenseNet201
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from huggingface_hub import hf_hub_download



class SynthBA:
    """
    Predict the brain age from any contrast and resolution using the SynthBA model.
    """

    def __init__(
        self,
        device: str,
        checkpoint: Optional[str] = None,
        model_type: str = 'g'
    ):
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
    def run(self, scan: Union[Nifti1Image,Nifti2Image]) -> float:
        """
        """
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
