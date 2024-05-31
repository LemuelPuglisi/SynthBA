import os
import argparse


DEVICE = 'cpu'
AVAILABLE_MODELS = {'u', 'g'}
AVAILABLE_TEMPLATES = {
    'T1w_1mm': 'MNI152_T1_1mm_Brain.nii.gz', 
    'T1w_2mm': 'MNI152_T1_2mm_Brain.nii.gz', 
    'T2w_1mm': 'MNI152_T2_1mm_Brain.nii.gz',
}


def str_available_models():
    tmp = ', '.join(AVAILABLE_MODELS)
    return f'[{tmp}]'

def str_available_templates():
    tmp = ', '.join(AVAILABLE_TEMPLATES)
    return f'[{tmp}]'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,      help='Input folder with nifti files.')
    parser.add_argument('-o', type=str, required=True,      help='Output folder where to store the predictions and (optionally) intermediate files')
    parser.add_argument('-c', type=str, required=True,      help='SynthBA checkpoints directory')
    parser.add_argument('-b', type=int, default=1,          help='Batch size')
    parser.add_argument('-m', type=str, default='g',        help=f'SynthBA model (available configurations = {str_available_models()})')
    parser.add_argument('-t', type=str, default='T1w_1mm',  help=f'Template for registration (available templates = {str_available_templates()})')
    return parser.parse_args()


def get_inputs(input_dir_path: str) -> list:
    is_nii = lambda f: f.endswith('.nii') or f.endswith('.nii.gz')
    files = os.listdir(input_dir_path)
    input_paths = [ os.path.join(input_dir_path, f) for f in files if is_nii(f) ]
    return input_paths


def preprocess(input_path, output_path, template_path):
    os.system(f'mri_synthstrip -i {input_path} -o {output_path}') # > /dev/null')
    reg_dirp = os.path.join(os.path.dirname(output_path), 'temp_')
    os.system(f'antsRegistrationSyNQuick.sh -d 3 -f {template_path} -m {output_path} -o {reg_dirp} -n {os.cpu_count()} -t a') #> /dev/null')
    reg_path = os.path.join(os.path.dirname(output_path), 'temp_Warped.nii.gz')
    rem_path = os.path.join(os.path.dirname(output_path), 'temp_*')
    os.system(f'mv {reg_path} {output_path}; rm {rem_path}')


if __name__ == '__main__':

    args = parse_args()    
    assert os.path.exists(args.i), 'Input dir does not exist.'
    assert os.path.exists(args.o), 'Output dir does not exist.'
    assert os.path.exists(args.c), 'Checkpoints dir does not exist.'
    assert args.m in AVAILABLE_MODELS, \
        'Invalid SynthBA model selected, please choose between: ' + str_available_models()
    assert args.t in AVAILABLE_TEMPLATES, \
        'Invalid template selected, please choose between: ' + str_available_templates()

    # Close the program if there are no images to process.
    input_paths = get_inputs(args.i)
    if len(input_paths) < 1: 
        print('No images to process. Exiting.')
        exit()

    # Check if the model weights exists
    model_path = os.path.join(args.c, f'synthba-{args.m}.pth')
    assert os.path.exists(model_path), f'Model not found at path {model_path}'

    # Select the template
    template_path = os.path.join('home', 'templates', AVAILABLE_TEMPLATES[args.t])

    # Loading the libraries only after checking
    # if all the arguments are correct.
    import torch
    from torch.utils.data import DataLoader
    from monai import transforms
    from monai.data import Dataset
    from monai.networks.nets.densenet import DenseNet201

    # setting up the folder that will contain the preprocessed images
    preprocess_output_path = os.path.join(args.o, 'preprocess_out')
    os.makedirs(preprocess_output_path, exist_ok=True)

    # preprocessing each input image
    preprocessed_paths = []
    for inp_path in input_paths:
        inp_fname = os.path.basename(inp_path)
        out_path = os.path.join(preprocess_output_path, inp_fname)
        preprocess(inp_path, out_path, template_path)
        preprocessed_paths.append(out_path)        

    # Load the model    
    model = DenseNet201(3, 1, 1, dropout_prob=0)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # Define the transforms
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'scan_path'}, names=['image']),
        transforms.LoadImageD(keys='image'),
        transforms.EnsureChannelFirstD(keys='image', channel_dim='no_channel'),
        transforms.SpacingD(keys='image', pixdim=1.4),
        transforms.ResizeWithPadOrCropD(keys='image', spatial_size=(130, 130, 130), mode='minimum'),
        transforms.ScaleIntensityD(keys='image', minv=0, maxv=1),
        transforms.Lambda(lambda d: d['image']),
    ])

    # prepare the dataset
    data = [ { 'scan_path': p } for p in preprocessed_paths ]
    dataset = Dataset(data=data, transform=transforms_fn)    
    loader = DataLoader(dataset=dataset, batch_size=args.b)
        
    # predict the brain age from all 
    # the preprocessed images.
    brain_age_list = []
    for i, images in enumerate(loader):
        print(f'processing batch n.{i}')
        with torch.no_grad():
            brain_ages = model(images.to(DEVICE))
            brain_age_list += list(brain_ages.view(-1).cpu().numpy() * 100)

    # save the predictions
    csv_path = os.path.join(args.o, 'predictions.csv')
    with open(csv_path, 'w') as csv:
        csv.write("input_id,brainage\n")
        for scan_path, brain_age in zip(preprocessed_paths, brain_age_list):
            csv.write(f"{scan_path},{brain_age}\n")