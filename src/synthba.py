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


def print_citation():
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

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', type=str, required=True,      help='Either a path to an input folder with input nifti files or to a csv listing the input paths (the latter won\'t work on Docker).')
    parser.add_argument('-o', type=str, required=True,      help='Path to output directory where to store the results.')
    parser.add_argument('-c', type=str, required=True,      help='SynthBA checkpoints directory')
    parser.add_argument('-b', type=int, default=1,          help='Batch size')
    parser.add_argument('-m', type=str, default='g',        help=f'SynthBA model (available configurations = {str_available_models()})')
    parser.add_argument('-t', type=str, default='T1w_1mm',  help=f'Template for registration (available templates = {str_available_templates()})')
    
    parser.add_argument('--templates-dir', type=str, default='/home/templates/',  help='Path to template directory')
    parser.add_argument('--skip-prep',     action='store_true', help='Skip all preprocessing steps.')
    return parser.parse_args()


def get_inputs(inputs_path: str) -> list:
    
    def get_inputs_from_dir(inputs_path: str) -> list:
        is_nii = lambda f: f.endswith('.nii') or f.endswith('.nii.gz')
        files = os.listdir(inputs_path)
        input_paths = [ os.path.join(inputs_path, f) for f in files if is_nii(f) ]
        return input_paths

    def get_inputs_from_csv(inputs_path: str) -> list:
        with open(inputs_path, 'r') as f:
            return [ p.strip() for p in f.readlines() ]
        
    return get_inputs_from_dir(inputs_path) if os.path.isdir(inputs_path) \
        else get_inputs_from_csv(inputs_path)
    

def preprocess(input_path, output_path, template_path):
    os.system(f'mri_synthstrip -i {input_path} -o {output_path}') # > /dev/null')
    reg_dirp = os.path.join(os.path.dirname(output_path), 'temp_')
    os.system(f'antsRegistrationSyNQuick.sh -d 3 -f {template_path} -m {output_path} -o {reg_dirp} -n {os.cpu_count()} -t a') #> /dev/null')
    reg_path = os.path.join(os.path.dirname(output_path), 'temp_Warped.nii.gz')
    rem_path = os.path.join(os.path.dirname(output_path), 'temp_*')
    os.system(f'mv {reg_path} {output_path}; rm {rem_path}')


if __name__ == '__main__':

    args = parse_args()

    if not os.path.exists(args.i):
        print('Input dir does not exist.')
        exit(1)
        
    if not os.path.exists(args.o):
        print('Output dir does not exist.')
        exit(1)
        
    if not os.path.isdir(args.o):
        print('Specified output dir it\'s not a directory.')
        exit(1)
        
    if not os.path.exists(args.c): 
        print('Checkpoints dir does not exist.')
        exit(1)
    
    if args.m not in AVAILABLE_MODELS:
        print('Invalid SynthBA model selected, please choose between: ' + str_available_models())
        exit(1)
        
    if not args.skip_prep and (not os.path.exists(args.templates_dir) or not os.path.isdir(args.templates_dir)):
        print('Invalid template directory. Download the correct `templates` directory from SynthBA\'s GitHub repo.')
        exit(1)
        
    if not args.skip_prep and args.t not in AVAILABLE_TEMPLATES:
        print('Invalid template selected, please choose between: ' + str_available_templates())
        exit(1)

    # Close the program if there are no images to process.
    input_paths = get_inputs(args.i)
    if len(input_paths) < 1: 
        print('No images to process. Exiting.')
        exit()

    print(f"""
==============================================================
    Device: {DEVICE}
    Selected Model: SynthBA-{args.m}
==============================================================
""")

    # Check if the model weights exists
    model_path = os.path.join(args.c, f'synthba-{args.m}.pth')
    assert os.path.exists(model_path), f'Model not found at path {model_path}'

    # Select the template
    template_path = os.path.join(args.templates_dir, AVAILABLE_TEMPLATES[args.t])

    # Loading the libraries only after checking
    # if all the arguments are correct.
    import torch
    from torch.utils.data import DataLoader
    from monai import transforms
    from monai.data import Dataset
    from monai.networks.nets.densenet import DenseNet201

    if not args.skip_prep:

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

    else:
        # Skip the preprocessing, assuming your MRIs are already
        # skull-stripped and properly aligned to the MNI space
        preprocessed_paths = input_paths

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
            
    print_citation()