![synthba](assets/synthba-readme.png)


<h4 align="center">SynthBA: Reliable Brain Age Estimation Across Multiple MRI Sequences and Resolutions</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2406.00365">Arxiv</a> •
  <a href="#usage">Usage</a> •
  <a href="#installation">Installation</a> •
  <a href="#citing">Cite</a>
</p>


> [!NOTE]  
> SynthBA has been peer-reviewed and accepted at IEEE MetroXRAINE 2024.


**NEWS**
* 🆕 Added support for providing inputs via a text file in non-Docker mode  
* 🆕 Added the `--skip-prep` flag to allow skipping preprocessing  
* 🆕 SynthBA is now available on [DockerHub](https://hub.docker.com/repository/docker/lemuelpuglisi/synthba/general)  



## Short description

SynthBA (Synthetic Brain Age) is a deep-learning model able to predict the biological age of the brain (referred to as **brain age**) through brain MRIs of arbitrary contrast and resolution. It follows the idea of using domain-randomization from the seminal work of [SynthSeg](https://github.com/BBillot/SynthSeg).

## Usage
Running SynthBA requires `docker` (see the [Docker installation page](https://docs.docker.com/engine/install/)) and nothing else. Once `docker` is installed, you can run the latest version of SynthBA (see [DockerHub](https://hub.docker.com/repository/docker/lemuelpuglisi/synthba/general)) in one command: 

```bash
./synthba --help
```

Place your MRIs (nifti format) in a folder (`/path/to/inputs`) and create a folder where to store the outputs (`/path/to/outputs`). Then run:

```
./synthba /path/to/inputs /path/to/outputs \
  -m <MODEL> -b <BATCHSIZE> -t <TEMPLATE>
```

Or directly using Docker:

```bash
docker run --rm \
    -v /path/to/inputs:/home/inputs \
    -v /path/to/outputs:/home/outputs \
    lemuelpuglisi/synthba:latest \
        -m <MODEL> -b <BATCHSIZE> -t <TEMPLATE>
```

Configure the batch size according to your available RAM. Templates are currently needed to align the scans to MNI space. We provide T1w and T2w templates, which should work for a wide range of MRI sequences. A list of `<MODEL>` and `<TEMPLATE>` options are provided below: 

| MODEL | Description | TEMPLATE | Description                   |
| ------- | ----------- | ---------- | ----------------------------- |
| `g`     | SynthBA-g   | `T1w_1mm`  | T1w 1mm brain MNI152 template |
| `u`     | SynthBA-u   | `T1w_2mm`  | T1w 2mm brain MNI152 template |
|         |             | `T2w_1mm`  | T2w 1mm brain MNI152 template |

The output folder will contain both the predicted brain age for each input (predictions.csv) and the preprocessed scans (in preprocessed).

> [!WARNING]  
> Before running SynthBA on your MRIs, you might need to manually increase the RAM limit applied to Docker containers from Docker Desktop (see [here](https://stackoverflow.com/questions/44417159/docker-process-killed-with-cryptic-killed-message)).



## Installation (Docker)

If you want to build the SynthBA's Docker image locally, run the following command at the root of the project:

```bash
docker build -t synthba .
```
Verify the installation by calling the `--help` option:

```
docker run --rm synthba --help
```

## Citing

Cite the preprint:

```
@misc{puglisi2024synthba,
      title={SynthBA: Reliable Brain Age Estimation Across Multiple MRI Sequences and Resolutions}, 
      author={Lemuel Puglisi and Alessia Rondinella and Linda De Meo and Francesco Guarnera and Sebastiano Battiato and Daniele Ravì},
      year={2024},
      eprint={2406.00365},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```




