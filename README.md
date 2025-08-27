# PlückeRF: A Line-based 3D Representation for Few-view Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-2506.03713-b31b1b.svg)](https://arxiv.org/abs/2506.03713)

## Overview

**PlückeRF** is a novel line-based 3D representation for few-view reconstruction that more effectively harnesses multi-view information.

### Key Contributions

- **PlückeRF Representation**: A structured, feature-augmented line-based 3D representation that connects pixel rays from input views
- **Enhanced Multi-view Information Sharing**: Preferential information sharing between nearby 3D locations and between 3D locations and nearby pixel rays  
- **Superior Performance**: Demonstrated improvements in reconstruction quality over equivalent triplane representations and state-of-the-art feedforward reconstruction methods

## Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended)

### Setup Instructions (Anaconda)

```bash
# Create and activate conda environment
conda create -p ./env python=3.10 -y
conda activate ./env

# Install dependencies
pip install -r requirements.txt
pip install -e .
```
 (tested August 21 2025)

## Dataset Setup

### ShapeNet-SRN Chairs Dataset

1. Download `srn_chairs.zip` from the [PixelNeRF data folder](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR)
2. Extract the dataset and organize the `train` and `val` samples in the *same* directory
3. Update the `root_dirs` parameter in your training configuration file to point to this directory

### Expected Directory Structure

```
dataset/
├── trainval/
│   ├── 1a6f615e8b1b5ae4dbbc9440457e303e/
│   ├── 1a74a83fa6d24b3cacd67ce2c72c02e/
│   └── ...
```

> **Note**: Training and validation splits are defined in the respective configuration files.
## Pretrained Model

Download our pretrained chairs model from [Hugging Face](https://huggingface.co/hugsam/PluckeRF/tree/main).
Download the entire `chairs_model` folder, as you will need to reference that folder when running inference with the model.

## Training

The model is trained from scratch using a multi-stage training approach. Update your configuration file to point to the correct dataset directory before training.

### Training Command

Update the directories and other parameters in the training config file (2-views-chairs-biased-100k-sample.yaml) before training. 
The accelerate-cluster.yaml is configured for a machine with 4 GPUs, adjust accordingly.

```bash
accelerate launch --config_file "./configs/accelerate-cluster.yaml" -m openlrm.launch train.lrm --config "./configs/2-views-chairs-biased-100k-sample.yaml"
```

The training follows a structured three-stage approach:

1. **Stage 1 (0-100k steps)**: Full image training without cropping to allow the model to learn complete object representations
2. **Stage 2**: Introduction of image cropping for detailed learning  
3. **Stage 3**: Incorporation of LPIPS loss for enhanced perceptual quality

> **Note**: The number of steps for stages 2 and 3 can be adjusted in the training configuration file. When transitioning to cropped training, the code needs to be restarted.

## Inference and Evaluation

Before running evaluation, you have two options: convert your own trained checkpoint using the conversion script below, or download and use our pretrained model from Hugging Face (see Pretrained Model section above).

### Model Conversion

Convert your trained checkpoint to the required format (not required if you downloaded our model):

```bash
python scripts/convert_hf.py --config ./configs/2-view-chairs-biased.yaml
```

This creates a `releases` folder in your experiments directory.

### Running Evaluation

1. Update `./configs/infer-dataset-metrics.yaml` to point to your test dataset directory
2. Run inference and evaluation:

```bash
python openlrm/launch.py infer.lrm --infer ./configs/infer-dataset-metrics.yaml model_name=./exps/releases/shapenet-chairs/2-view-chairs-biased/step_800000
```

For example if you use our pretrained model:

```bash
python openlrm/launch.py infer.lrm --infer ./configs/infer-dataset-metrics.yaml model_name=/path/to/chairs_model
```
This will create a new `evaluation_outputs` folder in the `chairs_model` folder with predictions for each sample in `chairs_test.json`

### Metrics Calculation

Calculate detailed metrics including PSNR, SSIM, and LPIPS:

```bash
# Update paths in the evaluation script to point to the evaluation outputs folder from the previous step
python scripts/pluckerf_evaluate_on_images.py
```

This generates a CSV file with per-view metrics and prints summary statistics for all views and extrapolated viewpoints.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@InProceedings{Bahrami_2025_CVPR,
    author    = {Bahrami, Sam and Campbell, Dylan},
    title     = {Pl\"{u}cker: A Line-based 3D Representation for Few-view Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {317-326}
}
```

This project builds upon [OpenLRM](https://github.com/3DTopia/OpenLRM). We maintain the original Apache 2.0 license from OpenLRM for all files that contain substantial portions of their original code. New contributions and modifications specific to PlückeRF are clearly identified at the top of the source files.
