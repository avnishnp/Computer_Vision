# Neural Radiance Fields (NeRF) Implementation

This repository contains a **PyTorch implementation** of Neural Radiance Fields (NeRF), a neural network-based approach for generating novel views of complex 3D scenes from a sparse set of 2D images.

## Features
- **Training and Testing Pipelines**: Includes functions for training and rendering.
- **Positional Encoding**: Implements sinusoidal positional encoding.
- **Density & Color Estimation Networks**: Modular blocks for density and color estimation.
- **Ray Marching & Volume Rendering**: Includes functions for rendering rays through a volumetric scene.

## Requirements
- `torch`
- `tqdm`
- `numpy`
- `matplotlib`

## Usage
1. Prepare training and testing data stored as `training_data.pkl` and `testing_data.pkl`.
2. Run the training script using:
python nerf.py

## Model Architecture
- **Positional Encoding Layer**: Projects input coordinates into higher dimensional space using sine and cosine functions.
- **MLP Blocks**: Multiple fully-connected layers for density and color prediction.
- **Ray Rendering**: Volume rendering using sampled points along rays.

## Training
The training process is configured with:
- **Epochs**: `16`
- **Learning Rate**: `5e-4`
- **Batch Size**: `1024`
- **Optimizer**: `Adam`
- **Scheduler**: `MultiStepLR`

## Output
Images are saved in the `novel_views/` directory during training.

## References
- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**, Mildenhall et al.




