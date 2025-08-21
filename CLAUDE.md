# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Den-MinGRU is a research project implementing minimal GRU networks with various architectural variations including DCLS (Delay Compensation Linear System), convolution layers, and channel mixing components. The project focuses on sequence modeling for computer vision tasks (MNIST, CIFAR) and NLP tasks (IMDB, ListOps, PathFinder).

## Core Architecture

### Main Components
- **Model Layers**: 
  - `HeinsenMinGeneralGRULayer`: Core recurrent layer with parallel computation using Heinsen's method
  - `DCLSLayer`: Delay compensation layer with Gaussian interpolation kernels and FFT support
  - `CausalDepthWiseConv1d`: Convolution layer with dilation support
  - `MLP` and `GLU`: Channel mixing components with configurable activations

- **Training Pipeline**: Located in `training.py` with JAX/Flax implementation
- **Data Loading**: Utilities in `utils.py` for MNIST, CIFAR, IMDB, ListOps, PathFinder32, and PathX datasets
- **Experiments**: YAML-based configuration system in `yaml_folder/`

### Key Files
- `main.py`: Main training script with command-line interface and WANDB integration
- `model.py`: Neural network architectures and layer definitions with comprehensive parameter control
- `training.py`: Training loops, optimization, validation functions, and extensive monitoring
- `utils.py`: Dataset loading, configuration parsing, experiment management, and utilities

## Development Commands

### Environment Setup
```bash
# Create conda environment and install dependencies
make install

# Activate environment (manual step)
conda activate .jax_conda_env_$(pwd)
```

### Running Experiments
```bash
# Basic training command
python main.py --dataset <dataset_name> --gpu <gpu_id> --conv_mode <conv_mode>

# Available datasets: mnist, cifar, imdb, listops, path, pathx
# Available conv_modes: dcls, rnn_eerf, rnn_lerf, vanilla, tcn_lerf, tcn_eerf
# GPU IDs: 0, 1, 2, 3

# Examples:
python main.py --dataset mnist --gpu 0 --conv_mode dcls
python main.py --dataset cifar --gpu 1 --conv_mode rnn_eerf --seed 42
python main.py --dataset listops --gpu 2 --conv_mode dcls --dcls_config 6
```

### Configuration System
- Experiments are defined in YAML files in `yaml_folder/`
- Naming convention: `{dataset}_{conv_mode}_{config_id}.yaml` or `{dataset}_{conv_mode}_{config_id}_wandb_{file_nb}.yaml`
- For DCLS mode, specify config with `--dcls_config` parameter
- WANDB integration for experiment tracking (requires `wandb_api_key.txt`)

### Sweep Mode
```bash
# Run hyperparameter sweeps with WANDB
python main.py --dataset mnist --conv_mode dcls --dcls_config 0 --file_nb 0 --sweep
```

### Resume Training
```bash
# Resume from checkpoint
python main.py --resume_from <checkpoint_path> --dataset <dataset> --gpu <gpu_id>
```

## Architecture Details

### Model Architecture (`BatchRNN_General`)
The main model is a flexible RNN backbone with multiple configurable blocks:

1. **Encoder Block**: Optional dense layer to project input to hidden dimension
2. **Sequence Processing Blocks** (per layer):
   - Convolution Block (DCLS/Causal Conv with dilation)
   - Recurrent Block (MinGRU with Heinsen's parallel method)
   - Channel Mixing Block (MLP/GLU)
   - Compression Block (dimensionality reduction)
3. **Output Block**: Dense layer for final predictions

### DCLS Layer Configuration
- **Delay Types**: 
  - `synaptic`: Output-dependent delays (kernel shape: [dim_out, dim_in, kernel_size])
  - `axonal`: Input-dependent delays (kernel shape: [dim_in, kernel_size])
  - `dendritic`: Pre-recurrent delays (skips recurrent dense layers)
- **Kernel Types**: Gaussian interpolation with configurable parameters
- **Heterogeneity**: Independent control over weights, positions, and standard deviations
- **FFT Support**: Fast convolution using FFT for efficiency

### Dilation Schedules (for convolution layers)
- **clip**: Dilation increases exponentially until boundary, then clips
- **wrap**: Dilation wraps back to initial value after boundary  
- **constant**: Fixed dilation factor across all layers

### Training Features
- **Advanced Optimization**: Multi-optimizer support with parameter-specific learning rates
- **Gradient Clipping**: Element-wise gradient clipping with configurable bounds
- **Comprehensive Monitoring**: WANDB logging for gradients, network dynamics, parameters
- **Early Stopping**: Patience-based stopping with dataset-specific criteria
- **Checkpoint Management**: Automatic best model saving and resuming

### Memory Management
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION` environment variable to control GPU memory usage
- Default values: 0.95 for sweeps, configurable per experiment via YAML

## Parameter Control

### DCLS Parameters
- `train_weights`, `train_positions`, `train_std`: Control which DCLS parameters are trainable
- `heterogeneous_*`: Enable parameter heterogeneity per kernel element
- `weight_init_scale`: Scale factor for weight initialization

### Optimization Control
- Layer-specific optimizers (MLP, GRU, LayerNorm parameters)
- Parameter-specific learning rates and optimizers
- Bias and scale parameter optimization control

### Initialization Options
- Multiple bias initialization schemes: zero, uniform gate init, constant
- Weight initialization scaling per layer type
- Position initialization: uniform or dilated patterns

## Dependencies
- JAX 0.4.25+ with CUDA support
- Flax 0.8.0+ for neural networks
- Optax 0.1.8+ for optimization
- PyTorch for data loading utilities
- WANDB for experiment tracking
- LRA datasets for long-range sequence tasks
- Scikit-learn for metrics
- Matplotlib/Seaborn for plotting
- Full dependency list in `requirements.txt`

## Experiment Organization
- Results saved in auto-generated directories with experiment IDs
- Checkpoints saved for best validation performance with backup checkpoints
- Training dynamics saved as `.npz` files
- Comprehensive WANDB logging for metrics, hyperparameters, and visualizations
- Configuration files saved with checkpoints for reproducibility