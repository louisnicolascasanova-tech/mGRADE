# UEA Dataset Integration

This document describes the integration of UEA (University of East Anglia) Time Series Archive datasets into the Den-minGRU training framework.

## Overview

Six multivariate time series classification datasets from the UEA archive have been integrated:

1. **EigenWorms** (`worms`) - 6 channels, 17984 timesteps, 5 classes
2. **SelfRegulationSCP1** (`scp1`) - 6 channels, 896 timesteps, 2 classes
3. **SelfRegulationSCP2** (`scp2`) - 7 channels, 1152 timesteps, 2 classes
4. **Heartbeat** (`heartbeat`) - 61 channels, 405 timesteps, 2 classes
5. **MotorImagery** (`motor`) - 64 channels, 3000 timesteps, 2 classes
6. **EthanolConcentration** (`ethanol`) - 3 channels, 1751 timesteps, 4 classes

## Data Location

Preprocessed data should be located at:
```
data_dir/processed/UEA/{dataset_name}/
├── data.pkl      # Shape: (n_samples, seq_length, n_channels)
└── labels.pkl    # Shape: (n_samples,)
```

## Usage

### Command Line Training

```bash
# Train on EigenWorms dataset
python main.py --dataset worms --gpu 0 --conv_mode dcls --file_nb 0

# Train on Heartbeat dataset
python main.py --dataset heartbeat --gpu 0 --conv_mode dcls --file_nb 0

# Train on MotorImagery dataset
python main.py --dataset motor --gpu 0 --conv_mode dcls --file_nb 0
```

### Available Dataset Names

| Command Line Arg | Full Dataset Name         | Shorthand |
|------------------|---------------------------|-----------|
| `worms`          | EigenWorms                | worms     |
| `scp1`           | SelfRegulationSCP1        | scp1      |
| `scp2`           | SelfRegulationSCP2        | scp2      |
| `heartbeat`      | Heartbeat                 | heartbeat |
| `motor`          | MotorImagery              | motor     |
| `ethanol`        | EthanolConcentration      | ethanol   |

### Python API

```python
from utils import create_uea_classification_dataset
import jax.numpy as jnp

# Create dataloaders
trainloader, valloader, testloader, n_classes, seq_len, in_dim = \
    create_uea_classification_dataset(
        dataset_name='EigenWorms',
        bsz=32,
        data_dir='./data_dir',
        dtype=jnp.float32,
        seed=42
    )
```

## Implementation Details

### Data Loading Function

Located in [utils.py:776-864](utils.py#L776-L864)

**Key Features:**
- Loads preprocessed pickle files
- Automatic train/val/test splitting (70%/15%/15%)
- Seed-based reproducibility for splits
- Custom collate function for JAX array conversion
- Support for both float16 and float32 dtypes
- Returns same format as vision datasets (MNIST, CIFAR)

### Main Script Integration

Located in [main.py:82-107](main.py#L82-L107)

**Changes Made:**
1. Added import for `create_uea_classification_dataset`
2. Added lambda wrappers in `dataset_fns` dictionary
3. Updated data loading logic to handle UEA datasets
4. Added UEA dataset choices to argument parser

### Dataset Characteristics

| Dataset   | Samples | Seq Length | Channels | Classes | Type              |
|-----------|---------|------------|----------|---------|-------------------|
| Worms     | 236     | 17,984     | 6        | 5       | Motion tracking   |
| SCP1      | 268     | 896        | 6        | 2       | Physiological     |
| SCP2      | 200     | 1,152      | 7        | 2       | Physiological     |
| Heartbeat | 204     | 405        | 61       | 2       | ECG signals       |
| Motor     | 278     | 3,000      | 64       | 2       | EEG signals       |
| Ethanol   | 500     | 1,751      | 3        | 4       | Chemical sensors  |

## Data Preprocessing

The UEA datasets are expected to be preprocessed into pickle format. Example preprocessing script structure:

```python
import pickle
import numpy as np
from pathlib import Path

# Load raw data (format depends on source)
data = load_raw_uea_data(dataset_name)  # Shape: (n_samples, seq_len, n_channels)
labels = load_raw_uea_labels(dataset_name)  # Shape: (n_samples,)

# Create output directory
output_dir = Path('data_dir/processed/UEA') / dataset_name
output_dir.mkdir(parents=True, exist_ok=True)

# Save as pickle
with open(output_dir / 'data.pkl', 'wb') as f:
    pickle.dump(data, f)
with open(output_dir / 'labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
```

## Testing

### Test Script

A test script is provided at [test_dataset_loader.py](test_dataset_loader.py)

```bash
# Activate conda environment
conda activate .jax_conda_env_pcRNN

# Test a specific dataset
python test_dataset_loader.py --dataset EigenWorms --batch_size 32

# Custom data directory
python test_dataset_loader.py --dataset Heartbeat --data_dir ./custom_data_dir
```

### Expected Output

```
[*] Generating EigenWorms Classification Dataset...
    Dataset size: 236 samples
    Sequence length: 17984
    Input dimension: 6
    Number of classes: 5

[*] Dataset Information:
    Number of classes: 5
    Sequence length: 17984
    Input dimension: 6

[*] Dataloader Information:
    Train batches: 5
    Val batches: 1
    Test batches: 1

[*] Testing batch loading...
    Batch sequences shape: (32, 17984, 6)
    Batch labels shape: (32,)
    Sequences dtype: float32
    Labels dtype: int32

[*] EigenWorms dataloader test successful!
```

## Data Flow

```
UEA Dataset
    ↓
Pickle Files (data.pkl, labels.pkl)
    ↓
create_uea_classification_dataset()
    ↓
PyTorch DataLoader
    ↓
Custom Collate Function
    ↓
JAX Arrays (batch_x, batch_y)
    ↓
Model Training
```

## Notes

1. **Data Format:** UEA datasets follow the same format as vision datasets (MNIST, CIFAR, GSC), returning fixed-length sequences without padding masks.

2. **Reproducibility:** Use the `seed` parameter to ensure consistent train/val/test splits across runs.

3. **Memory Considerations:** Some datasets (e.g., EigenWorms with 17,984 timesteps) have very long sequences. Adjust batch size accordingly.

4. **Dtype Support:** Both float16 and float32 are supported. Use float16 for memory-constrained scenarios.

5. **Class Distribution:** Check class balance for each dataset. Some may benefit from class weighting (see ListOps example in main.py).

## Future Enhancements

Potential improvements:
- [ ] Support for pre-defined train/test splits from UEA archive
- [ ] Automatic data download and preprocessing
- [ ] Data augmentation for time series
- [ ] Support for additional UEA datasets
- [ ] Class weight computation for imbalanced datasets
- [ ] Variable-length sequence support (if needed)
