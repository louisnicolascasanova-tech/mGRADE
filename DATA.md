# Data Loading and Model Input Pipeline

This document describes how data is loaded and fed to the model in the Den-minGRU project.

## Overview

The system supports multiple datasets through a unified pipeline that handles both fixed-length sequences (vision/audio tasks) and variable-length sequences (text tasks).

## Dataset Creation

### Supported Datasets ([main.py:82-104](main.py#L82-L104))

The system uses a dictionary of dataset creation functions:

**Vision/Audio Datasets:**
- `mnist`: MNIST handwritten digits (784 sequence length)
- `cifar`: CIFAR-10 grayscale images (1024 sequence length)
- `gsc`: Google Speech Commands v2 (16000 audio samples)

**LRA Benchmark Datasets:**
- `imdb`: IMDB movie reviews (variable length, max 4096)
- `listops`: ListOps compositional logic (variable length, max 2000)
- `path`: PathFinder-32 (1024 sequence length)
- `pathx`: PathFinder-128 (16384 sequence length)
- `aan`: AAN document retrieval (variable length, max 4000)

**UEA Time Series Archive Datasets:**
- `worms`: EigenWorms (multivariate time series)
- `scp1`: SelfRegulationSCP1 (physiological signals)
- `scp2`: SelfRegulationSCP2 (physiological signals)
- `heartbeat`: Heartbeat (ECG signals)
- `motor`: MotorImagery (EEG signals)
- `ethanol`: EthanolConcentration (chemical sensor data)

Each dataset function returns:
```python
trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM
```

## Data Loading by Dataset Type

### Vision/Audio Datasets (MNIST, CIFAR, GSC)

Located in [utils.py](utils.py):

**MNIST** ([utils.py:33-79](utils.py#L33-L79)):
- Uses `torchvision.datasets.MNIST`
- Normalization: mean=0.5, std=0.5
- Shape: `(batch_size, 784, 1)` for sequential version
- Custom collate function converts to NumPy arrays

**CIFAR-10 Grayscale** ([utils.py:82-124](utils.py#L82-L124)):
- Uses `torchvision.datasets.CIFAR10`
- Converts to grayscale
- Normalization: mean=122.6/255.0, std=61.0/255.0
- Shape: `(batch_size, 1024, 1)`

**Google Speech Commands** ([utils.py:712-773](utils.py#L712-L773)):
- Custom `SpeechCommandsDataset` class
- 35 classes (v2)
- Sample rate: 16000 Hz
- Audio waveforms padded/truncated to 16000 samples
- Shape: `(batch_size, 16000, 1)`

### LRA Datasets (IMDB, ListOps, AAN, Path, PathX)

**Common Pattern**:
- Uses `make_data_loader` helper ([utils.py:137-176](utils.py#L137-L176))
- Seed-based shuffling for reproducibility
- Returns batches with auxiliary data (sequence lengths)

**IMDB** ([utils.py:178-216](utils.py#L178-L216)):
- Vocabulary size: 135
- Max sequence length: 4096
- Binary classification (positive/negative reviews)

**ListOps** ([utils.py:218-266](utils.py#L218-L266)):
- Vocabulary size: 20
- Max sequence length: 2000
- 10-class classification
- Uses class weights for imbalanced data

**AAN** ([utils.py:268-321](utils.py#L268-L321)):
- Document retrieval task
- Variable vocabulary size
- Max sequence length: 4000
- Binary classification

**Path32/PathX** ([utils.py:491-545](utils.py#L491-L545)):
- PathFinder-32: 1024 sequence length
- PathFinder-128 (PathX): 16384 sequence length
- Binary classification (connected/disconnected paths)

**UEA Time Series Archive** ([utils.py:776-864](utils.py#L776-L864)):
- Multivariate time series classification
- Loads preprocessed data from pickle files in `data_dir/processed/UEA/{dataset_name}/`
- 70/15/15 train/val/test split with seed-based reproducibility
- Supports datasets: EigenWorms, SelfRegulationSCP1, SelfRegulationSCP2, Heartbeat, MotorImagery, EthanolConcentration
- Variable sequence lengths and input dimensions depending on dataset
- Custom collate function converts to JAX arrays with specified dtype

## Batch Preprocessing

### prep_batch Function ([utils.py:324-356](utils.py#L324-L356))

Handles conversion from PyTorch to JAX format for LRA datasets:

```python
def prep_batch(batch, seq_len, in_dim, dtype=jnp.float32):
    """
    Args:
        batch: (inputs, targets) or (inputs, targets, aux_data)
        seq_len: Maximum sequence length
        in_dim: Input dimension
        dtype: Target data type (float16 or float32)

    Returns:
        full_inputs: inputs or (inputs, lengths) for padded sequences
        targets: labels in JAX format
    """
```

**Key Operations:**
1. Extract inputs, targets, and optional sequence lengths
2. Convert PyTorch tensors to JAX arrays
3. Pad sequences to uniform length
4. Apply one-hot encoding if needed (when input shape doesn't match in_dim)
5. Bundle inputs with lengths for padded datasets

## Training Loop Data Flow

### run_epoch ([training.py:140-173](training.py#L140-L173))

```python
for batch in progress_bar:
    if len(batch) == 2:  # Pre-processed (MNIST, CIFAR, GSC)
        batch_x, batch_y = batch
    elif len(batch) == 3:  # Needs preprocessing (IMDB, ListOps, AAN, Path, PathX)
        batch_x, batch_y = prep_batch(batch, seq_len, in_dim, dtype=dtype)

    grads, loss, accuracy, aux_dict = apply_model(
        state, model, batch_x, batch_y, reg_factor, do_key, class_weights
    )
```

### Model Forward Pass ([training.py:48-93](training.py#L48-L93))

```python
def apply_model(state, model, x, y, reg_factor, do_key, class_weights, dtype=jnp.float32):
    """
    Args:
        x: Input data - can be:
           - JAX array (batch_size, seq_len, in_dim) for fixed-length
           - Tuple (inputs, lengths) for variable-length sequences
        y: Target labels (batch_size,)
        class_weights: Optional per-class weights for loss
    """
    def loss_fn(params):
        net_dyn, logits, monitor = model.apply(
            {'params': params},
            x.astype(dtype),  # Ensure correct dtype
            rngs={'dropout': do_key}
        )

        # Cross-entropy loss with optional class weighting
        one_hot = jax.nn.one_hot(y, model.out_dim, dtype=jnp.float32)
        batch_loss = optax.softmax_cross_entropy(
            logits=logits.astype(jnp.float32),
            labels=one_hot
        )

        if class_weights is not None:
            batch_loss = batch_loss * class_weights_jnp[y]

        # Regularization on hidden states
        reg = sum((layers[2]**2).mean() for layers in net_dyn)
        loss = jnp.mean(batch_loss) + reg_factor * reg
```

## Key Design Patterns

### 1. Data Type Handling ([main.py:74-76](main.py#L74-L76))

```python
dtype = getattr(args, 'dtype', 'float32')
dtype = jnp.float16 if dtype == 'float16' else jnp.float32
```

Data can be loaded and processed in either float16 or float32 for memory efficiency.

### 2. Padded Sequences ([utils.py:350-354](utils.py#L350-L354))

For variable-length sequences (IMDB, ListOps), inputs are bundled with lengths:

```python
if lengths is not None:
    lengths = np.asarray(lengths.numpy())
    full_inputs = (inputs.astype(dtype), lengths.astype(dtype))
else:
    full_inputs = inputs.astype(dtype)
```

The model handles this tuple format internally.

### 3. Automatic One-Hot Encoding ([utils.py:346-347](utils.py#L346-L347))

```python
if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
    inputs = one_hot(inputs, in_dim)
```

Automatically converts integer token IDs to one-hot vectors when needed.

### 4. Class Weighting ([main.py:114](main.py#L114))

```python
class_weights = compute_class_weights(trainloader, N_CLASSES) if args.dataset == 'listops' else None
```

ListOps uses class weighting to handle imbalanced data distribution.

### 5. Train/Val/Test Splits

**Vision/Audio datasets:**
- MNIST: 50k train / 10k val / 10k test
- CIFAR-10: 40k train / 10k val / 10k test
- GSC: Uses official train/validation/test splits from dataset

**LRA datasets:**
- Provide pre-defined splits in their data files
- No random splitting needed

## Custom Collate Functions

### Vision/Audio ([utils.py:60-65](utils.py#L60-L65), [utils.py:753-758](utils.py#L753-L758))

```python
def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))
    labels = np.array(transposed_data[1])
    images = np.array(transposed_data[0])
    return images, labels
```

Converts PyTorch batch format to NumPy arrays for JAX compatibility.

### LRA Datasets

Uses dataset object's internal `_collate_fn` method which handles:
- Variable-length sequences
- Padding masks
- Auxiliary data (sequence lengths)

## Data Format Summary

| Dataset | Input Shape | IN_DIM | SEQ_LENGTH | Format |
|---------|-------------|--------|------------|--------|
| MNIST | (B, 784, 1) | 1 | 784 | Fixed |
| CIFAR | (B, 1024, 1) | 1 | 1024 | Fixed |
| GSC | (B, 16000, 1) | 1 | 16000 | Fixed |
| IMDB | (B, ≤4096, 135) | 135 | 4096 | Padded |
| ListOps | (B, ≤2000, 20) | 20 | 2000 | Padded |
| AAN | (B, ≤4000, vocab) | vocab | 4000 | Padded |
| Path32 | (B, 1024, 2) | 2 | 1024 | Fixed |
| PathX | (B, 16384, 2) | 2 | 16384 | Fixed |
| Worms | (B, 17984, 6) | 6 | 17984 | Fixed |
| SCP1 | (B, 896, 6) | 6 | 896 | Fixed |
| SCP2 | (B, 1152, 7) | 7 | 1152 | Fixed |
| Heartbeat | (B, 405, 61) | 61 | 405 | Fixed |
| Motor | (B, 3000, 64) | 64 | 3000 | Fixed |
| Ethanol | (B, 1751, 3) | 3 | 1751 | Fixed |

**B** = batch size (configurable)

## Validation and Testing

Uses the same data flow but with `eval_model` instead of `apply_model`:

```python
def validate(state, model, testloader, seq_len, in_dim, out_dim, ...):
    for batch in testloader:
        if len(batch) == 2:
            inputs, labels = batch
        elif len(batch) == 3:
            inputs, labels = prep_batch(batch, seq_len, in_dim)

        loss, acc, predictions, max_probs, mean_confidence = eval_model(
            state, model, inputs, labels, out_dim
        )
```

No dropout is applied during validation (model created with `training=False`).
