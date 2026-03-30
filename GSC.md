# Google Speech Commands Dataset Loading

This document explains how Google Speech Commands (GSC) dataset loading is implemented in the codebase.

## Overview

The Google Speech Commands dataset contains short audio recordings of spoken commands (like "yes", "no", "up", "down", etc.). The codebase supports loading this dataset for sequence classification tasks.

## Dataset Structure

- **Version**: Speech Commands v0.02 (35 classes)
- **Audio Format**: 16kHz WAV files, ~1 second duration
- **Classes**: 35 command words including digits (0-9), directions (up, down, left, right), and other commands
- **Data Split**: Uses official train/validation/test splits defined in text files

## Implementation Files

### 1. **sc.py** - Core Dataset Implementation
Contains the `SpeechCommands` class that handles:
- Audio file loading and preprocessing
- Train/validation/test splits based on official lists
- MFCC feature extraction (optional)
- Data caching for faster loading

Key components:
```python
class SpeechCommands(SequenceDataset):
    # Loads from validation_list.txt and testing_list.txt for splits
    # Processes WAV files into tensors
    # Supports MFCC preprocessing or raw audio
```

### 2. **utils.py** - Dataset Factory Function
`create_speechcommands35_classification_dataset()` function (lines 536-579):
- Creates train/val/test DataLoaders
- Sets up dataset parameters:
  - `N_CLASSES = 35`
  - `SEQ_LENGTH = 16000` (samples at 16kHz = 1 second)
  - `IN_DIM = 1` (raw audio)
- Supports multiple resolutions with subsampling rates

### 3. **Notebook Implementation** - Alternative Approach
The `tmp/gsc.ipynb` notebook contains a PyTorch-based implementation:
- Direct WAV file loading with torchaudio
- Custom dataset class with fallback audio loading (torchaudio → librosa → soundfile)
- Handles audio resampling and padding to fixed length
- Custom collate function for batch processing

## Data Flow

### Method 1: Using sc.py (LRA-style)
```python
from utils import create_speechcommands35_classification_dataset

# Create dataloaders
trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE = \
    create_speechcommands35_classification_dataset(bsz=32, seed=42)

# Data format: (batch_size, seq_length, input_dim)
# Where seq_length=16000, input_dim=1 for raw audio
```

### Method 2: Using Notebook Implementation
```python
# From tmp/gsc.ipynb
train_loader, val_loader, test_loader = get_speech_commands_dataloaders(
    root='./data',
    version='v2',
    batch_size=32,
    sample_rate=16000,
    max_length=16000
)

# Returns numpy arrays via custom collate function
# Shape: (batch_size, channels, samples) = (32, 1, 16000)
```

## Data Preprocessing

### Audio Processing Pipeline:
1. **Loading**: Load WAV files at original sample rate
2. **Resampling**: Convert to 16kHz if needed
3. **Mono Conversion**: Convert stereo to mono by averaging
4. **Length Normalization**: Pad or truncate to exactly 16000 samples (1 second)
5. **Optional Features**: Extract MFCC features or use raw waveform

### Dataset Splits:
- Uses official Google splits defined in:
  - `validation_list.txt`: Files for validation set
  - `testing_list.txt`: Files for test set  
  - Remaining files: Training set

## File Structure Requirements

Expected directory structure:
```
data/speech_commands_v2/
├── validation_list.txt
├── testing_list.txt
├── _background_noise_/
├── yes/
│   ├── 004ae714_nohash_0.wav
│   └── ...
├── no/
│   ├── 012c8314_nohash_0.wav
│   └── ...
└── [other command directories]/
```

## Integration with Main Training Loop

### For LRA-style Integration:
```python
# In main.py, add to dataset_fns:
dataset_fns = {
    'speechcommands': create_speechcommands35_classification_dataset,
    # ... other datasets
}

# Usage:
# python main.py --dataset speechcommands --batch_size 32
```

### Data Preparation for JAX Models:
The `prep_batch()` function in utils.py converts PyTorch tensors to JAX arrays:
- Handles padding to maximum sequence length
- Converts to one-hot encoding if needed
- Returns format expected by JAX/Flax models

## Key Parameters

- **Sequence Length**: 16000 samples (1 second at 16kHz)
- **Input Dimension**: 1 (raw audio) or feature dimension (if using MFCC)
- **Output Classes**: 35 command categories
- **Sample Rate**: 16kHz
- **Audio Format**: Single channel WAV files

## Error Handling

The notebook implementation includes robust error handling:
- Fallback audio loading: torchaudio → librosa → soundfile
- Handles different audio formats and sample rates
- Graceful padding/truncation for variable-length files

## Performance Considerations

- **Caching**: Processed datasets are cached to avoid repeated preprocessing
- **Multiprocessing**: DataLoaders support multiple workers for faster loading
- **Memory**: Raw audio requires more memory than MFCC features
- **Background Loading**: Supports asynchronous data loading during training