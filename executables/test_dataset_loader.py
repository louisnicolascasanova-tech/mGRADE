"""
Test script for UEA dataset dataloaders with CLI argument support
"""
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def create_uea_classification_dataset(dataset_name, bsz=128, data_dir="./data_dir"):
    """
    Create PyTorch dataloaders for any UEA dataset.

    Args:
        dataset_name: Name of the UEA dataset (e.g., 'EigenWorms', 'Heartbeat')
        bsz: Batch size
        data_dir: Root directory containing processed data

    Returns:
        trainloader, valloader, testloader, n_classes, seq_length, in_dim
    """
    print(f"[*] Generating {dataset_name} Classification Dataset...")

    # Load the processed data
    dataset_path = Path(data_dir) / "processed" / "UEA" / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    with open(dataset_path / "data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(dataset_path / "labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # Convert to numpy arrays if needed
    data = np.array(data)
    labels = np.array(labels)

    # Get dataset constants
    N_SAMPLES, SEQ_LENGTH, IN_DIM = data.shape
    N_CLASSES = len(np.unique(labels))

    print(f"    Dataset size: {N_SAMPLES} samples")
    print(f"    Sequence length: {SEQ_LENGTH}")
    print(f"    Input dimension: {IN_DIM}")
    print(f"    Number of classes: {N_CLASSES}")

    # Create a simple Dataset class
    class SimpleDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Split the dataset into train, val, test (70%, 15%, 15%)
    n_train = int(N_SAMPLES * 0.7)
    n_val = int(N_SAMPLES * 0.15)

    # Shuffle indices
    indices = np.random.permutation(N_SAMPLES)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Create datasets
    train = SimpleDataset(data[train_idx], labels[train_idx])
    val = SimpleDataset(data[val_idx], labels[val_idx])
    test = SimpleDataset(data[test_idx], labels[test_idx])

    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        labels = np.array(transposed_data[1])
        sequences = np.array(transposed_data[0])

        return sequences, labels

    # Return data loaders, with the provided batch size
    trainloader = DataLoader(
        train, batch_size=bsz, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
    )
    valloader = DataLoader(
        val, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )
    testloader = DataLoader(
        test, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def plot_sample(sequences, labels, dataset_name, sample_idx=0, save_path=None):
    """
    Plot all channels of a single sample from any UEA dataset.

    Args:
        sequences: Array of shape (batch_size, seq_length, n_channels)
        labels: Array of labels for each sample
        dataset_name: Name of the dataset (for title)
        sample_idx: Index of the sample to plot (default: 0)
        save_path: Path to save the plot (default: {dataset_name}_sample.png)
    """
    if save_path is None:
        save_path = f"{dataset_name.lower()}_sample.png"

    sample = sequences[sample_idx]  # Shape: (seq_length, n_channels)
    label = labels[sample_idx]
    seq_length, n_channels = sample.shape

    # Determine figure size based on number of channels
    # Allocate at least 2 inches per channel for readability
    fig_height = max(n_channels * 2, 10)

    # Adjust width based on sequence length for better aspect ratio
    if seq_length > 5000:
        fig_width = 16
    elif seq_length > 1000:
        fig_width = 14
    else:
        fig_width = 12

    # Create a figure with subplots for each channel
    fig, axes = plt.subplots(n_channels, 1, figsize=(fig_width, fig_height), sharex=True)
    fig.suptitle(f'{dataset_name} Sample (Label: {label})', fontsize=16, fontweight='bold', y=0.995)

    # Handle case where there's only one channel
    if n_channels == 1:
        axes = [axes]

    # Define channel names
    channel_names = [f'Ch {i+1}' for i in range(n_channels)]

    # Adjust font sizes based on number of channels
    ylabel_fontsize = max(6, min(10, 60 // n_channels))
    text_fontsize = max(5, min(8, 50 // n_channels))

    # Plot each channel
    for i in range(n_channels):
        axes[i].plot(sample[:, i], linewidth=0.6, color=f'C{i % 10}')  # Cycle through 10 colors
        axes[i].set_ylabel(channel_names[i], fontsize=ylabel_fontsize, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, seq_length)

        # Reduce tick label size for many channels
        if n_channels > 20:
            axes[i].tick_params(axis='y', labelsize=text_fontsize)

        # Add min/max values as text
        y_min, y_max = sample[:, i].min(), sample[:, i].max()
        axes[i].text(0.02, 0.95, f'min: {y_min:.2f}\nmax: {y_max:.2f}',
                    transform=axes[i].transAxes, fontsize=text_fontsize,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')

    # Adjust layout with minimal spacing
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test UEA dataset dataloader')
    parser.add_argument('--dataset', type=str, default='EigenWorms',
                        help='Name of the UEA dataset (default: EigenWorms)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--data_dir', type=str, default='./data_dir',
                        help='Data directory (default: ./data_dir)')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save the plot (default: {dataset}_sample.png)')

    args = parser.parse_args()

    # Create the dataloader
    trainloader, valloader, testloader, n_classes, seq_length, in_dim = \
        create_uea_classification_dataset(
            dataset_name=args.dataset,
            bsz=args.batch_size,
            data_dir=args.data_dir
        )

    print(f"\n[*] Dataset Information:")
    print(f"    Number of classes: {n_classes}")
    print(f"    Sequence length: {seq_length}")
    print(f"    Input dimension: {in_dim}")

    print(f"\n[*] Dataloader Information:")
    print(f"    Train batches: {len(trainloader)}")
    print(f"    Val batches: {len(valloader)}")
    print(f"    Test batches: {len(testloader)}")

    # Test loading a batch
    print(f"\n[*] Testing batch loading...")
    for sequences, labels in trainloader:
        print(f"    Batch sequences shape: {sequences.shape}")
        print(f"    Batch labels shape: {labels.shape}")
        print(f"    Sequences dtype: {sequences.dtype}")
        print(f"    Labels dtype: {labels.dtype}")
        print(f"    Sample label values: {labels[:5]}")

        # Plot the first sample
        print(f"\n[*] Plotting the first sample...")
        plot_sample(
            sequences,
            labels,
            dataset_name=args.dataset,
            sample_idx=0,
            save_path=args.save_plot
        )
        break

    print(f"\n[*] {args.dataset} dataloader test successful!")


if __name__ == "__main__":
    main()
