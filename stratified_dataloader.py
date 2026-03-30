"""
Stratified dataloader utilities for balanced batch sampling.
This ensures each batch has exactly 50% of each class.
"""
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, List


class StratifiedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has balanced class representation.

    For a binary classification task with batch_size=8, each batch will have
    exactly 4 samples from class 0 and 4 samples from class 1.

    Args:
        labels: Array or list of labels for all samples in the dataset
        batch_size: Size of each batch
        shuffle: Whether to shuffle within each class
        seed: Random seed for reproducibility
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(self, labels, batch_size: int, shuffle: bool = True,
                 seed: int = None, drop_last: bool = True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Get indices for each class
        self.class_indices = {}
        unique_classes = np.unique(self.labels)
        for cls in unique_classes:
            self.class_indices[cls] = np.where(self.labels == cls)[0].tolist()

        self.num_classes = len(unique_classes)

        # Calculate samples per class per batch
        if self.batch_size % self.num_classes != 0:
            print(f"Warning: batch_size ({self.batch_size}) is not evenly divisible by "
                  f"num_classes ({self.num_classes}). Class balance may not be exact.")

        self.samples_per_class = self.batch_size // self.num_classes

        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class

        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState()

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        class_iters = {}
        for cls, indices in self.class_indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                self.rng.shuffle(indices_copy)
            class_iters[cls] = iter(indices_copy)

        # Generate batches
        for _ in range(self.num_batches):
            batch = []
            for cls in sorted(self.class_indices.keys()):
                # Get samples_per_class samples from this class
                for _ in range(self.samples_per_class):
                    try:
                        batch.append(next(class_iters[cls]))
                    except StopIteration:
                        if not self.drop_last:
                            # Try to fill from other classes if available
                            continue
                        else:
                            return

            # Shuffle samples within the batch to avoid class ordering
            if self.shuffle:
                self.rng.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        return self.num_batches


def create_stratified_dataloader(dataset, labels, batch_size: int,
                                 shuffle: bool = True, seed: int = None,
                                 drop_last: bool = True, num_workers: int = 0):
    """
    Create a DataLoader with stratified batch sampling.

    Args:
        dataset: PyTorch Dataset object
        labels: Array or list of labels for all samples in the dataset
        batch_size: Size of each batch
        shuffle: Whether to shuffle samples within each class
        seed: Random seed for reproducibility
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader with stratified sampling
    """
    sampler = StratifiedBatchSampler(
        labels=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last
    )

    # When using a custom batch sampler, we need to set batch_size=1
    # and use batch_sampler parameter
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers
    )


def extract_labels_from_dataset(dataset):
    """
    Extract all labels from a PyTorch dataset.

    Args:
        dataset: PyTorch Dataset or Subset

    Returns:
        numpy array of labels
    """
    # Handle different dataset types
    if hasattr(dataset, 'tensors'):
        # TensorDataset
        return dataset.tensors[1].numpy()
    elif hasattr(dataset, 'targets'):
        # Common datasets like MNIST, CIFAR
        if isinstance(dataset.targets, torch.Tensor):
            return dataset.targets.numpy()
        else:
            return np.array(dataset.targets)
    elif hasattr(dataset, 'dataset'):
        # Subset
        labels = extract_labels_from_dataset(dataset.dataset)
        return labels[dataset.indices]
    else:
        # Fallback: iterate through dataset
        print("Warning: Extracting labels by iterating through dataset. This may be slow.")
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return np.array(labels)


# Example usage function
if __name__ == "__main__":
    from utils import create_lra_pathx_classification_dataset
    from collections import Counter

    print("Creating PathX dataset with stratified sampling...")

    # First, create datasets using the standard approach
    BATCH_SIZE = 8
    SEED = 42

    # Load using the original function to get the underlying datasets
    from lra import PathFinder
    from pathlib import Path

    name = 'pathfinder'
    resolution = 128
    dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'
    cache_dir = Path("./cache_dir/") / name

    dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
    dataset_obj.cache_dir = cache_dir
    dataset_obj.setup()

    # Extract datasets
    train_dataset = dataset_obj.dataset_train
    val_dataset = dataset_obj.dataset_val
    test_dataset = dataset_obj.dataset_test

    # Extract labels
    print("\nExtracting labels from datasets...")
    train_labels = train_dataset.tensors[1].numpy()
    val_labels = val_dataset.tensors[1].numpy()
    test_labels = test_dataset.tensors[1].numpy()

    print(f"Train dataset: {len(train_labels)} samples")
    print(f"Val dataset: {len(val_labels)} samples")
    print(f"Test dataset: {len(test_labels)} samples")

    # Create stratified dataloaders
    print("\nCreating stratified dataloaders...")
    train_loader = create_stratified_dataloader(
        train_dataset, train_labels,
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED, drop_last=True
    )
    val_loader = create_stratified_dataloader(
        val_dataset, val_labels,
        batch_size=BATCH_SIZE, shuffle=False, seed=SEED, drop_last=False
    )
    test_loader = create_stratified_dataloader(
        test_dataset, test_labels,
        batch_size=BATCH_SIZE, shuffle=False, seed=SEED, drop_last=False
    )

    print(f"\nNumber of batches:")
    print(f"  Train: {len(train_loader)}")
    print(f"  Val: {len(val_loader)}")
    print(f"  Test: {len(test_loader)}")

    # Verify stratification by checking first 20 batches
    print("\n" + "="*80)
    print("Verifying stratified sampling (first 20 batches):")
    print("="*80 + "\n")

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 20:
            break

        inputs, targets = batch
        targets = targets.numpy()

        class_counts = Counter(targets)
        print(f"Batch {batch_idx + 1}: ", end="")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            prop = count / len(targets)
            print(f"Class {int(cls)}: {count}/{len(targets)} ({prop:.1%})  ", end="")
        print()

    # Calculate overall statistics
    print("\n" + "="*80)
    print("Overall training set statistics (all batches):")
    print("="*80 + "\n")

    total_counts = Counter()
    perfect_balance_count = 0

    for batch in train_loader:
        inputs, targets = batch
        targets = targets.numpy()
        class_counts = Counter(targets)
        total_counts.update(class_counts)

        # Check if perfectly balanced
        if all(count == BATCH_SIZE // 2 for count in class_counts.values()):
            perfect_balance_count += 1

    total_samples = sum(total_counts.values())
    print(f"Total batches: {len(train_loader)}")
    print(f"Perfectly balanced batches: {perfect_balance_count}/{len(train_loader)} "
          f"({perfect_balance_count/len(train_loader):.1%})")
    print(f"\nOverall class distribution:")
    for cls in sorted(total_counts.keys()):
        count = total_counts[cls]
        prop = count / total_samples
        print(f"  Class {int(cls)}: {count}/{total_samples} ({prop:.2%})")
