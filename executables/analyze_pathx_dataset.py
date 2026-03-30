"""
Analyze PathX dataset class distributions per batch.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import create_lra_pathx_classification_dataset

# Configuration
BATCH_SIZE = 8
SEED = 42

def analyze_batch_class_distribution(loader, split_name="train"):
    """
    Analyze and display class proportions for each batch in a dataloader.

    Args:
        loader: PyTorch DataLoader
        split_name: Name of the split (train/val/test)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {split_name} split")
    print(f"{'='*80}\n")

    # Statistics accumulators
    all_class_counts = []
    batch_class_distributions = []

    for batch_idx, batch in enumerate(loader):
        # Extract labels from batch
        # batch is a tuple: (inputs, targets) or (inputs, targets, aux_data)
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 3:
            inputs, targets, aux_data = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Convert targets to numpy if needed
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()
        else:
            targets = np.array(targets)

        # Count class occurrences in this batch
        class_counts = Counter(targets)
        batch_class_distributions.append(class_counts)

        # Display class proportions for this batch
        print(f"Batch {batch_idx + 1}:")
        print(f"  Total samples: {len(targets)}")

        # Calculate proportions
        for class_label in sorted(class_counts.keys()):
            count = class_counts[class_label]
            proportion = count / len(targets)
            print(f"    Class {int(class_label)}: {count}/{len(targets)} ({proportion:.2%})")

        # Calculate balance metric (entropy-based or std of proportions)
        proportions = np.array([class_counts.get(i, 0) / len(targets) for i in range(2)])
        balance_std = np.std(proportions)
        print(f"  Balance metric (std of proportions): {balance_std:.4f}")
        print()

        all_class_counts.append(class_counts)

    # Overall statistics
    print(f"\n{'='*80}")
    print(f"Overall {split_name} split statistics")
    print(f"{'='*80}\n")

    # Aggregate all class counts
    total_class_counts = Counter()
    for counts in all_class_counts:
        total_class_counts.update(counts)

    total_samples = sum(total_class_counts.values())
    print(f"Total batches: {len(all_class_counts)}")
    print(f"Total samples: {total_samples}")
    print(f"\nOverall class distribution:")
    for class_label in sorted(total_class_counts.keys()):
        count = total_class_counts[class_label]
        proportion = count / total_samples
        print(f"  Class {int(class_label)}: {count}/{total_samples} ({proportion:.2%})")

    # Calculate batch-wise balance statistics
    batch_balance_metrics = []
    for counts in all_class_counts:
        # Get batch size
        batch_size = sum(counts.values())
        proportions = np.array([counts.get(i, 0) / batch_size for i in range(2)])
        balance_std = np.std(proportions)
        batch_balance_metrics.append(balance_std)

    print(f"\nBatch balance statistics:")
    print(f"  Mean balance std: {np.mean(batch_balance_metrics):.4f}")
    print(f"  Median balance std: {np.median(batch_balance_metrics):.4f}")
    print(f"  Min balance std: {np.min(batch_balance_metrics):.4f}")
    print(f"  Max balance std: {np.max(batch_balance_metrics):.4f}")

    return all_class_counts, total_class_counts


def plot_class_distribution_summary(train_counts, val_counts, test_counts):
    """
    Create visualizations of class distributions across splits.

    Args:
        train_counts: List of Counter objects for each training batch
        val_counts: List of Counter objects for each validation batch
        test_counts: List of Counter objects for each test batch
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    splits = [
        ('Train', train_counts),
        ('Val', val_counts),
        ('Test', test_counts)
    ]

    for ax, (split_name, counts_list) in zip(axes, splits):
        # Calculate proportion of class 0 vs class 1 for each batch
        class_0_props = []
        class_1_props = []

        for counts in counts_list:
            batch_size = sum(counts.values())
            class_0_props.append(counts.get(0, 0) / batch_size)
            class_1_props.append(counts.get(1, 0) / batch_size)

        # Create stacked bar chart
        x = np.arange(len(counts_list))
        ax.bar(x, class_0_props, label='Class 0', color='skyblue')
        ax.bar(x, class_1_props, bottom=class_0_props, label='Class 1', color='salmon')

        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Class Proportion')
        ax.set_title(f'{split_name} Split - Class Distribution per Batch')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)

        # Show only every nth x-tick label for readability
        if len(x) > 20:
            tick_spacing = len(x) // 10
            ax.set_xticks(x[::tick_spacing])
            ax.set_xticklabels(x[::tick_spacing])

    plt.tight_layout()
    plt.savefig('pathx_class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: pathx_class_distribution_analysis.png")
    plt.close()


def main():
    """Main analysis function."""
    print("="*80)
    print("PathX Dataset Class Distribution Analysis")
    print("="*80)
    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Seed: {SEED}")

    # Create dataloaders
    print("\nLoading PathX dataset...")
    trainloader, val_loader, testloader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE = \
        create_lra_pathx_classification_dataset(bsz=BATCH_SIZE, seed=SEED)

    print(f"\nDataset info:")
    print(f"  Number of classes: {N_CLASSES}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Input dimension: {IN_DIM}")
    print(f"  Training set size: {TRAIN_SIZE}")
    print(f"  Number of training batches: {len(trainloader)}")
    print(f"  Number of validation batches: {len(val_loader)}")
    print(f"  Number of test batches: {len(testloader)}")

    # Analyze each split
    train_batch_counts, train_total = analyze_batch_class_distribution(trainloader, "Train")
    val_batch_counts, val_total = analyze_batch_class_distribution(val_loader, "Validation")
    test_batch_counts, test_total = analyze_batch_class_distribution(testloader, "Test")

    # Create visualization
    print("\nCreating visualization...")
    plot_class_distribution_summary(train_batch_counts, val_batch_counts, test_batch_counts)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
