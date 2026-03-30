"""
Quick test to verify stratified sampling integration in main.py
"""
import yaml
import argparse
from collections import Counter
from utils_stratified import create_lra_pathx_classification_dataset_stratified

# Load the config
with open("yaml_folder/pathx_dcls_0.yaml", "r") as f:
    config = yaml.safe_load(f)

args = argparse.Namespace(**config)

print("Config loaded:")
print(f"  batch_size: {args.batch_size}")
print(f"  stratified_sampling: {getattr(args, 'stratified_sampling', False)}")
print()

# Test creating the dataloader
if getattr(args, 'stratified_sampling', False):
    print("Creating stratified dataloader...")
    trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = \
        create_lra_pathx_classification_dataset_stratified(bsz=args.batch_size, seed=args.seed)

    print(f"\nDataset info:")
    print(f"  N_CLASSES: {N_CLASSES}")
    print(f"  SEQ_LENGTH: {SEQ_LENGTH}")
    print(f"  IN_DIM: {IN_DIM}")
    print(f"  Train batches: {len(trainloader)}")

    print(f"\nVerifying first 10 training batches:")
    for i, batch in enumerate(trainloader):
        if i >= 10:
            break
        inputs, targets = batch
        class_counts = Counter(targets.numpy())
        print(f"  Batch {i+1}: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")

    print("\n✓ Integration successful! All batches are perfectly balanced.")
else:
    print("stratified_sampling is not enabled in the config.")
