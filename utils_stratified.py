"""
Add this function to utils.py to create a stratified PathX dataloader.
This ensures perfectly balanced batches (50% class 0, 50% class 1).
"""
from pathlib import Path
from typing import Union
from stratified_dataloader import create_stratified_dataloader


def create_lra_pathx_classification_dataset_stratified(
    cache_dir: Union[str, Path] = "./cache_dir/",
    bsz: int = 50,
    seed: int = 42
):
    """
    Create PathX dataset with stratified batch sampling.
    Each batch will have exactly 50% of each class.

    Args:
        cache_dir: Directory for caching processed data
        bsz: Batch size (should be even for binary classification)
        seed: Random seed for reproducibility

    Returns:
        trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE
    """
    from lra import PathFinder

    print("[*] Generating LRA-PathX Classification Dataset (STRATIFIED SAMPLING)")

    if bsz % 2 != 0:
        print(f"Warning: batch_size ({bsz}) is odd. For perfect 50/50 split, use even batch size.")

    name = 'pathfinder'
    resolution = 128
    dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

    dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    # Extract datasets and labels
    train_dataset = dataset_obj.dataset_train
    val_dataset = dataset_obj.dataset_val
    test_dataset = dataset_obj.dataset_test

    train_labels = train_dataset.tensors[1].numpy()
    val_labels = val_dataset.tensors[1].numpy()
    test_labels = test_dataset.tensors[1].numpy()

    # Create stratified dataloaders
    trn_loader = create_stratified_dataloader(
        train_dataset, train_labels,
        batch_size=bsz, shuffle=True, seed=seed, drop_last=True
    )
    val_loader = create_stratified_dataloader(
        val_dataset, val_labels,
        batch_size=bsz, shuffle=False, seed=seed, drop_last=False
    )
    tst_loader = create_stratified_dataloader(
        test_dataset, test_labels,
        batch_size=bsz, shuffle=False, seed=seed, drop_last=False
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
    IN_DIM = dataset_obj.d_input
    TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

    aux_loaders = {}

    print(f"  Stratified batches created: Train={len(trn_loader)}, Val={len(val_loader)}, Test={len(tst_loader)}")
    print(f"  Each batch has exactly {bsz//2} samples from each class")

    return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE
