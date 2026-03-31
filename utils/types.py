"""Type definitions for training pipeline."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from flax.training import train_state
from torch.utils.data import DataLoader


@dataclass
class DatasetInfo:
    """Dataset information and loaders.

    Attributes:
        trainloader: Training data loader
        val_loader: Validation data loader
        testloader: Test data loader
        n_classes: Number of output classes
        seq_length: Sequence length
        in_dim: Input dimension
        class_weights: Optional class weights for imbalanced datasets
        sample_batch: Sample batch for model tabulation (inputs, labels)
    """
    trainloader: DataLoader
    val_loader: DataLoader
    testloader: DataLoader
    n_classes: int
    seq_length: int
    in_dim: int
    class_weights: Optional[jnp.ndarray] = None
    sample_batch: Optional[Tuple[Any, Any]] = None


@dataclass
class ExperimentDirs:
    """Experiment directory structure.

    Attributes:
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory for best model checkpoints
        warmup_dir: Directory for warmup checkpoints
        results_dir: Directory for results (training dynamics, etc.)
        plots_dir: Directory for plots
        backup_dir: Directory for backup checkpoints
    """
    experiment_id: str
    checkpoint_dir: str
    warmup_dir: str
    results_dir: str
    plots_dir: str
    backup_dir: str


@dataclass
class TrainingState:
    """Training state and configuration.

    Attributes:
        state: Flax training state with parameters and optimizer
        n_params: Total number of model parameters
        start_epoch: Starting epoch (0 for new training, > 0 for resumed)
        lr_fn: Learning rate schedule function
    """
    state: train_state.TrainState
    n_params: int
    start_epoch: int
    lr_fn: Callable


@dataclass
class TrainingResults:
    """Training results and metrics.

    Attributes:
        final_state: Final training state after all epochs
        train_losses: Training loss per epoch
        train_accuracies: Training accuracy per epoch
        val_losses: Validation loss per epoch
        val_accuracies: Validation accuracy per epoch
        test_losses: Test loss per epoch (only when val improves)
        test_accuracies: Test accuracy per epoch (only when val improves)
        best_val_acc: Best validation accuracy achieved
        best_test_acc: Test accuracy at best validation
        aux_dict_training: Auxiliary training information per epoch
    """
    final_state: train_state.TrainState
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    test_losses: List[float]
    test_accuracies: List[float]
    best_val_acc: float
    best_test_acc: float
    aux_dict_training: List[Dict[str, Any]]
