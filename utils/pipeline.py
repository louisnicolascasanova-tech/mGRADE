"""Training pipeline functions for Den-minGRU.

This module contains the core pipeline functions extracted from the main
training script to improve modularity and testability.
"""

import os
from dataclasses import dataclass
from functools import partial
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import checkpoints
import wandb

from . import (
    create_mnist_classification_dataset,
    create_cifar_gs_classification_dataset,
    create_lra_imdb_classification_dataset,
    create_lra_listops_classification_dataset,
    create_lra_path32_classification_dataset,
    create_lra_pathx_classification_dataset,
    create_lra_aan_classification_dataset,
    create_speechcommands35_classification_dataset,
    prep_batch,
    generate_experiment_id,
    create_experiment_directories,
    compute_class_weights,
    write_config_yaml,
)
from .types import DatasetInfo, ExperimentDirs, TrainingState, TrainingResults
from model import BatchRNN_General, RNN_General_Retrieval_Backbone
from training import (
    create_train_state, run_epoch, validate, create_learning_rate_map,
    EpochConfig, WandBLogConfig
)


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration with dataset-specific thresholds.

    Args:
        patience: Number of epochs to wait before stopping
        min_improvement: Minimum improvement threshold
        overflow_threshold: Maximum overfitting occurrences before stopping
    """
    patience: int
    min_improvement: float
    overflow_threshold: int = 5

    @classmethod
    def for_dataset(cls, dataset: str) -> 'EarlyStoppingConfig':
        """Get dataset-specific early stopping configuration.

        Args:
            dataset: Dataset name

        Returns:
            EarlyStoppingConfig with dataset-specific parameters
        """
        configs = {
            'listops': cls(patience=10, min_improvement=0.005),
            'aan': cls(patience=10, min_improvement=0.01),
            'imdb': cls(patience=10, min_improvement=0.005),
            'mnist': cls(patience=10, min_improvement=0.01),
            'cifar': cls(patience=10, min_improvement=0.01),
            'path': cls(patience=10, min_improvement=0.01),
            'pathx': cls(patience=10, min_improvement=0.01),
            'gsc': cls(patience=10, min_improvement=0.01),
        }
        return configs.get(dataset, cls(patience=10, min_improvement=0.01))


def setup_dataset(args, dtype) -> DatasetInfo:
    """Load and prepare dataset using appropriate loader.

    Args:
        args: Configuration containing dataset name, batch_size, seed, etc.
        dtype: Data type for tensors (float16 or float32)

    Returns:
        DatasetInfo with loaders, dimensions, metadata, and sample batch

    Raises:
        ValueError: If dataset name is not recognized
    """
    # Define dataset functions
    dataset_fns = {
        'cifar': create_cifar_gs_classification_dataset,
        'mnist': create_mnist_classification_dataset,
        'imdb': create_lra_imdb_classification_dataset,
        'listops': create_lra_listops_classification_dataset,
        'path': create_lra_path32_classification_dataset,
        'pathx': create_lra_pathx_classification_dataset,
        'aan': create_lra_aan_classification_dataset,
        'gsc': create_speechcommands35_classification_dataset,
    }

    # Get dataset function
    if args.dataset not in dataset_fns:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_fn = dataset_fns[args.dataset]

    # Create appropriate loader based on dataset type
    if args.dataset in ['cifar', 'mnist', 'gsc']:
        loader = MnistCifarGscDatasetLoader(args.dataset, dataset_fn)
    elif args.dataset in ['imdb', 'listops', 'aan']:
        loader = TextDatasetLoader(args.dataset, dataset_fn)
    elif args.dataset in ['path', 'pathx']:
        loader = PathDatasetLoader(args.dataset, dataset_fn)
    else:
        raise ValueError(f"No loader configured for dataset: {args.dataset}")

    # Load and return dataset
    return loader.load(args, dtype)


def _build_model_config(args, dataset_info: DatasetInfo,
                            hidden_dim: tuple, latent_dim: tuple) -> Dict:
    """Build common model configuration dictionary.

    Args:
        args: Model configuration parameters
        dataset_info: Dataset metadata
        hidden_dim: Hidden dimensions per layer
        latent_dim: Latent dimensions per layer

    Returns:
        Dictionary of model configuration parameters
    """
    # For 'aan' dataset, use direct attribute access (assumes they exist)
    # For others, use getattr with None defaults for DCLS params
    is_aan = args.dataset == 'aan'

    config = {
        # Architecture
        'n_layers': args.n_layers,
        'out_dim': dataset_info.n_classes,
        'hidden_dim': tuple(hidden_dim),
        'latent_dim': tuple(latent_dim),
        'do_rate': args.do_rate,
        'enable_monitoring': getattr(args, 'enable_monitoring', False),

        # Encoder
        'encoder': getattr(args, 'encoder', True),
        'encoder_scale': getattr(args, 'encoder_scale', 1.0),
        'encoder_bias': getattr(args, 'encoder_bias', True),

        # Skip connections
        'layer_skip': args.layer_skip,
        'element_skip': args.element_skip,

        # Convolution
        'enable_conv': args.enable_conv,
        'conv_layer': args.conv,
        'kernel_size': args.kernel_size,
        'kernel_n_elems': args.kernel_n_elems,
        'wavenet_dilation': args.wavenet_dilation,
        'dilation_schedule': args.dilation_schedule,
        'dilation_boundary': args.dilation_boundary,
        'dilation_offset': args.dilation_offset,
        'constant_dilation': args.constant_dilation,
        'dcls_fft': True,
        'weight_init_scale': getattr(args, 'weight_init_scale', 1.0),
        'conv_ln': getattr(args, 'conv_ln', False),

        # DCLS parameters (direct access for aan, getattr with None for others)
        'dcls_type': args.delay_type if is_aan \
            else getattr(args, 'delay_type', None),
        'dcls_kernel': args.delay_kernel if is_aan \
            else getattr(args, 'delay_kernel', None),
        'dcls_std': args.init_std if is_aan \
            else getattr(args, 'init_std', None),
        'dcls_heterogeneous_weights': args.heterogeneous_weights if is_aan \
            else getattr(args, 'heterogeneous_weights', None),
        'dcls_heterogeneous_positions': args.heterogeneous_positions if is_aan \
            else getattr(args, 'heterogeneous_positions', None),
        'dcls_heterogeneous_std': args.heterogeneous_std if is_aan \
            else getattr(args, 'heterogeneous_std', None),

        # Recurrent
        'enable_rec': args.enable_rec,
        'rec_act': args.rec_act,
        'rec_ln': getattr(args, 'rec_ln', False),
        'rec_dense_out': getattr(args, 'rec_dense_out', False),
        'rec_dense_out_act': getattr(args, 'rec_dense_out_act', False),
        'dense_z_weight_init_scale': \
            getattr(args, 'dense_z_weight_init_scale', 1.0),
        'dense_z_bias_init': getattr(args, 'dense_z_bias_init', 'zero'),
        'dense_h_weight_init_scale': \
            getattr(args, 'dense_h_weight_init_scale', 1.0),
        'dense_h_bias_init': getattr(args, 'dense_h_bias_init', 'zero'),

        # Channel mixing
        'enable_cm': args.enable_cm,
        'channel_mixing': args.channel_mixing,
        'cm_act': args.cm_act,
        'glu_type': args.glu_type,
        'cm_ln': getattr(args, 'cm_ln', False),

        # Compression
        'comp_act': args.comp_act,
        'postnorm': args.postnorm,
        'decoder_bias': getattr(args, 'decoder_bias', True),
    }

    return config


def create_model(args, dataset_info: DatasetInfo,
                    hidden_dim: tuple, latent_dim: tuple) -> Callable:
    """Configure and create model class.

    Args:
        args: Model configuration parameters
        dataset_info: Dataset metadata (n_classes, in_dim, etc.)
        hidden_dim: Hidden dimensions per layer (tuple)
        latent_dim: Latent dimensions per layer (tuple)

    Returns:
        model_cls: Partial function for model instantiation
    """
    # Build common configuration
    config = _build_model_config(args, dataset_info, hidden_dim, latent_dim)

    # Select model class and add dataset-specific parameters
    if args.dataset == 'aan':
        # AAN uses retrieval backbone (no padded parameter)
        model_cls = partial(RNN_General_Retrieval_Backbone, **config)
    else:
        # All other datasets use standard backbone with padding option
        config['padded'] = args.dataset in ['imdb', 'listops']
        model_cls = partial(BatchRNN_General, **config)

    return model_cls


def setup_training(key: jax.random.PRNGKey, model_cls: Callable,
                    dataset_info: DatasetInfo, args, dtype) -> TrainingState:
    """Create training state with optimizer and optional checkpoint loading.

    Args:
        key: JAX random key
        model_cls: Model class (partial function)
        dataset_info: Dataset information
        args: Training configuration
        dtype: Data type

    Returns:
        TrainingState with optimizer, parameters, start epoch, and lr function
    """
    steps_per_epoch = len(dataset_info.trainloader)
    lr_map, lr_fn = create_learning_rate_map(args, steps_per_epoch)
    sim_args = {
        'key': key, 'model_cls': model_cls, 'lr_map': lr_map,
        'dataset_version': 'sequential', 'in_dim': dataset_info.in_dim,
        'seq_len': dataset_info.seq_length,
        'batch_size': args.batch_size, 'wd': args.weight_decay, 'dtype': dtype
    }
    state, n_params, _ = create_train_state(**sim_args)

    # Load checkpoint if resume_from is provided
    start_epoch = 0
    if hasattr(args, 'resume_from') and args.resume_from is not None:
        print(f"Loading checkpoint from {args.resume_from}")
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=args.resume_from, target=state
        )
        if restored_state is not None:
            state = restored_state
            start_epoch = int(state.step // len(dataset_info.trainloader))
            print(f"Resumed training from epoch {start_epoch}, "
                    f"step {state.step}")
        else:
            print("Warning: Could not load checkpoint, starting from scratch")

    return TrainingState(
        state=state,
        n_params=n_params,
        start_epoch=start_epoch,
        lr_fn=lr_fn
    )


def setup_experiment_dirs(args, hidden_dim: tuple, latent_dim: tuple,
                            seed: int, config_file: str) -> ExperimentDirs:
    """Generate experiment ID and create directory structure.

    Args:
        args: Experiment configuration
        hidden_dim: Hidden dimensions
        latent_dim: Latent dimensions
        seed: Random seed
        config_file: Path to config file to copy

    Returns:
        ExperimentDirs with all directory paths
    """
    # Generate experiment ID and create directories
    base_id = generate_experiment_id(args, hidden_dim, latent_dim, seed)
    id_sim, CKPT_DIR, WU_DIR, RESULT_DIR, PLT_DIR = \
        create_experiment_directories(base_id)

    bu_dir = os.path.join(CKPT_DIR, "Backup")
    # Create directories
    os.makedirs(bu_dir, exist_ok=True)
    print(bu_dir)

    # Write configuration to YAML file
    write_config_yaml(config_file, CKPT_DIR)

    return ExperimentDirs(
        experiment_id=id_sim,
        checkpoint_dir=CKPT_DIR,
        warmup_dir=WU_DIR,
        results_dir=RESULT_DIR,
        plots_dir=PLT_DIR,
        backup_dir=bu_dir
    )


def tabulate_model(model_cls: Callable, sample_batch: Tuple,
                    key: jax.random.PRNGKey) -> None:
    """Print model architecture table.

    Args:
        model_cls: Model class (partial function)
        sample_batch: Sample batch (inputs, labels) for tabulation
        key: JAX random key
    """
    batch_x, batch_y = sample_batch
    key1, key2 = jax.random.split(key, 2)
    model_tab = model_cls(training=False)
    tabulate_fn = nn.tabulate(model_tab, {'params': key1, 'dropout': key2})
    print(tabulate_fn(batch_x))


def print_dcls_parameters(state, dataset: str) -> None:
    """Print DCLS layer parameters.

    Args:
        state: Training state with parameters
        dataset: Dataset name (determines parameter access path)
    """
    if dataset == 'aan':
        params = state.params['VmapRNN_General_Backbone_0']['DCLSLayer_0']
    else:
        params = state.params['DCLSLayer_0']

    print(params['positions'])
    print(params['weights'])
    print(params['std'])
    print(params['positions'].dtype)
    print(params['weights'].dtype)
    print(params['std'].dtype)


def run_validation_for_dataset(state, model_cls: Callable, val_loader,
                                testloader, dataset: str, seq_length: int,
                                in_dim: int, n_classes: int,
                                log_classification_report: bool) -> Tuple[float, float, Dict]:
    """Run validation on appropriate loader (imdb uses testloader as val).

    Args:
        state: Training state
        model_cls: Model class
        val_loader: Validation data loader
        testloader: Test data loader
        dataset: Dataset name
        seq_length: Sequence length
        in_dim: Input dimension
        n_classes: Number of classes
        log_classification_report: Whether to log classification report

    Returns:
        Tuple of (val_loss, val_acc, val_metrics)
    """
    loader = testloader if dataset == 'imdb' else val_loader
    return validate(
        state, model_cls, loader, seq_length, in_dim, n_classes,
        log_classification_report=log_classification_report,
        split_name="val"
    )


@dataclass
class ImprovementThresholds:
    """Dynamic improvement thresholds based on validation accuracy.

    These are ad-hoc empirical values that fit the training dynamics
    of each dataset.
    """
    # Accuracy thresholds and corresponding improvement values
    thresholds: list[tuple[float, float]]  # [(acc_threshold, improvement)]

    def get_threshold(self, val_acc: float, current: float) -> float:
        """Get improvement threshold based on validation accuracy.

        Args:
            val_acc: Current validation accuracy
            current: Current improvement threshold

        Returns:
            Updated improvement threshold
        """
        for acc_threshold, improvement in self.thresholds:
            if val_acc > acc_threshold:
                return improvement
        return current

    @classmethod
    def for_dataset(cls, dataset: str) -> 'ImprovementThresholds':
        """Get dataset-specific improvement thresholds.

        Args:
            dataset: Dataset name

        Returns:
            ImprovementThresholds configured for the dataset
        """
        configs = {
            'mnist': cls(thresholds=[(0.94, 0.001)]),
            'cifar': cls(thresholds=[(0.80, 0.001), (0.70, 0.005)]),
            'listops': cls(thresholds=[(0.55, 0.001), (0.50, 0.003)]),
            'path': cls(thresholds=[(0.88, 0.002)]),
            'pathx': cls(thresholds=[(0.93, 0.002), (0.88, 0.005)]),
            'imdb': cls(thresholds=[(0.83, 0.001)]),
            'aan': cls(thresholds=[(0.83, 0.001)]),
            'gsc': cls(thresholds=[(0.83, 0.001)]),
        }
        return configs.get(dataset, cls(thresholds=[]))


class DatasetLoader(ABC):
    """Abstract base class defining the interface for dataset loaders."""
    
    @abstractmethod
    def load(self, args, dtype):
        """Load and prepare dataset. Must be implemented by subclasses."""
        pass


class AbstractDatasetLoader(DatasetLoader):
    """Base class handling common initialization and DatasetInfo construction."""
    
    def __init__(self, dataset_name: str, dataset_fn):
        self.dataset_name = dataset_name
        self.dataset_fn = dataset_fn

    def _build_info(self, train, val, test, n_classes, seq_length, in_dim, 
                    batch_x, batch_y, class_weights=None):
        """Helper to print debugging info and construct the DatasetInfo object."""
        # Handle printing based on whether batch_x is a tuple/list (prep_batch output) or a single tensor
        if isinstance(batch_x, (tuple, list)):
            print(batch_x[0].shape, batch_x[1].shape)
            print(batch_x[0].dtype)
        else:
            print(batch_x.shape, batch_y.shape)
            print(batch_x.dtype)
            
        print(batch_y.dtype)

        return DatasetInfo(
            trainloader=train,
            val_loader=val,
            testloader=test,
            n_classes=n_classes,
            seq_length=seq_length,
            in_dim=in_dim,
            class_weights=class_weights,
            sample_batch=(batch_x, batch_y)
        )


class MnistCifarGscDatasetLoader(AbstractDatasetLoader):
    """Loader for base datasets (MNIST, CIFAR, GSC)."""

    def load(self, args, dtype):
        train, val, test, n_cls, seq_len, in_dim = self.dataset_fn(
            bsz=args.batch_size, root="data", dtype=dtype
        )
        batch_x, batch_y = next(iter(test))
        
        return self._build_info(train, val, test, n_cls, seq_len, in_dim, 
                                batch_x, batch_y)


class TextDatasetLoader(AbstractDatasetLoader):
    """Loader for text LRA datasets (IMDB, ListOps, AAN) that use prep_batch."""

    def load(self, args, dtype):
        train, val, test, _, n_cls, seq_len, in_dim, _ = self.dataset_fn(
            batch_size=args.batch_size, seed=args.seed
        )
        batch_x, batch_y = prep_batch(next(iter(test)), seq_len, in_dim)
        
        weights = compute_class_weights(train, n_cls) if \
            self.dataset_name == 'listops' else None
        
        return self._build_info(train, val, test, n_cls, seq_len, in_dim, 
                                batch_x, batch_y, weights)


class PathDatasetLoader(AbstractDatasetLoader):
    """Loader for Path datasets (Path32, PathX) with stratified sampling support."""

    def load(self, args, dtype):
        # Use a kwargs dictionary to cleanly handle the conditional stratified flag
        kwargs = {"bsz": args.batch_size, "seed": args.seed}
        
        if self.dataset_name == 'pathx' and \
            getattr(args, 'stratified_sampling', False):
            print("[*] Using STRATIFIED sampling for PathX (balanced batches)")
            kwargs["stratified"] = True

        train, val, test, _, n_cls, seq_len, in_dim, _ = \
            self.dataset_fn(**kwargs)
        batch_x, batch_y = prep_batch(next(iter(test)), seq_len, in_dim)

        return self._build_info(train, val, test, n_cls, seq_len, in_dim, 
                                batch_x, batch_y)


def update_improvement_threshold(dataset: str, best_val_acc: float,
                                    current_improvement: float) -> float:
    """Update improvement threshold based on dataset and current accuracy.

    These are ad-hoc empirical values that fit the dynamics.

    Args:
        dataset: Dataset name
        best_val_acc: Best validation accuracy so far
        current_improvement: Current improvement threshold

    Returns:
        Updated improvement threshold
    """
    thresholds = ImprovementThresholds.for_dataset(dataset)
    return thresholds.get_threshold(best_val_acc, current_improvement)


def check_early_stopping(dataset: str, epoch: int, patience: int,
                            val_acc: float, best_val_acc: float,
                            val_loss: float, best_val_acc_loss: float,
                            ovf_count: int, bad_count: int,
                            lim_patience: int
                        ) -> Tuple[bool, int, int, Optional[str]]:
    """Check if training should stop early.

    Args:
        dataset: Dataset name
        epoch: Current epoch
        patience: Current patience counter
        val_acc: Current validation accuracy
        best_val_acc: Best validation accuracy
        val_loss: Current validation loss
        best_val_acc_loss: Best validation loss
        ovf_count: Overfitting counter (for imdb)
        bad_count: Bad performance counter (for imdb)
        lim_patience: Patience limit

    Returns:
        Tuple of (should_stop, updated_ovf_count, updated_bad_count, reason)
    """
    # Check patience-based early stopping
    if patience >= lim_patience and best_val_acc < 0.45:
        return (True, ovf_count, bad_count,
                f"Low validation accuracy ({best_val_acc:.2f}) and "
                f"patience limit reached ({patience}/{lim_patience})")

    if epoch > 10 and patience >= 3 and val_acc < 0.55 and best_val_acc > 0.6:
        return (True, ovf_count, bad_count,
                f"Low validation accuracy ({val_acc:.2f}, "
                f"best: {best_val_acc:.2f}) and "
                f"patience limit reached ({patience}/{lim_patience})")

    # IMDB-specific early stopping
    if dataset == 'imdb':
        # Check overfitting
        if val_loss > 2 * best_val_acc_loss:
            ovf_count += 1
            if ovf_count > 5:
                return (True, ovf_count, bad_count, 'OVERFITTING')
            print(f'OVF COUNT: {ovf_count}')
        else:
            ovf_count = 0

        # Check low accuracy
        if best_val_acc < 0.7 and epoch > 15:
            bad_count += 1
            if bad_count > 5:
                return (True, ovf_count, bad_count, 'LOW ACCURACY ON IMDB')
            print(f'BAD COUNT: {bad_count}')
        else:
            bad_count = 0

    return (False, ovf_count, bad_count, None)


def train_model(training_state: TrainingState, model_cls: Callable,
                dataset_info: DatasetInfo, experiment_dirs: ExperimentDirs,
                args, key: jax.random.PRNGKey) -> TrainingResults:
    """Execute training loop with validation, checkpointing, and early stopping.

    Args:
        training_state: Initial training state
        model_cls: Model class
        dataset_info: Dataset loaders and metadata
        experiment_dirs: Directory paths
        args: Training hyperparameters
        key: JAX random key

    Returns:
        TrainingResults with metrics and final state
    """
    state = training_state.state
    start_epoch = training_state.start_epoch

    # Initialize tracking lists
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    keys_to_track = []
    inner_keys_to_track = ['net_dyn']
    aux_dict_training = []

    # Early stopping configuration
    best_val_acc = 0.0
    improvement = 0.01  # 1%, minimum improvement to save checkpoint
    async_manager = checkpoints.AsyncManager()
    test_loss = 2.5
    test_acc = 0.0
    test_metrics = {}
    ovf_count = 0
    bad_count = 0
    patience = 0
    lim_patience = 10

    # Initial validation if resuming
    if hasattr(args, 'resume_from') and args.resume_from is not None:
        val_loss, val_acc, val_metrics = run_validation_for_dataset(
            state, model_cls, dataset_info.val_loader, dataset_info.testloader,
            args.dataset, dataset_info.seq_length, dataset_info.in_dim,
            dataset_info.n_classes,
            getattr(args, 'log_model_behavior', True)
        )
        print(f"Resumed model validation | val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc*100:.2f}%")

    # Training loop
    for epoch in range(start_epoch, args.n_epochs):
        key, subkey = jax.random.split(key)

        # Get dtype from args (default to float32)
        dtype_str = getattr(args, 'dtype', 'float32')
        dtype = jnp.float16 if dtype_str == 'float16' else jnp.float32

        # Create epoch and wandb configs
        epoch_config = EpochConfig(
            lim_batch=None,
            keys_to_track=keys_to_track,
            inner_keys_to_track=inner_keys_to_track,
            lr_fn=training_state.lr_fn,
            grad_clip_norm=args.grad_clip_norm,
            log_model_behavior=args.log_model_behavior,
            epoch_num=epoch,
            class_weights=dataset_info.class_weights,
            dtype=dtype
        )

        wandb_config = WandBLogConfig(
            log_gradients=getattr(args, 'wandb_gradients', False),
            log_states=getattr(args, 'wandb_states', False),
            log_matrices=getattr(args, 'wandb_matrices', False),
            in_dim=dataset_info.in_dim,
            seq_len=dataset_info.seq_length
        )

        state, train_loss, train_acc, (break_flag, aux_dict_epoch) = \
            run_epoch(state, model_cls, dataset_info.trainloader, subkey,
                      reg_factor=args.reg_factor,
                      kernel_size=args.kernel_size,
                      config=epoch_config,
                      wandb_config=wandb_config,
                      dataset=args.dataset)
        aux_dict_training.append(aux_dict_epoch)

        if break_flag:
            break

        # Validation
        val_loss, val_acc, val_metrics = run_validation_for_dataset(
            state, model_cls, dataset_info.val_loader, dataset_info.testloader,
            args.dataset, dataset_info.seq_length, dataset_info.in_dim,
            dataset_info.n_classes,
            getattr(args, 'log_model_behavior', True)
        )

        # Check if model improved
        if val_acc > best_val_acc + improvement:
            patience = 0
            best_val_acc = val_acc
            best_val_acc_loss = val_loss

            # Update improvement threshold based on dataset and accuracy
            improvement = update_improvement_threshold(
                args.dataset, best_val_acc, improvement
            )

            # Run test evaluation
            if args.dataset != 'imdb':
                test_loss, test_acc, test_metrics = validate(
                    state, model_cls, dataset_info.testloader,
                    dataset_info.seq_length, dataset_info.in_dim,
                    dataset_info.n_classes,
                    log_classification_report=\
                        getattr(args, 'log_model_behavior', True),
                    split_name="test"
                )
                print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | "
                        f"train_acc: {train_acc*100:.2f}% | "
                        f"val_loss: {val_loss:.4f} | "
                        f"val_acc: {val_acc*100:.2f}% | "
                        f"test_loss: {test_loss:.4f} | "
                        f"test_acc: {test_acc*100:.2f}%")
            else:
                print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | "
                        f"train_acc: {train_acc*100:.2f}% | "
                        f"val_loss: {val_loss:.4f} | "
                        f"val_acc: {val_acc*100:.2f}%")

            print(f"Saving the model at epoch {epoch}, "
                    f"in directory {experiment_dirs.checkpoint_dir}")
            checkpoints.save_checkpoint(
                ckpt_dir=experiment_dirs.checkpoint_dir, target=state,
                step=state.step, overwrite=True, async_manager=async_manager
            )

        else:
            print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc*100:.2f}% | "
                    f"val_loss: {val_loss:.4f} | "
                    f"val_acc: {val_acc*100:.2f}%")
            patience += 1
            if args.dataset == 'listops' and epoch < 20:
                patience = 0

            # Check early stopping
            should_stop, ovf_count, bad_count, reason = check_early_stopping(
                args.dataset, epoch, patience, val_acc, best_val_acc,
                val_loss, best_val_acc_loss, ovf_count, bad_count, lim_patience
            )
            if should_stop:
                print(f"Early stopping at epoch {epoch} due to {reason}")
                break

        # Prepare main logging dictionary
        main_metrics = {
            "train/epoch_loss": train_loss, "train/epoch_acc": train_acc,
            "val/loss": val_loss, "val/acc": val_acc,
            "val/loss_best": best_val_acc_loss, "val/acc_best": best_val_acc,
        }
        if args.dataset != 'imdb':
            main_metrics.update({
                "test/loss_best": test_loss, "test/acc_best": test_acc
            })

        # Add validation metrics if available
        if val_metrics:
            main_metrics.update(val_metrics)

        # Add test metrics if available (only when model improves)
        if test_metrics:
            main_metrics.update(test_metrics)

        wandb.log(main_metrics)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Print DCLS parameters at epoch 0
        if epoch == 0 and args.conv == 'dcls':
            print_dcls_parameters(state, args.dataset)

        # Save warmup checkpoint
        if epoch == args.n_epochs * args.warmup_frac:
            print("Saving the warmed up model")
            checkpoints.save_checkpoint(
                ckpt_dir=experiment_dirs.warmup_dir, target=state,
                step=state.step, overwrite=True, async_manager=async_manager
            )

    return TrainingResults(
        final_state=state,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        best_val_acc=best_val_acc,
        best_test_acc=test_acc,
        aux_dict_training=aux_dict_training
    )
