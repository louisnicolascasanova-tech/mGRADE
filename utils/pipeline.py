"""Training pipeline functions for Den-minGRU.

This module contains the core pipeline functions extracted from the main
training script to improve modularity and testability.
"""

import os
from functools import partial
from typing import Tuple

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
from training import create_train_state, run_epoch, validate, create_learning_rate_map


def setup_dataset(args, dtype):
    """Load and prepare dataset.

    Args:
        args: Configuration containing dataset name, batch_size, seed, etc.
        dtype: Data type for tensors (float16 or float32)

    Returns:
        DatasetInfo with loaders, dimensions, metadata, and sample batch
    """
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

    # Load dataset based on type
    if args.dataset in ['cifar', 'mnist', 'gsc']:
        if args.dataset in ['cifar', 'mnist']:
            trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, \
                IN_DIM = dataset_fns[args.dataset](
                    bsz=args.batch_size, root="data", dtype=dtype
                )
        elif args.dataset == 'gsc':
            trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, \
                IN_DIM = dataset_fns[args.dataset](
                    bsz=args.batch_size, root="data", dtype=dtype
                )
        batch_x, batch_y = next(iter(testloader))

    elif args.dataset in ['imdb', 'listops', 'aan']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, \
            IN_DIM, _ = dataset_fns[args.dataset](
                batch_size=args.batch_size, seed=args.seed
            )
        batch = next(iter(testloader))
        batch_x, batch_y = prep_batch(batch, SEQ_LENGTH, IN_DIM)

    elif args.dataset in ['path', 'pathx']:
        # Check if stratified sampling is requested for PathX
        if args.dataset == 'pathx' and \
            getattr(args, 'stratified_sampling', False):
            print("[*] Using STRATIFIED sampling for PathX (balanced batches)")
            trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, \
                IN_DIM, _ = dataset_fns[args.dataset](
                    bsz=args.batch_size, seed=args.seed, stratified=True
                )
        else:
            trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, \
                IN_DIM, _ = dataset_fns[args.dataset](
                    bsz=args.batch_size, seed=args.seed
                )
        batch = next(iter(testloader))
        batch_x, batch_y = prep_batch(batch, SEQ_LENGTH, IN_DIM)

    # Print shapes for debugging
    if args.dataset in ['imdb', 'listops', 'aan']:
        print(batch_x[0].shape, batch_x[1].shape)
        print(batch_x[0].dtype)
    else:
        print(batch_x.shape, batch_y.shape)
        print(batch_x.dtype)
    print(batch_y.dtype)

    # Compute class weights if needed (only for listops)
    class_weights = compute_class_weights(trainloader, N_CLASSES) \
        if args.dataset == 'listops' else None

    return DatasetInfo(
        trainloader=trainloader,
        val_loader=val_loader,
        testloader=testloader,
        n_classes=N_CLASSES,
        seq_length=SEQ_LENGTH,
        in_dim=IN_DIM,
        class_weights=class_weights,
        sample_batch=(batch_x, batch_y)
    )


def _build_model_config(args, dataset_info, hidden_dim, latent_dim):
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


def create_model(args, dataset_info, hidden_dim, latent_dim):
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


def setup_training(key, model_cls, dataset_info, args, dtype):
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


def setup_experiment_dirs(args, hidden_dim, latent_dim, seed, config_file):
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


def tabulate_model(model_cls, sample_batch, key):
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


def print_dcls_parameters(state, dataset):
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


def run_validation_for_dataset(state, model_cls, val_loader, testloader,
                                dataset, seq_length, in_dim, n_classes,
                                log_classification_report):
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


def update_improvement_threshold(dataset, best_val_acc, current_improvement):
    """Update improvement threshold based on dataset and current accuracy.

    These are ad-hoc empirical values that fit the dynamics. 

    Args:
        dataset: Dataset name
        best_val_acc: Best validation accuracy so far
        current_improvement: Current improvement threshold

    Returns:
        Updated improvement threshold
    """
    if dataset == 'mnist':
        if best_val_acc > 0.94:
            return 0.001  # 0.1%
    elif dataset == 'cifar':
        if 0.8 > best_val_acc > 0.70:
            return 0.005  # 0.5%
        elif best_val_acc >= 0.80:
            return 0.001  # 0.1%
    elif dataset == 'listops':
        if best_val_acc > 0.55:
            return 0.001  # 0.3%
        elif best_val_acc > 0.50:
            return 0.003  # 0.2%
    elif dataset == 'path':
        if best_val_acc > 0.88:
            return 0.002  # 0.2%
    elif dataset == 'pathx':
        if best_val_acc > 0.93:
            return 0.002  # 0.2%
        elif best_val_acc > 0.88:
            return 0.005  # 0.5%
    elif dataset in ['imdb', 'aan', 'gsc']:
        if best_val_acc > 0.83:
            return 0.001  # 0.1%

    return current_improvement


def check_early_stopping(dataset, epoch, patience, val_acc, best_val_acc,
                            val_loss, best_val_acc_loss, ovf_count, bad_count,
                            lim_patience):
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


def train_model(training_state, model_cls, dataset_info, experiment_dirs,
                args, key):
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

        state, train_loss, train_acc, (break_flag, aux_dict_epoch) = \
            run_epoch(state, model_cls, dataset_info.trainloader, subkey,
                        reg_factor=args.reg_factor,
                        kernel_size=args.kernel_size,
                        lim_batch=None,
                        keys_to_track=keys_to_track,
                        inner_keys_to_track=inner_keys_to_track,
                        lr_fn=training_state.lr_fn,
                        wandb_gradients=getattr(args, 'wandb_gradients', False),
                        wandb_states=getattr(args, 'wandb_states', False),
                        wandb_matrices=getattr(args, 'wandb_matrices', False),
                        in_dim=dataset_info.in_dim,
                        seq_len=dataset_info.seq_length,
                        grad_clip_norm=args.grad_clip_norm,
                        log_model_behavior=args.log_model_behavior,
                        epoch_num=epoch,
                        class_weights=dataset_info.class_weights,
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
