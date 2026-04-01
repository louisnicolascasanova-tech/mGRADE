"""
Training loop and model update functions.
Contains the main run_epoch function and gradient/parameter update logic.
"""
from typing import Optional

import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from tqdm import tqdm
from flax.traverse_util import flatten_dict, unflatten_dict
import wandb

from .configs import EpochConfig, WandBLogConfig


def clip_gradients_elementwise(grads, min_val=-1.0, max_val=1.0):
    """Clip gradients element-wise."""
    return jax.tree_map(lambda g: jnp.clip(g, min_val, max_val), grads)


@jax.jit
def update_model(state, grads, kernel_size, grad_clip_norm=1.0):
    """
    Update model parameters with gradient clipping and position clamping.

    Args:
        state: Training state
        grads: Gradients
        kernel_size: Kernel size for position clamping
        grad_clip_norm: Gradient clipping threshold

    Returns:
        state: Updated training state
        grad_norm: Gradient norm before clipping
        grad_norm_post_clip: Gradient norm after clipping
    """
    # Apply gradient clipping
    grad_norm = optax.global_norm(grads)
    clipped_grads = \
        optax.clip_by_global_norm(grad_clip_norm).update(grads, None)[0]
    grad_norm_post_clip = optax.global_norm(clipped_grads)

    state = state.apply_gradients(grads=clipped_grads)

    def clip_negative_positions(params):
        """Clip DCLS positions to [0, kernel_size]."""
        def clip_fn(path, v):
            # path is a tuple of keys, e.g. ('Dense_0', 'DCLS', 'positions')
            if path[-1] == 'positions':
                v = jnp.where(v < 0, 0, v)
                v = jnp.where(v > kernel_size, kernel_size, v)
            return v
        flat_params = flatten_dict(params, sep='/')
        clipped_flat = \
            {k: clip_fn(k.split('/'), v) for k, v in flat_params.items()}
        return unflatten_dict(clipped_flat, sep='/')

    state = state.replace(params=clip_negative_positions(state.params))
    return state, grad_norm, grad_norm_post_clip


def _apply_model_common(state, model, x, y, reg_factor, do_key,
                         is_retrieval=False, class_weights=None,
                         dtype=jnp.float32):
    """
    Common implementation for apply_model and apply_retrieval_model.

    Computes gradients, loss and accuracy for a single batch, with optional
    regularization and class weighting based on model type.

    Args:
        state: Training state
        model: Model instance
        x: Input batch (tensor or tuple of (inputs, lengths))
        y: Target labels
        reg_factor: Regularization factor
        do_key: Dropout random key
        is_retrieval: Whether this is a retrieval model (no reg, no class weights)
        class_weights: Optional class weights for loss weighting (ignored if is_retrieval)
        dtype: Data type

    Returns:
        grads: Parameter gradients
        loss: Batch loss
        accuracy: Batch accuracy
        aux_dict: Auxiliary outputs (logits, predictions, confidence, etc.)
    """
    def loss_fn(params):
        # Handle case where x is a tuple (inputs, lengths) for datasets like AAN
        if isinstance(x, tuple):
            inputs, lengths = x
            model_input = (inputs.astype(dtype), lengths)
        else:
            model_input = x.astype(dtype)

        net_dyn, logits, monitor = model.apply(
            {'params': params}, model_input, rngs={'dropout': do_key}
        )
        one_hot = jax.nn.one_hot(y, model.out_dim, dtype=jnp.float32)
        batch_loss = optax.softmax_cross_entropy(
            logits=logits.astype(jnp.float32), labels=one_hot
        )

        # Retrieval models: simple mean loss, no class weights, no regularization
        if is_retrieval:
            loss = jnp.mean(batch_loss)
            return loss, {'logits': logits, 'net_dyn': net_dyn, 
                            'monitor': monitor}

        # Standard models: class weights + regularization
        if class_weights is not None:
            class_weights_jnp = jnp.array(class_weights, dtype=jnp.float32)
            batch_loss = batch_loss * class_weights_jnp[y]

        reg = 0.
        for layers in net_dyn:
            reg = reg + (layers[2]**2).mean()  # h_tilde_preact

        loss = jnp.mean(batch_loss) + reg_factor * reg
        return loss, {'logits': logits, 'batch_loss': batch_loss,
                        'net_dyn': net_dyn, 'reg': reg, 'monitor': monitor}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)

    # Compute prediction probabilities and confidence metrics (common to both)
    logits = aux_dict['logits']  # shape: (batch_size, out_dim)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == y)

    # Prediction confidence (max softmax probability)
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    confidence_std = jnp.std(max_probs)

    # Add model behavior metrics to aux_dict
    common_metrics = {
        'probs': probs,
        'predictions': predictions,
        'mean_confidence': mean_confidence,
        'confidence_std': confidence_std,
        'max_probs': max_probs
    }

    # Standard models also log loss and accuracy in aux_dict
    if not is_retrieval:
        common_metrics.update({
            'loss': loss,
            'accuracy': accuracy
        })

    aux_dict.update(common_metrics)
    return grads, loss, accuracy, aux_dict


@partial(jax.jit, static_argnames=('model','reg_factor', 'class_weights'))
def apply_model(state, model, x, y, reg_factor, do_key, class_weights,
                dtype=jnp.float32):
    """
    Computes gradients, loss and accuracy for a single batch.

    Args:
        state: Training state
        model: Model instance
        x: Input batch
        y: Target labels
        reg_factor: Regularization factor
        do_key: Dropout random key
        class_weights: Class weights for loss weighting
        dtype: Data type

    Returns:
        grads: Parameter gradients
        loss: Batch loss
        accuracy: Batch accuracy
        aux_dict: Auxiliary outputs (logits, predictions, confidence, etc.)
    """
    return _apply_model_common(
        state, model, x, y, reg_factor, do_key,
        is_retrieval=False, class_weights=class_weights, dtype=dtype
    )


def apply_retrieval_model(state, model, x, y, reg_factor, do_key,
                            dtype=jnp.float32):
    """
    Apply model for retrieval tasks (e.g., AAN).

    Similar to apply_model but simplified for retrieval-specific architectures:
    - No regularization
    - No class weights
    - Cleaner loss computation

    Args:
        state: Training state
        model: Model instance
        x: Input batch
        y: Target labels
        reg_factor: Regularization factor (ignored for retrieval)
        do_key: Dropout random key
        dtype: Data type

    Returns:
        grads: Parameter gradients
        loss: Batch loss
        accuracy: Batch accuracy
        aux_dict: Auxiliary outputs (logits, predictions, confidence, etc.)
    """
    return _apply_model_common(
        state, model, x, y, reg_factor, do_key,
        is_retrieval=True, class_weights=None, dtype=dtype
    )


def run_epoch(state, model_cls, train_dl, key, reg_factor, kernel_size,
                config: EpochConfig, 
                wandb_config: Optional[WandBLogConfig] = None,
                dataset: Optional[str] = None):
    """Train for a single epoch with cleaner parameter interface.

    Args:
        state: Training state
        model_cls: Model class (partial with config)
        train_dl: Training dataloader
        key: JAX random key
        reg_factor: Regularization factor
        kernel_size: Kernel size for position clamping
        config: Epoch configuration (batch limits, tracking, clipping, etc.)
        wandb_config: Optional WandB logging configuration
        dataset: Optional dataset name for dataset-specific logging

    Returns:
        state: Updated training state
        train_loss: Average training loss
        train_accuracy: Average training accuracy
        (break_flag, aux_dict_hist): Break flag and auxiliary history
    """
    # Import here to avoid circular import at module level
    from utils import prep_batch, LoggingConfig, log_training_batch

    # Set default wandb_config if not provided
    if wandb_config is None:
        wandb_config = WandBLogConfig()

    model = model_cls(training=True)
    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = \
        {k: [] for k in config.keys_to_track + config.inner_keys_to_track}
    batch_id = 0
    break_flag = False
    key, do_key = jax.random.split(key)

    # Initialize logging configuration from config objects
    logging_config = LoggingConfig(
        wandb_gradients=wandb_config.log_gradients,
        wandb_states=wandb_config.log_states,
        wandb_matrices=wandb_config.log_matrices,
        log_model_behavior=config.log_model_behavior
    )
    logging_config.grad_clip_norm = config.grad_clip_norm

    # Model behavior tracking
    all_predictions = []
    all_targets = []
    all_confidences = []

    for batch in progress_bar:
        if len(batch) == 2:  # If the batch is already preprocessed
            batch_x, batch_y = batch
        elif len(batch) == 3:  # If the batch contains mask
            batch_x, batch_y = prep_batch(
                batch, wandb_config.seq_len, wandb_config.in_dim, 
                dtype=config.dtype
            )

        grads, loss, accuracy, aux_dict = apply_model(
            state, model, batch_x, batch_y, reg_factor=reg_factor,
            do_key=do_key, class_weights=config.class_weights
        )

        for k in config.keys_to_track:
            aux_dict_hist[k].append(locals()[k])
        for k in config.inner_keys_to_track:
            if k != 'net_dyn':  # Skip net_dyn to save memory
                aux_dict_hist[k].append(aux_dict[k])

        # Use new logging system
        log_dict = log_training_batch(
            state=state, aux_dict=aux_dict, grads=grads, model=model,
            lr_fn=config.lr_fn, grad_norm=0.0, post_clip_grad_norm=0.0,  # Will be updated after gradient clipping
            batch_id=batch_id, config=logging_config, dataset=dataset
        )

        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

        if jnp.isnan(loss).sum() > 0:
            print('NAN Loss')
            flat_grads = flatten_dict(grads, sep='/')
            for key, grad in flat_grads.items():
                if grad is not None:
                    if jnp.isnan(grad).sum() > 0:
                        print(f'NaN in gradients: {key}')
            for l, ldyn in enumerate(aux_dict['net_dyn']):
                for i in range(3):
                    print(f"Layer {l} | {i} | {ldyn[i].shape} | "
                            f"{ldyn[i].min()} | {ldyn[i].max()} | "
                            f"{ldyn[i].mean()}")

            break_flag = True
            break

        state, grad_norm, post_clip_grad_norm = update_model(
            state, grads, kernel_size, config.grad_clip_norm
        )

        # Update gradient clipping info in the log dictionary
        log_dict.update({
            "train_grad/norm": grad_norm,
            "train_grad/norm_post_clip": post_clip_grad_norm,
            "train_grad/norm_clipped": int(grad_norm > config.grad_clip_norm)
        })

        # Single consolidated wandb log call
        wandb.log(log_dict, step=state.step)

        batch_id += 1

        if batch_id % 3 == 0:
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        if config.lim_batch is not None and batch_id >= config.lim_batch:
            break_flag = True
            break

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)

    # Compute epoch-level model behavior metrics using new logging system
    if config.log_model_behavior and len(all_predictions) > 0:
        from utils import log_classification_metrics
        epoch_metrics = log_classification_metrics(
            all_predictions, all_targets, all_confidences,
            split_name="train_epoch", config=logging_config
        )
        # Log epoch metrics with custom step to avoid conflicts
        wandb.log(epoch_metrics, step=state.step, commit=False)

    return state, train_loss, train_accuracy, (break_flag, aux_dict_hist)
