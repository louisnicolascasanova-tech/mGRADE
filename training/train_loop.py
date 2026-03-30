"""
Training loop and model update functions.
Contains the main run_epoch function and gradient/parameter update logic.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from tqdm import tqdm
from flax.traverse_util import flatten_dict, unflatten_dict
import wandb

from utils import prep_batch
from custom_logging import LoggingConfig, log_training_batch


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
    clipped_grads = optax.clip_by_global_norm(grad_clip_norm).update(grads, None)[0]
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
        clipped_flat = {k: clip_fn(k.split('/'), v) for k, v in flat_params.items()}
        return unflatten_dict(clipped_flat, sep='/')

    state = state.replace(params=clip_negative_positions(state.params))
    return state, grad_norm, grad_norm_post_clip


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
    def loss_fn(params):
        net_dyn, logits, monitor = model.apply(
            {'params': params}, x.astype(dtype), rngs={'dropout': do_key}
        )
        one_hot = jax.nn.one_hot(y, model.out_dim, dtype=jnp.float32)
        batch_loss = optax.softmax_cross_entropy(
            logits=logits.astype(jnp.float32), labels=one_hot
        )
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

    # Compute prediction probabilities and confidence metrics
    logits = aux_dict['logits']  # shape: (batch_size, out_dim)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == y)

    # Prediction confidence (max softmax probability)
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    confidence_std = jnp.std(max_probs)

    # Add model behavior metrics to aux_dict
    aux_dict.update({
        'loss': loss,
        'accuracy': accuracy,
        'probs': probs,
        'predictions': predictions,
        'mean_confidence': mean_confidence,
        'confidence_std': confidence_std,
        'max_probs': max_probs
    })
    return grads, loss, accuracy, aux_dict


def apply_retrieval_model(state, model, x, y, reg_factor, do_key, dtype=jnp.float32):
    """
    Apply model for retrieval tasks (e.g., AAN).

    Similar to apply_model but simplified for retrieval-specific architectures.
    """
    def loss_fn(params):
        net_dyn, logits, monitor = model.apply(
            {'params': params}, x.astype(dtype), rngs={'dropout': do_key}
        )
        one_hot = jax.nn.one_hot(y, model.out_dim, dtype=jnp.float32)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits.astype(jnp.float32), labels=one_hot)
        )
        return loss, {'logits': logits, 'net_dyn': net_dyn, 'monitor': monitor}
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)

    # Compute prediction probabilities and confidence metrics
    logits = aux_dict['logits']  # shape: (batch_size, out_dim)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == y)

    # Prediction confidence (max softmax probability)
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    confidence_std = jnp.std(max_probs)

    # Add model behavior metrics to aux_dict
    aux_dict.update({
        'probs': probs,
        'predictions': predictions,
        'mean_confidence': mean_confidence,
        'confidence_std': confidence_std,
        'max_probs': max_probs
    })
    return grads, loss, accuracy, aux_dict


def run_epoch(state, model_cls, train_dl, key, reg_factor, kernel_size,
                lim_batch=None, keys_to_track=None, inner_keys_to_track=None,
                lr_fn=None, wandb_gradients=False, wandb_states=False,
                wandb_matrices=False, in_dim=None, seq_len=None,
                grad_clip_norm=1.0, log_model_behavior=True, epoch_num=0,
                class_weights=None, retrieval=False, dataset=None,
                dtype=jnp.float32):
    """
    Train for a single epoch.

    Args:
        state: Training state
        model_cls: Model class (partial with config)
        train_dl: Training dataloader
        key: JAX random key
        reg_factor: Regularization factor
        kernel_size: Kernel size for position clamping
        lim_batch: Limit number of batches (for debugging)
        keys_to_track: Keys to track in aux_dict
        inner_keys_to_track: Inner keys to track in aux_dict
        lr_fn: Learning rate function
        wandb_gradients: Whether to log gradients to wandb
        wandb_states: Whether to log states to wandb
        wandb_matrices: Whether to log matrices to wandb
        in_dim: Input dimension
        seq_len: Sequence length
        grad_clip_norm: Gradient clipping threshold
        log_model_behavior: Whether to log model behavior
        epoch_num: Epoch number
        class_weights: Class weights for loss weighting
        retrieval: Whether using retrieval model
        dataset: Dataset name
        dtype: Data type

    Returns:
        state: Updated training state
        train_loss: Average training loss
        train_accuracy: Average training accuracy
        (break_flag, aux_dict_hist): Break flag and auxiliary history
    """
    model = model_cls(training=True)
    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = {k: [] for k in keys_to_track+inner_keys_to_track}
    batch_id = 0
    break_flag = False
    key, do_key = jax.random.split(key)

    # Initialize logging configuration
    logging_config = LoggingConfig(
        wandb_gradients=wandb_gradients,
        wandb_states=wandb_states,
        wandb_matrices=wandb_matrices,
        log_model_behavior=log_model_behavior
    )
    logging_config.grad_clip_norm = grad_clip_norm

    # Model behavior tracking
    all_predictions = []
    all_targets = []
    all_confidences = []

    for batch in progress_bar:
        if len(batch) == 2:  # If the batch is already preprocessed
            batch_x, batch_y = batch
        elif len(batch) == 3:  # If the batch contains mask
            batch_x, batch_y = prep_batch(batch, seq_len, in_dim, dtype=dtype)

        grads, loss, accuracy, aux_dict = apply_model(
            state, model, batch_x, batch_y, reg_factor=reg_factor,
            do_key=do_key, class_weights=class_weights
        )

        for k in keys_to_track:
            aux_dict_hist[k].append(locals()[k])
        for k in inner_keys_to_track:
            if k != 'net_dyn':  # Skip net_dyn to save memory
                aux_dict_hist[k].append(aux_dict[k])

        # Use new logging system
        log_dict = log_training_batch(
            state=state, aux_dict=aux_dict, grads=grads, model=model,
            lr_fn=lr_fn, grad_norm=0.0, post_clip_grad_norm=0.0,  # Will be updated after gradient clipping
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
                    print(f"Layer {l} | {i} | {ldyn[i].shape} | {ldyn[i].min()} | {ldyn[i].max()} | {ldyn[i].mean()}")

            break_flag = True
            break

        state, grad_norm, post_clip_grad_norm = update_model(
            state, grads, kernel_size, grad_clip_norm
        )

        # Update gradient clipping info in the log dictionary
        log_dict.update({
            "train_grad/norm": grad_norm,
            "train_grad/norm_post_clip": post_clip_grad_norm,
            "train_grad/norm_clipped": int(grad_norm > grad_clip_norm)
        })

        # Single consolidated wandb log call
        wandb.log(log_dict, step=state.step)

        batch_id += 1

        if batch_id % 3 == 0:
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        if lim_batch is not None and batch_id >= lim_batch:
            break_flag = True
            break

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)

    # Compute epoch-level model behavior metrics using new logging system
    if log_model_behavior and len(all_predictions) > 0:
        from custom_logging import log_classification_metrics
        epoch_metrics = log_classification_metrics(
            all_predictions, all_targets, all_confidences,
            split_name="train_epoch", config=logging_config
        )
        # Log epoch metrics with custom step to avoid conflicts
        wandb.log(epoch_metrics, step=state.step, commit=False)

    return state, train_loss, train_accuracy, (break_flag, aux_dict_hist)
