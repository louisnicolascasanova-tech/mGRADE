"""
Model evaluation and validation functions.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from tqdm import tqdm

from utils import prep_batch


@partial(jax.jit, static_argnames=('model', 'out_dim', 'dtype'))
def eval_model(state, model, images, labels, out_dim, dtype=jnp.float32):
    """
    Computes loss, accuracy, and predictions for a single batch.

    Args:
        state: Training state
        model: Model instance (training=False)
        images: Input batch
        labels: Target labels
        out_dim: Output dimension (number of classes)
        dtype: Data type

    Returns:
        loss: Batch loss
        accuracy: Batch accuracy
        predictions: Predicted labels
        max_probs: Maximum softmax probabilities
        mean_confidence: Mean prediction confidence
    """
    def loss_fn(params):
        net_dyn, logits, _ = model.apply({'params': params}, images.astype(dtype))
        one_hot = jax.nn.one_hot(labels, out_dim, dtype=jnp.float32)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits.astype(jnp.float32), labels=one_hot)
        )
        return loss, logits

    loss, logits = loss_fn(state.params)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == labels)

    # Compute confidence metrics
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)

    return loss, accuracy, predictions, max_probs, mean_confidence


def validate(state, model, testloader, seq_len, in_dim, out_dim,
                log_classification_report=True, split_name='val'):
    """
    Validate model on test/validation dataset.

    Args:
        state: Training state
        model: Model class (partial with config)
        testloader: Test/validation dataloader
        seq_len: Sequence length
        in_dim: Input dimension
        out_dim: Output dimension (number of classes)
        log_classification_report: Whether to log classification metrics
        split_name: Name of the split ('val', 'test')

    Returns:
        mean_loss: Average loss
        mean_accuracy: Average accuracy
        eval_metrics: Classification metrics dictionary
    """
    # Compute average loss & accuracy
    model = model(training=False)  # needed when using dropout
    losses, accuracies = [], []
    all_predictions, all_targets, all_confidences = [], [], []
    progress_bar = tqdm(testloader, desc="Validation", leave=True)

    for batch in progress_bar:
        if len(batch) == 2:  # If the batch is already preprocessed
            inputs, labels = batch
        elif len(batch) == 3:  # If the batch contains mask
            inputs, labels = prep_batch(batch, seq_len, in_dim)

        loss, acc, predictions, max_probs, mean_confidence = eval_model(
            state, model, inputs, labels, out_dim
        )

        losses.append(loss)
        accuracies.append(acc)

        # Collect for classification report
        if log_classification_report:
            all_predictions.extend(np.array(predictions))
            all_targets.extend(np.array(labels))
            all_confidences.extend(np.array(max_probs))

    # Compute and log classification metrics using new logging system
    if log_classification_report and len(all_predictions) > 0:
        from custom_logging import log_classification_metrics
        eval_metrics = log_classification_metrics(
            all_predictions, all_targets, all_confidences, split_name=split_name
        )
        return np.mean(losses), np.mean(accuracies), eval_metrics
    else:
        return np.mean(losses), np.mean(accuracies), {}
