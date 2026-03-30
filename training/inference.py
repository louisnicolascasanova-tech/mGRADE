"""
Inference-specific functions for model evaluation with detailed outputs.
"""
import jax
import jax.numpy as jnp
import optax
from functools import partial


@partial(jax.jit, static_argnames=('model', 'out_dim', 'dtype'))
def inf_model(state, model, images, labels, out_dim, dtype=jnp.float32):
    """
    Computes loss, accuracy, and network dynamics for inference.

    Similar to eval_model but returns full network dynamics (ndh) for analysis.

    Args:
        state: Training state
        model: Model instance
        images: Input batch
        labels: Target labels
        out_dim: Output dimension (number of classes)
        dtype: Data type

    Returns:
        loss: Batch loss
        accuracy: Batch accuracy
        ndh: Network dynamics history (for detailed analysis)
    """
    def loss_fn(params):
        ndh, logits, monitor = model.apply({'params': params}, images.astype(dtype))
        one_hot = jax.nn.one_hot(labels, out_dim, dtype=jnp.float32)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits.astype(jnp.float32), labels=one_hot)
        )
        return loss, ndh, logits

    loss, ndh, logits = loss_fn(state.params)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == labels)

    # Compute confidence metrics
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)

    return loss, accuracy, ndh
