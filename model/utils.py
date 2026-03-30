"""
Utility functions for sequence processing.
"""
import jax
import jax.numpy as jnp


def masked_meanpool(x, lengths):
    """
    Perform mean pooling across the sequence length for variable-length sequences.
    Only pools across the pre-padded sequence length.

    Args:
        x (float32): Input sequence of shape (L, d_model)
        lengths (int32): The original length of the sequence before padding

    Returns:
        float32: Mean pooled output sequence of shape (d_model)
    """
    L = x.shape[0]
    mask = jnp.arange(L) < lengths
    return jnp.sum(mask[..., None]*x, axis=0)/lengths


# Vectorize across batch dimension for batch processing
batch_masked_meanpool = jax.vmap(masked_meanpool)
