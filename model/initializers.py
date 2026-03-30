"""
Weight and bias initialization functions for neural network layers.
"""
import jax
import jax.numpy as jnp


def identity_weight_init():
    """Initialize weights as identity matrix scaled by 0.9, with last row set to 1.0."""
    def init(key, shape, dtype=jnp.float32):
        w = jnp.eye(shape[0], shape[1], dtype=dtype) * 0.9
        w = w.at[-1, :].set(1.0)
        return w
    return init


def uniform_init():
    """Initialize parameters from uniform distribution."""
    def init(key, shape, max_val, min_val=0, dtype=jnp.float32):
        p = jax.random.uniform(key, shape, dtype=dtype, minval=min_val, maxval=max_val)
        return p
    return init


def constant_init():
    """Initialize parameters as constant value (0.8 by default)."""
    def init(key, shape, offset, dtype=jnp.float32):
        p = jnp.ones(shape, dtype=dtype) * 0.8
        return p
    return init


def dilated_init():
    """Initialize positions with dilated pattern for kernel elements."""
    def init(key, shape, kernel_size, kernel_n_elems, dtype=jnp.float32):
        dilation = kernel_size // kernel_n_elems
        if len(shape) == 2:
            p = jnp.arange(0, kernel_size, dilation)[:, None] * jnp.ones(shape[-1])
        elif len(shape) == 3:
            p = jnp.arange(0, kernel_size, dilation)[:, None, None] * jnp.ones((shape[-2], shape[-1]))
        else:
            raise ValueError(f"Shape {shape} is not supported")
        return p
    return init


def uniform_gate_init(epsilon=1e-5):
    """
    Initialize the bias of the gate using the Uniform Gate Initialization method from the paper:
    "Improving the Gating Mechanism of Recurrent Neural Networks" by A. Gu et al., ICML 2020

    b_z ~ sigma^{-1}( U(0.1, 0.9) ) where sigma^{-1} is the inverse sigmoid function.
    i.e sigma^{-1}(x) = log(x/(1-x)).

    For stability, we use a small constant to avoid division by zero: epsilon = 1e-5

    Args:
        epsilon: Small constant to avoid division by zero

    Returns:
        Initializer function that takes (key, shape, dtype) and returns initialized bias
    """
    def init(key, shape, dtype=jnp.float32):
        b_tilde = jax.random.uniform(key, shape, dtype=dtype, minval=epsilon, maxval=1.-epsilon)
        b = jnp.log(b_tilde / (1 - b_tilde))
        return b
    return init


def constant_gate_init(scale=1.0):
    """Initialize gate bias as constant value."""
    def init(key, shape, dtype=jnp.float32):
        b = jnp.ones(shape, dtype=dtype) * scale
        return b
    return init


def uniform_bias_init():
    """Initialize bias from uniform distribution [-1, 1]."""
    def init(key, shape, dtype=jnp.float32):
        b = jax.random.uniform(key, shape, dtype=dtype, minval=-1, maxval=1)
        return b
    return init


def uniform_bias_init_enc():
    """Initialize encoder bias from uniform distribution [0.5, 1]."""
    def init(key, shape, dtype=jnp.float32):
        b = jax.random.uniform(key, shape, dtype=dtype, minval=0.5, maxval=1)
        return b
    return init
