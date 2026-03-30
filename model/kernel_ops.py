"""
Kernel construction and convolution operations for DCLS layers.
Includes Gaussian interpolation, fast kernel construction, and FFT-based convolution.
"""
import jax
import jax.numpy as jnp
from functools import partial


def gaussian_interpolation(w, p, s, kernel_size):
    """
    Recovers the Gaussian kernel from the parameters w, p, s.

    Args:
        w (float32): The weights of the kernel. Shape: (dim_out, dim_in, 1) or (dim_in, 1)
        p (int32): The positions of the kernel. Shape: (dim_out, dim_in, 1) or (dim_in, 1)
        s (float32): The standard deviations of the kernel. Shape: (dim_out, dim_in, 1) or (dim_in, 1)
        kernel_size (int32): The temporal size of the kernel.

    Returns:
        float32: The FLIPPED gaussian kernel. Shape: (dim_out, dim_in, kernel_size) or (dim_in, kernel_size)
    """
    k = jnp.arange(kernel_size-1, -1, -1)
    return w * jnp.exp(- (p - k)**2 / (2 * s**2))


def construct_kernel_fast(w, p, s, kernel_size, kernel_ndim):
    """
    Constructs the kernel used for the DCLS layer using fast broadcasting.

    Args:
        w (float32): The weights of the kernel. Shape: (dim_out, dim_in) or (dim_in,)
        p (int32): The positions of the kernel. Shape: (dim_out, dim_in) or (dim_in,)
        s (float32): The standard deviations of the kernel. Shape: (dim_out, dim_in) or (dim_in,)
        kernel_size (int32): The temporal size of the kernel.
        kernel_ndim (int32): The number of dimensions of the kernel (2 or 3).

    Returns:
        float32: The gaussian kernel. Shape: (dim_out, dim_in, kernel_size) or (dim_in, kernel_size)
    """
    # Reshape w, p, s to enable broadcasting:
    if kernel_ndim == 3:
        # (dim_out, dim_in) → (dim_out, dim_in, 1)
        w = w[:, :, None]
        p = p[:, :, None]
        s = s[:, :, None]
    elif kernel_ndim == 2:
        # (dim_in,) → (dim_in, 1)
        w = w[:, None]
        p = p[:, None]
        s = s[:, None]

    # Compute Gaussian interpolation
    return gaussian_interpolation(w, p, s, kernel_size)


@jax.jit
def convolve(x, kernel):
    """
    Basic convolution operation (legacy, not used in production).

    Args:
        x: Input tensor of shape (batch, channels, length)
        kernel: Convolution kernel

    Returns:
        Convolved output
    """
    x = jnp.pad(x, ((0, 0), (0, 0), (kernel.shape[-1]-1, 0)))
    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NCL', 'OIL', 'NCL')  # (input, kernel, output)
    )


@partial(jax.jit, static_argnums=(2,))
def convolve_dcls(x, kernel, group=1):
    """
    DCLS-specific convolution with group support.

    Args:
        x: Input tensor of shape (batch, length, channels)
        kernel: Convolution kernel
        group: Number of groups for grouped convolution (default=1)

    Returns:
        Convolved output of shape (batch, length+kernel_size, channels)
    """
    x = jnp.pad(x, ((0, 0), (kernel.shape[-1], 0), (0, 0)))
    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1,),
        padding='VALID',
        feature_group_count=group,
        dimension_numbers=('NLC', 'OIL', 'NLC')  # (input, kernel, output)
    )


# ============================================================================
# FFT-based Convolution Operations
# ============================================================================

def mich_fft_k(input, K):
    """
    FFT-based convolution for efficient delay computation.

    Args:
        input: 1D array of shape (sim_len,)
        K: 1D array of shape (d_max,) - kernel representing delays

    Returns:
        1D array of shape (sim_len,) - convolved output
    """
    d_max = K.shape[-1]
    sim_len = input.shape[-1]

    # Cast to float32 for FFT operations, then cast back
    input_dtype = input.dtype

    input_fft = jnp.fft.rfft(jnp.pad(input.astype(jnp.float32), (0, d_max)))
    K_fft = jnp.fft.rfft(jnp.pad(K.astype(jnp.float32), (0, sim_len)))
    I_fft = K_fft * input_fft
    I = jnp.fft.irfft(I_fft)[:sim_len]
    return I.astype(input_dtype)


# JIT and vmap wrappers for efficient batch processing
j_mich_fft_k = jax.jit(mich_fft_k)
vj_mich_fft_k = jax.vmap(j_mich_fft_k, in_axes=(0, 0))
jvj_mich_fft_k = jax.jit(vj_mich_fft_k)
vjvj_mich_fft_k = jax.vmap(jvj_mich_fft_k, in_axes=(None, 0))
jvjvj_mich_fft_k = jax.jit(vjvj_mich_fft_k)


def wrapper_jvjvj_mich_fft_k(input, K):
    """Wrapper for synaptic delays - sums over kernel dimension."""
    return jvjvj_mich_fft_k(input, K).sum(1)


j_wrapper_jvjvj_mich_fft_k = jax.jit(wrapper_jvjvj_mich_fft_k)


def wrapper_jvj_mich_fft_k(input, K):
    """Wrapper for axonal delays."""
    return jvj_mich_fft_k(input, K)


j_wrapper_jvj_mich_fft_k = jax.jit(wrapper_jvj_mich_fft_k)
