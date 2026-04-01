"""
Configuration dataclasses for training.

This module contains configuration dataclasses to reduce function parameter complexity
and improve code organization. These configs can be imported without circular dependencies.
"""
from dataclasses import dataclass
from typing import Optional, List, Callable

import jax.numpy as jnp


@dataclass
class EpochConfig:
    """Configuration for training epoch.

    Groups epoch-level training parameters to reduce function signature complexity.

    Args:
        lim_batch: Optional batch limit for debugging/testing
        keys_to_track: List of top-level keys to track in auxiliary history
        inner_keys_to_track: List of nested keys to track in auxiliary history
        lr_fn: Optional learning rate function for logging
        grad_clip_norm: Gradient clipping threshold (element-wise)
        log_model_behavior: Whether to log model behavior metrics
        epoch_num: Current epoch number
        class_weights: Optional class weights for loss weighting
        dtype: JAX data type to use
    """
    lim_batch: Optional[int] = None
    keys_to_track: Optional[List[str]] = None
    inner_keys_to_track: Optional[List[str]] = None
    lr_fn: Optional[Callable] = None
    grad_clip_norm: float = 1.0
    log_model_behavior: bool = True
    epoch_num: int = 0
    class_weights: Optional[jnp.ndarray] = None
    dtype: jnp.dtype = jnp.float32

    def __post_init__(self):
        """Initialize empty lists for tracking keys if None."""
        if self.keys_to_track is None:
            self.keys_to_track = []
        if self.inner_keys_to_track is None:
            self.inner_keys_to_track = []


@dataclass
class WandBLogConfig:
    """WandB logging configuration.

    Controls what gets logged to Weights & Biases during training.

    Args:
        log_gradients: Whether to log gradient statistics
        log_states: Whether to log hidden state statistics
        log_matrices: Whether to log parameter matrices as images
        in_dim: Input dimension (required for some logging)
        seq_len: Sequence length (required for some logging)
    """
    log_gradients: bool = False
    log_states: bool = False
    log_matrices: bool = False
    in_dim: Optional[int] = None
    seq_len: Optional[int] = None
