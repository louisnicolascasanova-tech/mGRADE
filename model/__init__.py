"""
Den-minGRU Model Package

Refactored model architecture with modular components:
- initializers: Weight and bias initialization functions
- kernel_ops: Kernel construction and convolution operations
- layers: Core layer implementations (DCLS, Conv, MLP, GLU, GRU)
- utils: Utility functions (pooling, etc.)
- backbone: Main RNN backbone models
- retrieval: Retrieval-specific models for document matching

Public API maintains backward compatibility with original model.py
"""

# ============================================================================
# Public API Exports (for backward compatibility)
# ============================================================================

# Main models (most commonly used)
from .backbone import BatchRNN_General, RNN_General_Backbone
from .retrieval import RNN_General_Retrieval_Backbone, RetrievalDecoder

# Utility functions
from .utils import batch_masked_meanpool, masked_meanpool

# Kernel operations (used by custom_logging.py and inf.py)
from .kernel_ops import construct_kernel_fast

# Layer classes (for advanced usage)
from .layers import (
    DCLSLayer,
    CausalDepthWiseConv1d,
    MLP,
    GLU,
    HeinsenMinGeneralGRULayer
)

# Initialization functions (for custom layers)
from .initializers import (
    identity_weight_init,
    uniform_init,
    constant_init,
    dilated_init,
    uniform_gate_init,
    constant_gate_init,
    uniform_bias_init,
    uniform_bias_init_enc
)

# Additional kernel operations
from .kernel_ops import (
    gaussian_interpolation,
    convolve,
    convolve_dcls,
    j_wrapper_jvjvj_mich_fft_k,
    j_wrapper_jvj_mich_fft_k
)

__all__ = [
    # Main models
    'BatchRNN_General',
    'RNN_General_Backbone',
    'RNN_General_Retrieval_Backbone',
    'RetrievalDecoder',

    # Utilities
    'batch_masked_meanpool',
    'masked_meanpool',
    'construct_kernel_fast',

    # Layers
    'DCLSLayer',
    'CausalDepthWiseConv1d',
    'MLP',
    'GLU',
    'HeinsenMinGeneralGRULayer',

    # Initializers
    'identity_weight_init',
    'uniform_init',
    'constant_init',
    'dilated_init',
    'uniform_gate_init',
    'constant_gate_init',
    'uniform_bias_init',
    'uniform_bias_init_enc',

    # Kernel operations
    'gaussian_interpolation',
    'convolve',
    'convolve_dcls',
    'j_wrapper_jvjvj_mich_fft_k',
    'j_wrapper_jvj_mich_fft_k',
]
