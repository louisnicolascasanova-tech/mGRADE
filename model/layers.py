"""
Neural network layer implementations for the Den-minGRU model.
Includes DCLS, Conv, MLP, GLU, and recurrent GRU layers.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

from .initializers import (
    constant_init, uniform_init, dilated_init,
    uniform_gate_init, constant_gate_init, uniform_bias_init
)
from .kernel_ops import (
    construct_kernel_fast, convolve_dcls,
    j_wrapper_jvjvj_mich_fft_k, j_wrapper_jvj_mich_fft_k
)


class DCLSLayer(nn.Module):
    """
    Delay Compensation Linear System (DCLS) layer.

    Implements learnable delays using Gaussian interpolation kernels.
    Supports both synaptic (output-dependent) and axonal (input-dependent) delays.
    Can use either standard convolution or FFT-based convolution for efficiency.

    Attributes:
        kernel_size: Temporal size of the kernel
        dim_in: Input dimension
        dim_out: Output dimension
        kernel_n_elems: Number of kernel elements (delays)
        fft: Whether to use FFT-based convolution
        delay_type: Type of delay ('synaptic' or 'axonal')
        delay_kernel: Type of kernel ('gaussian')
        init_std: Initial standard deviation for Gaussian kernels
        heterogeneous_weights: Whether weights vary per kernel element
        heterogeneous_positions: Whether positions vary per kernel element
        heterogeneous_std: Whether standard deviations vary per kernel element
        weight_init_scale: Scale factor for weight initialization
    """
    kernel_size: int
    dim_in: int
    dim_out: int
    kernel_n_elems: int = 1
    fft: bool = False
    delay_type: str = 'axonal'  # 'synaptic', 'axonal'
    delay_kernel: str = 'gaussian'  # 'impulse', WARNING: ONLY "gaussian" is implemented for now
    init_std: float = 5  # paper 0.23
    heterogeneous_weights: bool = True
    heterogeneous_positions: bool = True
    heterogeneous_std: bool = False
    weight_init_scale: float = 1.0  # scale for the weight initialization, only used if heterogeneous_weights is True

    def setup(self):
        params_shape = (self.kernel_n_elems, self.dim_out, self.dim_in) if self.delay_type == 'synaptic' else (self.kernel_n_elems, self.dim_in)
        kernel_ndim = 3 if self.delay_type == 'synaptic' else 2
        self.fft_fn = j_wrapper_jvjvj_mich_fft_k if self.delay_type == 'synaptic' else j_wrapper_jvj_mich_fft_k

        # Initialize weights
        if self.heterogeneous_weights:
            self.weights = self.param(f'weights', nn.initializers.variance_scaling(self.weight_init_scale, 'fan_in', 'truncated_normal'), params_shape)
        else:
            self.weights = self.param(f'weights', constant_init(), params_shape, 1.0)

        # Initialize positions
        if self.heterogeneous_positions:
            self.positions = self.param(f'positions', uniform_init(), params_shape, self.kernel_size-1)
        else:
            self.positions = self.param(f'positions', dilated_init(), params_shape, self.kernel_size, self.kernel_n_elems)

        # Initialize standard deviations
        if self.heterogeneous_std:
            self.std = self.param(f'std', uniform_init(), params_shape, self.init_std*0.8, self.init_std*1.2)
        else:
            self.std = self.param(f'std', constant_init(), params_shape, self.init_std)

        # Construct kernel by summing all elements
        for i in range(self.kernel_n_elems):
            if i == 0:
                self.kernel = construct_kernel_fast(self.weights[i], self.positions[i], self.std[i], self.kernel_size, kernel_ndim)
            else:
                self.kernel += construct_kernel_fast(self.weights[i], self.positions[i], self.std[i], self.kernel_size, kernel_ndim)

    def __call__(self, x):
        """
        Forward pass through DCLS layer.

        Args:
            x: Input tensor of shape (seq_len, dim_in). Batch dimension removed by vmap.

        Returns:
            Output tensor of shape (seq_len, dim_out)
        """
        if self.fft:
            x = x.T  # (dim_in, seq_len)
            out = self.fft_fn(x, self.kernel)
            out = jnp.swapaxes(out, 0, 1)  # (seq_len, dim_out)
        else:
            x = x[None, :, :]  # (1, seq_len, dim_in)
            if self.delay_type == 'synaptic':
                out = convolve_dcls(x, self.kernel)
                out = out[:, :-1]
            else:
                out = convolve_dcls(x, self.kernel[:, None, :], self.dim_in)  # kernel: (dim_out, dim_in/group=1, seq_len)
                out = out[:, :-1]
            out = out.squeeze(0)
        return out


class CausalDepthWiseConv1d(nn.Module):
    """
    Causal depthwise 1D convolution layer.

    Implements causal (left-padded) depthwise convolution with configurable dilation.
    Each input channel is convolved independently.

    Attributes:
        k_len: Kernel length
        in_channels: Number of input channels
        L: Sequence length
        dilation: Dilation factor for dilated convolution
    """
    k_len: int
    in_channels: int
    L: int  # sim_len
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through causal depthwise convolution.

        Args:
            x: Input tensor of shape (seq_len, in_channels)

        Returns:
            Output tensor of shape (seq_len, in_channels)
        """
        x = x[None, :, :]  # (1, seq_len, dim_in)

        # FULL convolution: manual padding
        print(self.dilation)  # needed for debugging
        padding = self.dilation * (self.k_len - 1)
        print(f'{padding=}')  # needed for debugging
        pad = [(0, 0), (padding, padding), (0, 0)]  # (batch, length, channels)
        x_padded = jnp.pad(x, pad)
        print(f'{x_padded.shape=}')

        x = nn.Conv(
            features=self.in_channels,  # 1 output per input channel
            kernel_size=(self.k_len,),
            feature_group_count=self.in_channels,  # Depthwise convolution
            use_bias=False,
            strides=(1,),
            padding="VALID",
            kernel_dilation=(self.dilation,),
        )(x_padded)

        return x[0][:self.L]


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with LayerNorm.

    Two-layer MLP with configurable activation function.
    Applies LayerNorm before processing.

    Attributes:
        dim: Output dimension (same as input)
        hidden_dim: Hidden layer dimension
        layer_act: Activation function ('linear', 'gelu', 'relu')
    """
    dim: int
    hidden_dim: int
    layer_act: str = 'gelu'  # 'linear', 'gelu', 'relu'

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through MLP.

        Args:
            x: Input tensor of shape (seq_len, dim)

        Returns:
            Output tensor of shape (seq_len, dim)
        """
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.variance_scaling(1, 'fan_in', 'truncated_normal'),
        )(x)

        if self.layer_act == 'linear':
            pass
        elif self.layer_act == 'relu':
            x = nn.relu(x)
        elif self.layer_act == 'gelu':
            x = nn.gelu(x)

        x = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.variance_scaling(1, 'fan_in', 'truncated_normal'),
        )(x)
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit with LayerNorm.

    Implements different gating strategies for channel mixing.

    Attributes:
        dim: Output dimension
        type: Gating type ('full', 'candidate', 'gate')
            - 'full': Both candidate and gate are learned
            - 'candidate': Input is used as gate, candidate is learned
            - 'gate': Input is candidate, gate is learned
    """
    dim: int
    type: str = 'full'  # 'full', 'candidate', 'gate'

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through GLU.

        Args:
            x: Input tensor of shape (seq_len, dim)

        Returns:
            Output tensor of shape (seq_len, dim)
        """
        x = nn.LayerNorm()(x)

        if self.type == 'full':
            x_g = nn.Dense(2*self.dim)(x)
            x, g = jnp.split(x_g, 2, axis=-1)
            x = x * nn.sigmoid(g)
        elif self.type == 'candidate':
            g = x
            x = nn.Dense(self.dim)(x)
            x = x * nn.sigmoid(g)
        elif self.type == 'gate':
            g = nn.Dense(self.dim)(x)
            x = x * nn.sigmoid(g)

        return x


class HeinsenMinGeneralGRULayer(nn.Module):
    """
    Parallel minimal GRU layer using Heinsen's method.

    Implements minGRU using parallel scan instead of sequential processing.
    This is much faster than traditional scan-based RNNs.

    Based on: "Efficient Parallelization of a Ubiquitous Sequential Computation" by Heinsen (2023)

    Attributes:
        hidden_dim: Hidden state dimension
        rec_act: Recurrent activation function ('linear', 'gelu', 'relu', 'sigmoid')
        training: Whether in training mode (for dropout)
        do_rate: Dropout rate
        skip_recurrent_dense: Skip dense layers (for dendritic DCLS)
        dense_z_weight_init_scale: Weight init scale for gate dense layer
        dense_z_bias_init: Bias initialization for gate ('zero', 'ugi', 'constant', 'ubi')
        dense_h_weight_init_scale: Weight init scale for candidate dense layer
        dense_h_bias_init: Bias initialization for candidate ('zero', 'ugi', 'constant', 'ubi')
    """
    hidden_dim: int
    rec_act: str  # 'linear', 'gelu', 'relu'
    training: bool
    do_rate: float
    skip_recurrent_dense: bool = False  # if the DCLS layer is dendritic, we need to skip the recurrent dense layer
    dense_z_weight_init_scale: float = 1.0  # scale for the weight initialization of the z preactivation
    dense_z_bias_init: str = 'zero'  # 'zero', 'ugi', 'constant'
    dense_h_weight_init_scale: float = 1.0
    dense_h_bias_init: str = 'zero'  # 'zero', 'ugi', 'constant'

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through minGRU layer.

        Args:
            x: Input tensor of shape (seq_len, input_dim)

        Returns:
            Tuple of (h_new, z_preact, h_tilde_preact, out) where:
                - h_new: Hidden states (seq_len, hidden_dim)
                - z_preact: Gate pre-activations (seq_len, hidden_dim)
                - h_tilde_preact: Candidate pre-activations (seq_len, hidden_dim)
                - out: Layer output with activation and dropout (seq_len, hidden_dim)
        """
        def softplus(x):
            return jnp.log(1 + jnp.exp(x))

        def safe_softplus(x):
            """Safe softplus that clips to prevent overflow."""
            x_safe = jnp.where(x > 7, 0, x)
            return jnp.where(x > 7, x, softplus(x_safe))

        def log_g(x):
            """
            Logarithm function with safe handling of edge cases.

            JAX's autodiff computes gradients for all branches in jnp.where,
            even the non-selected one (multiplied by 0). So we need to ensure
            all branches are well-defined to avoid NaN gradients.
            """
            x_safe = jnp.where(x == -0.5, x+1e-5, x)
            return jnp.where(x >= 0, jnp.log(x_safe+0.5), -safe_softplus(-x))

        def heinsen_update(z, h):
            """
            Heinsen's parallel scan for minGRU.

            Reformulates h_{t} = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
            as h_t = a_t * h_{t-1} + b_t using log-space cumulative operations.
            """
            x_0 = -10000.
            log_a = -safe_softplus(z)
            log_b = -safe_softplus(-z) + log_g(h)
            a_star = jnp.cumsum(log_a)

            c = log_b - a_star
            c = jnp.pad(c, (1, 0), constant_values=x_0)
            d = jax.lax.cumlogsumexp(c)
            d = d[1:]  # tailing d
            log_final = a_star + d
            return jnp.exp(log_final)

        vj_heinsen_update = jax.vmap(jax.jit(heinsen_update), in_axes=(1, 1), out_axes=1)

        def update(x):
            """Core GRU update logic."""
            if self.skip_recurrent_dense:
                z_preact, h_tilde_preact = jnp.split(x, 2, axis=-1)
            else:
                bias_inits = {
                    'zero': nn.initializers.zeros,
                    'ugi': uniform_gate_init(),
                    'constant': constant_gate_init(scale=-0.5),
                    'ubi': uniform_bias_init(),
                }
                if self.dense_z_bias_init not in bias_inits:
                    raise ValueError(f"Unknown dense_z_bias_init: {self.dense_z_bias_init}")

                if self.dense_h_bias_init not in bias_inits:
                    raise ValueError(f"Unknown dense_h_bias_init: {self.dense_h_bias_init}")

                z_preact = nn.Dense(
                    self.hidden_dim, name='Dense_z',
                    kernel_init=nn.initializers.variance_scaling(self.dense_z_weight_init_scale, 'fan_in', 'truncated_normal'),
                    bias_init=bias_inits[self.dense_z_bias_init]
                )(x)
                h_tilde_preact = nn.Dense(
                    self.hidden_dim, name='Dense_h',
                    kernel_init=nn.initializers.variance_scaling(self.dense_h_weight_init_scale, 'fan_in', 'truncated_normal'),
                    bias_init=bias_inits[self.dense_h_bias_init]
                )(x)

            h_new = vj_heinsen_update(z_preact, h_tilde_preact)  # h_new: (seq_len, hidden_dim)

            # Apply recurrent activation
            if self.rec_act == 'linear':
                out = h_new
            elif self.rec_act == 'sigmoid':
                out = nn.sigmoid((h_new-0.5)*10)
            elif self.rec_act == 'gelu':
                out = nn.gelu(h_new-1)
            elif self.rec_act == 'relu':
                out = nn.relu(h_new-2)

            # Apply dropout
            out = nn.Dropout(rate=self.do_rate, broadcast_dims=(0,), deterministic=not self.training)(out)

            return (h_new, z_preact, h_tilde_preact, out)

        state_hist = update(x)
        return state_hist
