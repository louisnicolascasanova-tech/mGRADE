import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import lecun_normal

from typing import Sequence
from collections import namedtuple

def identity_weight_init():
    def init(key, shape, dtype=jnp.float32):
        w = jnp.eye(shape[0], shape[1], dtype=dtype) * 0.9
        w = w.at[-1, :].set(1.0)
        return w
    return init


def uniform_init():
    def init(key, shape, max_val, min_val=0, dtype=jnp.float32):
        p = jax.random.uniform(key, shape, dtype=dtype, minval=min_val, maxval=max_val)
        return p
    return init


def constant_init():
    def init(key, shape, offset, dtype=jnp.float32):
        p = jnp.ones(shape, dtype=dtype) * offset
        return p
    return init

def dilated_init():
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

class minGRULayer(nn.Module):
    '''
    MGU Layer
    We use a modified version of the h_tilde calculation in the MGU cell for better performance.
    - $f_t = \sigma(W^{fx} x_t + U^{fh} h_{t-1} + b^f)$
    - $\~{h}_t = \tanh(W^{hx} x_t + f_t \odot (U^{hh} h_{t-1} + b^h))$
    
    Args:
        hidden_dim: int, size of hidden state
    '''
    hidden_dim: int

    @nn.compact
    def __call__(self, x):

        def update(self, state, x):
            h = state[0]
            z_htilde = nn.Dense(2*self.hidden_dim, name='Dense_x')(x)
            z_preact, h_tilde = jnp.split(z_htilde, 2, axis=-1)
            z = nn.sigmoid(z_preact)
            h_new = (1 - z) * h + z * h_tilde
            return (h_new,), (h_new, z, z_preact)

        scan_update = nn.scan(
            update,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=0,
            out_axes=0,
        )
        h = jnp.zeros((self.hidden_dim,))
        state = (h,)
        _, state_hist = scan_update(self, state, x)

        return state_hist


class HeinsenMinGRULayer(nn.Module):
    '''
    MGU Layer
    We use a modified version of the h_tilde calculation in the MGU cell for better performance.
    - $f_t = \sigma(W^{fx} x_t + U^{fh} h_{t-1} + b^f)$
    - $\~{h}_t = \tanh(W^{hx} x_t + f_t \odot (U^{hh} h_{t-1} + b^h))$
    
    Args:
        hidden_dim: int, size of hidden state
        out_dim: int, size of output
    '''
    hidden_dim: int
    latent_dim: int
    layer_act: str # 'linear', 'tanh', 'sigmoid', 'relu'
    training: bool
    do_rate: float 

    @nn.compact
    def __call__(self, x):


        def softplus(x):
            return jnp.log(1 + jnp.exp(x))
        def safe_softplus(x):
            x_safe = jnp.where(x > 10, 0, x)
            return jnp.where(x > 10, x, softplus(x_safe))
        def log_g(x):
            x_safe = jnp.where(x == -0.5, x+1e-5, x)
            return jnp.where(x >= 0, jnp.log(x_safe+0.5), -safe_softplus(-x))        

        def heinsen_update(z, h):
            '''
            we can formulate h_{t} = (1 - z_t) * h_{t-1} + z_t * h_tilde_t as 
            h_t = a_t * h_{t-1} + b_t with a_t = 1 - z_t and b_t = z_t * h_tilde_t which can be precomputed
            '''
            x_0 = -10000.
            log_a = -safe_softplus(z)
            log_b = -safe_softplus(-z) + log_g(h)
            a_star = jnp.cumsum(log_a)

            c = log_b - a_star 
            c = jnp.pad(c, (1, 0), constant_values=x_0)
            d = jax.lax.cumlogsumexp(c)
            d = d[1:] # tailing d
            log_final = a_star + d
            return jnp.exp(log_final)
        
        vj_heinsen_update = jax.vmap(jax.jit(heinsen_update), in_axes=(1, 1), out_axes=1)

        def update(x):
            '''
            As the x is not scanned anymore, x is 2D array (n_ts, n_features)
            - for sMNIST, the first layer will have x of shape (784, 1), the other layers will have x of shape (784, 10) assuming latent_dim = 10

            '''
            z_htilde = nn.Dense(2*self.hidden_dim, name='Dense_x')(x) # z_htilde: (784, 2*64), assuming hidden_dim = 64
            z_preact, h_tilde_preact = jnp.split(z_htilde, 2, axis=-1) # z_preact and h_tilde_preact: (784, 64)
            h_new = vj_heinsen_update(z_preact, h_tilde_preact) # h_new: (784, 64)
            if self.latent_dim is not None:
                out_preact = nn.Dense(self.latent_dim, name='Dense_latent')(h_new) # out: (784, 10), assuming latent_dim = 10
            else:
                out_preact = h_new 
            if self.layer_act == 'tanh':
                out = nn.tanh(out_preact) # out: (784, 10)
            elif self.layer_act == 'sigmoid':
                out = nn.sigmoid((out_preact-0.5)*10)
            elif self.layer_act == 'relu':
                out = nn.relu(out_preact-1)
            elif self.layer_act == 'linear':
                out = out_preact
            elif self.layer_act == 'gelu':
                out = nn.gelu(out_preact-1)
            else: 
                raise ValueError(f"Unknown activation type: {self.layer_act}")
            # define the dropout layer
            out = nn.Dropout(rate=self.do_rate, broadcast_dims=(0,), deterministic=not self.training)(out)
            return (h_new, z_preact, h_tilde_preact, out, out_preact)
        #{'h_new': h_new, 'z_preact': z_preact, 'h_tilde_preact': h_tilde_preact, 'out': out, 'out_preact': out_preact}
        
        state_hist = update(x)
        return state_hist






# NOTE: never use, it was just for testing
def construct_kernel(w, p, s, dim_out, dim_in, kernel_size):
    def _gaussian_interpolation(x, s):
        return jnp.exp(-x**2/(2*(s**2)))

    kernel = jnp.zeros((dim_out, dim_in, kernel_size))
    for i in range(dim_out):
        for j in range(dim_in):
            for k in range(kernel_size):
                kernel = kernel.at[i, j, k].set(_gaussian_interpolation(p[i, j] - k, s[i,j]) * w[i, j]) 
    return kernel

def gaussian_interpolation(w, p, s, kernel_size):
    """ Recovers the Gaussian kernel from the parameters w, p, s.
    Args:
        w (float32):    The weights of the kernel.              shape: (dim_out, dim_in, 1) or (dim_in, 1)
        p (int32):      The positions of the kernel.            shape: (dim_out, dim_in, 1) or (dim_in, 1)
        s (float32):    The standard deviations of the kernel.  shape: (dim_out, dim_in, 1) or (dim_in, 1)
        kernel_size (int32): The temporal size of the kernel.

    Returns:
        float32: The FLIPED gaussian kernel. shape: (dim_out, dim_in, kernel_size) or (dim_in, kernel_size)
    """
    k = jnp.arange(kernel_size-1, -1, -1)
    return w * jnp.exp(- (p - k)**2 / (2 * s**2))
    

def construct_kernel_fast(w, p, s, kernel_size, kernel_ndim):
    """ Constructs the kernel used for the DLCS layer.
    Args:
        w (float32):    The weights of the kernel.              shape: (dim_out, dim_in) or (dim_in,)
        p (int32):      The positions of the kernel.            shape: (dim_out, dim_in) or (dim_in,)
        s (float32):    The standard deviations of the kernel.  shape: (dim_out, dim_in) or (dim_in,)
        kernel_size (int32): The temporal size of the kernel.
        kernel_ndim (int32): The number of dimensions of the kernel.
    Returns:
        float32: The gaussian kernel. shape: (dim_out, dim_in, kernel_size) or (dim_in, kernel_size)
    """

    # Reshape w, p, s to enable broadcasting:
    if kernel_ndim == 3:
        # print(f'hello 3d')
        # (dim_out, dim_in) → (dim_out, dim_in, 1)
        w = w[:, :, None]
        p = p[:, :, None]
        s = s[:, :, None]
    elif kernel_ndim == 2:
        # print(f'hello 2d')
        # squeeze to remove first dimension of 
        # w = w.squeeze(0)
        # (dim_in,) → (dim_in, 1)
        # print(f'{w.shape=}, {p.shape=}, {s.shape=}')
        w = w[:, None]
        p = p[:, None]
        s = s[:, None]
        # print(f'{w.shape=}, {p.shape=}, {s.shape=}')

    # Compute Gaussian interpolation
    return gaussian_interpolation(w, p, s, kernel_size)  # shape (dim_out, dim_in, kernel_size)

# def construct_2d_kernel_fast(w, p, s, dim_in, kernel_size):
#     """
#     Constructs a 2D kernel of shape (dim_in, kernel_size)
#     using Gaussian interpolation with parameters per input channel.
#     """
#     # w, p, s: shape (dim_in,)
#     # squeeze to remove first dimension
#     w = w.squeeze(0)
#     k = jnp.arange(kernel_size)  # shape (kernel_size,)

#     # Reshape for broadcasting: (dim_in, 1) - (1, kernel_size)
#     w = w[:, None]
#     p = p[:, None]
#     s = s[:, None]

#     # Compute Gaussian interpolation
#     kernel = w * jnp.exp(- (p - k)**2 / (2 * s**2))  # shape (dim_in, kernel_size)
#     return kernel
@jax.jit
def convolve(x, kernel):
    x = jnp.pad(x, ((0, 0), (0, 0), (kernel.shape[-1]-1, 0)))
    # print(x.shape)
    return jax.lax.conv_general_dilated(
                lhs=x,
                rhs=kernel,
                window_strides=(1,),
                padding='VALID',
                dimension_numbers=('NCL', 'OIL', 'NCL')  # (input, kernel, output), N: batch, L: length, C: channels, I: n_in/group, O: n_out
)

from functools import partial
@partial(jax.jit, static_argnums=(2,))
def convolve_dcls(x, kernel, group=1):
    x = jnp.pad(x, ((0, 0), (kernel.shape[-1], 0), (0, 0)))
    # print(f'{x.shape=}')
    # print(f'{kernel.shape=}')
    # print(x.shape)
    return jax.lax.conv_general_dilated(
                lhs=x,
                rhs=kernel,
                window_strides=(1,),
                padding='VALID',
                feature_group_count=group,
                dimension_numbers=('NLC', 'OIL', 'NLC')  # (input, kernel, output), N: batch, L: length, C: channels, I: n_in/group, O: n_out
)


    
def mich_fft_k(input, K):
    '''
    :param input: 1D array of shape (sim_len,)
    :param K: 1D array of shape (d_max,). n_rep-hot encoded delays
    :return: 1D array of shape (sim_len+d_max,)
    '''
    # print(f'{input.shape=}')
    # print(f'{K.shape=}')
    d_max = K.shape[-1]
    sim_len = input.shape[-1]
    input_fft = jnp.fft.rfft(jnp.pad(input, (0, d_max)))
    K_fft = jnp.fft.rfft(jnp.pad(K, (0, sim_len)))
    I_fft = K_fft * input_fft
    I = jnp.fft.irfft(I_fft)[:sim_len]
    return I

j_mich_fft_k = jax.jit(mich_fft_k)
vj_mich_fft_k = jax.vmap(j_mich_fft_k, in_axes=(0, 0))
jvj_mich_fft_k = jax.jit(vj_mich_fft_k)
vjvj_mich_fft_k = jax.vmap(jvj_mich_fft_k, in_axes=(None, 0))
jvjvj_mich_fft_k = jax.jit(vjvj_mich_fft_k)

def wrapper_jvjvj_mich_fft_k(input, K):
    return jvjvj_mich_fft_k(input, K).sum(1)

j_wrapper_jvjvj_mich_fft_k = jax.jit(wrapper_jvjvj_mich_fft_k)

def wrapper_jvj_mich_fft_k(input, K):
    return jvj_mich_fft_k(input, K)

j_wrapper_jvj_mich_fft_k = jax.jit(wrapper_jvj_mich_fft_k)


class DCLSLayer(nn.Module):
    kernel_size: int
    dim_in: int
    dim_out: int
    kernel_n_elems: int = 1
    fft: bool = False
    delay_type: str = 'axonal' # 'synaptic', 'axonal'
    delay_kernel: str = 'gaussian' # 'impulse', WARNING: ONLY "gaussian" is implemented for now
    init_std: float = 5 # paper 0.23
    heterogeneous_weights: bool = True
    heterogeneous_positions: bool = True
    heterogeneous_std: bool = False
    def setup(self):

        params_shape = (self.kernel_n_elems, self.dim_out, self.dim_in) if self.delay_type == 'synaptic' else (self.kernel_n_elems, self.dim_in)
        kernel_ndim = 3 if self.delay_type == 'synaptic' else 2
        self.fft_fn = j_wrapper_jvjvj_mich_fft_k if self.delay_type == 'synaptic' else j_wrapper_jvj_mich_fft_k
        # weight_shape = params_shape if self.delay_type == 'synaptic' else (1, self.dim_in) # needed beacause lecun_normal() needs 2D shape
        if self.heterogeneous_weights:
            self.weights = self.param(f'weights', lecun_normal(), params_shape)
        else:
            self.weights = self.param(f'weights', constant_init(), params_shape, 1.0)
        # sample positions from uniform distribution [0, kernel_size]
        if self.heterogeneous_positions:
            self.positions = self.param(f'positions', uniform_init(), params_shape, self.kernel_size-1)
        else: 
            self.positions = self.param(f'positions', dilated_init(), params_shape, self.kernel_size, self.kernel_n_elems)
        if self.heterogeneous_std:
            self.std = self.param(f'std', uniform_init(), params_shape, self.init_std*0.8, self.init_std*1.2)
        else:
            self.std = self.param(f'std', constant_init(), params_shape, self.init_std)
        for i in range(self.kernel_n_elems):
            # print(self.weights.shape)
            # print(self.positions.shape)
            # print(self.std.shape)
            if i == 0:
                self.kernel = construct_kernel_fast(self.weights[i], self.positions[i], self.std[i], self.kernel_size, kernel_ndim)
            else: 
                self.kernel += construct_kernel_fast(self.weights[i], self.positions[i], self.std[i], self.kernel_size, kernel_ndim)


    def __call__(self, x):
        '''
        x: (seq_len, dim_in, ), the batch dimension has been removed from using VMAP from flax
        '''
        if self.fft:
            # print(f'{x.shape=}')
            x = x.T # (dim_in, seq_len)
            # print(f'self.kernel.shape={self.kernel.shape}')
            out = self.fft_fn(x, self.kernel)
            # print(f'{out.shape=}')
            out = jnp.swapaxes(out, 0, 1) # (seq_len, dim_out)
        else: 
            x = x[None, :, :] # (1, seq_len, dim_in)
            # print(f'x.shape={x.shape}')
            # print(f'self.kernel.shape={self.kernel.shape}')
            if self.delay_type == 'synaptic':
                # print(f'{x.shape=}')
                # print(f'{self.kernel.shape=}')
                out = convolve_dcls(x, self.kernel)
                out = out[:, :-1]
            else:
                # print(x.shape)
                out = convolve_dcls(x, self.kernel[:, None, :], self.dim_in) # kernel: (dim_out, dim_in/group=1, seq_len)
                out = out[:, :-1]
                # print(out.shape)
                
            out = out.squeeze(0)
        return out

class CausalDepthWiseConv1d(nn.Module):
    k_len: jnp.ndarray  # [kernel_len]
    in_channels: int
    L: int  # sim_len
    dilation: int = 1

    
    @nn.compact
    def __call__(self, x):

        x = x[None, :, :]  # (1, seq_len, dim_in)
        # FULL convolution: manual padding
        print(self.dilation) # needed for debugging 
        padding = self.dilation * (self.k_len - 1)
        print(f'{padding=}') # needed for debugging
        pad = [(0, 0), (padding, padding), (0, 0)]  # (batch, length, channels)
        x_padded = jnp.pad(x, pad)
        print(f'{x_padded.shape=}')
        x = nn.Conv(
            features=self.in_channels,  # 1 output per input channel
            kernel_size=(self.k_len,),
            feature_group_count=self.in_channels,  # <-- Key part!
            use_bias=False,
            strides=(1,),
            padding="VALID",
            kernel_dilation=(self.dilation,),
        )(x_padded) # axonal convolution
        # x = nn.Conv(
        #     features=self.in_channels,  # 1 output per input channel
        #     kernel_size=(1,),
        #     feature_group_count=1,  # <-- Key part!
        #     use_bias=False,
        #     strides=(1,),
        #     padding="VALID",
        # )(x)
        return x[0][:self.L]

class MLP(nn.Module):
    dim: int
    hidden_dim: int
    layer_act: str = 'gelu' # 'linear', 'gelu', 'relu'

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        if self.layer_act == 'linear':
            pass
        elif self.layer_act == 'relu':
            x = nn.relu(x)
        elif self.layer_act == 'gelu':
            x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x
    
class GLU(nn.Module):
    dim: int
    type: str = 'full' # 'full', 'candidate', 'gate'

    @nn.compact
    def __call__(self, x):
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
    '''
    MGU Layer
    We use a modified version of the h_tilde calculation in the MGU cell for better performance.
    - $f_t = \sigma(W^{fx} x_t + U^{fh} h_{t-1} + b^f)$
    - $\~{h}_t = \tanh(W^{hx} x_t + f_t \odot (U^{hh} h_{t-1} + b^h))$
    
    Args:
        hidden_dim: int, size of hidden state
        out_dim: int, size of output
    '''
    hidden_dim: int
    rec_act: str # 'linear', 'gelu', 'relu'
    training: bool
    do_rate: float 
    skip_recurrent_dense: bool = False # if the DCLS layer is dendritic, we need to skip the recurrent dense layer

    @nn.compact
    def __call__(self, x):


        def softplus(x):
            return jnp.log(1 + jnp.exp(x))
        def safe_softplus(x):
            x_safe = jnp.where(x > 10, 0, x)
            return jnp.where(x > 10, x, softplus(x_safe))
        def log_g(x):
            x_safe = jnp.where(x == -0.5, x+1e-5, x)
            return jnp.where(x >= 0, jnp.log(x_safe+0.5), -safe_softplus(-x))        

        def heinsen_update(z, h):
            '''
            we can formulate h_{t} = (1 - z_t) * h_{t-1} + z_t * h_tilde_t as 
            h_t = a_t * h_{t-1} + b_t with a_t = 1 - z_t and b_t = z_t * h_tilde_t which can be precomputed
            '''
            x_0 = -10000.
            log_a = -safe_softplus(z)
            log_b = -safe_softplus(-z) + log_g(h)
            a_star = jnp.cumsum(log_a)

            c = log_b - a_star 
            c = jnp.pad(c, (1, 0), constant_values=x_0)
            d = jax.lax.cumlogsumexp(c)
            d = d[1:] # tailing d
            log_final = a_star + d
            return jnp.exp(log_final)
        
        vj_heinsen_update = jax.vmap(jax.jit(heinsen_update), in_axes=(1, 1), out_axes=1)

        def update(x):
            '''
            As the x is not scanned anymore, x is 2D array (n_ts, n_features)
            - if the DCLS layer is dendritic, x will instead have shape (n_ts, 2*n_feaatures) and be equivalent to z_htilde
            - for sMNIST, the first layer will have x of shape (784, 1), the other layers will have x of shape (784, 10) assuming latent_dim = 10

            '''
            if self.skip_recurrent_dense:
                z_htilde = x
            else:
                z_htilde = nn.Dense(2*self.hidden_dim, name='Dense_x')(x) # z_htilde: (784, 2*64), assuming hidden_dim = 64
            z_preact, h_tilde_preact = jnp.split(z_htilde, 2, axis=-1) # z_preact and h_tilde_preact: (784, 64)
            # z_preact = DCLSLayer(kernel_size=50, dim_out=self.hidden_dim, dim_in=self.hidden_dim)(z_preact) 
            h_new = vj_heinsen_update(z_preact, h_tilde_preact) # h_new: (784, 64)
            # h_new needs to be scaled and shifted for the nonlinear activation to work properly
            if self.rec_act == 'linear':
                out = h_new
            elif self.rec_act == 'sigmoid':
                out = nn.sigmoid((h_new-0.5)*10)
            elif self.rec_act == 'gelu':
                out = nn.gelu(h_new-1)
            elif self.rec_act == 'relu':
                out = nn.relu(h_new-1)
            return (h_new, z_preact, h_tilde_preact, out)
        #{'h_new': h_new, 'z_preact': z_preact, 'h_tilde_preact': h_tilde_preact, 'out': out, 'out_preact': out_preact}
        
        state_hist = update(x)
        return state_hist


class RNN_General_Backbone(nn.Module):
    training: bool 
    n_layers: int
    out_dim: int
    hidden_dim: Sequence
    do_rate: float
    # ENCODER
    encoder: bool = True
    # SKIP
    layer_skip: bool = False
    element_skip: bool = False
    # CONVOLUTION
    enable_conv: bool = True
    conv_layer: str = 'dcls' # 'dcls', 'conv', None
    kernel_size: int = 50
    kernel_n_elems: int = 1 # number of elements in the kernel, for synaptic delays this is the number of delays
    dcls_std: float = 0.7
    dcls_type: str = 'synaptic' # 'synaptic', 'axonal'
    dcls_kernel: str = 'gaussian' # 'impulse', WARNING: ONLY "gaussian" is implemented for now
    dcls_fft: bool = False
    dcls_heterogeneous_weights: bool = True
    dcls_heterogeneous_positions: bool = True
    dcls_heterogeneous_std: bool = False
    wavenet_dilation: bool = False
    dilation_schedule: str = 'clip' # 'clip', 'wrap', 'linear
    dilation_boundary: int = None # when to start the dilation schedule
    dilation_offset: int = 0 # offset to the dilation schedule, 
    constant_dilation: int = 1 # if wavenet_dilation is False, this is the dilation factor
    # RECURRENT
    enable_rec: bool = True
    recurrent_layer: nn.Module = HeinsenMinGeneralGRULayer
    rec_act: str = 'linear' # 'linear', 'gelu', 'relu'
    # CHANNEL MIXING
    enable_cm: bool = True
    channel_mixing: str = 'none' # 'none', 'mlp', 'glu'
    cm_act: str = 'relu' # 'linear', 'gelu', 'relu'
    glu_type: str = 'full' # 'full', 'candidate', 'gate'
    # COMPRESSION
    latent_dim: Sequence = None
    comp_act: str = 'linear' # 'linear', 'gelu', 'relu'
    # NORMALIZATION
    postnorm: bool = False


    @nn.compact
    def __call__(self, x):

        # ========== ENCODER ==========
        if self.encoder:
            x = nn.Dense(self.hidden_dim[0], name='Encoder')(x)

        # ========== SEQUENCE BLOCKS ==========
        state_hist = []
        skip_recurrent_dense = False # if the DCLS layer is dendritic, we need to skip the recurrent dense layer
        for i in range(self.n_layers):

            # ========== LAYER SKIP: SOURCE ==========
            # print(f'{x.shape=}')
            if self.layer_skip: 
                layer_skip = x

            # ========== CONVOLUTION BLOCK ==========
            if self.enable_conv == True:
                if self.element_skip:
                    conv_skip = x

                if self.conv_layer == 'dcls':
                    output_dim = self.hidden_dim[i]
                    if self.dcls_type == 'dendritic':
                        x = nn.Dense(2 * self.hidden_dim[i])(x)
                        skip_recurrent_dense = True
                        output_dim = 2 * self.hidden_dim[i]
                    x = DCLSLayer(kernel_size=self.kernel_size, dim_out=output_dim, dim_in=x.shape[-1], kernel_n_elems=self.kernel_n_elems,
                                    fft=self.dcls_fft, delay_type=self.dcls_type, delay_kernel=self.dcls_kernel,
                                    init_std=self.dcls_std, 
                                    heterogeneous_weights=self.dcls_heterogeneous_weights,
                                    heterogeneous_positions=self.dcls_heterogeneous_positions,
                                    heterogeneous_std=self.dcls_heterogeneous_std,
                                    )(x)
                    
                elif self.conv_layer == 'conv':
                    if self.wavenet_dilation == True:
                        if self.dilation_schedule == 'clip':
                            dilation_i = min(self.dilation_boundary-1, i+self.dilation_offset) 
                        elif self.dilation_schedule == 'wrap':
                            dilation_i = self.dilation_offset + i % (self.dilation_boundary-self.dilation_offset)
                        else: 
                            dilation_i = 0 # if dilation_schedule is None, 
                        dilation = 2 ** dilation_i 
                    else: 
                        dilation = self.constant_dilation
                    in_channels = 1 if i == 0 and self.encoder == False else self.hidden_dim[i]
                    x = CausalDepthWiseConv1d(k_len=self.kernel_size, in_channels=in_channels,
                                                L=x.shape[0], dilation=dilation)(x)
                elif self.conv_layer is None:
                    pass
                if self.element_skip:
                    x = x + conv_skip
            
            # ========== RECURRENT BLOCK ==========
            if self.enable_rec == True:
                if self.element_skip:
                    # print(f'{x.shape=}')
                    # print(f'{conv_skip.shape=}')
                    rec_skip = x
                # print(f'{x.shape=}')
                out_dict = self.recurrent_layer(
                    self.hidden_dim[i],
                    self.rec_act,
                    self.training,
                    self.do_rate,
                    skip_recurrent_dense, # if the DCLS layer is dendritic, we need to skip the recurrent dense layer
                )(x)
                state_hist.append(out_dict)
                x = out_dict[3]
                # x = nn.Dropout(rate=self.do_rate, deterministic=not self.training)(x)
                if self.element_skip:
                    x = x + rec_skip

            # ========== CHANNEL MIXING BLOCK ==========
            if self.enable_cm == True: 
                if self.element_skip:
                    ch_mix_skip = x

                if self.channel_mixing == 'mlp':
                    x = MLP(self.hidden_dim[i], 
                            2*self.hidden_dim[i], 
                            self.cm_act)(x)
                elif self.channel_mixing == 'glu':
                    x = GLU(self.hidden_dim[i], 
                            self.glu_type)(x)
                    
                # x = nn.Dropout(rate=self.do_rate, deterministic=not self.training)(x)

                if self.element_skip:
                    x = x + ch_mix_skip

            
            # ========== COMPRESSION BLOCK ==========
            if self.latent_dim[i] is not None:
                x = nn.Dense(self.latent_dim[i])(x) # out: (784, 10), assuming latent_dim = 10
                if self.comp_act == 'linear':
                    pass
                elif self.comp_act == 'gelu':
                    x = nn.gelu(x)
                elif self.comp_act == 'relu':
                    x = nn.relu(x)

            # ========== LAYER SKIP: DESTINATION ==========
            if self.layer_skip: 
                x = x + layer_skip

            # ========== POST-NORMALIZATION ==========
            if self.postnorm: 
                x = nn.LayerNorm()(x) 
        
        # ========== OUTPUT BLOCK ==========
        out = nn.Dense(self.out_dim, name='Dense_Out')(x)#[0])
        return state_hist, out
    
# in_axes = 0: the first dimension of the input is the batch size
# out_axes = 0: the first dimension of the output is the batch size
# variable_axes = {'params': None}: the params are not batched, meaning everey sample in the batch will have the same params
# split_rngs = {'params': False}: the params are not split, meaning every sample in the batch will have the same params
BatchRNN_General = nn.vmap(RNN_General_Backbone, in_axes=0, out_axes=0, 
                           variable_axes={'params': None, 'dropout': None}, 
                           split_rngs={'params': False, 'dropout': False})

