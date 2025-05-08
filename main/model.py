import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from typing import Sequence
from utils import bimodal_gaussian
from functools import partial
from jaxlib import xla_extension as xlx
from types import SimpleNamespace
from jax.nn.initializers import lecun_normal



class minGRULayer(nn.Module):
    '''
    MGU Layer
    We use a modified version of the h_tilde calculation in the MGU cell for better performance.
    - $f_t = \sigma(W^{fx} x_t + U^{fh} h_{t-1} + b^f)$
    - $\~{h}_t = \tanh(W^{hx} x_t + f_t \odot (U^{hh} h_{t-1} + b^h))$
    
    Args:
        hidden_size: int, size of hidden state
        output_size: int, size of output
    '''
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):

        def update(self, state, x):
            h = state[0]
            z_htilde = nn.Dense(2*self.hidden_size, name='Dense_x')(x)
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
        h = jnp.zeros((self.hidden_size,))
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
        hidden_size: int, size of hidden state
        output_size: int, size of output
    '''
    hidden_size: int
    decay_gate: bool
    decay_candidate: bool
    # timedecay_mean: Sequence[float]
    key: jax.random.key
    # hidden_nonlinearity: xlx.PjitFunction
    args: SimpleNamespace

    def get_time_constants(self):
        key_gate, key_cand = jax.random.split(self.key)
        tcg = jax.random.uniform(key_gate, (self.hidden_size,), minval=self.args.time_decay_mean[1], maxval=self.args.time_decay_mean[0])
        tcc = jax.random.uniform(key_cand, (self.hidden_size,), minval=self.args.time_decay_mean[1], maxval=self.args.time_decay_mean[0])
        if self.args.train_gate_decay:
            print("Trainable gate time constants, get from state")
            tcg = None
        if self.args.train_candidate_decay:
            print("Trainable candidate time constants, get from state")
            tcc = None          
        return {"Time_constants_gate": tcg,
                "Time_constants_candidate": tcc}
        

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
        
        def exponential_decay_kernel(time_constants, length):
            """
            Generate an exponential decay kernel for each time constant.
            Args:
                time_constants: Array of time constants (shape: [hidden_size]).
                length: Length of the kernel.
            Returns:
                A kernel matrix of shape [hidden_size, length].
            """
            t = jnp.arange(length)
            kernel = jnp.exp(-t[:, None] / time_constants[None, :])
            return kernel.T  # Shape: [hidden_size, length]


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

        # Define  time constants
        
        # time_constants = self.param(
        #     'time_constants',
        #     lambda rng, shape: jax.random.uniform(self.key, (self.hidden_size,), minval=self.timedecay_mean[1], maxval=self.timedecay_mean[0]),
        #     (self.hidden_size,)
        # )

        # time_constants = jax.random.uniform(self.key, (self.hidden_size,), minval=self.timedecay_mean[1], maxval=self.timedecay_mean[0])
        key_gate, key_candidate = jax.random.split(self.key)
        if self.decay_gate:
            if self.args.train_gate_decay:
                time_constants_gate = self.param(
                    'Time_constants_gate',
                    lambda rng, shape: jnp.abs(jax.random.uniform(key_gate, (self.hidden_size,), minval=self.args.time_decay_mean[1], maxval=self.args.time_decay_mean[0])),
                    (self.hidden_size,)
                )
            else:
                time_constants_gate = self.get_time_constants()["Time_constants_gate"]
                
            # Generate decay kernels
            decay_kernels_gate = exponential_decay_kernel(time_constants_gate, 50)
        
        if self.decay_candidate:
            if self.args.train_candidate_decay:
                time_constants_cand = self.param(
                    'Time_constants_candidate',
                    lambda rng, shape: jnp.abs(jax.random.uniform(key_candidate, (self.hidden_size,), minval=self.args.time_decay_mean[1], maxval=self.args.time_decay_mean[0])),
                    (self.hidden_size,)
                )
            else:
                time_constants_cand= self.get_time_constants()["Time_constants_candidate"]
                
            # Generate decay kernels
            decay_kernels_cand = exponential_decay_kernel(time_constants_cand, 50)

        def update(x):
            '''
            As the x is not scanned anymore, x is 2D array (n_ts, n_features)
            - for sMNIST, the first layer will have x of shape (784, 1)

            '''
            z_htilde = nn.Dense(2*self.hidden_size, name='Dense_x')(x) # z_htilde: (784, 128)
            z_preact, h_tilde_preact = jnp.split(z_htilde, 2, axis=-1) # z_preact and h_tilde_preact: (784, 64)
            
            #Convolve z_preact with decay kernels
            if self.decay_gate:
                z_preact = jax.vmap(
                    lambda z_, k: jax.scipy.signal.convolve(z_, k, mode='same'),
                    in_axes=(1, 0), out_axes=1
                )(z_preact, decay_kernels_gate)
            
            #Convolve h_tilde_preact with decay kernels
            if self.decay_candidate:
                h_tilde_preact = jax.vmap(
                    lambda h_, k: jax.scipy.signal.convolve(h_, k, mode='same'),
                    in_axes=(1, 0), out_axes=1
                )(h_tilde_preact, decay_kernels_cand)

            h_new = vj_heinsen_update(z_preact, h_tilde_preact) # h_new: (784, 64)
            
            if self.args.pre_mixing:
                h_new = nn.Dense(self.hidden_size, name='Dense_h')(h_new)
            if self.args.hidden_nonlinearity is not None:
                h_new = self.args.hidden_nonlinearity(h_new)
            
            return (h_new, z_preact, h_tilde_preact)
        
        state_hist = update(x)
        return state_hist
    

class RNNBackbone(nn.Module):
    # hidden_size: int
    output_size: int
    # n_layers: int
    # decay: Sequence[bool]
    # timedecay_mean: Sequence[float]
    # seed: int
    # hidden_nonlinearity: xlx.PjitFunction
    args: SimpleNamespace
    recurrent_layer: nn.Module = HeinsenMinGRULayer

    def get_time_constants(self, args, layer_index):
        key = jax.random.PRNGKey(args.seed)
        key_list = [key]*args.n_layers
        key_list = jax.random.split(key,args.n_layers)
        key_gate, key_cand = jax.random.split(key_list[layer_index])

        tcg = jax.random.uniform(key_gate, (args.hidden_dim,), minval=args.time_decay_mean[1], maxval=args.time_decay_mean[0])
        tcc = jax.random.uniform(key_cand, (args.hidden_dim,), minval=args.time_decay_mean[1], maxval=args.time_decay_mean[0])
        
        if args.train_gate_decay:
            print("Trainable gate time constants, get from state")
            tcg = None
        if args.train_candidate_decay:
            print("Trainable candidate time constants, get from state")
            tcc = None          
        return {"Time_constants_gate": tcg,
                "Time_constants_candidate": tcc}

    @nn.compact
    def __call__(self, x):
        key = jax.random.PRNGKey(self.args.seed)
        key_list = [key]*self.args.n_layers
        key_list = jax.random.split(key,self.args.n_layers)
        state_hist = []
        
        for i in range(self.args.n_layers-1):
            # print("Layer: ", i, "Decay: ", self.args.decay[i])
            x = self.recurrent_layer(self.args.hidden_dim, self.args.decay_gate[i], self.args.decay_candidate[i], key_list[i], self.args)(x)
            state_hist.append(x)
            x = x[0]
        # print("Layer: ", self.args.n_layers-1, "Decay: ", self.decay[self.args.n_layers-1])
        x = self.recurrent_layer(self.args.hidden_dim, self.args.decay_gate[self.args.n_layers-1], self.args.decay_candidate[self.args.n_layers-1], key_list[i], self.args)(x)
        state_hist.append(x)
        out = nn.Dense(self.output_size, name='Dense_Out')(x[0])
        return state_hist, out
    
BatchRNN = nn.vmap(RNNBackbone, in_axes=0, out_axes=0, variable_axes={'params': None}, split_rngs={'params': False})




@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)



@partial(jax.jit, static_argnames=('reg_factor',))
def apply_model(state, x, y, reg_factor):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        net_dyn, out_hist = state.apply_fn({'params': params}, x)
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(y, 10)
        batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        reg = 0.0
        for layers in net_dyn:
            # 2 is the candidate, layers[1] would regularize the gate
            reg += jnp.where(jnp.abs(layers[2]) > 1, layers[2]**2, 0.0).sum()
        reg += jnp.where(jnp.abs(out_hist) > 1, out_hist**2, 0.0).sum()
        loss = jnp.mean(batch_loss) + reg_factor * reg
        return loss, {'logits': logits, 'batch_loss': batch_loss, 'net_dyn': net_dyn}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(aux_dict['logits'], -1) == y)
    aux_dict.pop('logits')
    return grads, loss, accuracy, aux_dict


@jax.jit
def eval_model(state, images, labels):
    
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        _, out_hist = state.apply_fn({'params': params}, images)
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    loss, logits = loss_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy

def uniform_init():
    def init(key, shape, kernel_size, dtype=jnp.float32):
        p = jax.random.uniform(key, shape, dtype=dtype, minval=0, maxval=1) * (kernel_size-1)
        return p
    return init
def constant_init():
    def init(key, shape, std, dtype=jnp.float32):
        p = jnp.ones(shape, dtype=dtype) * std
        return p
    return init

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
        float32: The gaussian kernel. shape: (dim_out, dim_in, kernel_size) or (dim_in, kernel_size)
    """
    k = jnp.arange(kernel_size)
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
        w = w.squeeze(0)
        # (dim_in,) → (dim_in, 1)
        w = w[:, None]
        p = p[:, None]
        s = s[:, None]
        # print(f'{w.shape=}, {p.shape=}, {s.shape=}')

    # Compute Gaussian interpolation
    return gaussian_interpolation(w, p, s, kernel_size)  # shape (dim_out, dim_in, kernel_size)


def convolve(x, kernel):
    return jax.lax.conv_general_dilated(
                lhs=x,
                rhs=kernel,
                window_strides=(1,),
                padding='VALID',
                dimension_numbers=('NCH', 'OIH', 'NCH')  # (input, kernel, output), N: batch, C: channel, O: output, H: height
)

    
def mich_fft_k(input, K):
    '''
    :param input: 1D array of shape (sim_len,)
    :param K: 1D array of shape (d_max,). n_rep-hot encoded delays
    :return: 1D array of shape (sim_len+d_max,)
    '''
    print(f'{input.shape=}')
    print(f'{K.shape=}')
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
    init_std: float = 5 # paper 0.23
    fft: bool = False
    delay_type: str = 'synaptic' # 'synaptic', 'axonal'
    delay_kernel: str = 'gaussian' # 'impulse', WARNING: ONLY "gaussian" is implemented for now
    def setup(self):
        if self.delay_type == 'synaptic':
            self.weights = self.param('weights', lecun_normal(), (self.dim_out, self.dim_in))
            # sample positions from uniform distribution [0, kernel_size]
            self.positions = self.param('positions', uniform_init(), (self.dim_out, self.dim_in), self.kernel_size)
            self.std = self.param('std', constant_init(), (self.dim_out, self.dim_in), self.init_std)
            self.kernel = construct_kernel_fast(self.weights, self.positions, self.std, self.kernel_size, 3)
            # print(f'{self.kernel.shape=}')
            self.fft_fn = j_wrapper_jvjvj_mich_fft_k
        elif self.delay_type == 'axonal':
            self.weights = self.param('weights', lecun_normal(), (1, self.dim_in))
            # sample positions from uniform distribution [0, kernel_size]
            self.positions = self.param('positions', uniform_init(), (self.dim_in,), self.kernel_size)
            self.std = self.param('std', constant_init(), (self.dim_in,), self.init_std)
            self.kernel = construct_kernel_fast(self.weights, self.positions, self.std, self.kernel_size, 2)
            # print(f'{self.kernel.shape=}')
            self.fft_fn = j_wrapper_jvj_mich_fft_k

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
            x = x.T[None, :, :] # (seq_len, dim_in)
            # print(f'x.shape={x.shape}')
            # print(f'self.kernel.shape={self.kernel.shape}')
            if self.delay_type == 'axonal':
                out = convolve(x, self.kernel[None, :, :]) # (1, seq_len, dim_out)
            else:
                out = convolve(x, self.kernel)
            out = out.squeeze(0)
            out = out.T # (seq_len, dim_out)
        return out

class HeinsenMinGRULayerOriginal(nn.Module):
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
                out = nn.sigmoid(out_preact)
            elif self.layer_act == 'relu':
                out = nn.relu(out_preact)
            elif self.layer_act == 'linear':
                out = out_preact
            else: 
                raise ValueError(f"Unknown activation type: {self.layer_act}")
            # define the dropout layer
            out = nn.Dropout(rate=self.do_rate, deterministic=not self.training)(out)
            return (h_new, z_preact, h_tilde_preact, out, out_preact)
        #{'h_new': h_new, 'z_preact': z_preact, 'h_tilde_preact': h_tilde_preact, 'out': out, 'out_preact': out_preact}
        
        state_hist = update(x)
        return state_hist


class RNN_Delayed_Backbone(nn.Module):
    n_layers: int
    out_dim: int
    hidden_dim: Sequence
    latent_dim: Sequence 
    training: bool 
    do_rate: float
    layer_act: str = 'linear' # 'tanh', 'sigmoid', 'relu'
    recurrent_layer: nn.Module = HeinsenMinGRULayerOriginal
    delay_layer: nn.Module = DCLSLayer
    kernel_size: int = 50
    fft: bool = False
    delay_type: str = 'synaptic' # 'synaptic', 'axonal'
    delay_kernel: str = 'gaussian' # 'impulse', WARNING: ONLY "gaussian" is implemented for now

    @nn.compact
    def __call__(self, x):
        state_hist = []
        for i in range(self.n_layers):
            # print(f'{x.shape=}')
            x = self.delay_layer(kernel_size=self.kernel_size, dim_out=x.shape[-1], dim_in=x.shape[-1], 
                                 fft=self.fft, delay_type=self.delay_type, delay_kernel=self.delay_kernel,
                                 )(x)
            # print(f'{x.shape=}')
            out_dict = self.recurrent_layer(
                self.hidden_dim[i],
                self.latent_dim[i],
                self.layer_act,
                self.training,
                self.do_rate,
            )(x)
            state_hist.append(out_dict)
            x = out_dict[3]
        out = nn.Dense(self.out_dim, name='Dense_Out')(x)#[0])
        return state_hist, out
    
# in_axes = 0: the first dimension of the input is the batch size
# out_axes = 0: the first dimension of the output is the batch size
# variable_axes = {'params': None}: the params are not batched, meaning everey sample in the batch will have the same params
# split_rngs = {'params': False}: the params are not split, meaning every sample in the batch will have the same params
BatchRNN_Delayed = nn.vmap(RNN_Delayed_Backbone, in_axes=0, out_axes=0, variable_axes={'params': None, 'dropout': None}, 
                   split_rngs={'params': False, 'dropout': False})
