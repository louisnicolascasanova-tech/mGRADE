import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from typing import Sequence
from utils import bimodal_gaussian
from functools import partial
from jaxlib import xla_extension as xlx
from types import SimpleNamespace



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