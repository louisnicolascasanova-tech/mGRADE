"""
Main RNN backbone models for Den-minGRU.
Includes configurable encoder, convolution, recurrent, and channel mixing blocks.
"""
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

from .layers import DCLSLayer, CausalDepthWiseConv1d, MLP, GLU, HeinsenMinGeneralGRULayer
from .utils import masked_meanpool

class RNN_General_Backbone(nn.Module):
    training: bool 
    padded: bool
    n_layers: int
    out_dim: int
    hidden_dim: Sequence
    do_rate: float
    # ENCODER
    encoder: bool = True
    encoder_scale: float = 1.0
    encoder_bias: bool = False
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
    weight_init_scale: float = 1.0 # scale for the weight initialization of the convolution layer
    conv_ln: bool = False # whether to apply LayerNorm before the convolution layer
    # RECURRENT
    enable_rec: bool = True
    recurrent_layer: nn.Module = HeinsenMinGeneralGRULayer
    rec_act: str = 'linear' # 'linear', 'gelu', 'relu'
    rec_ln: bool = False # whether to apply LayerNorm before the recurrent layer
    rec_dense_out: bool = False # whether to add Dense layer after recurrent layer
    rec_dense_out_act: bool = False # whether to add ReLU activation after dense out layer
    dense_z_weight_init_scale: float = 1.0 # scale for the weight initialization of the z preactivation
    dense_z_bias_init: str = 'zero' # 'zero', 'ugi', 'constant'
    dense_h_weight_init_scale: float = 1.0 # scale for the weight initialization of the h preactivation
    dense_h_bias_init: str = 'zero' # 'zero', 'ugi', 'constant'
    # CHANNEL MIXING
    enable_cm: bool = True
    channel_mixing: str = 'none' # 'none', 'mlp', 'glu'
    cm_act: str = 'relu' # 'linear', 'gelu', 'relu'
    glu_type: str = 'full' # 'full', 'candidate', 'gate'
    cm_ln: bool = False # whether to apply LayerNorm before the channel mixing layer
    # COMPRESSION
    latent_dim: Sequence = None
    comp_act: str = 'linear' # 'linear', 'gelu', 'relu'
    # NORMALIZATION
    postnorm: bool = False
    # DECODER
    decoder_bias: bool = False
    # MONITORING
    enable_monitoring: bool = False


    @nn.compact
    def __call__(self, x):

        if self.padded:
            x, length = x
        
        # Initialize monitoring dictionary
        if self.enable_monitoring:
            monitor = {
                'input': x,
                'encoder_out': None,
                'layers': [],
                'final_output': None
            }
        else:
            monitor = {}

        # ========== ENCODER ==========
        if self.encoder:
            # make a Dense layer without bias
            x = nn.Dense(self.hidden_dim[0], name='Encoder', 
                         kernel_init=nn.initializers.variance_scaling(self.encoder_scale, 'fan_in', 'truncated_normal'),
                         #bias_init=uniform_bias_init_enc(),
                         use_bias=self.encoder_bias)(x)
            if self.enable_monitoring:
                monitor['encoder_out'] = x

        # x = nn.LayerNorm(name='LayerNormInput')(x) # normalize the input

        # ========== SEQUENCE BLOCKS ==========
        state_hist = []
        skip_recurrent_dense = False # if the DCLS layer is dendritic, we need to skip the recurrent dense layer
        for i in range(self.n_layers):
            
            if self.enable_monitoring:
                layer_monitor = {
                    'layer_id': i,
                    'input': x,
                    'layer_skip_source': None,
                    'conv_input': None,
                    'conv_output': None,
                    'conv_skip': None,
                    'rec_input': None,
                    'rec_ln_output': None,
                    'rec_dense_output': None,
                    'rec_output': None,
                    'rec_skip': None,
                    'cm_input': None,
                    'cm_output': None,
                    'cm_skip': None,
                    'compression_output': None,
                    'layer_skip_output': None,
                    'postnorm_output': None,
                    'final_layer_output': None
                }

            # ========== LAYER SKIP: SOURCE ==========
            # print(f'{x.shape=}')
            if self.layer_skip: 
                layer_skip = x
                if self.enable_monitoring:
                    layer_monitor['layer_skip_source'] = layer_skip

            # ========== CONVOLUTION BLOCK ==========
            if self.enable_conv == True:
                # if self.conv_ln:
                #     x = nn.LayerNorm(name=f'LayerNormConv_{i}')(x) # normalize the output of the convolution layer

                if self.enable_monitoring:
                    layer_monitor['conv_input'] = x
                    
                if self.element_skip:
                    conv_skip = x
                    if self.enable_monitoring:
                        layer_monitor['conv_skip'] = conv_skip

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
                                    weight_init_scale=self.weight_init_scale
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
                if self.conv_ln:
                    x = nn.LayerNorm(name=f'LayerNormConv_{i}')(x) # normalize the output of the convolution layer
                
                if self.enable_monitoring:
                    layer_monitor['conv_output'] = x
                    
                if self.element_skip:
                    x = x + conv_skip
            
            # ========== RECURRENT BLOCK ==========
            if self.enable_rec == True:
                # if self.rec_ln:
                #     x = nn.LayerNorm(name=f'LayerNormRec_{i}')(x)

                if self.enable_monitoring:
                    layer_monitor['rec_input'] = x

                if self.element_skip:
                    # print(f'{x.shape=}')
                    # print(f'{conv_skip.shape=}')
                    rec_skip = x
                    if self.enable_monitoring:
                        layer_monitor['rec_skip'] = rec_skip
                # print(f'{x.shape=}')
                out_dict = self.recurrent_layer(
                    self.hidden_dim[i],
                    self.rec_act,
                    self.training,
                    self.do_rate,
                    skip_recurrent_dense, # if the DCLS layer is dendritic, we need to skip the recurrent dense layer
                    dense_z_weight_init_scale=self.dense_z_weight_init_scale,
                    dense_z_bias_init=self.dense_z_bias_init,
                    dense_h_weight_init_scale=self.dense_h_weight_init_scale,
                    dense_h_bias_init=self.dense_h_bias_init,
                )(x)
                state_hist.append(out_dict)
                x = out_dict[3]
                if self.enable_monitoring:
                    layer_monitor['rec_output'] = x

                if self.rec_dense_out:
                    x = nn.Dense(self.hidden_dim[i], name=f'DenseRec_{i}')(x) # project back to hidden_dim if needed
                if self.rec_dense_out_act:
                    x = nn.relu(x)
                if self.enable_monitoring:
                    layer_monitor['rec_dense_output'] = x

                if self.rec_ln:
                    x = nn.LayerNorm(name=f'LayerNormRec_{i}')(x)
                # x = nn.Dropout(rate=self.do_rate, deterministic=not self.training)(x)
                if self.element_skip:
                    x = x + rec_skip

            # ========== CHANNEL MIXING BLOCK ==========
            if self.enable_cm == True: 
                if self.enable_monitoring:
                    layer_monitor['cm_input'] = x
                    
                if self.element_skip:
                    ch_mix_skip = x
                    if self.enable_monitoring:
                        layer_monitor['cm_skip'] = ch_mix_skip

                if self.channel_mixing == 'mlp':
                    x = MLP(self.hidden_dim[i], 
                            2*self.hidden_dim[i], 
                            self.cm_act)(x)
                elif self.channel_mixing == 'glu':
                    x = GLU(self.hidden_dim[i], 
                            self.glu_type)(x)
                    
                # x = nn.Dropout(rate=self.do_rate, deterministic=not self.training)(x)
                if self.cm_ln:
                    x = nn.LayerNorm(name=f'LayerNormCM_{i}')(x)

                if self.enable_monitoring:
                    layer_monitor['cm_output'] = x
                    
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
                if self.enable_monitoring:
                    layer_monitor['compression_output'] = x

            # ========== LAYER SKIP: DESTINATION ==========
            if self.layer_skip: 
                x = x + layer_skip
                if self.enable_monitoring:
                    layer_monitor['layer_skip_output'] = x

            # ========== POST-NORMALIZATION ==========
            if self.postnorm: # and not i == self.n_layers - 1:  
                x = nn.LayerNorm(name=f'LayerNormPost_{i}')(x)
                if self.enable_monitoring:
                    layer_monitor['postnorm_output'] = x
            
            if self.enable_monitoring:
                layer_monitor['final_layer_output'] = x
                monitor['layers'].append(layer_monitor) 
        
        # ========== OUTPUT BLOCK ==========
        x = nn.Dense(self.out_dim, use_bias=self.decoder_bias,
                       #kernel_init=nn.initializers.variance_scaling(0.05, 'fan_in', 'truncated_normal'), 
                       name='Dense_Out')(x)#[0])
        
        if self.enable_monitoring:
            monitor['final_output'] = x
        
        # ========== LOGITS ==========
        if self.padded: 
            x = masked_meanpool(x, length)
        else: 
            x = jnp.mean(x, axis=0) # axis is 0 because x is (L, d_model), as we vmapped the batch dim

        return state_hist, x, monitor
    
# in_axes = 0: the first dimension of the input is the batch size
# out_axes = 0: the first dimension of the output is the batch size
# variable_axes = {'params': None}: the params are not batched, meaning everey sample in the batch will have the same params
# split_rngs = {'params': False}: the params are not split, meaning every sample in the batch will have the same params
BatchRNN_General = nn.vmap(RNN_General_Backbone, in_axes=0, out_axes=0, 
                           variable_axes={'params': None, 'dropout': None}, 
                           split_rngs={'params': False, 'dropout': False})
