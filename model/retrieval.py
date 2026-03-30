"""
Retrieval-specific models for document matching tasks (e.g., AAN).
Includes RetrievalDecoder and RNN_General_Retrieval_Backbone.
"""
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

from .backbone import RNN_General_Backbone
from .layers import HeinsenMinGeneralGRULayer
from .utils import batch_masked_meanpool

class RetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
    """
    d_model: int
    d_output: int

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        self.layer1 = nn.Dense(self.d_model)
        self.layer2 = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)


class RNN_General_Retrieval_Backbone(nn.Module):
    """ 
    RNN General Retrieval classification model. This consists of the RNN_General_Backbone
    encoder (which consists of convolution, recurrent, and channel mixing blocks), mean pooling
    across the sequence length, constructing 4 features which are fed into a MLP,
    and a softmax operation. Note that unlike the standard classification model above,
    this model operates directly on the batch of data for document pair processing.
    
    The model processes document pairs for retrieval/matching tasks like AAN (Academic Article Network).
    Input format: (2*bsz, seq_len, d_input) where pairs of documents are concatenated in batch dimension.
    
    Args:
        All parameters from RNN_General_Backbone plus:
        retrieval_d_model (int): Hidden dimension for retrieval features (defaults to first hidden_dim)
    """
    training: bool 
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
    kernel_n_elems: int = 1
    dcls_std: float = 0.7
    dcls_type: str = 'synaptic' # 'synaptic', 'axonal'
    dcls_kernel: str = 'gaussian'
    dcls_fft: bool = False
    dcls_heterogeneous_weights: bool = True
    dcls_heterogeneous_positions: bool = True
    dcls_heterogeneous_std: bool = False
    wavenet_dilation: bool = False
    dilation_schedule: str = 'clip' # 'clip', 'wrap', 'linear'
    dilation_boundary: int = None
    dilation_offset: int = 0
    constant_dilation: int = 1
    weight_init_scale: float = 1.0
    conv_ln: bool = False
    # RECURRENT
    enable_rec: bool = True
    recurrent_layer: nn.Module = HeinsenMinGeneralGRULayer
    rec_act: str = 'linear' # 'linear', 'gelu', 'relu'
    rec_ln: bool = False
    rec_dense_out: bool = False # whether to add Dense layer after recurrent layer
    rec_dense_out_act: bool = False # whether to add ReLU activation after dense out layer
    dense_z_weight_init_scale: float = 1.0
    dense_z_bias_init: str = 'zero' # 'zero', 'ugi', 'constant'
    dense_h_weight_init_scale: float = 1.0
    dense_h_bias_init: str = 'zero' # 'zero', 'ugi', 'constant'
    # CHANNEL MIXING
    enable_cm: bool = True
    channel_mixing: str = 'none' # 'none', 'mlp', 'glu'
    cm_act: str = 'relu' # 'linear', 'gelu', 'relu'
    glu_type: str = 'full' # 'full', 'candidate', 'gate'
    cm_ln: bool = False
    # COMPRESSION
    latent_dim: Sequence = None
    comp_act: str = 'linear' # 'linear', 'gelu', 'relu'
    # NORMALIZATION
    postnorm: bool = False
    # DECODER
    decoder_bias: bool = False
    # RETRIEVAL SPECIFIC
    retrieval_d_model: int = None  # If None, uses hidden_dim[0]
    padded: bool = True
    # MONITORING
    enable_monitoring: bool = False

    @nn.compact
    def __call__(self, input):  # input is a tuple of x and lengths
        """
        Compute the size d_output log softmax output given a
        document pair input sequence. The encoded features are constructed as in
        Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
        Args:
             input (float32, int32): tuple of input sequence and prepadded sequence lengths
                input sequence is of shape (2*bsz, L, d_input) (includes both documents) and
                lengths is (2*bsz,)
        Returns:
            output (float32): (bsz, d_output) - log softmax logits
        """
        x, lengths = input  # x: (2*bsz, seq_len, d_input), lengths: (2*bsz,)
        
        # Use hidden_dim[0] as retrieval d_model if not specified
        d_model = self.retrieval_d_model if self.retrieval_d_model is not None else self.hidden_dim[0]
        
        # Create vmapped encoder backbone
        BatchRNNEncoder = nn.vmap(
            RNN_General_Backbone,
            in_axes=0,
            out_axes=0,
            variable_axes={'params': None, 'dropout': None},
            split_rngs={'params': False, 'dropout': True},
            axis_name='batch'
        )

        encoder = BatchRNNEncoder(
            padded=False,
            training=self.training,
            n_layers=self.n_layers,
            out_dim=d_model,  # Output d_model for pooling
            hidden_dim=self.hidden_dim,
            do_rate=self.do_rate,
            enable_monitoring=self.enable_monitoring,
            encoder=self.encoder,
            encoder_scale=self.encoder_scale,
            encoder_bias=self.encoder_bias,
            layer_skip=self.layer_skip,
            element_skip=self.element_skip,
            enable_conv=self.enable_conv,
            conv_layer=self.conv_layer,
            kernel_size=self.kernel_size,
            kernel_n_elems=self.kernel_n_elems,
            dcls_std=self.dcls_std,
            dcls_type=self.dcls_type,
            dcls_kernel=self.dcls_kernel,
            dcls_fft=self.dcls_fft,
            dcls_heterogeneous_weights=self.dcls_heterogeneous_weights,
            dcls_heterogeneous_positions=self.dcls_heterogeneous_positions,
            dcls_heterogeneous_std=self.dcls_heterogeneous_std,
            wavenet_dilation=self.wavenet_dilation,
            dilation_schedule=self.dilation_schedule,
            dilation_boundary=self.dilation_boundary,
            dilation_offset=self.dilation_offset,
            constant_dilation=self.constant_dilation,
            weight_init_scale=self.weight_init_scale,
            conv_ln=self.conv_ln,
            enable_rec=self.enable_rec,
            recurrent_layer=self.recurrent_layer,
            rec_act=self.rec_act,
            rec_ln=self.rec_ln,
            dense_z_weight_init_scale=self.dense_z_weight_init_scale,
            dense_z_bias_init=self.dense_z_bias_init,
            dense_h_weight_init_scale=self.dense_h_weight_init_scale,
            dense_h_bias_init=self.dense_h_bias_init,
            enable_cm=self.enable_cm,
            channel_mixing=self.channel_mixing,
            cm_act=self.cm_act,
            glu_type=self.glu_type,
            cm_ln=self.cm_ln,
            latent_dim=self.latent_dim,
            comp_act=self.comp_act,
            postnorm=self.postnorm,
            decoder_bias=self.decoder_bias  # No decoder bias for encoder
        )
        
        # Create vmapped retrieval decoder
        BatchRetrievalDecoder = nn.vmap(
            RetrievalDecoder,
            in_axes=0,
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False},
        )

        decoder = BatchRetrievalDecoder(
            d_model=d_model,
            d_output=self.out_dim
        )
        
        # Encode sequences through RNN backbone
        state_hist, encoded_x, monitor = encoder(x)  # encoded_x: (2*bsz, seq_len, d_model)
        
        # Mean pool across sequence dimension with masking for variable lengths
        pooled = batch_masked_meanpool(encoded_x, lengths)  # pooled: (2*bsz, d_model)
        
        # Split pooled representations into document pairs
        pooled_0, pooled_1 = jnp.split(pooled, 2)  # each: (bsz, d_model)
        
        # Construct 4 comparison features as in Tay et al. 2020
        features = jnp.concatenate([
            pooled_0,                    # First document embedding
            pooled_1,                    # Second document embedding  
            pooled_0 - pooled_1,         # Element-wise difference
            pooled_0 * pooled_1          # Element-wise product
        ], axis=-1)  # features: (bsz, 4*d_model)
        
        # Pass through retrieval decoder MLP
        logits = decoder(features)  # logits: (bsz, d_output)
        
        return state_hist, logits, monitor

