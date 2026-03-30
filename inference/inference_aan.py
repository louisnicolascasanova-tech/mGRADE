"""
Inference script for AAN (Academic Article Network) dataset using RNN_General_Retrieval_Backbone.
This script allows step-by-step debugging and verification of the retrieval model.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from model import RNN_General_Retrieval_Backbone, batch_masked_meanpool
from utils import create_lra_aan_classification_dataset, prep_batch
import flax.linen as nn
from flax.training import train_state
import optax


def create_dummy_aan_data(batch_size=4, seq_len=100, vocab_size=98):
    """
    Create dummy AAN data for testing the retrieval model.
    
    Args:
        batch_size: Number of document pairs
        seq_len: Sequence length
        vocab_size: Vocabulary size for AAN
        
    Returns:
        Tuple of (inputs, targets, aux_data) formatted for AAN retrieval
    """
    # Create two sets of dummy documents (pairs)
    doc1 = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    doc2 = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # Concatenate document pairs in batch dimension (AAN format)
    inputs = np.concatenate([doc1, doc2], axis=0)  # Shape: (2*batch_size, seq_len)
    
    # Create random binary labels for citation prediction
    targets = np.random.randint(0, 2, size=(batch_size,))
    
    # Create random lengths for variable-length sequences
    lengths1 = np.random.randint(seq_len//2, seq_len, size=(batch_size,))
    lengths2 = np.random.randint(seq_len//2, seq_len, size=(batch_size,))
    lengths = np.concatenate([lengths1, lengths2], axis=0)  # Shape: (2*batch_size,)
    
    aux_data = {'lengths': lengths}
    
    return inputs, targets, aux_data


def initialize_retrieval_model(
    vocab_size=98,
    hidden_dim=64,
    n_layers=2,
    seq_len=100,
    batch_size=4
):
    """
    Initialize the RNN_General_Retrieval_Backbone model with test parameters.
    """
    # Model configuration
    model_config = {
        'training': False,
        'n_layers': n_layers,
        'out_dim': 2,  # Binary classification for AAN
        'hidden_dim': [hidden_dim] * n_layers,
        'do_rate': 0.1,
        # Encoder
        'encoder': True,
        'encoder_scale': 1.0,
        'encoder_bias': False,
        # Skip connections
        'layer_skip': False,
        'element_skip': False,
        # Convolution (DCLS)
        'enable_conv': True,
        'conv_layer': 'dcls',
        'kernel_size': 16,
        'kernel_n_elems': 4,
        'dcls_std': 0.7,
        'dcls_type': 'synaptic',
        'dcls_kernel': 'gaussian',
        'dcls_fft': False,
        'dcls_heterogeneous_weights': True,
        'dcls_heterogeneous_positions': True,
        'dcls_heterogeneous_std': False,
        'weight_init_scale': 1.0,
        'conv_ln': False,
        # Recurrent
        'enable_rec': True,
        'rec_act': 'linear',
        'rec_ln': False,
        'dense_z_weight_init_scale': 1.0,
        'dense_z_bias_init': 'zero',
        'dense_h_weight_init_scale': 1.0,
        'dense_h_bias_init': 'zero',
        # Channel mixing
        'enable_cm': True,
        'channel_mixing': 'mlp',
        'cm_act': 'relu',
        'glu_type': 'full',
        'cm_ln': False,
        # Compression
        'latent_dim': [None] * n_layers,
        'comp_act': 'linear',
        # Normalization
        'postnorm': False,
        # Retrieval specific
        'retrieval_d_model': hidden_dim
    }
    
    # Create model
    model = RNN_General_Retrieval_Backbone(**model_config)
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    init_key, dropout_key = jax.random.split(key, 2)
    
    # Create dummy input for initialization
    dummy_inputs = jnp.ones((2 * batch_size, seq_len, vocab_size))  # One-hot encoded
    dummy_lengths = jnp.ones((2 * batch_size,)) * seq_len
    dummy_input = (dummy_inputs, dummy_lengths)
    
    # Initialize model parameters
    variables = model.init({
        'params': init_key,
        'dropout': dropout_key
    }, dummy_input)
    
    params = variables['params']
    
    return model, params


def debug_step_by_step(model, params, inputs, lengths, targets):
    """
    Debug the retrieval model step by step for detailed analysis.
    
    Args:
        model: RNN_General_Retrieval_Backbone instance
        params: Model parameters
        inputs: Input sequences (2*bsz, seq_len, vocab_size)
        lengths: Sequence lengths (2*bsz,)
        targets: Target labels (bsz,)
    """
    print("="*60)
    print("STEP-BY-STEP DEBUGGING OF RNN_General_Retrieval_Backbone")
    print("="*60)
    
    batch_size = targets.shape[0]
    seq_len = inputs.shape[1]
    vocab_size = inputs.shape[2]
    
    print(f"Input shapes:")
    print(f"  inputs: {inputs.shape} (2*bsz={2*batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    print(f"  lengths: {lengths.shape} (2*bsz={2*batch_size})")
    print(f"  targets: {targets.shape} (bsz={batch_size})")
    print(f"  lengths values: {lengths}")
    print()
    
    # Step 1: Forward pass through the model
    print("Step 1: Forward pass through retrieval model")
    print("-" * 40)
    
    input_tuple = (inputs, lengths)
    logits = model.apply({'params': params}, input_tuple)
    
    print(f"Final logits shape: {logits.shape}")  # Should be (bsz, 2)
    print(f"Final logits values:\n{logits}")
    print()
    
    # Step 2: Manual step-by-step breakdown
    print("Step 2: Manual breakdown of retrieval process")
    print("-" * 40)
    
    # Access the encoder and decoder from the model
    # Note: This requires accessing internal components, for debugging purposes
    
    # Simulate the encoder forward pass
    print("2a: Encoding document pairs through RNN backbone...")
    
    # The encoder is vmapped, so we need to simulate its behavior
    # For debugging, let's create a simplified version
    
    # Simulate encoded representations (this would come from the RNN backbone)
    d_model = 64  # Should match retrieval_d_model
    simulated_encoded = jnp.ones((2 * batch_size, seq_len, d_model)) * 0.5
    print(f"Simulated encoded shape: {simulated_encoded.shape}")
    
    # Step 3: Mean pooling with masking
    print("\n2b: Applying masked mean pooling...")
    
    pooled = batch_masked_meanpool(simulated_encoded, lengths)
    print(f"Pooled shape after mean pooling: {pooled.shape}")  # Should be (2*bsz, d_model)
    print(f"Pooled values sample:\n{pooled[:4, :8]}")  # Show first few values
    
    # Step 4: Split into document pairs
    print("\n2c: Splitting into document pairs...")
    
    pooled_0, pooled_1 = jnp.split(pooled, 2)
    print(f"Document 1 embeddings shape: {pooled_0.shape}")  # Should be (bsz, d_model)
    print(f"Document 2 embeddings shape: {pooled_1.shape}")  # Should be (bsz, d_model)
    
    # Step 5: Construct comparison features
    print("\n2d: Constructing 4 comparison features...")
    
    features = jnp.concatenate([
        pooled_0,                    # First document embedding
        pooled_1,                    # Second document embedding  
        pooled_0 - pooled_1,         # Element-wise difference
        pooled_0 * pooled_1          # Element-wise product
    ], axis=-1)
    
    print(f"Comparison features shape: {features.shape}")  # Should be (bsz, 4*d_model)
    print(f"Feature components:")
    print(f"  - Document 1 embeddings: {pooled_0.shape}")
    print(f"  - Document 2 embeddings: {pooled_1.shape}")
    print(f"  - Difference (doc1 - doc2): {(pooled_0 - pooled_1).shape}")
    print(f"  - Product (doc1 * doc2): {(pooled_0 * pooled_1).shape}")
    
    # Step 6: Predictions and accuracy
    print("\n2e: Final predictions...")
    
    predictions = jnp.argmax(logits, axis=-1)
    probabilities = jnp.exp(logits)  # Convert from log-softmax
    
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities:\n{probabilities}")
    
    # Calculate accuracy
    accuracy = jnp.mean(predictions == targets)
    print(f"Accuracy: {accuracy:.4f}")
    
    return logits, features, pooled, predictions


def test_with_real_data():
    """
    Test the model with actual AAN dataset (if available).
    """
    print("="*60)
    print("TESTING WITH REAL AAN DATA")
    print("="*60)
    
    try:
        # Try to load real AAN data
        cache_dir = Path("./cache_dir/")
        trainloader, valloader, testloader, aux_loaders, n_classes, seq_len, in_dim, train_size = \
            create_lra_aan_classification_dataset(cache_dir=cache_dir, batch_size=4, seed=42)
        
        print(f"Successfully loaded AAN dataset:")
        print(f"  n_classes: {n_classes}")
        print(f"  seq_len: {seq_len}")
        print(f"  in_dim (vocab_size): {in_dim}")
        print(f"  train_size: {train_size}")
        print()
        
        # Initialize model with real parameters
        model, params = initialize_retrieval_model(
            vocab_size=in_dim,
            hidden_dim=64,
            n_layers=2,
            seq_len=seq_len,
            batch_size=4
        )
        
        # Get a batch of real data
        batch_iter = iter(testloader)
        batch = next(batch_iter)
        
        # Prepare batch (convert to JAX format with one-hot encoding)
        inputs, targets, aux_data = batch
        full_inputs, targets_jax, _ = prep_batch(batch, seq_len, in_dim)
        
        # Extract inputs and lengths
        if isinstance(full_inputs, tuple):
            inputs_jax, lengths_jax = full_inputs
        else:
            inputs_jax = full_inputs
            lengths_jax = jnp.ones((inputs_jax.shape[0],)) * seq_len
        
        print(f"Real data batch shapes:")
        print(f"  inputs_jax: {inputs_jax.shape}")
        print(f"  lengths_jax: {lengths_jax.shape}")
        print(f"  targets_jax: {targets_jax.shape}")
        
        # Debug with real data
        debug_step_by_step(model, params, inputs_jax, lengths_jax, targets_jax)
        
    except Exception as e:
        print(f"Could not load real AAN data: {e}")
        print("Falling back to dummy data...")
        return False
    
    return True


def main():
    """
    Main function to run the inference and debugging.
    """
    print("RNN_General_Retrieval_Backbone Inference and Debugging")
    print("="*60)
    
    # Set JAX to use GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(f"Using GPU 1 (CUDA_VISIBLE_DEVICES=1)")
    print(f"JAX devices: {jax.devices()}")
    
    # Test 1: Try with real data first
    real_data_success = test_with_real_data()
    
    if not real_data_success:
        # Test 2: Use dummy data
        print("\n" + "="*60)
        print("TESTING WITH DUMMY DATA")
        print("="*60)
        
        # Create dummy data
        inputs, targets, aux_data = create_dummy_aan_data(batch_size=4, seq_len=100, vocab_size=98)
        lengths = aux_data['lengths']
        
        # Convert to JAX arrays and one-hot encode
        inputs_jax = jax.nn.one_hot(inputs, 98).astype(jnp.float32)
        targets_jax = jnp.array(targets)
        lengths_jax = jnp.array(lengths).astype(jnp.float32)
        
        # Initialize model
        model, params = initialize_retrieval_model(
            vocab_size=98,
            hidden_dim=64,
            n_layers=2,
            seq_len=100,
            batch_size=4
        )
        
        # Debug step by step
        debug_step_by_step(model, params, inputs_jax, lengths_jax, targets_jax)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("You can now set breakpoints and step through the debug_step_by_step function")
    print("to verify each component of the RNN_General_Retrieval_Backbone.")


if __name__ == "__main__":
    # Create inference directory if it doesn't exist
    os.makedirs("inference", exist_ok=True)
    main()