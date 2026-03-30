# Network Dynamics Monitoring System - Complete Documentation

## Overview

This document provides comprehensive documentation of the network dynamics monitoring system developed for the Den-MinGRU project. The system enables deep introspection into all intermediate variables and transformations within the recurrent neural network architecture, providing unprecedented visibility into the model's internal behavior.

## Project Context

### Den-MinGRU Architecture
The Den-MinGRU (Delay-Enhanced Minimal GRU) is a research project implementing minimal GRU networks with various architectural enhancements:

- **Core Layer**: `HeinsenMinGRULayer` - Recurrent layer using Heinsen's parallel computation method
- **Delay Compensation**: `DCLSLayer` (Delay Compensation Linear System) with Gaussian interpolation kernels
- **Convolution Support**: `CausalDepthWiseConv1d` with dilation capabilities
- **Channel Mixing**: `MLP` and `GLU` components for feature transformation
- **Multiple Datasets**: MNIST, CIFAR, IMDB, ListOps, Path tasks

### Key Innovation
The project focuses on sequence modeling with delay-compensated recurrent layers, exploring how temporal delays in neural computation can improve performance on various tasks.

## Development Timeline

### Phase 1: Understanding Network Dynamics History (NDH)

**Objective**: Understand what `ndh` represents in the original inference script.

**Discovery**: 
- `ndh` stands for "Network Dynamics History"
- Contains internal states from each recurrent layer: `(h_new, z_preact, h_tilde_preact, out)`
- `h_new`: Hidden states after Heinsen update
- `z_preact`: Gate preactivations (before sigmoid)
- `h_tilde_preact`: Candidate state preactivations
- `out`: Layer outputs after activation functions

**Implementation**: Modified `main_inf.py` to create basic visualizations of these dynamics.

### Phase 2: Initial Visualization System

**Objective**: Create 6x1 subplot visualizations showing network dynamics across layers.

**Features Implemented**:
- Single sample visualization
- First 5 dimensions plotted per variable
- 6 layers + input/output structure
- Separate files for each variable type

**Files Generated**:
- `main_inf_h_new.png`
- `main_inf_z_preact.png` 
- `main_inf_h_tilde_preact.png`
- `main_inf_out.png`

### Phase 3: Multi-Sample Comparison

**Objective**: Compare dynamics across multiple batch samples (5 samples).

**Enhancement**: 
- Changed from 6x1 to 8x5 subplot grid
- Added input at top row, final output at bottom row
- 5 columns showing different batch samples side-by-side
- Enabled comparative analysis of network behavior

**Insight**: This revealed how the network processes different inputs through the same architectural pipeline.

### Phase 4: Comprehensive Monitoring Architecture

**Objective**: Create a monitored version of the network that captures ALL intermediate variables, not just recurrent dynamics.

#### 4.1 RNN_General_Backbone_Monitored Class

**Location**: `model.py` (lines 786-1011)

**Architecture**: Extended the original `RNN_General_Backbone` with comprehensive monitoring capabilities.

**Monitored Variables Per Layer**:
```python
layer_monitor = {
    'layer_id': i,
    'input': x,                      # Input to this layer
    'layer_skip_source': None,       # Source for layer skip connection
    'conv_input': None,              # Input to convolution block
    'conv_output': None,             # Output from convolution block
    'conv_skip': None,               # Skip connection for convolution
    'rec_input': None,               # Input to recurrent block
    'rec_ln_output': None,           # Output after LayerNorm (if enabled)
    'rec_output': None,              # Output from recurrent block
    'rec_skip': None,                # Skip connection for recurrent
    'cm_input': None,                # Input to channel mixing block
    'cm_output': None,               # Output from channel mixing block
    'cm_skip': None,                 # Skip connection for channel mixing
    'compression_output': None,      # Output after compression/latent projection
    'layer_skip_output': None,       # Output after applying layer skip
    'postnorm_output': None,         # Output after post-normalization
    'final_layer_output': None       # Final output of this layer
}
```

**Global Monitoring**:
```python
monitor = {
    'input': x,                      # Raw network input
    'encoder_out': None,             # Output from encoder Dense layer
    'layers': [],                    # List of layer_monitor dicts
    'final_output': None             # Final network output (logits/predictions)
}
```

**Return Signature**: `return state_hist, out, monitor`
- `state_hist`: Original recurrent dynamics (NDH)
- `out`: Final predictions
- `monitor`: Comprehensive monitoring data

#### 4.2 BatchRNN_General_Monitored

**Purpose**: Vectorized version of the monitored backbone using `nn.vmap`

**Configuration**:
- `in_axes=0, out_axes=0`: Batch dimension handling
- `variable_axes={'params': None, 'dropout': None}`: Shared parameters
- `split_rngs={'params': False, 'dropout': False}`: RNG handling

### Phase 5: Advanced Visualization System (main_inf_2.py)

**Objective**: Create comprehensive plotting system for all monitored variables with organized output structure.

#### 5.1 Plot Categories

**Recurrent Variables** (from NDH):
- `h_new`: Hidden states evolution
- `z_preact`: Gate preactivations
- `h_tilde_preact`: Candidate preactivations  
- `out`: Layer outputs

**Monitored Variables** (from monitoring system):
- `encoder_out`: Encoder transformations
- `conv_input`/`conv_output`: Convolution processing
- `rec_input`/`rec_output`: Recurrent processing
- `cm_input`/`cm_output`: Channel mixing
- `compression_output`: Latent space projections
- `layer_skip_output`: Skip connection effects
- `postnorm_output`: Normalization effects
- `final_layer_output`: Complete layer transformations

#### 5.2 Visualization Features

**Multi-Sample Analysis**:
- 5 batch samples shown simultaneously
- Side-by-side comparison of network behavior
- Reveals sample-specific vs. general patterns

**Comprehensive Coverage**:
- Input → Encoder → Layers → Final Output pipeline
- Every transformation stage captured and visualized
- Complete signal flow documentation

**File Organization**:
```
images/{sim_name}/main_inf_2_recurrent_h_new_all_dims.png
images/{sim_name}/main_inf_2_recurrent_z_preact_all_dims.png
images/{sim_name}/main_inf_2_recurrent_h_tilde_preact_all_dims.png
images/{sim_name}/main_inf_2_recurrent_out_all_dims.png
images/{sim_name}/main_inf_2_monitored_encoder_out_all_dims.png
images/{sim_name}/main_inf_2_monitored_conv_input_all_dims.png
images/{sim_name}/main_inf_2_monitored_conv_output_all_dims.png
... (12+ files total)
```

**Plot Organization System**:
- All plots are automatically saved in organized subfolders: `images/{sim_name}/`
- The `{sim_name}` parameter is provided via the `--sim_name` CLI argument
- Directory structure is created automatically if it doesn't exist
- Default simulation name is 'default' if not specified

### Phase 6: Full-Dimensional Visualization

**Objective**: Plot ALL dimensions (up to 128) instead of limiting to first 5.

### Phase 7: Organized Output Structure (Latest Enhancement)

**Objective**: Implement organized file structure for better experiment management.

**Features Implemented**:
- **Automatic Directory Creation**: Creates `images/{sim_name}/` subfolder structure
- **CLI Integration**: Uses existing `--sim_name` argument for folder naming
- **Default Handling**: Falls back to 'default' folder when sim_name not specified
- **Cross-Platform Compatibility**: Uses `os.path.join()` for proper path handling

**Benefits**:
- **Experiment Organization**: Each simulation run gets its own dedicated folder
- **Batch Processing**: Easy to run multiple experiments without file conflicts
- **Result Comparison**: Side-by-side comparison of different experiment outputs
- **Clean Workspace**: No clutter of plot files in root directory

#### 6.1 Enhanced Color Schemes

**Smart Coloring Strategy**:
- **≤20 dimensions**: `tab20` colormap, high opacity (α=0.8)
- **>20 dimensions**: `viridis`/`plasma` colormaps, lower opacity (α=0.3)
- **Special cases**: 
  - Encoder: `Reds` colormap
  - Output classes: `Set1` colormap with labels

#### 6.2 Scalability Features

**Adaptive Visualization**:
- Automatic dimension counting in titles: `"Layer 0 (128 dims)"`
- Opacity adjustment for visual clarity with many lines
- Legend management (show only for ≤10 output classes)
- Grid overlay for better readability

**Performance Optimizations**:
- Reduced line width (0.8) for high-dimensional data
- Color cycling for >20 dimensions
- Efficient matplotlib rendering

## Technical Implementation Details

### Core Architecture Components

#### HeinsenMinGeneralGRULayer
```python
# Key computation in the recurrent layer
def heinsen_update(z, h):
    '''
    Formulates h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t as 
    h_t = a_t * h_{t-1} + b_t with precomputed coefficients
    '''
    log_a = -safe_softplus(z)              # 1 - z_t in log space
    log_b = -safe_softplus(-z) + log_g(h)  # z_t * h_tilde_t in log space
    # Parallel computation using cumulative operations
    a_star = jnp.cumsum(log_a)
    c = log_b - a_star 
    c = jnp.pad(c, (1, 0), constant_values=x_0)
    d = jax.lax.cumlogsumexp(c)
    return jnp.exp(a_star + d[1:])
```

#### DCLSLayer (Delay Compensation)
```python
# Gaussian kernel construction for temporal delays
def gaussian_interpolation(w, p, s, kernel_size):
    k = jnp.arange(kernel_size-1, -1, -1)  # Reversed time indices
    return w * jnp.exp(- (p - k)**2 / (2 * s**2))  # Gaussian interpolation
```

### Data Flow Architecture

**Input Processing**:
1. Raw input (e.g., 784 MNIST pixels)
2. Encoder Dense layer (input_dim → hidden_dim[0])
3. Layer sequence processing

**Per-Layer Processing**:
1. **Convolution Block**: DCLS or standard convolution
2. **Recurrent Block**: Heinsen MinGRU computation
3. **Channel Mixing Block**: MLP or GLU transformation
4. **Compression Block**: Dimensionality reduction (if specified)
5. **Skip Connections**: Element-wise and layer-wise residuals
6. **Normalization**: LayerNorm and post-normalization

**Output Processing**:
1. Final Dense layer (hidden_dim[-1] → num_classes)
2. Logits/predictions for classification

### Configuration System

**YAML-based Configuration**:
```yaml
# Example configuration structure
n_layers: 6
hidden_dim: [128, 128, 128, 128, 128, 128]
latent_dim: [null, null, null, null, null, null]
conv: 'dcls'
delay_type: 'axonal'
kernel_size: 50
enable_conv: true
enable_rec: true
enable_cm: false
```

**Command-line Interface**:
```bash
python main_inf_2.py --dataset mnist --gpu 0 --conv_mode dcls --dcls_config 6 --file_nb 63
```

## Key Insights and Discoveries

### Network Dynamics Patterns

**Recurrent Variables**:
- `h_new`: Shows memory formation and forgetting patterns
- `z_preact`: Reveals gating decisions (what to remember/forget)
- `h_tilde_preact`: Candidate information that could be stored
- `out`: Final layer-wise feature representations

**Skip Connections Impact**:
- Element-wise skips: Preserve gradient flow, prevent vanishing gradients
- Layer-wise skips: Enable deeper architectures, ResNet-like behavior

**Delay Compensation Effects**:
- DCLS layers create temporal receptive fields
- Gaussian kernels smooth temporal dependencies
- Different delay types (axonal/synaptic) show distinct patterns

### Sample Variability Analysis

**Cross-Sample Patterns**:
- Some dimensions show consistent behavior across samples
- Other dimensions exhibit sample-specific activations
- Output layer convergence varies by sample difficulty

**Class-Specific Dynamics**:
- Different input classes activate different hidden dimensions
- Gate patterns (`z_preact`) vary significantly by input type
- Memory utilization (`h_new`) shows class-dependent strategies

## Files and Structure

### Core Implementation Files

**model.py**:
- `RNN_General_Backbone`: Original architecture
- `RNN_General_Backbone_Monitored`: Enhanced monitoring version
- `BatchRNN_General_Monitored`: Vectorized monitored version
- All layer definitions (DCLS, Heinsen, MLP, GLU)

**main_inf_2.py**:
- `plot_monitored_data()`: Comprehensive visualization function
- `main()`: Experiment runner with monitoring
- Complete integration with existing training pipeline

**Configuration System**:
- `yaml_folder/`: Dataset-specific configuration files
- Pattern: `{dataset}_{conv_mode}_c{config}_{file_nb}.yaml`
- Examples: `mnist_dcls_c0_0.yaml`, `path_dcls_c6_63.yaml`

### Generated Outputs

**Recurrent Dynamics** (4 files):
- Hidden states, gate preactivations, candidates, outputs
- 8×5 subplots (input + 6 layers + output × 5 samples)
- All dimensions visualized with smart coloring

**Monitored Variables** (12+ files):
- Every intermediate transformation captured
- Encoder, convolution, recurrent, channel mixing stages
- Skip connections and normalization effects

## Usage Examples

### Basic Monitoring Run
```bash
python main_inf_2.py --dataset mnist --gpu 0 --conv_mode dcls --dcls_config 0 --file_nb 0
```

### Advanced Configuration with Custom Simulation Name
```bash
python main_inf_2.py --dataset path --gpu 1 --conv_mode dcls --dcls_config 6 --file_nb 63 --seed 42 --sim_name experiment_1
```

### Organized Output Structure
```bash
# Generated files are organized in simulation-specific folders:
images/
├── experiment_1/
│   ├── main_inf_2_recurrent_h_new_all_dims.png
│   ├── main_inf_2_recurrent_z_preact_all_dims.png
│   ├── main_inf_2_monitored_encoder_out_all_dims.png
│   └── ... (12+ files total)
├── mnist_baseline/
│   └── ... (another experiment's plots)
└── default/
    └── ... (plots when --sim_name not specified)

# Each subfolder provides:
# 1. Complete signal flow visualization
# 2. Multi-sample comparative analysis  
# 3. Full-dimensional representation (all 128 hidden dims)
# 4. Skip connection impact assessment
# 5. Delay compensation effectiveness
```

## Future Directions and Potential Extensions

### Immediate Enhancements

**Dynamic Analysis**:
- Temporal evolution across training epochs
- Convergence pattern visualization
- Learning dynamics monitoring

**Statistical Analysis**:
- Dimension importance ranking
- Activation distribution analysis
- Cross-layer correlation studies

### Research Applications

**Architecture Optimization**:
- Skip connection necessity assessment
- Layer depth optimization
- Hidden dimension utilization analysis

**Delay Compensation Studies**:
- Optimal kernel parameters
- Delay type effectiveness comparison
- Temporal receptive field analysis

**Comparative Studies**:
- Different architectures side-by-side
- Dataset-specific pattern analysis
- Hyperparameter sensitivity studies

### Technical Improvements

**Performance Optimization**:
- Memory-efficient monitoring for larger models
- Selective variable monitoring (user-configurable)
- Real-time visualization during training

**Visualization Enhancements**:
- Interactive plots with dimension selection
- 3D visualization for temporal-spatial patterns
- Animation support for dynamic analysis

**Integration Features**:
- Weights & Biases integration for monitoring
- Automatic report generation
- Batch processing for multiple configurations

## Conclusion

The monitoring system provides unprecedented insight into the Den-MinGRU architecture, enabling:

1. **Complete Transparency**: Every variable and transformation visible
2. **Multi-Sample Analysis**: Understanding of variability and consistency
3. **Full-Dimensional Coverage**: No hidden information in high-dimensional spaces
4. **Scalable Visualization**: From simple to complex architectures
5. **Research Foundation**: Platform for systematic architecture studies

This comprehensive monitoring capability transforms the black-box neural network into a fully observable system, enabling deep understanding of how delay-enhanced recurrent architectures process sequential information across various domains.

The system is production-ready and can be applied to any configuration within the Den-MinGRU framework, providing researchers with the tools needed to understand, analyze, and optimize their models at an unprecedented level of detail.