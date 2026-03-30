# Layer-Specific Learning Rate and Optimizer System

This document explains the comprehensive layer-specific learning rate and optimizer system implemented in Den-minGRU, allowing fine-grained control over optimization for different layer types and individual layers.

## Overview

The system provides three levels of optimization control:

1. **Parameter-Type Control**: Different optimizers for `bias`, `scale`, `kernel`, `weights`, `positions`, `std`
2. **Layer-Type Control**: Different optimizers for `DCLSLayer`, `MLP`, `HeinsenMinGeneralGRULayer`, `LayerNormPost`
3. **Layer-Index Control**: Different learning rates for each layer depth (Layer 0, 1, 2, etc.)

## Architecture Components

### Model Parameter Structure

```python
# Example parameter tree structure:
{
    'DCLSLayer_0': {'positions': (8, 96), 'std': (8, 96), 'weights': (8, 96)},
    'DCLSLayer_1': {'positions': (8, 96), 'std': (8, 96), 'weights': (8, 96)},
    'MLP_0': {
        'Dense_0': {'bias': (192,), 'kernel': (96, 192)}, 
        'Dense_1': {'bias': (96,), 'kernel': (192, 96)},
        'LayerNorm_0': {'bias': (96,), 'scale': (96,)}
    },
    'HeinsenMinGeneralGRULayer_0': {
        'Dense_h': {'bias': (96,), 'kernel': (96, 96)},
        'Dense_z': {'bias': (96,), 'kernel': (96, 96)}
    },
    'LayerNormPost_0': {'bias': (96,), 'scale': (96,)},
    'Encoder': {'bias': (96,), 'kernel': (1, 96)},
    'Dense_Out': {'bias': (2,), 'kernel': (96, 2)}
}
```

## Configuration

### YAML Configuration Example

```yaml
# Basic learning rate settings
lr: 0.005
scheduler: True
weight_decay: 0.01

# Layer-specific learning rate factors
lr_factors: [3, 2, 1.5, 1.1, 1, 0.5]  # Per-layer LR multipliers

# Global optimizer defaults
bias_optim: adam
scale_optim: adam

# Layer-specific optimizer overrides
dcls_weights_optim: adamw      # DCLS weights use AdamW with weight decay
dcls_positions_optim: adam_big # DCLS positions use Adam with 5x base LR
mlp_bias_optim: adam          # MLP bias parameters
mlp_kernel_optim: adamw       # MLP kernel parameters  
mlp_scale_optim: adam         # MLP LayerNorm scale parameters
gru_bias_optim: adam          # GRU bias parameters
gru_kernel_optim: adamw       # GRU kernel parameters
postnorm_bias_optim: adam     # Post-LayerNorm bias parameters
postnorm_scale_optim: adam    # Post-LayerNorm scale parameters

# DCLS parameter training control
train_weights: True
train_positions: True  
train_std: False
```

## Implementation Details

### 1. Learning Rate Creation (`create_learning_rate_map`)

```python
def create_learning_rate_map(args, steps_per_epoch):
    # Base learning rates
    lr_fn = create_learning_rate_fn(args, args.lr, steps_per_epoch) 
    lr_big_fn = create_learning_rate_fn(args, args.lr * 5, steps_per_epoch)
    
    # Create layer-specific learning rates
    layer_lr_fns = {}
    if hasattr(args, 'lr_factors') and args.lr_factors is not None:
        for layer_idx, factor in enumerate(args.lr_factors):
            layer_lr_fns[layer_idx] = create_learning_rate_fn(
                args, args.lr * factor, steps_per_epoch
            ) if args.scheduler else args.lr * factor
```

### 2. Parameter Rules Definition

```python
# DCLS-specific parameters (with train/freeze control)
if args.train_std:
    param_rules.append(('DCLSLayer', 'std', 'adam'))
else: 
    param_rules.append(('DCLSLayer', 'std', 'none'))

if args.train_weights:
    dcls_weights_optim = getattr(args, 'dcls_weights_optim', 'adamw')
    param_rules.append(('DCLSLayer', 'weights', dcls_weights_optim))
else:
    param_rules.append(('DCLSLayer', 'weights', 'none'))

# Layer-specific bias optimization
mlp_bias_optim = getattr(args, 'mlp_bias_optim', args.bias_optim)
gru_bias_optim = getattr(args, 'gru_bias_optim', args.bias_optim)

param_rules.extend([
    ('MLP', 'bias', mlp_bias_optim),
    ('HeinsenMinGeneralGRULayer', 'bias', gru_bias_optim),
    ('LayerNormPost', 'bias', postnorm_bias_optim),
])
```

### 3. Optimizer Creation

```python
# Base optimizers
lr_map = {
    'none': {'tx': optax.set_to_zero()},
    'adam': {'tx': optax.adam(lr_fn)},
    'adam_big': {'tx': optax.adam(lr_big_fn)},
    'adamw': {'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay)},
    'adamw_big': {'tx': optax.adamw(lr_big_fn, weight_decay=args.weight_decay)},
}

# Add layer-specific optimizers
if layer_lr_fns:
    for layer_idx, layer_lr_fn in layer_lr_fns.items():
        lr_map[f'adam_layer_{layer_idx}'] = {'tx': optax.adam(layer_lr_fn)}
        lr_map[f'adamw_layer_{layer_idx}'] = {'tx': optax.adamw(layer_lr_fn, weight_decay=args.weight_decay)}
```

### 4. Parameter Labeling (`label_fn`)

```python
def label_fn(params):
    flat = flatten_dict(params, sep='/')
    labels = {}
    
    for path, _ in flat.items():
        path_parts = path.split('/')
        param_name = path_parts[-1]  # e.g., 'bias', 'kernel'
        
        # Extract layer type and index
        layer_type = None
        layer_idx = None
        for part in path_parts:
            for pattern in ['DCLSLayer', 'MLP', 'HeinsenMinGeneralGRULayer', 'LayerNormPost']:
                if pattern in part:
                    layer_type = pattern
                    if '_' in part and part.split('_')[-1].isdigit():
                        layer_idx = int(part.split('_')[-1])
                    break
        
        # Match against parameter rules
        matched_optimizer = 'adamw'  # default
        for rule_layer_type, rule_param_name, optimizer_type in param_rules:
            if layer_type and rule_layer_type in layer_type:
                if param_name == rule_param_name:
                    matched_optimizer = optimizer_type
                    break
        
        # Apply layer-specific learning rate
        if layer_lr_fns and layer_idx is not None and layer_idx in layer_lr_fns:
            if matched_optimizer == 'adam':
                matched_optimizer = f'adam_layer_{layer_idx}'
            elif matched_optimizer == 'adamw':
                matched_optimizer = f'adamw_layer_{layer_idx}'
        
        labels[path] = matched_optimizer
```

## Learning Rate Scaling Examples

### Example 1: Decreasing Learning Rate by Depth

```yaml
lr: 0.001
lr_factors: [5, 3, 2, 1.5, 1, 0.5]
```

**Result:**
- Layer 0: `0.001 × 5 = 0.005` (early layers learn fast)
- Layer 1: `0.001 × 3 = 0.003`
- Layer 2: `0.001 × 2 = 0.002`
- Layer 3: `0.001 × 1.5 = 0.0015`
- Layer 4: `0.001 × 1 = 0.001` (base rate)
- Layer 5: `0.001 × 0.5 = 0.0005` (final layer learns slowly)

### Example 2: Parameter-Specific Assignments

For a parameter path `DCLSLayer_2/weights`:
1. **Layer Type**: `DCLSLayer`
2. **Parameter Name**: `weights`
3. **Layer Index**: `2`
4. **Base Rule**: `('DCLSLayer', 'weights', 'adamw')` → `adamw`
5. **Layer Scaling**: `adamw` + `layer_2` → `adamw_layer_2`
6. **Final Optimizer**: AdamW with `lr × lr_factors[2]` learning rate

## Optimizer Types

### Available Base Optimizers

- **`adam`**: Standard Adam optimizer
- **`adam_big`**: Adam with 5× base learning rate
- **`adamw`**: AdamW with weight decay
- **`adamw_big`**: AdamW with 5× base learning rate + weight decay
- **`none`**: Frozen parameters (no updates)

### Layer-Specific Optimizers

- **`adam_layer_X`**: Adam with `lr × lr_factors[X]` learning rate
- **`adamw_layer_X`**: AdamW with `lr × lr_factors[X]` learning rate + weight decay

## Parameter Coverage

### DCLSLayer Parameters
- **`std`**: Controlled by `train_std` flag, uses Adam
- **`weights`**: Configurable via `dcls_weights_optim` (default: AdamW)
- **`positions`**: Configurable via `dcls_positions_optim` (default: `adam_big`)

### MLP Parameters
- **`kernel`**: Configurable via `mlp_kernel_optim` (default: AdamW)
- **`bias`**: Configurable via `mlp_bias_optim` (default: `args.bias_optim`)
- **`scale`**: Only in LayerNorm_0, configurable via `mlp_scale_optim`

### HeinsenMinGeneralGRULayer Parameters
- **`kernel`**: Configurable via `gru_kernel_optim` (default: AdamW)
- **`bias`**: Configurable via `gru_bias_optim` (default: `args.bias_optim`)

### LayerNormPost Parameters
- **`scale`**: Configurable via `postnorm_scale_optim`
- **`bias`**: Configurable via `postnorm_bias_optim`

### Encoder/Dense_Out Parameters
- **`kernel`**: Uses default AdamW
- **`bias`**: Uses global `args.bias_optim`

## Advanced Usage

### Fine-Tuning Specific Components

```yaml
# Focus optimization on DCLS positioning
dcls_positions_optim: adam_big  # High LR for position learning
dcls_weights_optim: adamw       # Regularized weight learning
train_std: False                # Freeze standard deviations

# Conservative GRU training
gru_bias_optim: adam           # Standard Adam for GRU biases
gru_kernel_optim: adamw        # Weight decay for GRU kernels

# Layer-specific scaling for gradient flow
lr_factors: [2, 1.8, 1.5, 1.2, 1, 0.8]  # Gradual decrease
```

### Freezing Parameters

```yaml
train_weights: False    # Freeze DCLS weights
train_positions: False  # Freeze DCLS positions
train_std: False       # Freeze DCLS standard deviations
```

## Debugging and Monitoring

The system logs comprehensive information:

```python
print("Learning rate map optimizers:", {k: v for k, v in lr_map.items() if k != 'param_rules'})
print(f"Parameter rules: {len(param_rules)} rules defined")
```

This provides visibility into:
- All created optimizers
- Layer-specific learning rate functions
- Number of parameter matching rules

## Best Practices

1. **Early Layer Focus**: Use higher learning rates for earlier layers to improve gradient flow
2. **Parameter-Type Matching**: Use AdamW for weights/kernels (regularization) and Adam for biases/scales
3. **DCLS Specialization**: Use `adam_big` for positions (discrete optimization) and `adamw` for weights
4. **Conservative Final Layers**: Lower learning rates for output layers to maintain stability
5. **Scheduler Integration**: All layer-specific rates follow the same warmup + cosine decay schedule

## Troubleshooting

### Common Issues

1. **Variable Name Conflicts**: Ensure loop variables don't overwrite JAX random keys
2. **Missing Parameters**: Use `getattr(args, 'param_name', default)` for optional parameters
3. **Layer Index Extraction**: Verify layer naming follows `LayerType_X` pattern
4. **Optimizer Registration**: Ensure all layer-specific optimizers are added to `multi_transform`

### Debug Commands

```python
# Print parameter paths and their assigned optimizers
flat_params = flatten_dict(params, sep='/')
for path in flat_params.keys():
    print(f"{path} -> {label_fn(params)[path]}")
```

This comprehensive system enables precise control over optimization dynamics across the entire network depth and parameter types.