"""
Learning rate schedules and training state initialization.
Contains optimizer configuration and train state creation.
"""
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict


def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""
    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }
    return map_fn


def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch)

    cosine_epochs = max(config.n_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch,
        alpha=config.alpha_cosine)
    print(config.alpha_cosine)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])
    return schedule_fn


def create_learning_rate_schedule(args, steps_per_epoch):
    """
    Create learning rate schedules.

    Handles both scheduled and constant learning rates, including:
    - Base learning rate
    - Large learning rate (5x base, for certain parameters)
    - Layer-specific learning rates (via lr_factors)

    Args:
        args: Configuration containing lr, lr_big, scheduler settings
        steps_per_epoch: Number of training steps per epoch

    Returns:
        Tuple of (lr_fn, lr_big_fn, layer_lr_fns):
            - lr_fn: Base learning rate (schedule or constant)
            - lr_big_fn: Large learning rate (schedule or constant)
            - layer_lr_fns: Dict mapping layer_idx to learning rate
    """
    # Create base and large learning rate schedules
    if args.scheduler:
        lr_fn = create_learning_rate_fn(args, args.lr, steps_per_epoch)
        lr_big_fn = create_learning_rate_fn(args, args.lr * 5, steps_per_epoch)
    else:
        lr_fn = args.lr
        lr_big_fn = args.lr_big

    # Create layer-specific learning rates if lr_factors is provided
    layer_lr_fns = {}
    if hasattr(args, 'lr_factors') and args.lr_factors is not None:
        for layer_idx, factor in enumerate(args.lr_factors):
            if args.scheduler:
                layer_lr_fns[layer_idx] = create_learning_rate_fn(
                    args, args.lr * factor, steps_per_epoch
                )
            else:
                layer_lr_fns[layer_idx] = args.lr * factor

    print(lr_fn)
    return lr_fn, lr_big_fn, layer_lr_fns


def create_parameter_rules(args):
    """
    Define parameter-specific optimizer rules.

    Rules specify which optimizer to use for specific (layer_type, param_name)
    combinations. Supports:
    - DCLS-specific parameters (weights, positions, std) with train/freeze control
    - Layer-specific bias optimization
    - Layer-specific scale optimization (LayerNorm)
    - Layer-specific kernel optimization

    Args:
        args: Configuration containing optimizer settings

    Returns:
        List of (layer_pattern, param_name, optimizer_type) tuples
    """
    param_rules = []

    # DCLS-specific parameters (with train/freeze control)
    if args.conv == 'dcls':
        if args.train_std:
            param_rules.append(('DCLSLayer', 'std', 'adam'))
        else:
            param_rules.append(('DCLSLayer', 'std', 'none'))

    # Layer-specific bias optimization (only for layers that have bias)
    default_bias_optim = getattr(args, 'bias_optim', 'adam')
    mlp_bias_optim = getattr(args, 'mlp_bias_optim', default_bias_optim)
    gru_bias_optim = getattr(args, 'gru_bias_optim', default_bias_optim)
    postnorm_bias_optim = \
        getattr(args, 'postnorm_bias_optim', default_bias_optim)

    param_rules.extend([
        ('MLP', 'bias', mlp_bias_optim),
        ('HeinsenMinGeneralGRULayer', 'bias', gru_bias_optim),
        ('LayerNormPost', 'bias', postnorm_bias_optim),
        ('Encoder', 'bias', args.bias_optim),
        ('Dense_Out', 'bias', args.bias_optim),
    ])

    # DCLS weights and positions (with train/freeze control)
    if args.conv == 'dcls':
        if args.train_weights:
            dcls_weights_optim = getattr(args, 'dcls_weights_optim', 'adamw')
            param_rules.append(('DCLSLayer', 'weights', dcls_weights_optim))
        else:
            param_rules.append(('DCLSLayer', 'weights', 'none'))

        if args.train_positions:
            dcls_positions_optim = \
                getattr(args, 'dcls_positions_optim', 'adam_big')
            param_rules.append(('DCLSLayer', 'positions', dcls_positions_optim))
        else:
            param_rules.append(('DCLSLayer', 'positions', 'none'))

    # Layer-specific scale optimization (only for layers that have scale)
    default_scale_optim = getattr(args, 'scale_optim', 'adam')
    mlp_scale_optim = getattr(args, 'mlp_scale_optim', default_scale_optim)
    postnorm_scale_optim = \
        getattr(args, 'postnorm_scale_optim', default_scale_optim)

    param_rules.extend([
        ('MLP', 'scale', mlp_scale_optim),           # MLP LayerNorm scale
        ('LayerNormPost', 'scale', postnorm_scale_optim),  # Post-layer normalization scale
    ])

    # Layer-specific kernel optimization
    mlp_kernel_optim = getattr(args, 'mlp_kernel_optim', 'adamw')
    gru_kernel_optim = getattr(args, 'gru_kernel_optim', 'adamw')

    param_rules.extend([
        ('MLP', 'kernel', mlp_kernel_optim),
        ('HeinsenMinGeneralGRULayer', 'kernel', gru_kernel_optim),
        ('Encoder', 'kernel', 'adamw'),
        ('Dense_Out', 'kernel', 'adamw'),
    ])

    return param_rules


def create_learning_rate_map(args, steps_per_epoch):
    """
    Create learning rate map with parameter-specific optimizers.

    Combines learning rate schedules and parameter rules into an optimizer map
    that can be used with optax.multi_transform.

    Supports:
    - Layer-specific learning rates (via lr_factors)
    - Parameter-specific optimizers (bias, kernel, scale)
    - DCLS-specific parameter control (weights, positions, std)

    Args:
        args: Configuration containing all optimizer settings
        steps_per_epoch: Number of training steps per epoch

    Returns:
        Tuple of (lr_map, lr_fn):
            - lr_map: Dict mapping optimizer names to optax transformations
            - lr_fn: Base learning rate schedule
    """
    # Extract learning rate schedules
    lr_fn, lr_big_fn, layer_lr_fns = create_learning_rate_schedule(
        args, steps_per_epoch
    )

    # Extract parameter rules
    param_rules = create_parameter_rules(args)

    # Build optimizer map
    lr_map = {
        'none': {'tx': optax.set_to_zero()},
        'adam': {'tx': optax.adam(lr_fn)},
        'adam_big': {'tx': optax.adam(lr_big_fn)},
        'adamw': {'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay)},
        'adamw_big': \
            {'tx': optax.adamw(lr_big_fn, weight_decay=args.weight_decay)},
        'adamw_small': \
            {'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay / 10)},
        'param_rules': param_rules,  # Store rules for label_fn
        'layer_lr_fns': layer_lr_fns  # Store layer-specific learning rates
    }

    # Add layer-specific optimizers if lr_factors is provided
    if layer_lr_fns:
        for layer_idx, layer_lr_fn in layer_lr_fns.items():
            lr_map[f'adam_layer_{layer_idx}'] = {'tx': optax.adam(layer_lr_fn)}
            lr_map[f'adamw_layer_{layer_idx}'] = \
                {'tx': optax.adamw(layer_lr_fn, weight_decay=args.weight_decay)}

    print("Learning rate map optimizers:", \
            {k: v for k, v in lr_map.items() if k != 'param_rules'})
    print(f"Parameter rules: {len(param_rules)} rules defined")
    return lr_map, lr_fn


def init_model(key, model_cls, dataset_version, in_dim, seq_len, batch_size,
                dtype=jnp.float32):
    """
    Initialize model parameters.

    Args:
        key: JAX random key
        model_cls: Model class (partial with config)
        dataset_version: "sequential" or other
        in_dim: Input dimension
        seq_len: Sequence length
        batch_size: Batch size for initialization
        dtype: Data type for parameters

    Returns:
        model: Initialized model instance
        params: Model parameters
    """
    if dataset_version == "sequential":
        init_x = jnp.ones((batch_size, seq_len, in_dim), dtype=dtype)
    else:
        init_x = jnp.ones((batch_size, jnp.sqrt(seq_len), jnp.sqrt(seq_len)))

    model = model_cls(training=True)
    if model.padded:
        init_x = (init_x, jnp.ones((batch_size, 1), dtype=dtype))  # add dummy mask for initialization if model expects padding mask
    key, pkey, do_key = jax.random.split(key, 3)
    params = model.init({'params': pkey, 'dropout': do_key}, init_x)['params']
    if dtype != jnp.float32:
        params = jax.tree_map(
            lambda x: x.astype(dtype) if x.dtype == jnp.float32 else x, params
        )
    return model, params


def create_train_state(key, model_cls, lr_map, dataset_version, in_dim, seq_len,
                        batch_size, wd=0.05, dtype=jnp.float32):
    """
    Creates initial `TrainState` with parameter-specific optimizers.

    Args:
        key: JAX random key
        model_cls: Model class (partial with config)
        lr_map: Learning rate map from create_learning_rate_map
        dataset_version: "sequential" or other
        in_dim: Input dimension
        seq_len: Sequence length
        batch_size: Batch size for initialization
        wd: Weight decay (unused, kept for compatibility)
        dtype: Data type for parameters

    Returns:
        state: Flax TrainState
        n_params: Total number of parameters
        params: Model parameters
    """
    model, params = init_model(
        key, model_cls, dataset_version, in_dim, seq_len, batch_size,
        dtype=dtype
    )

    # Debugging: Print parameter structure
    print("Initialized parameter structure:", \
            jax.tree_util.tree_map(jnp.shape, params))

    param_sizes = map_nested_fn(lambda k, param: param.size)(params)
    n_params = sum(jax.tree_util.tree_leaves(param_sizes))
    print(f"[*] Trainable Parameters: {n_params}")

    def label_fn(params):
        """Assign optimizer labels to each parameter."""
        flat = flatten_dict(params, sep='/')
        labels = {}
        param_rules = lr_map['param_rules']
        layer_lr_fns = lr_map['layer_lr_fns']

        for path, _ in flat.items():
            path_parts = path.split('/')
            param_name = path_parts[-1]  # last part (e.g., 'bias', 'kernel')

            # Find layer type and layer index from path
            layer_type = None
            layer_idx = None
            for part in path_parts:
                for layer_pattern in ['DCLSLayer', 'MLP', 'Encoder',
                                        'HeinsenMinGeneralGRULayer',
                                        'LayerNormPost', 'Dense_Out']:
                    if layer_pattern in part:
                        layer_type = layer_pattern
                        # Extract layer index (e.g., 'DCLSLayer_0' -> 0)
                        if '_' in part and part.split('_')[-1].isdigit():
                            layer_idx = int(part.split('_')[-1])
                        break
                if layer_type:
                    break

            # Match against parameter rules
            matched_optimizer = 'adamw'  # default fallback
            for rule_layer_type, rule_param_name, optimizer_type in param_rules:
                if layer_type and rule_layer_type in layer_type:
                    if param_name == rule_param_name:
                        matched_optimizer = optimizer_type
                        break

            # Apply layer-specific learning rate if available
            if layer_lr_fns and layer_idx is not None and layer_idx in layer_lr_fns:
                # Convert base optimizer to layer-specific version
                if matched_optimizer == 'adam':
                    matched_optimizer = f'adam_layer_{layer_idx}'
                elif matched_optimizer == 'adamw':
                    matched_optimizer = f'adamw_layer_{layer_idx}'
                # Keep other optimizers (none, adam_big, adamw_big) as-is for now

            labels[path] = matched_optimizer

        return unflatten_dict(labels, sep='/')

    # Define optimizer transformations
    optimizer_transforms = {
        'adamw': lr_map['adamw']['tx'],  # weight decay
        'adamw_big': lr_map['adamw_big']['tx'],  # weight decay with big learning rate
        'adamw_small': lr_map['adamw_small']['tx'],  # weight decay / 10
        'adam': lr_map['adam']['tx'],  # adam
        'adam_big': lr_map['adam_big']['tx'],  # adam with big learning rate
        'none': lr_map['none']['tx'],  # frozen
    }

    # Add layer-specific optimizers if they exist
    for opt_key, opt_value in lr_map.items():
        if opt_key.startswith('adam_layer_') or opt_key.startswith('adamw_layer_'):
            optimizer_transforms[opt_key] = opt_value['tx']

    tx = optax.multi_transform(
        optimizer_transforms,
        param_labels=label_fn  # returns pytree of labels
    )

    key, do_key = jax.random.split(key)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), n_params, params
