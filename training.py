import jax 
import jax.numpy as jnp
from functools import partial
import optax
from tqdm import tqdm
import numpy as np
from flax.training import train_state
from time import time
import wandb
from flax.traverse_util import flatten_dict, unflatten_dict
from utils import prep_batch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings for cleaner output

from custom_logging import LoggingConfig, log_training_batch, log_classification_metrics


def clip_gradients_elementwise(grads, min_val=-1.0, max_val=1.0):
    return jax.tree_map(lambda g: jnp.clip(g, min_val, max_val), grads)

@jax.jit
def update_model(state, grads, kernel_size, grad_clip_norm=1.0):
    # Apply gradient clipping
    grad_norm = optax.global_norm(grads)
    # clipped_grads = optax.clip_by_global_norm(grad_clip_norm).update(grads, None)[0]
    clipped_grads = clip_gradients_elementwise(grads, -grad_clip_norm, grad_clip_norm)
    grad_norm_post_clip = optax.global_norm(clipped_grads)
    
    state = state.apply_gradients(grads=clipped_grads)
    def clip_negative_positions(params):
        def clip_fn(path, v):
            # path is a tuple of keys, e.g. ('Dense_0', 'DCLS', 'positions')
            if path[-1] == 'positions':
                v = jnp.where(v < 0, 0, v)
                v = jnp.where(v > kernel_size, kernel_size, v)
            return v
        flat_params = flatten_dict(params, sep='/')
        clipped_flat = {k: clip_fn(k.split('/'), v) for k, v in flat_params.items()}
        return unflatten_dict(clipped_flat, sep='/')

    state = state.replace(params=clip_negative_positions(state.params))
    return state, grad_norm, grad_norm_post_clip


@partial(jax.jit, static_argnames=('model','reg_factor', 'class_weights'))
def apply_model(state, model, x, y, reg_factor, do_key, class_weights):
    """Computes gradients, loss and accuracy for a single batch."""
    # do_key = jax.random.fold_in(do_key, state.step)
    def loss_fn(params):
        net_dyn, logits, monitor = model.apply({'params': params}, x, rngs={'dropout': do_key})
        one_hot = jax.nn.one_hot(y, model.out_dim)
        batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        if class_weights is not None:
            class_weights_jnp = jnp.array(class_weights, dtype=jnp.float32) # Need to create a new variable to avoid a shadowing error
            batch_loss = batch_loss * class_weights_jnp[y]
        reg = 0.
        for layers in net_dyn:
            # reg = reg + jnp.where(jnp.abs(layers[2]) > 1, (layers[2]-1)**2, 0.0).sum() # h_tilde_preact
            reg = reg + (layers[2]**2).mean() # h_tilde_preact
            # reg += jnp.where(jnp.abs(layers[1]) > 1, layers[1]**2, 0.0).sum() # z_preact
        # reg += jnp.where(jnp.abs(out_hist) > 1, out_hist**2, 0.0).sum()
        loss = jnp.mean(batch_loss) + reg_factor * reg
        return loss, {'logits': logits, 'batch_loss': batch_loss, 'net_dyn': net_dyn, 'reg': reg, 'monitor': monitor}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)
    
    # Compute prediction probabilities and confidence metrics
    logits = aux_dict['logits'] # shape: (batch_size, out_dim)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == y)
    
    # Prediction confidence (max softmax probability)
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    confidence_std = jnp.std(max_probs)
    
    # Add model behavior metrics to aux_dict
    aux_dict.update({
        'loss': loss,
        'accuracy': accuracy,
        'probs': probs,
        'predictions': predictions,
        'mean_confidence': mean_confidence,
        'confidence_std': confidence_std,
        'max_probs': max_probs
    })
    # aux_dict.pop('logits')
    return grads, loss, accuracy, aux_dict

def apply_retrieval_model(state, model, x, y, reg_factor, do_key):
    def loss_fn(params):
        net_dyn, logits, monitor = model.apply({'params': params}, x, rngs={'dropout': do_key})
        one_hot = jax.nn.one_hot(y, model.out_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, {'logits': logits, 'net_dyn': net_dyn, 'monitor': monitor}
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)

    # Compute prediction probabilities and confidence metrics
    logits = aux_dict['logits'] # shape: (batch_size, out_dim)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == y)

    # Prediction confidence (max softmax probability)
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    confidence_std = jnp.std(max_probs)
    
    # Add model behavior metrics to aux_dict
    aux_dict.update({
        'probs': probs,
        'predictions': predictions,
        'mean_confidence': mean_confidence,
        'confidence_std': confidence_std,
        'max_probs': max_probs
    })
    # aux_dict.pop('logits')
    return grads, loss, accuracy, aux_dict



def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def run_epoch(state, model_cls, train_dl, key, reg_factor, kernel_size, lim_batch=None, keys_to_track=None, inner_keys_to_track=None, lr_fn=None,
              wandb_gradients=False, wandb_states=False, wandb_matrices=False, in_dim=None, seq_len=None, grad_clip_norm=1.0, log_model_behavior=True, epoch_num=0, 
              class_weights=None, retrieval=False, dataset=None):
    """Train for a single epoch."""
    model = model_cls(training=True)
    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = {k: [] for k in keys_to_track+inner_keys_to_track}
    batch_id = 0
    break_flag = False
    key, do_key = jax.random.split(key)
    
    # Initialize logging configuration
    logging_config = LoggingConfig(
        wandb_gradients=wandb_gradients,
        wandb_states=wandb_states,
        wandb_matrices=wandb_matrices,
        log_model_behavior=log_model_behavior
    )
    logging_config.grad_clip_norm = grad_clip_norm
    
    # Model behavior tracking
    all_predictions = []
    all_targets = []
    all_confidences = []
    for batch in progress_bar:
        if len(batch) == 2:  # If the batch is already preprocessed
            batch_x, batch_y = batch
        elif len(batch) == 3:  # If the batch contains mask
            batch_x, batch_y = prep_batch(batch, seq_len, in_dim)
        # start = time()

        grads, loss, accuracy, aux_dict = apply_model(state, model, batch_x, batch_y, reg_factor=reg_factor, do_key=do_key, class_weights=class_weights)

        # stop = time()
        # print("forward pass time:", stop-start)
        for k in keys_to_track:
            aux_dict_hist[k].append(locals()[k])
        for k in inner_keys_to_track:
            if k != 'net_dyn':  # Skip net_dyn to save memory
                aux_dict_hist[k].append(aux_dict[k])
        
        # Collect model behavior data
        if log_model_behavior:
            all_predictions.extend(np.array(aux_dict['predictions']))
            all_targets.extend(np.array(batch_y))
            all_confidences.extend(np.array(aux_dict['max_probs']))
            
            
            
            # find the prediction class distribution
            preds = np.array(aux_dict['predictions'])
            targets = np.array(batch_y)
            unique_classes = np.unique(targets) # np.linspace(0, 9, num=10, dtype=int)  # Assuming classes are 0-9 for classification tasks
            if len(unique_classes) > 4: 
                unique_classes = np.linspace(0, 9, num=10, dtype=int)  # For larger classes, use a fixed range
            else: 
                unique_classes = np.linspace(0, 1, num=2, dtype=int)  # For binary classification, use 0 and 1
            unique_pred_classes = np.unique(preds)
            pred_dist = np.bincount(preds, minlength=len(unique_classes))
            target_dist = np.bincount(targets, minlength=len(unique_classes))
                    
        # Use new logging system
        log_dict = log_training_batch(
            state=state, aux_dict=aux_dict, grads=grads, model=model, lr_fn=lr_fn,
            grad_norm=0.0, post_clip_grad_norm=0.0,  # Will be updated after gradient clipping
            batch_id=batch_id, config=logging_config, dataset=dataset
        )
        
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
        
        if jnp.isnan(loss).sum() > 0:
            print('NAN Loss')
            # check if ther is a NaN in the gradients
            # if grads_previous is not None:
            #     for k, v in grads_previous.items():
            #         if jnp.isnan(v).sum() > 0:
            #             print(f'NaN in gradients: {k}')
            flat_grads = flatten_dict(grads, sep='/')
            for key, grad in flat_grads.items():
                if grad is not None:
                    if jnp.isnan(grad).sum() > 0:
                        print(f'NaN in gradients: {key}')
                    # flat_grad = jnp.ravel(grad)
                    # wandb.log({f"gradients/{key}": wandb.Histogram(flat_grad)}, step=state.step)
            for l, ldyn in enumerate(aux_dict['net_dyn']):
                for i in range(3):
                    print(f"Layer {l} | {i} | {ldyn[i].shape} | {ldyn[i].min()} | {ldyn[i].max()} | {ldyn[i].mean()}")

            break_flag = True
            break
        # start = time()
        state, grad_norm, post_clip_grad_norm = update_model(state, grads, kernel_size, grad_clip_norm)
        
        # Update gradient clipping info in the log dictionary
        log_dict.update({
            "train_grad/norm": grad_norm,
            "train_grad/norm_post_clip": post_clip_grad_norm,
            "train_grad/norm_clipped": int(grad_norm > grad_clip_norm)
        })
        
        # Single consolidated wandb log call
        wandb.log(log_dict, step=state.step)
        
        grads_previous = grads
        # print(f"lr: {lr_fn(state.step)}")
        # stop = time()
        # print("update pass time:", stop-start)
        batch_id += 1

        if batch_id % 3 == 0:
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())       

        if lim_batch is not None and batch_id >= lim_batch:
            break_flag = True
            break 
        
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    
    # Compute epoch-level model behavior metrics using new logging system
    if log_model_behavior and len(all_predictions) > 0:
        epoch_metrics = log_classification_metrics(
            all_predictions, all_targets, all_confidences, 
            split_name="train_epoch", config=logging_config
        )
        # Log epoch metrics with custom step to avoid conflicts  
        wandb.log(epoch_metrics, step=state.step, commit=False)
    
    return state, train_loss, train_accuracy, (break_flag, aux_dict_hist)

@partial(jax.jit, static_argnames=('model', 'out_dim'))
def eval_model(state, model, images, labels, out_dim):
    """Computes loss, accuracy, and predictions for a single batch."""

    def loss_fn(params):
        net_dyn, logits, _ = model.apply({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, out_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    loss, logits = loss_fn(state.params)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == labels)
    
    # Compute confidence metrics
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    
    return loss, accuracy, predictions, max_probs, mean_confidence

@partial(jax.jit, static_argnames=('model', 'out_dim'))
def inf_model(state, model, images, labels, out_dim):
    """Computes loss, accuracy, and predictions for a single batch."""

    def loss_fn(params):
        ndh, logits, monitor = model.apply({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, out_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, ndh, logits

    loss, ndh, logits = loss_fn(state.params)
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == labels)
    
    # Compute confidence metrics
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    
    return loss, accuracy, ndh


def validate(state, model, testloader, seq_len, in_dim, out_dim, log_classification_report=True, split_name='val'):
    # Compute average loss & accuracy
    model = model(training=False) # needed when using dropout
    losses, accuracies = [], []
    all_predictions, all_targets, all_confidences = [], [], []
    progress_bar = tqdm(testloader, desc="Validation", leave=True)
    
    for batch in progress_bar:
        if len(batch) == 2:  # If the batch is already preprocessed
            inputs, labels = batch
        elif len(batch) == 3:  # If the batch contains mask
            inputs, labels = prep_batch(batch, seq_len, in_dim)
        
        loss, acc, predictions, max_probs, mean_confidence = eval_model(
            state, model, inputs, labels, out_dim
        )
        
        losses.append(loss)
        accuracies.append(acc)
        
        # Collect for classification report
        if log_classification_report:
            all_predictions.extend(np.array(predictions))
            all_targets.extend(np.array(labels))
            all_confidences.extend(np.array(max_probs))
    
    # Compute and log classification metrics using new logging system
    if log_classification_report and len(all_predictions) > 0:
        eval_metrics = log_classification_metrics(
            all_predictions, all_targets, all_confidences, split_name=split_name
        )
        return np.mean(losses), np.mean(accuracies), eval_metrics
    else:
        return np.mean(losses), np.mean(accuracies), {}


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


def create_learning_rate_map(args, steps_per_epoch):
    lr_fn = create_learning_rate_fn(args, args.lr, steps_per_epoch) if args.scheduler else args.lr
    lr_big_fn = create_learning_rate_fn(args, args.lr * 5, steps_per_epoch) if args.scheduler else args.lr_big
    
    # Create layer-specific learning rates if lr_factors is provided
    layer_lr_fns = {}
    if hasattr(args, 'lr_factors') and args.lr_factors is not None:
        for layer_idx, factor in enumerate(args.lr_factors):
            layer_lr_fns[layer_idx] = create_learning_rate_fn(args, args.lr * factor, steps_per_epoch) if args.scheduler else args.lr * factor
    
    print(lr_fn)
    
    # Layer-type and parameter-specific rules
    # Format: (layer_pattern, param_name) -> optimizer_type
    param_rules = []
    
    # DCLS-specific parameters (with train/freeze control)
    if args.train_std:
        param_rules.append(('DCLSLayer', 'std', 'adam'))
    else: 
        param_rules.append(('DCLSLayer', 'std', 'none'))
    
    # Layer-specific bias optimization (only for layers that have bias)
    default_bias_optim = getattr(args, 'bias_optim', 'adam') 
    mlp_bias_optim = getattr(args, 'mlp_bias_optim', default_bias_optim) 
    gru_bias_optim = getattr(args, 'gru_bias_optim', default_bias_optim)
    postnorm_bias_optim = getattr(args, 'postnorm_bias_optim', default_bias_optim)
    
    param_rules.extend([
        ('MLP', 'bias', mlp_bias_optim),
        ('HeinsenMinGeneralGRULayer', 'bias', gru_bias_optim),
        ('LayerNormPost', 'bias', postnorm_bias_optim),
        ('Encoder', 'bias', args.bias_optim),
        ('Dense_Out', 'bias', args.bias_optim),
    ])
    
    # DCLS weights and positions (with train/freeze control)
    if args.train_weights:
        dcls_weights_optim = getattr(args, 'dcls_weights_optim', 'adamw')
        param_rules.append(('DCLSLayer', 'weights', dcls_weights_optim))
    else:
        param_rules.append(('DCLSLayer', 'weights', 'none'))
        
    if args.train_positions:
        dcls_positions_optim = getattr(args, 'dcls_positions_optim', 'adam_big')
        param_rules.append(('DCLSLayer', 'positions', dcls_positions_optim))
    else:
        param_rules.append(('DCLSLayer', 'positions', 'none'))
    
    # Layer-specific scale optimization (only for layers that have scale)
    default_scale_optim = getattr(args, 'scale_optim', 'adam')
    mlp_scale_optim = getattr(args, 'mlp_scale_optim', default_scale_optim)
    postnorm_scale_optim = getattr(args, 'postnorm_scale_optim', default_scale_optim)
    
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

    lr_map = {
        'none': {'tx': optax.set_to_zero()},
        'adam': {'tx': optax.adam(lr_fn)},
        'adam_big': {'tx': optax.adam(lr_big_fn)},
        'adamw': {'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay)},
        'adamw_big': {'tx': optax.adamw(lr_big_fn, weight_decay=args.weight_decay)},
        'adamw_small': {'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay / 10)},
        'param_rules': param_rules,  # Store rules for label_fn
        'layer_lr_fns': layer_lr_fns  # Store layer-specific learning rates
    }
    
    # Add layer-specific optimizers if lr_factors is provided
    if layer_lr_fns:
        for layer_idx, layer_lr_fn in layer_lr_fns.items():
            lr_map[f'adam_layer_{layer_idx}'] = {'tx': optax.adam(layer_lr_fn)}
            lr_map[f'adamw_layer_{layer_idx}'] = {'tx': optax.adamw(layer_lr_fn, weight_decay=args.weight_decay)}
    
    print("Learning rate map optimizers:", {k: v for k, v in lr_map.items() if k != 'param_rules'})
    print(f"Parameter rules: {len(param_rules)} rules defined")
    return lr_map, lr_fn


def init_model(key, model_cls, dataset_version, in_dim, seq_len, batch_size):
    
    
    init_x = jnp.ones((batch_size, seq_len, in_dim)) if dataset_version == "sequential" else jnp.ones((batch_size, jnp.sqrt(seq_len), jnp.sqrt(seq_len)))

    model = model_cls(training=True)
    if model.padded: 
        init_x = (init_x, jnp.ones((batch_size, 1)))  # add dummy mask for initialization if model expects padding mask
    key, pkey, do_key = jax.random.split(key, 3)
    params = model.init({'params': pkey, 'dropout': do_key}, init_x)['params']
    return model, params


def create_train_state(key, model_cls, lr_map, dataset_version, in_dim, seq_len, batch_size, wd=0.05):
    
    """Creates initial `TrainState`."""
    model, params = init_model(key, model_cls, dataset_version, in_dim, seq_len, batch_size)
    
    # Debugging: Print parameter structure
    print("Initialized parameter structure:", jax.tree_util.tree_map(jnp.shape, params))

    param_sizes = map_nested_fn(lambda k, param: param.size)(params)
    n_params = sum(jax.tree_util.tree_leaves(param_sizes))
    print(f"[*] Trainable Parameters: {n_params}")

    def label_fn(params):
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
                for layer_pattern in ['DCLSLayer', 'MLP', 'HeinsenMinGeneralGRULayer', 'LayerNormPost', 'Encoder', 'Dense_Out']:
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

    # 2. Define optimizer transformations
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
        # key=do_key
    ), n_params, params