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

@jax.jit
def update_model(state, grads, kernel_size, grad_clip_norm=1.0):
    # Apply gradient clipping
    grad_norm = optax.global_norm(grads)
    clipped_grads = optax.clip_by_global_norm(grad_clip_norm).update(grads, None)[0]
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
        net_dyn, out_hist = model.apply({'params': params}, x, rngs={'dropout': do_key})
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(y, model.out_dim)
        batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        if class_weights is not None:
            class_weights_jnp = jnp.array(class_weights, dtype=jnp.float32) # Need to create a new variable to avoid a shadowing error
            batch_loss = batch_loss * class_weights_jnp[y]
        # reg = 0.0
        # for layers in net_dyn:
        #     reg += jnp.where(jnp.abs(layers[2]) > 1, layers[2]**2, 0.0).sum() # h_tilde_preact
        #     # reg += jnp.where(jnp.abs(layers[1]) > 1, layers[1]**2, 0.0).sum() # z_preact
        # reg += jnp.where(jnp.abs(out_hist) > 1, out_hist**2, 0.0).sum()
        loss = jnp.mean(batch_loss) #+ reg_factor * reg
        return loss, {'logits': logits, 'batch_loss': batch_loss, 'net_dyn': net_dyn}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)
    
    # Compute prediction probabilities and confidence metrics
    logits = aux_dict['logits']
    probs = jax.nn.softmax(logits)
    predictions = jnp.argmax(logits, -1)
    accuracy = jnp.mean(predictions == y)
    
    # Prediction confidence (max softmax probability)
    max_probs = jnp.max(probs, axis=-1)
    mean_confidence = jnp.mean(max_probs)
    
    # Add model behavior metrics to aux_dict
    aux_dict.update({
        'probs': probs,
        'predictions': predictions,
        'mean_confidence': mean_confidence,
        'max_probs': max_probs
    })
    aux_dict.pop('logits')
    return grads, loss, accuracy, aux_dict

def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn

def plt_confusion_matrix(cm, class_labels):
    """Create a confusion matrix plot for wandb logging."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Convert to PIL image for wandb
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    from PIL import Image
    return Image.open(buf)

def run_epoch(state, model_cls, train_dl, key, reg_factor, kernel_size, lim_batch=None, keys_to_track=None, inner_keys_to_track=None, lr_fn=None,
              wandb_gradients=False, in_dim=None, seq_len=None, grad_clip_norm=1.0, log_model_behavior=True, epoch_num=0, class_weights=None):
    """Train for a single epoch."""
    model = model_cls(training=True)
    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = {k: [] for k in keys_to_track+inner_keys_to_track}
    batch_id = 0
    break_flag = False
    key, do_key = jax.random.split(key)
    
    # Model behavior tracking
    all_predictions = []
    all_targets = []
    all_confidences = []
    for batch in progress_bar:
        if len(batch) == 2:  # If the batch is already preprocessed
            batch_x, batch_y = batch
        elif len(batch) == 3:  # If the batch contains mask
            batch_x, batch_y, mask = prep_batch(batch, seq_len, in_dim)
        # start = time()
        grads, loss, accuracy, aux_dict = apply_model(state, model, batch_x, batch_y, reg_factor=reg_factor, do_key=do_key, class_weights=class_weights)
        # stop = time()
        # print("forward pass time:", stop-start)
        for k in keys_to_track:
            aux_dict_hist[k].append(locals()[k])
        for k in inner_keys_to_track:
            aux_dict_hist[k].append(aux_dict[k])
        
        # Collect model behavior data
        if log_model_behavior:
            all_predictions.extend(np.array(aux_dict['predictions']))
            all_targets.extend(np.array(batch_y))
            all_confidences.extend(np.array(aux_dict['max_probs']))
        
        # Compute gradient statistics per layer
        flat_grads = flatten_dict(grads, sep='/')
        grad_variances = {}
        grad_norms = {}
        grad_histograms = {}
        for key, grad in flat_grads.items():
            if grad is not None:
                grad_variances[key] = jnp.var(grad)
                grad_norms[key] = jnp.linalg.norm(grad)
                if wandb_gradients:
                    if jnp.isnan(grad).sum() > 0:
                        print(f'NaN in gradients: {key}')
                    flat_grad = jnp.ravel(grad)
                    grad_histograms[f"train/gradients/{key}"] = wandb.Histogram(flat_grad)
        
        # Prepare all logging data in a single dictionary
        current_lr = lr_fn(state.step) if lr_fn is not None else 0.0
        log_dict = {
            "train/learning_rate": current_lr,
        }
        
        # Add model behavior metrics if enabled
        if log_model_behavior:
            log_dict.update({
                "train/mean_confidence": aux_dict['mean_confidence'],
                "train/confidence_std": jnp.std(aux_dict['max_probs']),
            })
        
        # Add gradient variances and norms per layer
        for key, var in grad_variances.items():
            log_dict[f"grad_variance/{key}"] = var
        for key, norm in grad_norms.items():
            log_dict[f"grad_norm/{key}"] = norm
            
        # Add gradient histograms if enabled
        if wandb_gradients:
            log_dict.update(grad_histograms)

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
        
        # Add gradient clipping info to the log dictionary
        log_dict.update({
            "train/grad_norm": grad_norm,
            "train/grad_norm_log": jnp.log10(grad_norm + 1e-8),
            "train/grad_norm_post_clip": post_clip_grad_norm,
            "train/grad_clipped": int(grad_norm > grad_clip_norm)  # Convert bool to int
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
    
    # Compute epoch-level model behavior metrics
    if log_model_behavior and len(all_predictions) > 0:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        # Class distribution analysis
        unique_classes = np.unique(all_targets)
        pred_dist = np.bincount(all_predictions, minlength=len(unique_classes))
        target_dist = np.bincount(all_targets, minlength=len(unique_classes))
        
        # Confidence statistics
        epoch_mean_confidence = np.mean(all_confidences)
        epoch_confidence_std = np.std(all_confidences)
        
        # Log epoch-level metrics
        epoch_metrics = {
            "train/mean_confidence": epoch_mean_confidence,
            "train/confidence_std": epoch_confidence_std,
        }
        
        # Add class distribution metrics
        for i, class_idx in enumerate(unique_classes):
            epoch_metrics[f"train/pred_class_{class_idx}_ratio"] = pred_dist[i] / len(all_predictions)
            epoch_metrics[f"train/target_class_{class_idx}_ratio"] = target_dist[i] / len(all_targets)
        
        # Classification report and confusion matrix (only for small number of classes to avoid clutter)
        if len(unique_classes) <= 10:
            cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
            # Log confusion matrix as a wandb table or image
            epoch_metrics["train/confusion_matrix"] = wandb.Image(
                plt_confusion_matrix(cm, unique_classes)
            )
            
            # Generate and log classification report
            class_report = classification_report(
                all_targets, all_predictions, 
                labels=unique_classes, 
                output_dict=True,
                zero_division=0
            )
            
            # Log per-class metrics from classification report
            for class_idx in unique_classes:
                class_key = str(class_idx)
                if class_key in class_report:
                    epoch_metrics[f"train/precision_class_{class_idx}"] = class_report[class_key]['precision']
                    epoch_metrics[f"train/recall_class_{class_idx}"] = class_report[class_key]['recall']
                    epoch_metrics[f"train/f1_class_{class_idx}"] = class_report[class_key]['f1-score']
            
            # Log macro and weighted averages
            if 'macro avg' in class_report:
                epoch_metrics["train/macro_avg_precision"] = class_report['macro avg']['precision']
                epoch_metrics["train/macro_avg_recall"] = class_report['macro avg']['recall']
                epoch_metrics["train/macro_avg_f1"] = class_report['macro avg']['f1-score']
            
            if 'weighted avg' in class_report:
                epoch_metrics["train/weighted_avg_precision"] = class_report['weighted avg']['precision']
                epoch_metrics["train/weighted_avg_recall"] = class_report['weighted avg']['recall']
                epoch_metrics["train/weighted_avg_f1"] = class_report['weighted avg']['f1-score']
        
        # Log epoch metrics with custom step to avoid conflicts  
        wandb.log(epoch_metrics, step=state.step, commit=False)
    
    return state, train_loss, train_accuracy, (break_flag, aux_dict_hist)

@partial(jax.jit, static_argnames=('model', 'out_dim'))
def eval_model(state, model, images, labels, out_dim):
    """Computes loss, accuracy, and predictions for a single batch."""

    def loss_fn(params):
        _, out_hist = model.apply({'params': params}, images)
        logits = out_hist.mean(axis=1)
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

def validate(state, model, testloader, seq_len, in_dim, out_dim, log_classification_report=True, dataset_name="val"):
    # Compute average loss & accuracy
    model = model(training=False) # needed when using dropout
    losses, accuracies = [], []
    all_predictions, all_targets, all_confidences = [], [], []
    
    for batch_idx, batch in enumerate(testloader):
        if len(batch) == 2:  # If the batch is already preprocessed
            inputs, labels = batch
        elif len(batch) == 3:  # If the batch contains mask
            inputs, labels, _ = prep_batch(batch, seq_len, in_dim)
        
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
    
    # Compute and log classification metrics
    if log_classification_report and len(all_predictions) > 0:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        # Class distribution analysis
        unique_classes = np.unique(all_targets)
        pred_dist = np.bincount(all_predictions, minlength=len(unique_classes))
        target_dist = np.bincount(all_targets, minlength=len(unique_classes))
        
        # Confidence statistics
        mean_confidence = np.mean(all_confidences)
        confidence_std = np.std(all_confidences)
        
        # Prepare metrics dictionary
        eval_metrics = {
            f"{dataset_name}/mean_confidence": mean_confidence,
            f"{dataset_name}/confidence_std": confidence_std,
        }
        
        # Add class distribution metrics
        for i, class_idx in enumerate(unique_classes):
            eval_metrics[f"{dataset_name}/pred_class_{class_idx}_ratio"] = pred_dist[i] / len(all_predictions)
            eval_metrics[f"{dataset_name}/target_class_{class_idx}_ratio"] = target_dist[i] / len(all_targets)
        
        # Classification report and confusion matrix (only for small number of classes)
        if len(unique_classes) <= 10:
            cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
            # Log confusion matrix
            eval_metrics[f"{dataset_name}/confusion_matrix"] = wandb.Image(
                plt_confusion_matrix(cm, unique_classes)
            )
            
            # Generate and log classification report
            class_report = classification_report(
                all_targets, all_predictions, 
                labels=unique_classes, 
                output_dict=True,
                zero_division=0
            )
            
            # Log per-class metrics from classification report
            for class_idx in unique_classes:
                class_key = str(class_idx)
                if class_key in class_report:
                    eval_metrics[f"{dataset_name}/precision_class_{class_idx}"] = class_report[class_key]['precision']
                    eval_metrics[f"{dataset_name}/recall_class_{class_idx}"] = class_report[class_key]['recall']
                    eval_metrics[f"{dataset_name}/f1_class_{class_idx}"] = class_report[class_key]['f1-score']
            
            # Log macro and weighted averages
            if 'macro avg' in class_report:
                eval_metrics[f"{dataset_name}/macro_avg_precision"] = class_report['macro avg']['precision']
                eval_metrics[f"{dataset_name}/macro_avg_recall"] = class_report['macro avg']['recall']
                eval_metrics[f"{dataset_name}/macro_avg_f1"] = class_report['macro avg']['f1-score']
            
            if 'weighted avg' in class_report:
                eval_metrics[f"{dataset_name}/weighted_avg_precision"] = class_report['weighted avg']['precision']
                eval_metrics[f"{dataset_name}/weighted_avg_recall"] = class_report['weighted avg']['recall']
                eval_metrics[f"{dataset_name}/weighted_avg_f1"] = class_report['weighted avg']['f1-score']
        
        return np.mean(losses), np.mean(accuracies), eval_metrics
    else:
        return np.mean(losses), np.mean(accuracies), {}


def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., 
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
    print(lr_fn)
    none_params = []
    adam_params = []
    adamw_params = []
    if args.train_std:
        adam_params.append('std')
    else: 
        none_params.append('std')
    if args.train_weights:
        adamw_params.append('weights')
    else:
        none_params.append('weights')
    if args.train_positions:
        adam_params.append('positions')
    else:
        none_params.append('positions')
    adam_params.append('bias')
    adamw_params.append('kernel')
    lr_map = {
        'none': {'keys': none_params, 'tx': optax.set_to_zero()},
        'adam': {'keys': adam_params, 'tx': optax.adam(lr_fn)},
        'adamw': {'keys': adamw_params, 'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay)}
    }
    return lr_map, lr_fn


def init_model(key, model_cls, dataset_version, in_dim, seq_len, batch_size):
    
    
    init_x = jnp.ones((batch_size, seq_len, in_dim)) if dataset_version == "sequential" else jnp.ones((batch_size, jnp.sqrt(seq_len), jnp.sqrt(seq_len)))

    model = model_cls(training=True)
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
        for path, _ in flat.items():
            name = path.split('/')[-1]  # last part of the path
            if name in lr_map['none']['keys']:
                labels[path] = 'none'
            elif name in lr_map['adam']['keys']:
                labels[path] = 'adam'
            else:
                labels[path] = 'adamw'
        return unflatten_dict(labels, sep='/')

    # 2. Define optimizer transformations
    tx = optax.multi_transform(
        {
            'adamw': lr_map['adamw']['tx'],  # weight decay
            'adam': lr_map['adam']['tx'],  # adam
            'none': lr_map['none']['tx'],  # frozen
        },
        param_labels=label_fn  # returns pytree of labels
    )

    key, do_key = jax.random.split(key)
        
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        # key=do_key
    ), n_params, params