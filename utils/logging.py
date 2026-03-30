import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax.traverse_util import flatten_dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LoggingConfig:
    """Configuration for training logging behavior"""
    wandb_gradients: bool = False
    wandb_states: bool = False
    wandb_matrices: bool = False
    log_model_behavior: bool = True
    
    # Logging frequencies
    gradient_freq: int = 1  # Log gradients every batch
    dynamics_freq: int = 1500  # Log dynamics every N batches
    matrix_freq: int = 2400  # Log parameter matrices every N batches
    
    # Classification logging
    max_classes_for_report: int = 10  # Max classes to generate detailed reports for


def create_histogram_and_stats(data: jnp.ndarray, prefix: str) -> Dict[str, Any]:
    """
    Create wandb histogram and statistics for a given tensor.
    
    Args:
        data: JAX array to analyze
        prefix: Prefix for the metric names (e.g., "train_dynamics_stats/z_layer_0")
    
    Returns:
        Dictionary with histogram and statistics
    """
    metrics = {}
    
    # Create histogram
    flat_data = jnp.ravel(data)
    metrics[f"{prefix}_histogram"] = wandb.Histogram(flat_data)
    
    # Create statistics
    metrics[f"{prefix}_mean"] = float(jnp.mean(data))
    metrics[f"{prefix}_std"] = float(jnp.std(data))
    metrics[f"{prefix}_max"] = float(jnp.max(data))
    metrics[f"{prefix}_min"] = float(jnp.min(data))
    metrics[f"{prefix}_mean_abs"] = float(jnp.mean(jnp.abs(data)))
    
    return metrics


def log_gradient_histograms(grads: Dict, config: LoggingConfig, batch_id: int) -> Dict[str, Any]:
    """
    Log gradient histograms and statistics.
    
    Args:
        grads: Gradient dictionary from training
        config: Logging configuration
        batch_id: Current batch ID
    
    Returns:
        Dictionary of gradient metrics for wandb
    """
    
    grad_metrics = {}
    flat_grads = flatten_dict(grads, sep='/')
    grad_variances = {}
    grad_norms = {}
    
    for key, grad in flat_grads.items():
        if grad is not None:
            grad_variances[key] = jnp.var(grad)
            grad_norms[key] = jnp.linalg.norm(grad)
            
            if jnp.isnan(grad).sum() > 0:
                print(f'NaN in gradients: {key}')
            
            flat_grad = jnp.ravel(grad)
            grad_metrics[f"train/gradients/{key}"] = wandb.Histogram(flat_grad)
    
    # Add gradient variances and norms per layer
    for key, var in grad_variances.items():
        grad_metrics[f"grad_variance/{key}"] = var
    for key, norm in grad_norms.items():
        grad_metrics[f"grad_norm/{key}"] = norm
    
    return grad_metrics


def log_network_dynamics(aux_dict: Dict, config: LoggingConfig, batch_id: int) -> Dict[str, Any]:
    """
    Log network dynamics (net_dyn) histograms and statistics.
    
    Args:
        aux_dict: Auxiliary data from training step
        config: Logging configuration
        batch_id: Current batch ID
    
    Returns:
        Dictionary of network dynamics metrics for wandb
    """
    
    dynamics_metrics = {}
    net_dyn = aux_dict['net_dyn']
    
    for layer_idx, layer_dynamics in enumerate(net_dyn):
        h_new, z_preact, h_tilde_preact, out = layer_dynamics
        
        # Compute gate values z = sigmoid(z_preact)
        z = jax.nn.sigmoid(z_preact)
        
        # Log histograms for h_new and z (gates)
        dynamics_metrics.update(create_histogram_and_stats(
            h_new, f"train_dynamics/h_new_layer_{layer_idx}"))
        dynamics_metrics.update(create_histogram_and_stats(
            h_tilde_preact, f"train_dynamics/h_tilde_preact_layer_{layer_idx}"))
        dynamics_metrics.update(create_histogram_and_stats(
            z, f"train_dynamics/z_layer_{layer_idx}"))
        dynamics_metrics.update(create_histogram_and_stats(
            z_preact, f"train_dynamics/z_preact_layer_{layer_idx}"))
    
    return dynamics_metrics


def log_monitor_data(aux_dict: Dict, config: LoggingConfig, batch_id: int) -> Dict[str, Any]:
    """
    Log monitor dictionary data from RNN_General_Backbone_Monitored.
    
    Args:
        aux_dict: Auxiliary data from training step
        config: Logging configuration
        batch_id: Current batch ID
    
    Returns:
        Dictionary of monitor metrics for wandb
    """
    
    monitor_metrics = {}
    monitor = aux_dict['monitor']
    
    # Log encoder output if available
    if monitor.get('encoder_out') is not None:
        monitor_metrics.update(create_histogram_and_stats(
            monitor['encoder_out'], "train_monitor/encoder_out"))
    
    # Log final output if available
    if monitor.get('final_output') is not None:
        monitor_metrics.update(create_histogram_and_stats(
            monitor['final_output'], "train_monitor/final_output"))
    
    # Log per-layer monitor data
    if 'layers' in monitor:
        for layer_info in monitor['layers']:
            layer_idx = int(layer_info['layer_id'][0])
            
            for key, value in layer_info.items():
                if key != 'layer_id' and value is not None:
                    monitor_metrics.update(create_histogram_and_stats(
                        value, f"train_monitor/{key}_layer_{layer_idx}"))
    
    return monitor_metrics


def create_confusion_matrix_plot(cm: np.ndarray, class_labels: List) -> Any:
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


def log_parameter_matrices(state, model, config: LoggingConfig, batch_id: int, dataset: str) -> Dict[str, Any]:
    """
    Log parameter matrices as images (DCLS kernels, Dense weights, etc.).
    
    Args:
        state: Training state with parameters
        model: Model instance
        config: Logging configuration
        batch_id: Current batch ID
        dataset: Dataset name for DCLS parameter structure
    
    Returns:
        Dictionary of parameter visualization metrics for wandb
    """
    if not config.wandb_states or batch_id % config.dynamics_freq != 0:
        return {}
    
    param_metrics = {}
    flat_params = flatten_dict(state.params, sep='/')
    
    for param_name, param_value in flat_params.items():
        # Special handling for DCLS layers - reconstruct and plot the actual kernels
        if 'DCLSLayer' in param_name and param_name.endswith('weights'):
            param_metrics.update(_log_dcls_kernel(
                state, model, param_name, dataset, config.wandb_matrices, batch_id))
                
        elif 'DCLSLayer' not in param_name:  # Regular parameters (not DCLS)
            param_metrics.update(_log_regular_parameter(
                param_name, param_value, config.wandb_matrices, batch_id))
    
    return param_metrics


def _log_dcls_kernel(state, model, param_name: str, dataset: str, 
                    log_matrices: bool, batch_id: int) -> Dict[str, Any]:
    """Log DCLS kernel reconstruction and statistics."""
    from model import construct_kernel_fast
    
    metrics = {}
    
    # Get DCLS parameters - handle different parameter structures
    if dataset == 'aan':
        subnet_name = param_name.split('/')[0]
        layer_name = param_name.split('/')[1]
        weights = state.params[subnet_name][layer_name]['weights']
        positions = state.params[subnet_name][layer_name]['positions']
        std = state.params[subnet_name][layer_name]['std']
    else:
        layer_name = param_name.split('/')[0]
        weights = state.params[layer_name]['weights']
        positions = state.params[layer_name]['positions']
        std = state.params[layer_name]['std']
    
    # Determine kernel dimensions and size
    kernel_size = model.kernel_size
    n_dim = 3 if weights.ndim == 3 else 2  # synaptic vs axonal
    
    # Reconstruct the actual kernel
    kernel = None
    for j in range(weights.shape[0]):  # iterate over kernel elements
        k_j = construct_kernel_fast(weights[j], positions[j], std[j], kernel_size, n_dim)
        kernel = k_j if kernel is None else kernel + k_j
    
    # Flip kernel for visualization
    kernel = np.flip(np.array(kernel), axis=-1)
    
    # Plot with dual scale
    if log_matrices and batch_id % 2400 == 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        max_val = 0.5
        
        # Full scale
        im1 = ax1.imshow(kernel, cmap='RdBu_r', vmin=-max_val, vmax=max_val, aspect='auto')
        ax1.set_title(f"{layer_name} Kernel (full scale)")
        ax1.set_ylabel('Hidden Dimension')
        plt.colorbar(im1, ax=ax1, fraction=0.02)
        
        # Half scale
        im2 = ax2.imshow(kernel, cmap='RdBu_r', vmin=-max_val/2, vmax=max_val/2, aspect='auto')
        ax2.set_title(f"{layer_name} Kernel (half scale)")
        ax2.set_ylabel('Hidden Dimension')
        ax2.set_xlabel('Kernel Position')
        plt.colorbar(im2, ax=ax2, fraction=0.02)
        
        metrics[f"train_params_plot/{layer_name}_kernel"] = wandb.Image(fig)
        plt.close(fig)
    
    # Log kernel statistics
    metrics.update({
        f"train_params_stats/{layer_name}_weights_mean": float(np.mean(weights)),
        f"train_params_stats/{layer_name}_weights_mean_abs": float(np.mean(np.abs(weights))),
        f"train_params_stats/{layer_name}_weights_std": float(np.std(weights)),
        f"train_params_stats/{layer_name}_weights_max": float(np.max(weights)),
        f"train_params_stats/{layer_name}_weights_min": float(np.min(weights)),
    })
    
    return metrics


def _log_regular_parameter(param_name: str, param_value: jnp.ndarray, 
                          log_matrices: bool, batch_id: int) -> Dict[str, Any]:
    """Log regular (non-DCLS) parameter statistics and visualizations."""
    metrics = {}
    
    if log_matrices and batch_id % 2400 == 0:
        if param_value.ndim == 2:  # 2D matrices (Dense weights, etc.)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(np.array(param_value), aspect='auto', cmap='RdBu_r', 
                          interpolation='nearest', vmin=-0.5, vmax=0.5)
            ax.set_title(f"{param_name}")
            plt.colorbar(im, ax=ax)
            metrics[f"train_params_plot/{param_name}"] = wandb.Image(fig)
            plt.close(fig)
        elif param_value.ndim == 1:  # 1D vectors (biases, layer norm scales/shifts)
            fig, ax = plt.subplots(figsize=(10, 4))
            param_array = np.array(param_value)
            # Plot as a horizontal bar or line plot
            if len(param_array) <= 256:  # For small vectors, use bar plot
                ax.bar(range(len(param_array)), param_array, color='steelblue', alpha=0.7)
                ax.set_xlabel('Parameter Index')
            else:  # For large vectors, use line plot
                ax.plot(param_array, 'steelblue', linewidth=1)
                ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Value')
            ax.set_title(f"{param_name}")
            ax.grid(True, alpha=0.3)
            metrics[f"train_params_plot/{param_name}"] = wandb.Image(fig)
            plt.close(fig)
    
    # Log parameter statistics for all parameter types
    metrics.update({
        f"train_params_stats/{param_name}_mean": float(jnp.mean(param_value)),
        f"train_params_stats/{param_name}_mean_abs": float(jnp.mean(jnp.abs(param_value))),
        f"train_params_stats/{param_name}_std": float(jnp.std(param_value)),
        f"train_params_stats/{param_name}_max": float(jnp.max(param_value)),
        f"train_params_stats/{param_name}_min": float(jnp.min(param_value)),
    })
    
    return metrics


def log_batch_metrics(aux_dict: Dict, state, lr_fn, grad_norm: float, 
                     post_clip_grad_norm: float, grad_clip_norm: float, 
                     config: LoggingConfig) -> Dict[str, Any]:
    """
    Log basic per-batch training metrics.
    
    Args:
        aux_dict: Auxiliary data from training step
        state: Training state
        lr_fn: Learning rate function
        grad_norm: Gradient norm before clipping
        post_clip_grad_norm: Gradient norm after clipping
        grad_clip_norm: Gradient clipping threshold
        config: Logging configuration
    
    Returns:
        Dictionary of batch metrics for wandb
    """
    current_lr = lr_fn(state.step) if lr_fn is not None else 0.0
    
    batch_metrics = {
        "train/learning_rate": current_lr,
        "train/reg": aux_dict['reg'],
        "train_grad/norm": grad_norm,
        "train_grad/norm_log": jnp.log10(grad_norm + 1e-8),
        "train_grad/norm_post_clip": post_clip_grad_norm,
        "train_grad/norm_clipped": int(grad_norm > grad_clip_norm)
    }
    
    # Add model behavior metrics if enabled
    if config.log_model_behavior:
        batch_metrics.update({
            "train/loss": aux_dict['loss'],
            "train/accuracy": aux_dict['accuracy'],
            "train/mean_confidence": aux_dict['mean_confidence'],
            "train/confidence_std": aux_dict['confidence_std'],
        })
    
    return batch_metrics


def log_classification_metrics(all_predictions: List, all_targets: List, 
                             all_confidences: List, split_name: str = "train_epoch",
                             config: LoggingConfig = None) -> Dict[str, Any]:
    """
    Log comprehensive classification metrics including confusion matrix and per-class metrics.
    
    Args:
        all_predictions: List of predictions from all batches
        all_targets: List of targets from all batches
        all_confidences: List of confidence scores from all batches
        split_name: Prefix for metric names (e.g., "train_epoch", "val", "test")
        config: Logging configuration
    
    Returns:
        Dictionary of classification metrics for wandb
    """
    if len(all_predictions) == 0:
        return {}
    
    if config is None:
        config = LoggingConfig()
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    
    metrics = {}
    
    # Class distribution analysis
    
    unique_classes = np.unique(all_targets.astype(int))
    pred_dist = np.bincount(all_predictions.astype(int), minlength=len(unique_classes))
    target_dist = np.bincount(all_targets.astype(int), minlength=len(unique_classes))
    
    # # Confidence statistics
    # mean_confidence = np.mean(all_confidences)
    # confidence_std = np.std(all_confidences)
    
    # metrics.update({
    #     f"{split_name}/mean_confidence": mean_confidence,
    #     f"{split_name}/confidence_std": confidence_std,
    # })
    
    # Add class distribution metrics
    for i, class_idx in enumerate(unique_classes):
        metrics[f"{split_name}/pred_class_{class_idx}_ratio"] = pred_dist[i] / len(all_predictions)
        metrics[f"{split_name}/target_class_{class_idx}_ratio"] = target_dist[i] / len(all_targets)
    
    # Classification report and confusion matrix (only for small number of classes)
    if len(unique_classes) <= config.max_classes_for_report:
        cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
        # Log confusion matrix
        metrics[f"{split_name}/confusion_matrix"] = wandb.Image(
            create_confusion_matrix_plot(cm, unique_classes)
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
                metrics[f"{split_name}/precision_class_{class_idx}"] = class_report[class_key]['precision']
                metrics[f"{split_name}/recall_class_{class_idx}"] = class_report[class_key]['recall']
                metrics[f"{split_name}/f1_class_{class_idx}"] = class_report[class_key]['f1-score']
        
        # Log macro and weighted averages
        if 'macro avg' in class_report:
            metrics[f"{split_name}/macro_avg_precision"] = class_report['macro avg']['precision']
            metrics[f"{split_name}/macro_avg_recall"] = class_report['macro avg']['recall']
            metrics[f"{split_name}/macro_avg_f1"] = class_report['macro avg']['f1-score']
        
        if 'weighted avg' in class_report:
            metrics[f"{split_name}/weighted_avg_precision"] = class_report['weighted avg']['precision']
            metrics[f"{split_name}/weighted_avg_recall"] = class_report['weighted avg']['recall']
            metrics[f"{split_name}/weighted_avg_f1"] = class_report['weighted avg']['f1-score']
    
    return metrics


def log_training_batch(state, aux_dict: Dict, grads, model, lr_fn, 
                      grad_norm: float, post_clip_grad_norm: float, 
                      batch_id: int, config: LoggingConfig, dataset: str) -> Dict[str, Any]:
    """
    Single entry point for all batch-level logging during training.
    
    Args:
        state: Training state
        aux_dict: Auxiliary data from training step
        grads: Gradients from training step
        model: Model instance
        lr_fn: Learning rate function
        grad_norm: Gradient norm before clipping
        post_clip_grad_norm: Gradient norm after clipping
        batch_id: Current batch ID
        config: Logging configuration
        dataset: Dataset name
    
    Returns:
        Dictionary of all metrics for wandb logging
    """
    log_dict = {}
    
    # Basic batch metrics
    log_dict.update(log_batch_metrics(
        aux_dict, state, lr_fn, grad_norm, post_clip_grad_norm, 
        config.grad_clip_norm if hasattr(config, 'grad_clip_norm') else 1.0, config))
    
    # Gradient logging
    if not config.wandb_gradients or batch_id % config.gradient_freq != 0:
        pass
    else:
        log_dict.update(log_gradient_histograms(grads, config, batch_id))
    
    # Network dynamics logging (minGRU dynamics)
    if not config.wandb_states or batch_id % config.dynamics_freq != 0 or 'net_dyn' not in aux_dict:
        pass
    else: 
        log_dict.update(log_network_dynamics(aux_dict, config, batch_id))
    
    # Monitor data logging (backbone monitoring)
    if not config.wandb_states or batch_id % config.dynamics_freq != 0 or 'monitor' not in aux_dict:
        pass
    else:
        log_dict.update(log_monitor_data(aux_dict, config, batch_id))
    
    # Parameter matrix logging
    log_dict.update(log_parameter_matrices(state, model, config, batch_id, dataset))
    
    return log_dict