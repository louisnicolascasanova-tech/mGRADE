import jax
import numpy as np
import torch
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import matplotlib.pyplot as plt
px = 1 / plt.rcParams['figure.dpi']
jnp.set_printoptions(precision=3, suppress=True, linewidth=10000000)
from utils import create_mnist_classification_dataset, create_cifar_gs_classification_dataset, write_config_yaml, \
        create_lra_imdb_classification_dataset, create_lra_listops_classification_dataset, \
        create_lra_path32_classification_dataset, create_lra_pathx_classification_dataset, \
        prep_batch, setup_random_seeds, parse_experiment_config, generate_experiment_id, create_experiment_directories, \
        compute_class_weights 
from plots import plot_dynamics

from model import BatchRNN_General_Monitored
from training import create_train_state, run_epoch, validate, create_learning_rate_map
from functools import partial

import os
import argparse
import yaml
import wandb


def plot_monitored_data(ndh, monitor, inputs, out_hist, labels, sim_name='default'):
    """
    Create comprehensive plots for all monitored variables across 5 batch samples.
    Now plots ALL dimensions (up to 128) instead of just first 5.
    Saves plots in images/{sim_name}/ subfolder.
    """
    # Create output directory
    output_dir = os.path.join('images', sim_name)
    os.makedirs(output_dir, exist_ok=True)
    num_samples = min(5, inputs.shape[0])
    
    # Define all variables to plot from recurrent layers
    rec_variables = ['h_new', 'z_preact', 'h_tilde_preact', 'out']
    
    # Define all variables to plot from monitoring
    monitor_variables = [
        'encoder_out',
        'conv_input', 'conv_output', 
        'rec_input', 'rec_ln_output', 'rec_output',
        'cm_input', 'cm_output',
        'compression_output',
        'layer_skip_output',
        'postnorm_output',
        'final_layer_output'
    ]
    
    # Plot recurrent variables (from ndh) - ALL dimensions
    for var_name in rec_variables:
        fig, axes = plt.subplots(8, num_samples, figsize=(4*num_samples, 20))
        fig.suptitle(f'Recurrent Dynamics: {var_name} (5 Batch Samples - ALL DIMS)', fontsize=16)
        
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for sample_idx in range(num_samples):
            
            # Plot input at the top (row 0) - ALL input dimensions
            ax_input = axes[0, sample_idx]
            max_input_dims = inputs.shape[-1]  # Plot ALL input dimensions
            
            # Use colormap for many dimensions
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, max_input_dims)))
            if max_input_dims > 20:
                colors = plt.cm.viridis(np.linspace(0, 1, max_input_dims))
            
            for dim in range(max_input_dims):
                color = colors[dim % len(colors)]
                alpha = 0.8 if max_input_dims <= 20 else 0.3
                ax_input.plot(inputs[sample_idx, :, dim], 
                             color=color, alpha=alpha, linewidth=0.8)
            
            ax_input.set_title(f'Input ({max_input_dims} dims) - Sample {sample_idx}')
            ax_input.set_xlabel('Time Steps')
            ax_input.set_ylabel('Input Values')
            ax_input.grid(True, alpha=0.3)
            
            # Plot network layers (rows 1-6) - ALL dimensions
            for layer_idx, layer_data in enumerate(ndh):
                if layer_idx >= 6:
                    break
                    
                h_new, z_preact, h_tilde_preact, out = layer_data
                
                if var_name == 'h_new':
                    data = h_new
                elif var_name == 'z_preact':
                    data = z_preact
                elif var_name == 'h_tilde_preact':
                    data = h_tilde_preact
                elif var_name == 'out':
                    data = out
                
                
                ax = axes[layer_idx + 1, sample_idx]
                max_dims = data.shape[-1]  # Plot ALL dimensions
                
                # Use colormap for many dimensions
                colors = plt.cm.tab20(np.linspace(0, 1, min(20, max_dims)))
                if max_dims > 20:
                    colors = plt.cm.viridis(np.linspace(0, 1, max_dims))
                
                for dim in range(max_dims):
                    color = colors[dim % len(colors)]
                    alpha = 0.8 if max_dims <= 20 else 0.3
                    ax.plot(data[sample_idx, :, dim], 
                           color=color, alpha=alpha, linewidth=0.8)
                
                ax.set_title(f'Layer {layer_idx} ({max_dims} dims) - Sample {sample_idx}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'{var_name}')
                ax.grid(True, alpha=0.3)
            
            # Plot final output at the bottom (row 7) - ALL output dimensions
            ax_output = axes[7, sample_idx]
            max_output_dims = out_hist.shape[-1]  # Plot ALL output dimensions
            
            colors = plt.cm.Set1(np.linspace(0, 1, max_output_dims))
            
            for dim in range(max_output_dims):
                color = colors[dim % len(colors)]
                ax_output.plot(out_hist[sample_idx, :, dim], 
                              color=color, alpha=0.8, linewidth=2,
                              label=f'class_{dim}' if max_output_dims <= 10 else None)
            
            ax_output.set_title(f'Output ({max_output_dims} dims) - Sample {sample_idx}')
            ax_output.set_xlabel('Time Steps')
            ax_output.set_ylabel('Output Values')
            if sample_idx == num_samples - 1 and max_output_dims <= 10:
                ax_output.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_output.grid(True, alpha=0.3)
        
        # Remove empty subplot rows if fewer than 6 layers
        for i in range(len(ndh) + 2, 8):
            for j in range(num_samples):
                fig.delaxes(axes[i, j])
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'main_inf_2_recurrent_{var_name}_all_dims.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved recurrent {var_name} dynamics plot to {filename} (ALL {max_dims if 'max_dims' in locals() else 'N'} dimensions)")
    
    # Plot monitored variables for each layer - ALL dimensions
    for var_name in monitor_variables:
        # Check if this variable exists in any layer
        has_data = False
        for layer_monitor in monitor['layers']:
            if layer_monitor.get(var_name) is not None:
                has_data = True
                break
        
        # Also check global variables
        if var_name == 'encoder_out' and monitor.get('encoder_out') is not None:
            has_data = True
            
        if not has_data:
            continue
            
        # Determine number of rows needed
        n_layers = len(monitor['layers'])
        n_rows = n_layers + 2  # +1 for input, +1 for encoder/final output
        
        fig, axes = plt.subplots(n_rows, num_samples, figsize=(4*num_samples, 2.5*n_rows))
        fig.suptitle(f'Monitored Variable: {var_name} (5 Batch Samples - ALL DIMS)', fontsize=16)
        
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for sample_idx in range(num_samples):
            
            # Plot input at the top (row 0) - ALL input dimensions
            ax_input = axes[0, sample_idx]
            max_input_dims = inputs.shape[-1]  # Plot ALL input dimensions
            
            # Use colormap for many dimensions
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, max_input_dims)))
            if max_input_dims > 20:
                colors = plt.cm.viridis(np.linspace(0, 1, max_input_dims))
            
            for dim in range(max_input_dims):
                color = colors[dim % len(colors)]
                alpha = 0.8 if max_input_dims <= 20 else 0.3
                ax_input.plot(inputs[sample_idx, :, dim], 
                             color=color, alpha=alpha, linewidth=0.8)
            
            ax_input.set_title(f'Input ({max_input_dims} dims) - Sample {sample_idx}')
            ax_input.set_xlabel('Time Steps')
            ax_input.set_ylabel('Input Values')
            ax_input.grid(True, alpha=0.3)
            
            # Plot encoder output if this is encoder_out variable - ALL dimensions
            if var_name == 'encoder_out' and monitor.get('encoder_out') is not None:
                ax = axes[1, sample_idx]
                data = monitor['encoder_out']
                max_dims = data.shape[-1]  # Plot ALL dimensions
                
                colors = plt.cm.Reds(np.linspace(0.3, 1, max_dims))
                
                for dim in range(max_dims):
                    color = colors[dim % len(colors)]
                    alpha = 0.8 if max_dims <= 20 else 0.3
                    ax.plot(data[sample_idx, :, dim], 
                           color=color, alpha=alpha, linewidth=1.5)
                
                ax.set_title(f'Encoder Output ({max_dims} dims) - Sample {sample_idx}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Encoder Output')
                ax.grid(True, alpha=0.3)
            
            # Plot layer variables - ALL dimensions
            for layer_idx, layer_monitor in enumerate(monitor['layers']):
                data = layer_monitor.get(var_name)
                if var_name == 'rec_output':
                    print(data.min(), data.max(), data.mean(), data.std(), data.shape)
                if data is None:
                    continue
                
                row_idx = layer_idx + 1  # +1 for input row
                ax = axes[row_idx, sample_idx]
                max_dims = data.shape[-1]  # Plot ALL dimensions
                
                # Use colormap for many dimensions
                colors = plt.cm.tab20(np.linspace(0, 1, min(20, max_dims)))
                if max_dims > 20:
                    colors = plt.cm.plasma(np.linspace(0, 1, max_dims))
                
                for dim in range(max_dims):
                    color = colors[dim % len(colors)]
                    alpha = 0.8 if max_dims <= 20 else 0.3
                    ax.plot(data[sample_idx, :, dim], 
                           color=color, alpha=alpha, linewidth=0.8)
                
                ax.set_title(f'Layer {layer_idx} {var_name} ({max_dims} dims) - Sample {sample_idx}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'{var_name}')
                ax.grid(True, alpha=0.3)
            
            # Plot final output at the bottom - ALL output dimensions
            ax_output = axes[-1, sample_idx]
            max_output_dims = out_hist.shape[-1]  # Plot ALL output dimensions
            
            colors = plt.cm.Set1(np.linspace(0, 1, max_output_dims))
            
            for dim in range(max_output_dims):
                color = colors[dim % len(colors)]
                ax_output.plot(out_hist[sample_idx, :, dim], 
                              color=color, alpha=0.8, linewidth=2,
                              label=f'class_{dim}' if max_output_dims <= 10 else None)
            
            ax_output.set_title(f'Final Output ({max_output_dims} dims) - Sample {sample_idx}')
            ax_output.set_xlabel('Time Steps')
            ax_output.set_ylabel('Output Values')
            if sample_idx == num_samples - 1 and max_output_dims <= 10:
                ax_output.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_output.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'main_inf_2_monitored_{var_name}_all_dims.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved monitored {var_name} plot to {filename} (ALL dimensions)")


def main(args=None):
    if args is None:
        wandb.init()
        args = wandb.config
        args._from_wandb = True
    
    # Setup random seeds and configuration
    key, SEED = setup_random_seeds(args.seed)
    HIDDEN_DIM, LATENT_DIM = parse_experiment_config(args)

    dataset_fns = {
        'cifar': create_cifar_gs_classification_dataset,
        'mnist': create_mnist_classification_dataset,
        'imdb': create_lra_imdb_classification_dataset,
        'listops': create_lra_listops_classification_dataset,
        'path': create_lra_path32_classification_dataset,
        'pathx': create_lra_pathx_classification_dataset
    }
    if args.dataset in ['cifar', 'mnist']:
        trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = dataset_fns[args.dataset](bsz=args.batch_size, root="data")
        batch_x, batch_y = next(iter(testloader))
    elif args.dataset in ['imdb', 'listops']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = dataset_fns[args.dataset](batch_size=args.batch_size, seed=args.seed)
        batch = next(iter(testloader))
        batch_x, batch_y, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM)
    elif args.dataset in ['path', 'pathx']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = dataset_fns[args.dataset](bsz=args.batch_size, seed=args.seed)
        batch = next(iter(testloader))
        batch_x, batch_y, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM)
        
    print(batch_x.shape, batch_y.shape)
    print(batch_y.dtype)
    
    class_weights = compute_class_weights(trainloader, N_CLASSES) if args.dataset == 'listops' else None

    # Create monitored model
    model_cls = partial(
        BatchRNN_General_Monitored, 
        n_layers=args.n_layers, out_dim=N_CLASSES, hidden_dim=tuple(HIDDEN_DIM), do_rate=args.do_rate,
        encoder=args.encoder, encoder_scale=args.encoder_scale, encoder_bias=args.encoder_bias,
        layer_skip=args.layer_skip, element_skip=args.element_skip,
        enable_conv=args.enable_conv, conv_layer=args.conv, kernel_size=args.kernel_size, kernel_n_elems=args.kernel_n_elems,
        wavenet_dilation=args.wavenet_dilation, dilation_schedule=args.dilation_schedule, dilation_boundary=args.dilation_boundary,
        dilation_offset=args.dilation_offset, constant_dilation=args.constant_dilation,
        weight_init_scale=getattr(args, 'weight_init_scale', 1.0),
        conv_ln=getattr(args, 'conv_ln', False),
        dcls_fft=True, dcls_type=args.delay_type, dcls_kernel=args.delay_kernel, dcls_std=args.init_std,
        dcls_heterogeneous_weights=args.heterogeneous_weights, 
        dcls_heterogeneous_positions=args.heterogeneous_positions,
        dcls_heterogeneous_std=args.heterogeneous_std,
        enable_rec=args.enable_rec, rec_act=args.rec_act, rec_ln=args.rec_ln, 
        dense_z_weight_init_scale=args.dense_z_weight_init_scale, dense_z_bias_init=args.dense_z_bias_init,
        dense_h_weight_init_scale=getattr(args, 'dense_h_weight_init_scale', 1.0),
        dense_h_bias_init=getattr(args, 'dense_h_bias_init', 'zero'),
        enable_cm=args.enable_cm, channel_mixing=args.channel_mixing, cm_act=args.cm_act, glu_type=args.glu_type,
        cm_ln=getattr(args, 'cm_ln', False),
        latent_dim=tuple(LATENT_DIM), comp_act=args.comp_act,
        postnorm=args.postnorm,
        decoder_bias=args.decoder_bias
    )
                        
    steps_per_epoch = len(trainloader) 
    lr_map, lr_fn = create_learning_rate_map(args, steps_per_epoch)
    sim_args = {'key':key, 'model_cls': model_cls, 'lr_map':lr_map, 'dataset_version':'sequential', 'in_dim': IN_DIM, 'seq_len': SEQ_LENGTH, 'batch_size':args.batch_size, 'wd':args.weight_decay}
    state, n_params, _ = create_train_state(**sim_args)
    wandb.log({"n_params": n_params})
    del lr_map, sim_args

    if args.conv == 'dcls':
        print(state.params['DCLSLayer_0']['positions'])
        print(state.params['DCLSLayer_0']['weights'])
        print(state.params['DCLSLayer_0']['std'])

    key, key1, key2 = jax.random.split(key, 3)
    model_tab = model_cls(training=False)
    tabulate_fn = nn.tabulate(model_tab, {'params': key1, 'dropout': key2})
    print(tabulate_fn(batch_x))
    del model_tab, tabulate_fn, key1, key2

    print(args)
    if args.conv == 'dcls':
        print(state.params['DCLSLayer_0']['positions'])
        print(state.params['DCLSLayer_0']['weights'])
        print(state.params['DCLSLayer_0']['std'])

    model = model_cls(training=False)

    batch = next(iter(testloader))
    if len(batch) == 2:
        inputs, labels = batch
    elif len(batch) == 3:
        inputs, labels, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM)

    # Run the monitored model
    ndh, out_hist, monitor = model.apply({'params': state.params}, inputs)

    print("Network Dynamics Shape:", len(ndh), "layers")
    print("Monitor keys:", monitor.keys())
    print("Number of layer monitors:", len(monitor['layers']))
    if len(monitor['layers']) > 0:
        print("Layer 0 monitor keys:", list(monitor['layers'][0].keys()))
    
    # Create comprehensive plots
    plot_monitored_data(ndh, monitor, inputs, out_hist, labels, args_cli.sim_name)
    print("All plots saved successfully!")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GRU model")
    parser.add_argument("--dataset", type=str, default="mnist", choices=['mnist', 'cifar', 'imdb', 'listops', 'path', 'pathx'], help="Dataset version: mnist or cifar")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--conv_mode", type=str, default="dcls", choices=['dcls', 'rnn_eerf', 'rnn_lerf', 'vanilla', 'tcn_lerf', 'tcn_eerf'], help="Convolution mode: dcls, causal_eerf, or causal_lerf")
    parser.add_argument("--dcls_config", type=int, default=0, help="config to use for DCLS. 0: homP_onesW, 1: hetP_onesW, ...")
    parser.add_argument("--seed", type=int, default=None, help="Seed to use for random number generation")
    parser.add_argument("--file_nb", type=int, default=0, help="File number to load the configuration from")
    parser.add_argument("--sweep", action="store_true", help="Run in sweep mode using wandb sweep configuration")
    parser.add_argument("--sim_name", type=str, default="gen", help="Simulation name for wandb")
    args_cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    
    # check if the wandb_api_key.txt file exists
    if not os.path.exists("wandb_api_key.txt"):
        print("\n")
        raise FileNotFoundError("Please create a wandb_api_key.txt file with your WANDB API key.")
    # Read WANDB API key from external file
    with open("wandb_api_key.txt", "r") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    file_path = os.path.abspath(__file__)
    os.environ["WANDB_NOTEBOOK_NAME"] = file_path
    
    wandb.login()
    
    if args_cli.sweep:
        # Sweep mode: load sweep config and run sweep
        def load_config():
            conv_str = f'{args_cli.conv_mode}'
            if 'dcls' in args_cli.conv_mode:
                conv_str += f'_c{args_cli.dcls_config}'
            with open(f"yaml_folder/{args_cli.dataset}_{conv_str}_wandb_{args_cli.file_nb}.yaml", "r") as file:
                config = yaml.safe_load(file)
            return config
        
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
        sweep_config = load_config()
        print(sweep_config)
        
        sweep_id = wandb.sweep(sweep_config, project="Den-minGRU_sweeps") 
        wandb.agent(sweep_id, main)
    else:
        # Regular mode: load config from yaml and run single experiment
        def parse_args():
            conv_str = f'{args_cli.conv_mode}'
            if args_cli.conv_mode == 'dcls':
                conv_str += f'_c{args_cli.dcls_config}'
            with open(f"yaml_folder/{args_cli.dataset}_{conv_str}_{args_cli.file_nb}.yaml", "r") as file:
                config = yaml.safe_load(file)
            return argparse.Namespace(**config)

        args = parse_args()
        print(args)

        # add the CLI arguments to the args object
        args.dataset = args_cli.dataset
        args.gpu = args_cli.gpu
        if args_cli.seed is not None:
            args.seed = args_cli.seed
        
        # set jax XLA_PYTHON_CLIENT_MEM_FRACTION=.XX
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"0.4"
        
        wandb.init(project="DenGRU_general", name=f"{args_cli.sim_name}_monitored")
        wandb.config.update(args)
        
        main(args)