import jax
import numpy as np
import torch
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import os
import argparse
import yaml

from utils import create_mnist_classification_dataset, create_cifar_gs_classification_dataset, \
        create_lra_imdb_classification_dataset, create_lra_listops_classification_dataset, \
        create_lra_path32_classification_dataset, create_lra_pathx_classification_dataset
from model import BatchRNN_General
from training import create_train_state, validate, create_learning_rate_map, prep_batch
from functools import partial


def main(args):
    SEED = args.seed if args.seed is not None else 0
    key = jax.random.PRNGKey(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    HIDDEN_DIM = [args.hidden_dim]*args.n_layers
    LATENT_DIM = [args.latent_dim]*args.n_layers if args.latent_dim is not None else [None]*args.n_layers
    print(f"Using hidden dimensions: {HIDDEN_DIM}, latent dimensions: {LATENT_DIM}")

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
        batch_x, batch_y = next(iter(testloader)) # used for the tabulate function
    elif args.dataset in ['imdb', 'listops']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = dataset_fns[args.dataset](batch_size=args.batch_size, seed=SEED)
        batch = next(iter(testloader))
        batch_x, batch_y, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM) # used for the tabulate function
    elif args.dataset in ['path', 'pathx']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = dataset_fns[args.dataset](bsz=args.batch_size, seed=SEED)
        batch = next(iter(testloader))
        batch_x, batch_y, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM) # used for the tabulate function

    # Model partial
    model_cls = partial(
        BatchRNN_General, 
        n_layers=args.n_layers, out_dim=N_CLASSES, hidden_dim=tuple(HIDDEN_DIM), do_rate=args.do_rate,
        encoder=args.encoder,
        layer_skip=args.layer_skip, element_skip=args.element_skip,
        enable_conv=args.enable_conv, conv_layer=args.conv, kernel_size=args.kernel_size, kernel_n_elems=args.kernel_n_elems,
        wavenet_dilation=args.wavenet_dilation, dilation_schedule=args.dilation_schedule, dilation_boundary=args.dilation_boundary,
        dilation_offset=args.dilation_offset, constant_dilation=args.constant_dilation,
        dcls_fft=False, dcls_type=args.delay_type, dcls_kernel=args.delay_kernel, dcls_std=args.init_std,
        dcls_heterogeneous_weights=args.heterogeneous_weights, 
        dcls_heterogeneous_positions=args.heterogeneous_positions,
        dcls_heterogeneous_std=args.heterogeneous_std,
        enable_rec=args.enable_rec, rec_act=args.rec_act,
        enable_cm=args.enable_cm, channel_mixing=args.channel_mixing, cm_act=args.cm_act, glu_type=args.glu_type,
        latent_dim=tuple(LATENT_DIM), comp_act=args.comp_act,
        postnorm=args.postnorm)

    steps_per_epoch = len(trainloader)
    lr_map, lr_fn = create_learning_rate_map(args, steps_per_epoch)
    sim_args = {'key':key, 'model_cls': model_cls, 'lr_map':lr_map, 'dataset_version':'sequential', 'in_dim': IN_DIM, 'seq_len': SEQ_LENGTH, 'batch_size':args.batch_size, 'wd':args.weight_decay}
    state, n_params, _ = create_train_state(**sim_args)

    # Restore checkpoint
    CKPT_DIR = args.ckpt_dir
    state_trained = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
    print(f"Loaded checkpoint from {CKPT_DIR}")


    key, key1, key2 = jax.random.split(key, 3)
    model_tab = model_cls(training=False)
    tabulate_fn = nn.tabulate(model_tab, {'params': key1, 'dropout': key2})
    print(tabulate_fn(batch_x))
    del model_tab, tabulate_fn, batch_x, batch_y, key1, key2

    if args.conv == 'dcls':
        print(state.params['DCLSLayer_0']['positions'])
        print(state.params['DCLSLayer_0']['weights'])
        print(state.params['DCLSLayer_0']['std'])
        print(state_trained.params['DCLSLayer_0']['positions'])
        print(state_trained.params['DCLSLayer_0']['weights'])
        print(state_trained.params['DCLSLayer_0']['std'])



    # Evaluate on test set
    # test_loss, test_acc, test_metrics = validate(state_trained, model_cls, testloader, SEQ_LENGTH, IN_DIM, N_CLASSES, 
    #                                             log_classification_report=getattr(args, 'log_model_behavior', True), dataset_name="test") if args.dataset != 'imdb' \
    #                                                 else validate(state_trained, model_cls, testloader, SEQ_LENGTH, IN_DIM, N_CLASSES, log_classification_report=getattr(args, 'log_model_behavior', True), dataset_name="test")
    # print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc*100:.2f}%")

    from model import construct_kernel_fast
    import matplotlib.pyplot as plt
    def get_kernels(params, n_dim, kernel_size):
        ks = []
        all_k = []
        for i in range(args.n_layers):
            w = params[f'DCLSLayer_{i}']['weights']
            p = params[f'DCLSLayer_{i}']['positions']
            s = params[f'DCLSLayer_{i}']['std']
            ks_individual = []
            for j in range(w.shape[0]):
                if j == 0:
                    k = construct_kernel_fast(w[j], p[j], s[j], kernel_size, n_dim)
                    ks_individual.append(k)
                else:
                    k_j = construct_kernel_fast(w[j], p[j], s[j], kernel_size, n_dim)
                    ks_individual.append(k_j)
                    k += k_j

            ks.append(k)
            all_k.append(ks_individual)
        return ks, all_k
    n_dim = 3 if args.delay_type == 'synaptic' else 2
    ks, all_k = get_kernels(state.params, n_dim, args.kernel_size)
    ks_trained, all_k_trained = get_kernels(state_trained.params, n_dim, args.kernel_size)

    # # reverse the axis=1 of ks and ks_trained
    ks = [np.flip(k, axis=1) for k in ks]
    ks_trained = [np.flip(k, axis=1) for k in ks_trained]

    def plot_kernels_axonal(ks, init):
        PX = 1 / plt.rcParams['figure.dpi']
        factor_hight = np.log2(args.hidden_dim) - 5 if args.hidden_dim > 32 else 1
        factor_width = np.log2(args.kernel_size) - 5 if args.kernel_size > 32 else 1
        fig, axes = plt.subplots(2, args.n_layers, figsize=(300 * PX * args.n_layers, 300*factor_hight*PX), constrained_layout=True)
        print(f"Plotting axonal kernels for {args.n_layers} layers")
        # print figure size
        print(f"Figure size: {fig.get_size_inches()} inches")
        
        # Calculate max value for colorbar scaling
        max_val = max(np.max(np.abs(ks[i])) for i in range(args.n_layers))
        
        for i in range(args.n_layers):
            # Top row - full scale
            ax_top = axes[0, i] if args.n_layers > 1 else axes[0]
            im_top = ax_top.imshow(ks[i], cmap='RdBu_r', vmin=-max_val, vmax=max_val, aspect='auto')
            ax_top.set_title(f"Visualization of ks[{i}] (full scale)")
            ax_top.set_yticks(np.arange(0, ks[i].shape[0], 8))
            
            # Add y-label only to leftmost plots
            if i == 0:
                ax_top.set_ylabel('Hidden Dimension')
            
            # Bottom row - half scale
            ax_bottom = axes[1, i] if args.n_layers > 1 else axes[1]
            im_bottom = ax_bottom.imshow(ks[i], cmap='RdBu_r', vmin=-max_val/2, vmax=max_val/2, aspect='auto')
            ax_bottom.set_title(f"ks[{i}] (half scale)")
            ax_bottom.set_yticks(np.arange(0, ks[i].shape[0], 8))
            
            # Add y-label only to leftmost plots
            if i == 0:
                ax_bottom.set_ylabel('Hidden Dimension')
            
            # Add x-label to bottom row plots
            ax_bottom.set_xlabel('Kernel Position')
        
        # Add colorbars
        fig.colorbar(im_top, ax=axes[0], orientation='vertical', fraction=0.02, pad=0.04, label='Full scale')
        fig.colorbar(im_bottom, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.04, label='Half scale')
        
        # save figure
        fig_name = f'kernels_axonal_{args.dataset}_H{args.hidden_dim}_K{args.kernel_size}_E{args.kernel_n_elems}_init_{init}.png' if init else f'kernels_axonal_{args.dataset}_H{args.hidden_dim}_K{args.kernel_size}_E{args.kernel_n_elems}.png'
        print(f"Saving figure to {fig_name}")
        fig.savefig(fig_name, dpi=300)#, bbox_inches='tight')

    if args.delay_type == 'axonal':
        print("Plotting axonal kernels")
        plot_kernels_axonal(ks, init=True)
        plot_kernels_axonal(ks_trained, init=False)
    else:
        raise ValueError(f"Unsupported delay type: {args.delay_type}. Only 'axonal' is supported for plotting.")
        #plot_kernels_syn(ks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for GRU model")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory (the folder containing the checkpoint file)")
    # dataset
    parser.add_argument("--dataset", type=str, choices=['cifar', 'mnist', 'imdb', 'listops', 'path', 'pathx'], default='cifar', help="Dataset to use for inference")
    # gpu 
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    args_cli = parser.parse_args()

    # Load config
    config_path = os.path.join(args_cli.ckpt_dir, "config.yaml") #
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    args = argparse.Namespace(**config)
    args.ckpt_dir = args_cli.ckpt_dir
    args.dataset = args_cli.dataset
    args.warmup_epochs = args.n_epochs * args.warmup_frac

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    # set jax XLA_PYTHON_CLIENT_MEM_FRACTION=.XX
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"0.25"


    main(args) 