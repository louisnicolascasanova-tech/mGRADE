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
from utils import create_mnist_classification_dataset, create_cifar_gs_classification_dataset, write_config_yaml
from plots import plot_dynamics

from model import BatchRNN_General
from training import create_train_state, run_epoch, validate, create_learning_rate_map
from functools import partial

import os
import argparse
import yaml
import wandb

def main():
    wandb.init()
    args = wandb.config

    SEED = args.seed if args.seed is not None else 42
    # fix random seed
    key = jax.random.PRNGKey(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    args.n_layers = args.n_layers
    HIDDEN_DIM = [args.hidden_dim]*args.n_layers
    hidden_dim_str = HIDDEN_DIM[0]
    LATENT_DIM = [args.latent_dim]*args.n_layers
    latent_dim_str = LATENT_DIM[0] if LATENT_DIM[0] is not None else 'F'
    print(f"Hidden dim: {HIDDEN_DIM}, Latent dim: {LATENT_DIM}")
    assert len(HIDDEN_DIM) == len(LATENT_DIM) == args.n_layers, "Hidden and latent dimensions must match the number of layers"

    args.warmup_epochs = args.warmup_frac * args.n_epochs

    dataset_fns = {
        'cifar': create_cifar_gs_classification_dataset,
        'mnist': create_mnist_classification_dataset
    }
    trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = dataset_fns[args.dataset](bsz=args.batch_size, root="data")

    batch_x, batch_y = next(iter(testloader))
    print(batch_x.shape, batch_y.shape)
    print(batch_y.dtype)

    # if args.wavenet_dilation:
    #     if args.dataset == 'cifar':
    #         if args.kernel_size == 4:
    #             dilation_boundary = 7+1
    #         elif args.kernel_size == 8:
    #             dilation_boundary = 6+1
    #         elif args.kernel_size == 16:
    #             dilation_boundary = 5+1
    #         elif args.kernel_size == 32: 
    #             dilation_boundary = 4+1
    #         elif args.kernel_size == 64:
    #             dilation_boundary = 3+1
    #     elif args.dataset == 'mnist':
    #         if args.kernel_size <= 32:
    #             dilation_boundary = 4
    #         elif args.kernel_size == 64:
    #             dilation_boundary = 3
    # else:
    #     dilation_boundary = None

    
    

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
                        
    # model_mingru = BatchRNN(HIDDEN_DIM, 10, args.n_layers, recurrent_layer=minGRULayer)
    steps_per_epoch = len(trainloader) 
    lr_map, lr_fn = create_learning_rate_map(args, steps_per_epoch)
    sim_args = {'key':key, 'model_cls': model_cls, 'lr_map':lr_map, 'dataset_version':'sequential', 'seq_len': SEQ_LENGTH, 'batch_size':args.batch_size, 'wd':args.weight_decay}
    state, n_params, _ = create_train_state(**sim_args)
    wandb.log({"n_params": n_params})
    # print(state.opt_state)
    del lr_map, sim_args

    if args.conv == 'dcls':
        print(state.params['DCLSLayer_0']['positions'])
        print(state.params['DCLSLayer_0']['weights'])
        print(state.params['DCLSLayer_0']['std'])


    key, key1, key2 = jax.random.split(key, 3)
    model_tab = model_cls(training=False)
    tabulate_fn = nn.tabulate(model_tab, {'params': key1, 'dropout': key2})
    print(tabulate_fn(batch_x))
    del model_tab, tabulate_fn, batch_x, batch_y, key1, key2

    print(args)
    if args.conv == 'dcls':
        print(state.params['DCLSLayer_0']['positions'])
        print(state.params['DCLSLayer_0']['weights'])
        print(state.params['DCLSLayer_0']['std'])

    
    layer_skip_str = 'T' if args.layer_skip else 'F'
    element_skip_str = 'T' if args.element_skip else 'F'
    skip_str = f"skipL{layer_skip_str}E{element_skip_str}"
    postnorm_str = 'T' if args.postnorm else 'F'
    if args.conv == 'dcls':
        delay_type_str = 'syn' if args.delay_type == 'synaptic' else 'ax'
        delay_ker_str = 'gaus' if args.delay_kernel == 'gaussian' else 'exp'
        hete_pos = 'T' if args.heterogeneous_positions else 'F'
        hete_pos_str = f'hetP{hete_pos}'
        conv_str = f'DCLS{args.kernel_size}{delay_type_str}{delay_ker_str}{args.init_std}{hete_pos_str}{args.kernel_n_elems}'
    else:
        wavenet_str = 'eerf' if args.wavenet_dilation else 'lerf'
        schedule_str = args.dilation_schedule if args.dilation_schedule is not None else 'F'
        conv_str = f'conv{args.kernel_size}{wavenet_str}sch{schedule_str}'
    if args.channel_mixing == 'glu':
        cm_str = f"cm{args.channel_mixing}{args.glu_type}"
    elif args.channel_mixing == 'mlp':
        cm_str = f"cm{args.channel_mixing}{args.cm_act}"
    elif args.channel_mixing is None:
        cm_str = f"cmF"
    het_w_str = 'T' if args.heterogeneous_weights else 'F'
    het_std_str = 'T' if args.heterogeneous_std else 'F'
    train_w_str = 'T' if args.train_weights else 'F'
    train_std_str = 'T' if args.train_std else 'F'
    train_pos_str = 'T' if args.train_positions else 'F'
    hetero_str = f"hetW{het_w_str}S{het_std_str}trainW{train_w_str}S{train_std_str}P{train_pos_str}"
    id_sim = f"general/{args.dataset}/H{hidden_dim_str}L{args.n_layers}B{args.batch_size}_do{args.do_rate}_" + \
                f"lr{args.lr}wd{args.weight_decay}we{args.warmup_epochs}_" + \
                f"{skip_str}_{conv_str}_" + \
                f"rec{args.rec_act}_{cm_str}_" + \
                f"HpreReg{args.reg_factor}_C{latent_dim_str}{args.comp_act}_p{postnorm_str}_" + \
                f"{hetero_str}_s{SEED}"
    CKPT_DIR = os.path.join(os.getcwd(), f"checkpoints/{id_sim}")
    WU_DIR = os.path.join(os.getcwd(), f"checkpoints/{id_sim}/warmup")
    if os.path.exists(CKPT_DIR):
        id_sim += f"_training_1"
    i = 2
    while os.path.exists(CKPT_DIR):
        # pop the last number from the id_sim
        id_sim = id_sim[:-1]
        id_sim += f"{i}"
        CKPT_DIR = os.path.join(os.getcwd(), f"checkpoints/{id_sim}")
        i += 1
    print(CKPT_DIR)
    print(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)
    RESULT_DIR = os.path.join(os.getcwd(), f"results/{id_sim}")
    print(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)
    PLT_DIR = os.path.join(os.getcwd(), f"plots/{id_sim}")
    print(PLT_DIR)
    os.makedirs(PLT_DIR, exist_ok=True)

    # Write configuration to YAML file
    write_config_yaml(args, CKPT_DIR)



    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    # 'batch_loss' = [loss_sample1, loss_sample2, ..., loss_sampleBS]
    # 'loss' = [loss_batch1, loss_batch2, ..., loss_batchN] where N is the number of batches and loss_batchX = mean([loss_sample1, loss_sample2, ..., loss_sampleBS])
    keys_to_track = [] #['grads', 'loss', 'state', 'batch_x', 'batch_y'] # ['batch_x', 'batch_y', 'state', 'grads']
    inner_keys_to_track = [] #['batch_loss', 'net_dyn']
    aux_dict_training = []

    best_val_acc = 0.0
    improvement = 0.01 # 1%, minimum improvement to save checkpoint

    async_manager = checkpoints.AsyncManager()
    test_loss = 2.5
    test_acc = 0.0
    for epoch in range(args.n_epochs):
        key, subkey = jax.random.split(key) # not used in run_epoch (TODO: remove?)
        state, train_loss, train_acc, (break_flag, aux_dict_epoch) = \
            run_epoch(state, model_cls, trainloader, subkey, reg_factor=args.reg_factor, kernel_size=args.kernel_size,
                        lim_batch=None, keys_to_track=keys_to_track, inner_keys_to_track=inner_keys_to_track,
                        lr_fn=lr_fn, wandb_gradients=args.wandb_gradients)
        aux_dict_training.append(aux_dict_epoch)
        if break_flag:
            break
        val_loss, val_acc  = validate(state, model_cls, val_loader)
        if val_acc > best_val_acc + improvement:
            best_val_acc = val_acc
            best_val_acc_loss = val_loss
            if args.dataset == 'mnist':
                if best_val_acc > 0.94: improvement = 0.001 # 0.1%
            elif args.dataset == 'cifar':
                if 0.8 > best_val_acc > 0.70: improvement = 0.005 # 0.5%
                elif best_val_acc >= 0.80: improvement = 0.001 # 0.1%
            test_loss, test_acc = validate(state, model_cls, testloader)
            checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=state.step, overwrite=True, async_manager=async_manager)
            print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | test_loss: {test_loss:.4f} | test_acc: {test_acc*100:.2f}%")
        else: 
            print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%")
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_val_acc, "best_val_acc_loss": best_val_acc_loss})
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        if (epoch == 0 or epoch == 95) and args.conv == 'dcls':
            print(state.params['DCLSLayer_0']['positions'])
            print(state.params['DCLSLayer_0']['weights'])
            print(state.params['DCLSLayer_0']['std'])
        if epoch == args.n_epochs*args.warmup_frac: 
            print("Saving the warmed up model")
            checkpoints.save_checkpoint(ckpt_dir=WU_DIR, target=state, step=state.step, overwrite=True, async_manager=async_manager)


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GRU model")
    parser.add_argument("--dataset", type=str, default="mnist", choices=['mnist', 'cifar'], help="Dataset version: mnist or cifar")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--conv_mode", type=str, default="dcls", choices=['rnn_dcls', 'rnn_eerf', 'rnn_lerf', 'vanilla', 'tcn_lerf', 'tcn_eerf'], help="Convolution mode: dcls, causal_eerf, or causal_lerf")
    parser.add_argument("--dcls_config", type=int, default=0, help="config to use for DCLS. 0: homP_onesW, 1: hetP_onesW, ...")
    parser.add_argument("--file_nb", type=int, default=0, help="File number to load the configuration from")
    args_cli = parser.parse_args()

    def load_config():
        conv_str = f'{args_cli.conv_mode}'
        print(args_cli.conv_mode)
        if 'dcls' in args_cli.conv_mode:
            conv_str += f'_c{args_cli.dcls_config}'
        with open(f"yaml_folder/{args_cli.dataset}_{conv_str}_wandb_{args_cli.file_nb}.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    # set jax XLA_PYTHON_CLIENT_MEM_FRACTION=.XX
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"0.9"

    # check if the wandb_api_key.txt file exists
    if not os.path.exists("wandb_api_key.txt"):
        print("\n")
        raise FileNotFoundError("Please create a wandb_api_key.txt file with your WANDB API key.")
    # Read WANDB API key from external file
    with open("wandb_api_key.txt", "r") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    file_path = os.path.abspath(__file__)
    os.environ["WANDB_NOTEBOOK_NAME"] = file_path

    sweep_config = load_config()
    print(sweep_config)

    wandb.login() 
    sweep_id = wandb.sweep(sweep_config, project="Den-minGRU_sweeps") 
    wandb.agent(sweep_id, main)