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

from model import BatchRNN_General
from training import create_train_state, run_epoch, validate, create_learning_rate_map
from functools import partial

import os
import argparse
import yaml
import wandb


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
        batch_x, batch_y = next(iter(testloader)) # used for the tabulate function
    elif args.dataset in ['imdb', 'listops']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = dataset_fns[args.dataset](batch_size=args.batch_size, seed=args.seed)
        batch = next(iter(testloader))
        batch_x, batch_y, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM) # used for the tabulate function
    elif args.dataset in ['path', 'pathx']:
        trainloader, val_loader, testloader, _, N_CLASSES, SEQ_LENGTH, IN_DIM, _ = dataset_fns[args.dataset](bsz=args.batch_size, seed=args.seed)
        batch = next(iter(testloader))
        batch_x, batch_y, _ = prep_batch(batch, SEQ_LENGTH, IN_DIM) # used for the tabulate function
        

    print(batch_x.shape, batch_y.shape)
    print(batch_y.dtype)

    
    class_weights = compute_class_weights(trainloader, N_CLASSES) if args.dataset == 'listops' else None

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
        encoder=getattr(args, 'encoder', True), 
        encoder_scale=getattr(args, 'encoder_scale', 1.0),
        encoder_bias=getattr(args, 'encoder_bias', True),
        layer_skip=args.layer_skip, element_skip=args.element_skip,
        # CONVOLUTION
        enable_conv=args.enable_conv, conv_layer=args.conv, kernel_size=args.kernel_size, kernel_n_elems=args.kernel_n_elems,
        wavenet_dilation=args.wavenet_dilation, dilation_schedule=args.dilation_schedule, dilation_boundary=args.dilation_boundary,
        dilation_offset=args.dilation_offset, constant_dilation=args.constant_dilation,
        dcls_fft=True, dcls_type=args.delay_type, dcls_kernel=args.delay_kernel, dcls_std=args.init_std,
        dcls_heterogeneous_weights=args.heterogeneous_weights, 
        dcls_heterogeneous_positions=args.heterogeneous_positions,
        dcls_heterogeneous_std=args.heterogeneous_std,
        weight_init_scale=getattr(args, 'weight_init_scale', 1.0),
        conv_ln=getattr(args, 'conv_ln', False),  # whether to apply LayerNorm before the convolution layer
        # RECURRENT
        enable_rec=args.enable_rec, rec_act=args.rec_act, 
        rec_ln=getattr(args, 'rec_ln', False),
        dense_z_weight_init_scale=getattr(args, 'dense_z_weight_init_scale', 1.0), 
        dense_z_bias_init=getattr(args, 'dense_z_bias_init', 'zero'),
        dense_h_weight_init_scale=getattr(args, 'dense_h_weight_init_scale', 1.0),
        dense_h_bias_init=getattr(args, 'dense_h_bias_init', 'zero'),
        # CHANNEL MIXING
        enable_cm=args.enable_cm, channel_mixing=args.channel_mixing, cm_act=args.cm_act, glu_type=args.glu_type,
        cm_ln=getattr(args, 'cm_ln', False),
        # COMPRESSION
        latent_dim=tuple(LATENT_DIM), comp_act=args.comp_act,
        postnorm=args.postnorm,
        decoder_bias=getattr(args, 'decoder_bias', True),
    )
                        
    # model_mingru = BatchRNN(HIDDEN_DIM, 10, args.n_layers, recurrent_layer=minGRULayer)
    steps_per_epoch = len(trainloader) 
    lr_map, lr_fn = create_learning_rate_map(args, steps_per_epoch)
    sim_args = {'key':key, 'model_cls': model_cls, 'lr_map':lr_map, 'dataset_version':'sequential', 'in_dim': IN_DIM, 'seq_len': SEQ_LENGTH, 'batch_size':args.batch_size, 'wd':args.weight_decay}
    state, n_params, _ = create_train_state(**sim_args)
    
    # Load checkpoint if resume_from is provided
    start_epoch = 0
    if hasattr(args, 'resume_from') and args.resume_from is not None:
        print(f"Loading checkpoint from {args.resume_from}")
        restored_state = checkpoints.restore_checkpoint(ckpt_dir=args.resume_from, target=state)
        if restored_state is not None:
            state = restored_state
            start_epoch = int(state.step // len(trainloader))
            print(f"Resumed training from epoch {start_epoch}, step {state.step}")
        else:
            print("Warning: Could not load checkpoint, starting from scratch")
    
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

    # Generate experiment ID and create directories
    base_id = generate_experiment_id(args, HIDDEN_DIM, LATENT_DIM, SEED)
    id_sim, CKPT_DIR, WU_DIR, RESULT_DIR, PLT_DIR = create_experiment_directories(base_id)

    bu_dir = os.path.join(CKPT_DIR, "Backup")    
    # Create directories
    os.makedirs(bu_dir, exist_ok=True)
    print(bu_dir)


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
    inner_keys_to_track = ['net_dyn'] #['batch_loss', 'net_dyn']
    aux_dict_training = []

    best_val_acc = 0.0
    improvement = 0.01 # 1%, minimum improvement to save checkpoint

    async_manager = checkpoints.AsyncManager()
    test_loss = 2.5
    test_acc = 0.0
    test_metrics = {}
    ovf_count = 0
    bad_count = 0
    patience = 0
    lim_patience = 10
    for epoch in range(start_epoch, args.n_epochs):
        key, subkey = jax.random.split(key) # not used in run_epoch (TODO: remove?)
        
        state, train_loss, train_acc, (break_flag, aux_dict_epoch) = \
            run_epoch(state, model_cls, trainloader, subkey, reg_factor=args.reg_factor, kernel_size=args.kernel_size,
                        lim_batch=None, keys_to_track=keys_to_track, inner_keys_to_track=inner_keys_to_track,
                        lr_fn=lr_fn, 
                        wandb_gradients=getattr(args, 'wandb_gradients', False),
                        wandb_states=getattr(args, 'wandb_states', False), 
                        wandb_matrices=getattr(args, 'wandb_matrices', False),
                        in_dim=IN_DIM, seq_len=SEQ_LENGTH,
                        grad_clip_norm=args.grad_clip_norm,
                        log_model_behavior=args.log_model_behavior, epoch_num=epoch,
                        class_weights=class_weights)
        aux_dict_training.append(aux_dict_epoch)
        
        if break_flag:
            break
        
        if args.dataset != 'imdb':
            val_loss, val_acc, val_metrics = validate(state, model_cls, val_loader, SEQ_LENGTH, IN_DIM, N_CLASSES, 
                                                        log_classification_report=getattr(args, 'log_model_behavior', True), dataset_name="val") 
        else: 
            val_loss, val_acc, val_metrics = validate(state, model_cls, testloader, SEQ_LENGTH, IN_DIM, N_CLASSES, 
                                                        log_classification_report=getattr(args, 'log_model_behavior', True), dataset_name="val") # TODO: create a val loader for imdb
        
        if val_acc > best_val_acc + improvement: 
            patience = 0
            best_val_acc = val_acc
            best_val_acc_loss = val_loss
            
            if args.dataset == 'mnist':
                if best_val_acc > 0.94: improvement = 0.001 # 0.1%
            elif args.dataset == 'cifar':
                if 0.8 > best_val_acc > 0.70: improvement = 0.005 # 0.5%
                elif best_val_acc >= 0.80: improvement = 0.001 # 0.1%
            elif args.dataset == 'listops':
                if best_val_acc > 0.55: improvement = 0.001 # 0.3%
                elif best_val_acc > 0.50: improvement = 0.003 # 0.2%
            elif args.dataset == 'path':
                if best_val_acc > 0.88: improvement = 0.002 # 0.2%
            elif args.dataset == 'pathx':
                if best_val_acc > 0.93: improvement = 0.002 # 0.2%
                elif best_val_acc > 0.88: improvement = 0.005 # 0.5%
            elif args.dataset == 'imdb':
                if best_val_acc > 0.80: improvement = 0.003 # 0.3%
                elif best_val_acc > 0.83: improvement = 0.001 # 0.1%


            if args.dataset != 'imdb':
                test_loss, test_acc, test_metrics = validate(state, model_cls, testloader, SEQ_LENGTH, IN_DIM, N_CLASSES, 
                                                            log_classification_report=getattr(args, 'log_model_behavior', True), dataset_name="test") if args.dataset != 'imdb' else validate(state, model_cls, testloader, SEQ_LENGTH, IN_DIM, N_CLASSES, log_classification_report=getattr(args, 'log_model_behavior', True), dataset_name="test")
                print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | test_loss: {test_loss:.4f} | test_acc: {test_acc*100:.2f}%")
            else:
                print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%")
            
            print(f"Saving the model at epoch {epoch}, in directory {CKPT_DIR}")
            checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=state.step, overwrite=True, async_manager=async_manager)

        else: 
            print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%")
            patience += 1
            if args.dataset == 'listops' and epoch < 20:
                patience = 0
            if patience >= lim_patience and best_val_acc < 0.5:
                print(f"Early stopping at epoch {epoch} due to low validation accuracy ({best_val_acc:.2f}) and patience limit reached ({patience}/{lim_patience})")
                break
            if epoch > 10 and patience >= 3 and val_acc < 0.55 and best_val_acc > 0.6:
                print(f"Early stopping at epoch {epoch} due to low validation accuracy ({val_acc:.2f}, best: {best_val_acc:.2f}) and patience limit reached ({patience}/{lim_patience})")
                break
            #checkpoints.save_checkpoint(ckpt_dir=bu_dir, target=state, step=state.step, overwrite=False, async_manager=async_manager)

        
        # Prepare main logging dictionary
        main_metrics = {
            "train/loss": train_loss, "train/acc": train_acc, 
            "val/loss": val_loss, "val/acc": val_acc, 
            "val/loss_best": best_val_acc_loss, "val/acc_best": best_val_acc,
        }
        if args.dataset != 'imdb':
            main_metrics.update({
                "test/loss_best": test_loss, "test/acc_best": test_acc
            })
        
        # Add validation metrics if available
        if val_metrics:
            main_metrics.update(val_metrics)
            
        # Add test metrics if available (only when model improves)
        if test_metrics:
            main_metrics.update(test_metrics)
            
        wandb.log(main_metrics)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        if epoch == 0 and args.conv == 'dcls':
            print(state.params['DCLSLayer_0']['positions'])
            print(state.params['DCLSLayer_0']['weights'])
            print(state.params['DCLSLayer_0']['std'])
        if epoch == args.n_epochs*args.warmup_frac: 
            print("Saving the warmed up model")
            checkpoints.save_checkpoint(ckpt_dir=WU_DIR, target=state, step=state.step, overwrite=True, async_manager=async_manager)
        
        if args.dataset == 'imdb' and val_loss > 2*best_val_acc_loss:
            if ovf_count > 5:
                print('CANCELLING: OVERFITTING')
                break
            ovf_count += 1
            print(f'OVF COUNT: {ovf_count}')
        else:
            ovf_count = 0

        if args.dataset == 'imdb' and best_val_acc < 0.7 and epoch > 15:
            if bad_count > 5:
                print('CANCELLING: LOW ACCURACY ON IMDB')
                break
            bad_count += 1
            print(f'BAD COUNT: {bad_count}')
        else:
            bad_count = 0
        

        # if epoch == 4: 
        #     print("CANCELLING: REACHED 5 EPOCHS")
        #     print(f'Saving the model at epoch {epoch}, in directory {WU_DIR}')
        #     checkpoints.save_checkpoint(ckpt_dir=WU_DIR, target=state, step=state.step, overwrite=True, async_manager=async_manager)
        #     break


    # Save training dynamics (only in non-sweep mode)
    if not hasattr(args, '_from_wandb'):
        np.savez(os.path.join(RESULT_DIR, 'training_dynamics.npz'), 
                train_losses=train_losses, 
                train_accuracies=train_accuracies, 
                val_losses=val_losses, 
                val_accuracies=val_accuracies,
                test_losses=test_losses,
                test_accuracies=test_accuracies)
    
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
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume training from")
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
        
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"0.95"
        sweep_config = load_config()
        print(sweep_config)
        
        sweep_id = wandb.sweep(sweep_config, project="Den-minGRU_sweeps") 
        wandb.agent(sweep_id, main)
    else:
        # Regular mode: load config from yaml or checkpoint and run single experiment
        def parse_args():
            if args_cli.resume_from is not None:
                # Load config from checkpoint directory
                config_path = os.path.join(args_cli.resume_from, 'config.yaml')
                if os.path.exists(config_path):
                    print(f"Loading config from checkpoint: {config_path}")
                    with open(config_path, "r") as file:
                        config = yaml.safe_load(file)
                    return argparse.Namespace(**config)
                else:
                    raise FileNotFoundError(f"No config.yaml found in checkpoint directory: {args_cli.resume_from}")
            else:
                # Load config from yaml_folder
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
        args.resume_from = args_cli.resume_from
        
        # set jax XLA_PYTHON_CLIENT_MEM_FRACTION=.XX
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.mem_frac}"
        
        wandb.init(project="DenGRU_general", name=f"{args_cli.sim_name}")
        wandb.config.update(args)
        
        main(args)


    # The conv layers should take an input of length: 
    # 1024 + 2*(kernel_size-1)
    # if the the conv layer has dilation, the input length should be:
    # 1024 + 2*(kernel_size-1)*dilation

    # For a kernel size of 32 with dilation: 1, 2, 4, 8, 16, 32, 64 (7-layer network)
    # The input length should be: 1024 + 2*(32-1)*dilation
    # i.e 1086 -> 1148 -> 1272 -> 1520 -> 2016 -> 3008 -> 4992

    # For a kernel size of 64 with dilation: 1, 2, 4, 8, 16, 32
    # The input length should be: 1024 + 2*(64-1)*dilation
    # i.e 1150 -> 1276 -> 1528 -> 2032 -> 3040 -> 5056 -> 9098

    # If we use the clip dilation schedule, the input length should be (6-layer network):
    # layer_id                     0       1       2       3       4           5
    # k=32, clip at layer_id=5: 1086 -> 1148 -> 1272 -> 1520 -> 2016     -> 2016 (c)  # c: clipped
    # k=64, clip at layer_id=4: 1150 -> 1276 -> 1528 -> 2032 -> 2032 (c) -> 2032
    
    # If we use the wrap dilation schedule, the input length should be (6-layer network):
    # layer_id                     0       1       2       3       4           5
    # k=32, clip at layer_id=5: 1086 -> 1148 -> 1272 -> 1520 -> 2016     -> 1086 (w)  # w: wrapped
    # k=64, clip at layer_id=4: 1150 -> 1276 -> 1528 -> 2032 -> 1150 (w) -> 1276

    # If we use the clip dilation with offset, the input length should be (6-layer network):
    # layer_id                                        0       1       2       3           4       5
    # k=32, offset=2, clip at layer_id=5-offset=3: 1272 -> 1520 -> 2016 -> 2016 (c) -> 2016 -> 2016


    # If we use the constant dilation schedule, the input length should be (6-layer network):
    # layer_id                         0       1       2       3       4       5
    # k=32, constant dilation of 1: 1086 -> 1086 -> 1086 -> 1086 -> 1086 -> 1086 
    # k=32, constant dilation of 2: 1148 -> 1148 -> 1148 -> 1148 -> 1148 -> 1148
    # k=32, constant dilation of 4: 1272 -> 1272 -> 1272 -> 1272 -> 1272 -> 1272
    # ... maximum dilation is 16
    # k=64, constant dilation of 1: 1150 -> 1150 -> 1150 -> 1150 -> 1150 -> 1150
    # k=64, constant dilation of 2: 1276 -> 1276 -> 1276 -> 1276 -> 1276 -> 1276
    # k=64, constant dilation of 4: 1528 -> 1528 -> 1528 -> 1528 -> 1528 -> 1528
    # ... maximum dilation is 8



    # local receptive field size:
    # R = 1 + (K - 1) * d
    #  i | 0   | 1   | 2   | 3   | 4    | 5    | 6    | 7     | 8     | 9     | 10    |
    #  K | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | d=64 | d=128 | d=256 | d=512 | d=1024|
    #  4 | 5   | 9   | 13  | 25  | 49   | 97   | 193  | 385   | 769   | 1537  | 3073  |
    #  8 | 9   | 17  | 29  | 57  | 113  | 225  | 449  | 897   | 1793  | 3585  | 7169  |
    #  16| 17  | 31  | 61  | 121 | 241  | 481  | 961  | 1921  | 3841  | 7681  | 15361 |
    #  32| 33  | 63  | 125 | 249 | 497  | 993  | 1985 | 3969  | 7937  | 15873 | 31745 |
    #  64| 65  | 127 | 253 | 505 | 1009 | 2017 | 4033 | 8065  | 16129 | 32257 | 64513 |
    # 128| 129 | 255 | 509 |1017 | 2033 | 4065 | 8129 | 16257 | 32513 | 65025 |130049 |


# python main.py --resume_from checkpoints/general/listops_H128L6B64_do0_lr0.004wd0.1we15.0_skipLFET_DCLS64axgaus0.7hetPF8_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0 --dataset listops --gpu 0