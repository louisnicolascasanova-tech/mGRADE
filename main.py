import argparse
import os
import sys

# Parse GPU argument BEFORE importing JAX to avoid backend initialization issues
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
args_gpu, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_gpu.gpu)

# Now safe to import JAX - it will only see the specified GPU
import jax
# IMPORTANT: Force backend initialization NOW 
# (before other module imports trigger it)
# JAX uses lazy initialization - without this, backend may initialize during 
# utils/model imports, potentially before CUDA_VISIBLE_DEVICES takes effect
print(jax.__version__)
print(jax.devices())

import jax
from jax import numpy as jnp
import numpy as np
import os
import yaml
import wandb

jnp.set_printoptions(precision=3, suppress=True, linewidth=10000000)

from utils import setup_random_seeds, parse_experiment_config
from utils.pipeline import (
    setup_dataset,
    create_model,
    setup_training,
    setup_experiment_dirs,
    tabulate_model,
    print_dcls_parameters,
    train_model,
)


def load_config_with_dcls(config_file):
    """Load config file and merge with DCLS and optimizer configurations if needed."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    # Check if this is a DCLS config that needs merging
    if config.get('conv') == 'dcls':
        assert config.get('dcls_config') is not None, "DCLS config number must be specified in the main config"
        dcls_config_file = f"yaml_folder/dcls_c{config.get('dcls_config')}.yaml"
        
        if os.path.exists(dcls_config_file):
            with open(dcls_config_file, "r") as dcls_file:
                dcls_config = yaml.safe_load(dcls_file)
                
            # Merge DCLS config into main config (main config takes precedence)
            for key, value in dcls_config.items():
                if key not in config:
                    config[key] = value
                    
            print(f"Merged DCLS configuration from {dcls_config_file}")
    
    # Load optimizer configuration if it exists
    optimizer_config_file = "yaml_folder/optimizer_config.yaml"
    if os.path.exists(optimizer_config_file):
        with open(optimizer_config_file, "r") as opt_file:
            opt_config = yaml.safe_load(opt_file)
            
        # Merge optimizer config into main config (main config takes precedence)
        for key, value in opt_config.items():
            if key not in config:
                config[key] = value
                
        print(f"Merged optimizer configuration from {optimizer_config_file}")
    
    return config


def main(args=None):
    """Main training orchestration."""
    # Handle WandB sweep mode
    if args is None:
        wandb.init()
        args = wandb.config
        args._from_wandb = True

    # Basic setup
    dtype = getattr(args, 'dtype', 'float32')
    dtype = jnp.float16 if dtype == 'float16' else jnp.float32
    print(f"Using dtype: {dtype}")

    key, seed = setup_random_seeds(args.seed)
    hidden_dim, latent_dim = parse_experiment_config(args)

    # Setup pipeline
    dataset_info = setup_dataset(args, dtype)
    model_cls = create_model(args, dataset_info, hidden_dim, latent_dim)
    training_state = setup_training(key, model_cls, dataset_info, args, dtype)

    # Log parameters
    wandb.log({"n_params": training_state.n_params})

    # Model inspection
    key, subkey = jax.random.split(key)
    tabulate_model(model_cls, dataset_info.sample_batch, subkey)
    print(args)

    if args.conv == 'dcls':
        print_dcls_parameters(training_state.state, args.dataset)

    # Setup experiment directories
    experiment_dirs = setup_experiment_dirs(
        args, hidden_dim, latent_dim, seed, config_file
    )

    # Train
    results = train_model(
        training_state, model_cls, dataset_info,
        experiment_dirs, args, key
    )

    # Save training dynamics (non-sweep mode only)
    if not hasattr(args, '_from_wandb'):
        np.savez(
            os.path.join(experiment_dirs.results_dir, 'training_dynamics.npz'),
            train_losses=results.train_losses,
            train_accuracies=results.train_accuracies,
            val_losses=results.val_losses,
            val_accuracies=results.val_accuracies,
            test_losses=results.test_losses,
            test_accuracies=results.test_accuracies
        )
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GRU model")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=['mnist', 'cifar', 'imdb', 'listops', 'path', \
                                    'pathx', 'aan', 'gsc'],
                        help="Dataset to use for training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--conv_mode", type=str, default="dcls", 
                        choices=['dcls', 'rnn_eerf', 'rnn_lerf', 'vanilla', \
                                    'tcn_lerf', 'tcn_eerf'], 
                        help="Convolution mode: dcls, causal_eerf, or causal_lerf")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Seed to use for random number generation")
    parser.add_argument("--file_nb", type=int, default=0, 
                        help="File number to load the configuration from")
    parser.add_argument("--sweep", action="store_true", 
                        help="Run in sweep mode using wandb sweep configuration")
    parser.add_argument("--sim_name", type=str, default="gen", 
                        help="Simulation name for wandb")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to checkpoint directory to resume training from")
    args_cli = parser.parse_args()

    # GPU and JAX backend already configured at top of file

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
            config_file = f"yaml_folder/{args_cli.dataset}_{args_cli.conv_mode}_wandb_{args_cli.file_nb}.yaml"
            config = load_config_with_dcls(config_file)
            return config, config_file
        
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"0.95"
        sweep_config, config_file = load_config()
        print(sweep_config)
        
        sweep_id = wandb.sweep(sweep_config, project="Den-minGRU_sweeps") 
        wandb.agent(sweep_id, main)
    else:
        # Regular mode: load config from yaml or checkpoint and run single experiment
        def parse_args():
            if args_cli.resume_from is not None:
                # Load config from checkpoint directory
                config_file = os.path.join(args_cli.resume_from, 'config.yaml')
                if os.path.exists(config_file):
                    print(f"Loading config from checkpoint: {config_file}")
                    config = load_config_with_dcls(config_file)
                    return argparse.Namespace(**config), config_file
                else:
                    raise FileNotFoundError(f"No config.yaml found in checkpoint directory: {args_cli.resume_from}")
            else:
                # Load config from yaml_folder
                config_file = f"yaml_folder/{args_cli.dataset}_{args_cli.conv_mode}_{args_cli.file_nb}.yaml"
                config = load_config_with_dcls(config_file)
                return argparse.Namespace(**config), config_file

        args, config_file = parse_args()
        print(args)

        # add the CLI arguments to the args object
        args.dataset = args_cli.dataset
        args.gpu = args_cli.gpu
        if args_cli.seed is not None:
            args.seed = args_cli.seed
        args.resume_from = args_cli.resume_from
        args.sim_name = args_cli.sim_name
        
        # set jax XLA_PYTHON_CLIENT_MEM_FRACTION=.XX
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.mem_frac}"
        
        wandb.init(project="DenGRU_general", name=f"{args.sim_name}")
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
# python main.py --resume_from checkpoints/general/listops_H64L6B64_do0_lr0.003wd0.1we10.0_skipLTEF_DCLS64axgaus0.7hetPF8_recsigmoid_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0 --dataset listops --gpu 3
# python main.py --dataset aan --gpu 2 --conv_mode dcls --dcls_config 6 --file_nb 0
# python main.py --dataset listops --gpu 1 --conv_mode dcls --dcls_config 6 --file_nb 0 --sim_name listops_sigmoid_64dcls8_lr0.003_bs64_wd0.1_s0
# python main.py --dataset listops --gpu 3 --conv_mode dcls --dcls_config 6 --file_nb 0 --sim_name listops_sigmoid_64dcls8_lr0.003_bs64_wd0.1_s0


# 0: 0.005, rec_ln, conv_ln, wd0.02
# 1: 0.005, wd0.02
# 0: 0.005, rec_ln, conv_ln, wd0.1
# 1: 0.005, wd0.1
# 0: 0.003, rec_ln, conv_ln, wd0.02
# 1: 0.003, wd0.02
