import os
import yaml
import argparse

parser = argparse.ArgumentParser(description="Generate sweep YAMLs for different kernel sizes.")
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension value for the sweep YAMLs (default: 32)')
parser.add_argument('--mode', type=str, choices=['tcn', 'rnn'], default='tcn', help="Prefix for YAML files: 'tcn' or 'rnn' (default: 'tcn')")
parser.add_argument('--dcls_config', type=int, default=0, help='DCLS config value for the sweep YAMLs (default: 1)')
args = parser.parse_args()

# # Directory to save YAML files
# yaml_dir = "yaml_folder"
# os.makedirs(yaml_dir, exist_ok=True)

SEEDS = [0, 1, 2]
# Kernel sizes and their allowed kernel_n_elems values
kernel_configs = {
    8:   [4],
    16:  [4, 8],
    32:  [4, 8, 16],
    64:  [4, 8, 16, 32],
    128: [4, 8, 16, 32, 64],
    256: [4, 8, 16, 32, 64, 128],
    512: [4, 8, 16, 32, 64, 128, 256]
}

dcls_configs = {
    0: { # train homogeneously spaced positions with constant ones weights
        "heterogeneous_positions": False,
        "heterogeneous_weights": False,
        "train_weights": False,
    },
    1: { # train heterogeneously spaced positions with constant ones weights
        "heterogeneous_positions": True,
        "heterogeneous_weights": False,
        "train_weights": False,
    },
    2: { # train homogeneously spaced positions with constant heterogeneously initialised weights
        "heterogeneous_positions": False,
        "heterogeneous_weights": True,
        "train_weights": False,
    },
    3: { # train heterogeneously spaced positions with constant heterogeneously initialised weights
        "heterogeneous_positions": True,
        "heterogeneous_weights": True,
        "train_weights": False,
    },
    4: { # train homogeneously spaced positions with trainable homogeneously initialised weights
        "heterogeneous_positions": False,
        "heterogeneous_weights": False,
        "train_weights": True,
    },
    5: { # train heterogeneously spaced positions with trainable homogeneously initialised weights
        "heterogeneous_positions": True,
        "heterogeneous_weights": False,
        "train_weights": True,
    },
    6: { # train homogeneously spaced positions with trainable heterogeneously initialised weights
        "heterogeneous_positions": False,
        "heterogeneous_weights": True,
        "train_weights": True,
    },
    7: { # train heterogeneously spaced positions with trainable heterogeneously initialised weights
        "heterogeneous_positions": True,
        "heterogeneous_weights": True,
        "train_weights": True,
    }
}

# hetW (2) - hetTrainW (6) - hetp_hettrainw (7)

# Common sweep config
base_config = {
    "name": "sweep",
    "metric": {
        "name": "test_ac",
        "goal": "maximize"
    },
    "method": "grid",
    "parameters": {
        "dataset": {"value": "cifar"},
        "n_epochs": {"value": 100},
        "mem_frac": {"value": 0.5},
        "batch_size": {"value": 32},
        "lr": {"value": 0.004},
        "scheduler": {"value": True},
        "alpha_cosine": {"value": 0},
        "warmup_frac": {"value": 0.5},
        "weight_decay": {"value": 0.1},
        "n_layers": {"value": 6},
        "encoder": {"value": True},
        "layer_skip": {"value": False},
        "element_skip": {"value": True},
        "enable_rec": {"value": True if args.mode == 'rnn' else False},
        "rec_act": {"value": 'relu' if args.mode == 'rnn' else None},
        "enable_conv": {"value": True},
        "conv": {"value": "dcls"},
        "constant_dilation": {"value": None},
        "wavenet_dilation": {"value": None},
        "dilation_schedule": {"value": None},
        "dilation_offset": {"value": None},
        "dilation_boundary": {"value": None},
        # ============ DCLS ============
        "delay_type": {"value": 'axonal'},
        "delay_kernel": {"value": 'gaussian'},
        "init_std": {"value": 0.7},
        # "kernel_n_elems": {"value": None},
        # "heterogeneous_weights": {"value": None},
        # "heterogeneous_positions": {"value": None},
        "heterogeneous_std": {"value": False},
        # "train_weights": {"value": None},
        "train_positions": {"value": True},
        "train_std": {"value": False},
        "enable_cm": {"value": True},
        # ============ CM ============
        "channel_mixing": {"value": "mlp"},
        "cm_act": {"value": "relu"},
        "glu_type": {"value": None},
        "latent_dim": {"value": None},
        "comp_act": {"value": None},
        "postnorm": {"value": True},
        "do_rate": {"value": 0},
        "reg_factor": {"value": 0},
        "wandb_gradients": {"value": False},
    }
}

class FlowStyleList(list): pass

def flow_style_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowStyleList, flow_style_list_representer)

for kernel_size, kernel_n_elems in kernel_configs.items():
    config = base_config.copy()
    config["parameters"] = base_config["parameters"].copy()
    config["parameters"]["hidden_dim"] = {"value": args.hidden_dim}
    config["parameters"]["kernel_size"] = {"value": kernel_size}
    config["parameters"]["heterogeneous_positions"] = {"value": dcls_configs[args.dcls_config]["heterogeneous_positions"]}
    config["parameters"]["heterogeneous_weights"] = {"value": dcls_configs[args.dcls_config]["heterogeneous_weights"]}
    config["parameters"]["train_weights"] = {"value": dcls_configs[args.dcls_config]["train_weights"]}
    # Use FlowStyleList for constant_dilation and seeds
    config["parameters"]["seed"] = {"values": FlowStyleList(SEEDS)} 
    config["parameters"]["kernel_n_elems"] = {"values": FlowStyleList(kernel_n_elems)}
    filename = f"cifar_{args.mode}_dcls_c{args.dcls_config}_wandb_{kernel_size}.yaml"
    # filepath = os.path.join(yaml_dir, filename)
    filepath = filename
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Generated {filepath}")

print("All DCLS sweep YAMLs generated.") 