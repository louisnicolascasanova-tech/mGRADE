import os
import yaml
import argparse

parser = argparse.ArgumentParser(description="Generate sweep YAMLs for different kernel sizes.")
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension value for the sweep YAMLs (default: 32)')
parser.add_argument('--mode', type=str, choices=['tcn', 'rnn'], default='tcn', help="Prefix for YAML files: 'tcn' or 'rnn' (default: 'tcn')")
args = parser.parse_args()

# Directory to save YAML files
yaml_dir = "yaml_folder"
os.makedirs(yaml_dir, exist_ok=True)

SEEDS = [0, 1, 2]
# Kernel sizes and their allowed constant_dilation values
kernel_configs = {
    4:   [1, 2, 4, 8, 16, 32, 64, 128],
    8:   [1, 2, 4, 8, 16, 32, 64],
    16:  [1, 2, 4, 8, 16, 32],
    32:  [1, 2, 4, 8, 16],
    64:  [1, 2, 4, 8],
    128: [1, 2, 4],
    256: [1, 2],
    512: [1],
}

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
        "conv": {"value": "conv"},
        # kernel_size and constant_dilation will be set per file
        "wavenet_dilation": {"value": False},
        "dilation_schedule": {"value": None},
        "dilation_offset": {"value": None},
        "dilation_boundary": {"value": None},
        "delay_type": {"value": None},
        "delay_kernel": {"value": None},
        "init_std": {"value": None},
        "kernel_n_elems": {"value": None},
        "heterogeneous_weights": {"value": None},
        "heterogeneous_positions": {"value": None},
        "heterogeneous_std": {"value": None},
        "train_weights": {"value": None},
        "train_positions": {"value": None},
        "train_std": {"value": None},
        "enable_cm": {"value": True},
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

for kernel_size, dilations in kernel_configs.items():
    config = base_config.copy()
    config["parameters"] = base_config["parameters"].copy()
    config["parameters"]["hidden_dim"] = {"value": args.hidden_dim}
    config["parameters"]["kernel_size"] = {"value": kernel_size}
    # Use FlowStyleList for constant_dilation and seeds
    config["parameters"]["seed"] = {"values": FlowStyleList(SEEDS)} 
    config["parameters"]["constant_dilation"] = {"values": FlowStyleList(dilations)}
    filename = f"cifar_{args.mode}_lerf_wandb_{kernel_size}.yaml"
    filepath = os.path.join(yaml_dir, filename)
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Generated {filepath}")

print("All sweep YAMLs generated.") 