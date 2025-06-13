import os
import yaml
import argparse

parser = argparse.ArgumentParser(description="Generate sweep YAMLs for EERF (Exponential Effective Receptive Field) configs.")
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension value for the sweep YAMLs (default: 32)')
parser.add_argument('--mode', type=str, choices=['tcn', 'rnn'], default='tcn', help="Prefix for YAML files: 'tcn' or 'rnn' (default: 'tcn')")
args = parser.parse_args()

# # Directory to save YAML files
# yaml_dir = "yaml_folder"
# os.makedirs(yaml_dir, exist_ok=True)

SEEDS = [0, 1, 2]
# Map each kernel_size to its allowed dilation_offset values and corresponding dilation_boundary
kernel_configs = {
    4:   {"dilation_boundary": 8, "dilation_offset": [0, 1, 2, 3, 4, 5, 6]},
    8:   {"dilation_boundary": 7, "dilation_offset": [0, 1, 2, 3, 4, 5]},
    16:  {"dilation_boundary": 6, "dilation_offset": [0, 1, 2, 3, 4]},
    32:  {"dilation_boundary": 5, "dilation_offset": [0, 1, 2, 3]},
    64:  {"dilation_boundary": 4, "dilation_offset": [0, 1, 2]},
    128: {"dilation_boundary": 3, "dilation_offset": [0, 1]},
}

class FlowStyleList(list): pass

def flow_style_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowStyleList, flow_style_list_representer)

base_config = {
    "name": "sweep",
    "metric": {
        "name": "test_ac",
        "goal": "maximize"
    },
    "method": "grid",
    "parameters": {
        # seed
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
        "hidden_dim": {"value": args.hidden_dim},
        "encoder": {"value": True},
        "layer_skip": {"value": False},
        "element_skip": {"value": True},
        "enable_rec": {"value": True if args.mode == 'rnn' else False},
        "rec_act": {"value": 'relu' if args.mode == 'rnn' else None},
        # CONV
        "enable_conv": {"value": True},
        "conv": {"value": "conv"},
        # kernel_size
        "wavenet_dilation": {"value": True},
        "constant_dilation": {"value": None},
        "dilation_schedule": {"value": 'clip'},
        # dilation_offset
        # dilation_boundary
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

for kernel_size, config_dict in kernel_configs.items():
    config = base_config.copy()
    config["parameters"] = base_config["parameters"].copy()
    config["parameters"]["seed"] = {"values": FlowStyleList(SEEDS)}
    config["parameters"]["kernel_size"] = {"value": kernel_size}
    config["parameters"]["dilation_offset"] = {"values": FlowStyleList(config_dict["dilation_offset"])}
    config["parameters"]["dilation_boundary"] = {"value": config_dict["dilation_boundary"]}
    filename = f"cifar_{args.mode}_eerf_wandb_{kernel_size}.yaml"
    # filepath = os.path.join(yaml_dir, filename)
    filepath = filename  # Save in the current directory
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Generated {filepath}")

print("All EERF sweep YAMLs generated.") 