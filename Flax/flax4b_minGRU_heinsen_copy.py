import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]='0.95'
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]='false'
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]='platform'
import argparse
import jax
import numpy as np
import torch
from jax import numpy as jnp
from utils import create_mnist_classification_dataset, create_cifar_gs_classification_dataset, bimodal_gaussian, plot_dynamics
from model import BatchRNN, BatchRNNFirstLayer
from trainer import train, create_train_state
import json
import glob




def parse():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")

    # Define arguments
    parser.add_argument("--id_note", default='', type=str, required=False, 
                        help="String that denotes a unique checkpoint.")

    parser.add_argument("--firstlayer", default=False, type=bool, required=False, 
                        help="Boolean representing whether to use first layer decay or not.")
    parser.add_argument("--time_decay_mean", default=(10.0, 1.0), type=float, nargs=2, required=False, 
                        help="Two floats representing the time decay means.")
    parser.add_argument("--decay", default=False, type=bool, required=False, 
                        help="Boolean indicating whether decay is enabled.")
    parser.add_argument("--dataset", default='mnist', type=str, required=False)
    parser.add_argument("--dataset_version", default='sequential', type=str, required=False, 
                        help="String representing the dataset version.")
    parser.add_argument("--hidden_dim", default=64, type=int, required=False, 
                        help="Integer representing the hidden dimension.")
    parser.add_argument("--batch_size", default=128,type=int, required=False, 
                        help="Integer representing the batch size.")
    parser.add_argument("--n_layers", default=2, type=int, required=False, 
                        help="Integer representing the number of layers.")
    parser.add_argument("--n_epochs", default=50, type=int, required=False, 
                        help="Integer representing the number of epochs.")
    parser.add_argument("--lr", default=1e-3, type=float, required=False, 
                        help="Float representing the learning rate.")
    parser.add_argument("--reg_factor", default=1e-8, type=float, required=False, 
                        help="Float representing the regularization factor.")
    parser.add_argument("--seed", default=0, type=int, required=False, 
                        help="Integer representing the random seed.")
    
    parser.add_argument("--gpu_index", default='1', type=str, required=False, 
                        help="Index of GPU.")
    parser.add_argument("--gpu_mem_fraction", default='0.95', type=str, required=False, 
                        help="GPU memory fraction.")

    # Parse arguments
    args = parser.parse_args()

    # Print parsed arguments
    print("Parsed arguments:")
    print(f"Time Decay Mean: {args.time_decay_mean}")
    print(f"Decay: {args.decay}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset Version: {args.dataset_version}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Layers: {args.n_layers}")
    print(f"Number of Epochs: {args.n_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Regularization Factor: {args.reg_factor}")
    print(f"First Layer:", {args.firstlayer})
    print(f"Seed: {args.seed}")

    return args


def save_results(result_dir, decay, seed, test_loss, test_acc, n_layers, n_params, hidden_dim, dataset, firstlayer):
    """Save results to a JSON file, uniquely identified by n_layers, n_params, and decay."""
    results = {
        "seed": np.float64(seed),
        # "train_losses": np.float64(train_losses),
        # "train_accuracies": np.float64(train_accuracies),
        # "val_losses": np.float64(val_losses),
        # "val_accuracies": np.float64(val_accuracies),
        "test_loss": np.float64(test_loss),
        "test_acc": np.float64(test_acc),
        "n_layers": np.float64(n_layers),
        "n_params": np.float64(n_params),
    }
    if firstlayer:
        result_file = os.path.join(result_dir, f"results_{dataset}firstlayer_{seed}.json")
    else:
        result_file = os.path.join(result_dir, f"results_{dataset}_{seed}.json")
    key = (n_layers, hidden_dim, decay)

    print(result_file)

    # Check if the file exists and load existing data
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # Convert the key to a string (JSON keys must be strings)
    key_str = str(key)
    print(key_str)

    # Append results to the corresponding key
    if key_str not in existing_data:
        existing_data[key_str] = []
        existing_data[key_str].append(results)
    else:
        existing_data[key_str] = [results]


    # Write updated data back to the file
    with open(result_file, "w") as f:
        json.dump(existing_data, f, indent=4)
    print(f"Results saved to {result_file}")


def aggregate_results(result_dir):
    """Aggregate and average results across different seeds, returning 2D arrays for test accuracy and standard deviation."""
    result_files = glob.glob(os.path.join(result_dir, "results_seed_*.json"))
    
    if not result_files:
        print("No results files found.")
        return None, None, None, None

    # Organize results by (n_layers, n_params, decay)
    aggregated = {}
    for file in result_files:
        with open(file, "r") as f:
            data = json.load(f)
            for key_str, results in data.items():
                key = eval(key_str)  # Convert string back to tuple (n_layers, hidden_dim, decay)
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].extend([result["test_acc"] for result in results])

    # Extract unique values for layers and parameters
    unique_layers = sorted(set(key[0] for key in aggregated.keys()))
    unique_dims = sorted(set(key[1] for key in aggregated.keys()))

    # Initialize 2D arrays
    acc_decay_true = np.zeros((len(unique_layers), len(unique_dims)))
    std_decay_true = np.zeros((len(unique_layers), len(unique_dims)))
    acc_decay_false = np.zeros((len(unique_layers), len(unique_dims)))
    std_decay_false = np.zeros((len(unique_layers), len(unique_dims)))

    # Populate arrays
    for (n_layers, hidden_dim, decay), test_accuracies in aggregated.items():
        layer_idx = unique_layers.index(n_layers)
        param_idx = unique_dims.index(hidden_dim)
        mean_acc = np.mean(test_accuracies)
        std_acc = np.std(test_accuracies)

        if decay:
            acc_decay_true[layer_idx, param_idx] = mean_acc
            std_decay_true[layer_idx, param_idx] = std_acc
        else:
            acc_decay_false[layer_idx, param_idx] = mean_acc
            std_decay_false[layer_idx, param_idx] = std_acc

    return acc_decay_true, std_decay_true, acc_decay_false, std_decay_false


def main():
    args = parse()

    # fix random seed
    key = jax.random.PRNGKey(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # set printoptions
    jnp.set_printoptions(precision=3, suppress=True)

    # set devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.gpu_mem_fraction

    # set hyperparameters
    TIMEDECAY_MEAN = args.time_decay_mean
    DECAY = args.decay

    DATASET_VERSION = args.dataset_version
    HIDDEN_DIM = args.hidden_dim #2*x*(1+x*(n-1)+n+5+10/64)
    BATCH_SIZE = args.batch_size
    N_LAYERS = args.n_layers
    N_EPOCHS = args.n_epochs
    LR = args.lr
    REG_FACTOR = args.reg_factor # 1e-8, 1e-6, 1e-7, 1e-9, 0.0

    # get dataset
    key, subkey = jax.random.split(key)
    if args.dataset=="cifar10":
        trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_cifar_gs_classification_dataset(bsz=BATCH_SIZE)
    else:
        trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_mnist_classification_dataset(bsz=BATCH_SIZE, root="/home/christian/LearningJAX/data/MNIST",  version=DATASET_VERSION)
    

    batch_x, batch_y = next(iter(testloader))
    print("Batch size:", batch_x.shape, batch_y.shape)

    # set result and checkpoint directories
    id_sim = f"mingru_heinsen/{args.dataset}/{DATASET_VERSION}_h{HIDDEN_DIM}_l{N_LAYERS}_lr{LR}_bsz{BATCH_SIZE}_reg{REG_FACTOR}{args.id_note}"
    CKPT_DIR = os.path.join(os.getcwd(), f"checkpoints/{id_sim}")
    print(CKPT_DIR)
    RESULT_DIR = os.path.join(os.getcwd(), f"results/{id_sim}")
    RESULT_IMM = os.path.join(os.getcwd(), f"results")
    print(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)
    PLT_DIR = os.path.join(os.getcwd(), f"plots/{id_sim}")
    print(PLT_DIR)
    os.makedirs(PLT_DIR, exist_ok=True)


    # set model and create train state
    if args.firstlayer:
        model = BatchRNNFirstLayer(hidden_size=HIDDEN_DIM, output_size=10, n_layers=N_LAYERS, decay=DECAY, timedecay_mean=TIMEDECAY_MEAN)
        state = create_train_state(key, BatchRNNFirstLayer, LR, DATASET_VERSION, HIDDEN_DIM, N_LAYERS, BATCH_SIZE, DECAY, TIMEDECAY_MEAN)
    else:
        model = BatchRNN(hidden_size=HIDDEN_DIM, output_size=10, n_layers=N_LAYERS, decay=DECAY, timedecay_mean=TIMEDECAY_MEAN)
        state = create_train_state(key, BatchRNN, LR, DATASET_VERSION, HIDDEN_DIM, N_LAYERS, BATCH_SIZE, DECAY, TIMEDECAY_MEAN)

    # initialise parameters
    params = model.init(key, jnp.zeros_like(batch_x[:BATCH_SIZE]))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params["params"]))

    # train model
    train_losses, train_accuracies, val_losses, val_accuracies, test_loss, test_acc = train(key, state, trainloader, val_loader, testloader, N_EPOCHS, REG_FACTOR, RESULT_DIR, CKPT_DIR)

    # save test accuracy
    print("Test accuracy:", test_acc)
    # save results
    save_results(RESULT_IMM, args.decay, args.seed, test_loss, test_acc, N_LAYERS, n_params, HIDDEN_DIM, args.dataset, args.firstlayer)




if __name__ == "__main__":
    main()
