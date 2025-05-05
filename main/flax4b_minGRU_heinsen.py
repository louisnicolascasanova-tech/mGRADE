import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]='0.95'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]='false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]='platform'
import argparse
import jax
import numpy as np
import torch
from jax import numpy as jnp
from utils import create_mnist_classification_dataset, create_cifar_gs_classification_dataset, scaled_shifted_sigmoid, scaled_shifted_tanh, threshold, plot_dynamics
from model import BatchRNN
from trainer import train, create_train_state
import json


def parse():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")

    # Define arguments
    parser.add_argument("--id_note", default=None, type=str, required=False, 
                        help="String that denotes a unique checkpoint.")
    
    parser.add_argument("--model_code", default='alllayer_dyn', type=str, required=False,
                        help="String that denotes a unique type of training. Default options are alllayer_dyn, firstlayer_dyn, nolayer_dyn.")
    parser.add_argument("--sweep_steepness", default=False, type=bool, required=False,
                        help="Whether to add steepness to model_code.")

    parser.add_argument("--time_decay_mean", default=(10.0, 1.0), type=float, nargs=2, required=False, 
                        help="Two floats representing the time decay means.")
    parser.add_argument("--decay", default=11, type=str, required=False, 
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
    parser.add_argument("--hidden_activation", default=None, type=str, required=False, 
                        help="Nonlinearity between hidden layers. Default options are None, sigmoid, relu, tanh, selu, scaled_sigmoid, and threshold.")

    parser.add_argument("--hidden_nonlinearity", default=None, type=str, required=False, 
                        help="Helper argument IGNORE.")
    parser.add_argument("--seed", default=0, type=int, required=False, 
                        help="Integer representing the random seed.")
    
    parser.add_argument("--gpu_index", default='1', type=str, required=False, 
                        help="Index of GPU.")
    parser.add_argument("--gpu_mem_fraction", default='0.95', type=str, required=False, 
                        help="GPU memory fraction.")

    parser.add_argument("--sigmoid_scale", default=10.0, type=float, required=False, 
                        help="Scaling factor for the scaled sigmoid function.")
    parser.add_argument("--sigmoid_center", default=0.5, type=float, required=False, 
                        help="Center shift for the scaled sigmoid function.")

    # Parse arguments
    args = parser.parse_args()

    for c in args.decay:
        if c not in ['0', '1']:
            raise ValueError("Decay must be a list of 0s and 1s.")
    args.decay = [bool(int(c)) for c in args.decay]

    if args.model_code == 'firstlayer_dyn':
        args.decay = [False]*args.n_layers
        args.decay[0] = True
        args.reg_factor = 0.0
    elif args.model_code == 'nolayer_dyn':
        args.decay = [False]*args.n_layers
        args.reg_factor = 0.0
    elif args.model_code == 'alllayer_dyn':
        args.decay = [True]*args.n_layers
    


    # Print parsed arguments
    print("Parsed arguments:")
    print(f"Time Decay Mean: {args.time_decay_mean}")
    print(f"Model Code: {args.model_code}")
    print(f"Decay: {args.decay}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset Version: {args.dataset_version}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Layers: {args.n_layers}")
    print(f"Number of Epochs: {args.n_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Regularization Factor: {args.reg_factor}")
    print(f"Hidden Activation: {args.hidden_activation}")
    print(f"Seed: {args.seed}")

    if args.hidden_activation is not None:
        if args.hidden_activation == "sigmoid":
            args.hidden_nonlinearity = jax.nn.sigmoid
        elif args.hidden_activation == "scaled_sigmoid":
            print("Scale of scaled sigmoid:", args.sigmoid_scale)
            args.hidden_nonlinearity = lambda x: scaled_shifted_sigmoid(x, scale=args.sigmoid_scale, center=args.sigmoid_center)
        elif args.hidden_activation == "scaled_tanh":
            args.hidden_nonlinearity = lambda x: scaled_shifted_tanh(x, scale=args.sigmoid_scale, center=args.sigmoid_center)
        elif args.hidden_activation == "relu":
            args.hidden_nonlinearity = jax.nn.relu
        elif args.hidden_activation == "tanh":
            args.hidden_nonlinearity = jax.nn.tanh
        elif args.hidden_activation == "selu":
            args.hidden_nonlinearity = jax.nn.selu
        elif args.hidden_activation == "threshold":
            args.hidden_nonlinearity = lambda x: threshold(x, thr=args.sigmoid_center)
        else:
            print("Invalid hidden nonlinearity. Using default (None).")
            args.hidden_nonlinearity = None
    else:
        print("Using default hidden nonlinearity (None).")

    return args


def save_results(result_dir, model_code, seed, test_loss, test_acc, train_loss, train_acc, n_layers, n_params, hidden_dim, steepness, dataset, sweep_steepness):
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
        "final_train_loss": np.float64(train_loss),
        "final_train_acc": np.float64(train_acc),
        "steepness": np.float64(steepness)
    }
    
    if sweep_steepness:
        model_code = model_code + f"_{steepness}"

    result_file = os.path.join(result_dir, f"results_{dataset}_{seed}.json")
    key = (n_layers, hidden_dim, model_code)

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
    if args.dataset=="cifar10":
        trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_cifar_gs_classification_dataset(bsz=BATCH_SIZE)
    else:
        trainloader, val_loader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_mnist_classification_dataset(bsz=BATCH_SIZE, root="/home/christian/LearningJAX/data/MNIST",  version=DATASET_VERSION)
    

    batch_x, batch_y = next(iter(testloader))
    print("Batch size:", batch_x.shape, batch_y.shape)

    # set result and checkpoint directories
    id_sim = f"mingru_heinsen/{args.dataset}/{DATASET_VERSION}_h{HIDDEN_DIM}_l{N_LAYERS}_dyn_{args.model_code}_lr{LR}_act_{args.hidden_activation}{args.sigmoid_scale}{args.id_note}"
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
    model = BatchRNN(hidden_size=HIDDEN_DIM, output_size=10, n_layers=N_LAYERS, decay=DECAY, timedecay_mean=TIMEDECAY_MEAN, seed=args.seed, hidden_nonlinearity=args.hidden_nonlinearity)
    state = create_train_state(key, BatchRNN, LR, DATASET_VERSION, HIDDEN_DIM, N_LAYERS, BATCH_SIZE, DECAY, TIMEDECAY_MEAN, args.seed, args.hidden_nonlinearity)

    # initialise parameters
    params = model.init(key, jnp.zeros_like(batch_x[:BATCH_SIZE]))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params["params"]))

    # train model
    train_losses, train_accuracies, val_losses, val_accuracies, test_loss, test_acc = train(key, state, trainloader, val_loader, testloader, N_EPOCHS, REG_FACTOR, RESULT_DIR, CKPT_DIR)

    # save test accuracy
    print("Test accuracy:", test_acc)
    # save results
    steepness = args.sigmoid_scale
    if args.hidden_nonlinearity == "threshold":
        steepness = -1
    save_results(RESULT_IMM, args.model_code, args.seed, test_loss, test_acc, train_losses[-1], train_accuracies[-1], N_LAYERS, n_params, HIDDEN_DIM, args.sigmoid_scale ,args.dataset, args.sweep_steepness)




if __name__ == "__main__":
    main()
