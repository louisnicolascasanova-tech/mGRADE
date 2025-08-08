import jax
import jax.numpy as jnp
import torch
import torchvision
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd
import yaml
from typing import Union, Callable, Tuple
from pathlib import Path
from flax.linen import one_hot
from lra import IMDB, AAN, ListOps, PathFinder

PX = 1/plt.rcParams['figure.dpi']
DEFAULT_CACHE_DIR_ROOT = Path("./cache_dir/")



# WARNING: this code is from QSSM project and won't be updated 
def create_mnist_classification_dataset(bsz=128, root="./data", version="sequential"):
    print("[*] Generating MNIST Classification Dataset...")
    assert version in ["sequential", "row"], "Invalid version for MNIST dataset"

    # Constants
    if version == "sequential":
        SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    elif version == "row":
        SEQ_LENGTH, N_CLASSES, IN_DIM = 28, 10, 28
    tf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ]

    tf.append(transforms.Lambda(lambda x: x.view(SEQ_LENGTH, IN_DIM)))
    tf = transforms.Compose(tf)

    train = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=tf
    )

    # split the dataset into train and val 
    train, val = torch.utils.data.random_split(train, [50000, 10000])

    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        labels = np.array(transposed_data[1])
        images = np.array(transposed_data[0])

        return images, labels       


    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def create_cifar_gs_classification_dataset(bsz=128, root="./data"):
    
    print("[*] Generating CIFAR-10 Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 1
    tf = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(1, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )
    train, val = torch.utils.data.random_split(train, [40000, 10000])


    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        labels = np.array(transposed_data[1])
        images = np.array(transposed_data[0])

        return images, labels

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def write_config_yaml(args, CKPT_DIR):
    """
    Write a YAML configuration file to CKPT_DIR using the values from args.
    Uses the same template structure as the reference YAML files.
    """
    config = {
        'seed': args.seed,
        'mem_frac': args.mem_frac,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'scheduler': args.scheduler,
        'alpha_cosine': args.alpha_cosine,
        'warmup_frac': args.warmup_frac,
        'weight_decay': args.weight_decay,
        'n_layers': args.n_layers,
        'hidden_dim': args.hidden_dim,
        # ENCODER
        'encoder': args.encoder,
        # SKIP
        'layer_skip': args.layer_skip,
        'element_skip': args.element_skip,
        # RECURRENT
        'enable_rec': args.enable_rec,
        'rec_act': args.rec_act,
        # CONVOLUTION
        'enable_conv': args.enable_conv,
        'conv': args.conv,
        'kernel_size': args.kernel_size,
        # LERF and EERF
        'wavenet_dilation': args.wavenet_dilation,
        # LERF
        'constant_dilation': args.constant_dilation,
        # EERF
        'dilation_schedule': args.dilation_schedule,
        'dilation_offset': args.dilation_offset,
        'dilation_boundary': args.dilation_boundary,
        # DCLS
        'delay_type': args.delay_type,
        'delay_kernel': args.delay_kernel,
        'init_std': args.init_std,
        'kernel_n_elems': args.kernel_n_elems,
        'heterogeneous_weights': args.heterogeneous_weights,
        'heterogeneous_positions': args.heterogeneous_positions,
        'heterogeneous_std': args.heterogeneous_std,
        'train_weights': args.train_weights,
        'train_positions': args.train_positions,
        'train_std': args.train_std,
        # CHANNEL MIXING
        'enable_cm': args.enable_cm,
        'channel_mixing': args.channel_mixing,
        'cm_act': args.cm_act,
        'glu_type': args.glu_type,
        # COMPRESSION
        'latent_dim': args.latent_dim,
        'comp_act': args.comp_act,
        # NORMALIZATION
        'postnorm': args.postnorm,
        # REGULARIZATION
        'do_rate': args.do_rate,
        'reg_factor': args.reg_factor,
        # Wandb gradients
        'wandb_gradients': args.wandb_gradients
    }
    
    # Create the config file path
    config_path = os.path.join(CKPT_DIR, 'config.yaml')
    
    # Write the YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to: {config_path}")


def make_data_loader(
    dset,
    dobj,
    seed: int,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    collate_fn: callable = None,
):
    """

    :param dset: 			(PT dset):		PyTorch dataset object.
    :param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
    :param seed: 			(int):			Int for seeding shuffle.
    :param batch_size: 		(int):			Batch size for batches.
    :param shuffle:         (bool):			Shuffle the data loader?
    :param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
    :return:
    """

    # Create a generator for seeding random number draws.
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    if dobj is not None:
        assert collate_fn is None
        collate_fn = dobj._collate_fn

    # Generate the dataloaders.
    return torch.utils.data.DataLoader(
        dataset=dset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=rng,
    )

def create_lra_imdb_classification_dataset(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, batch_size: int = 50, seed: int = 42
):
    print("[*] Generating LRA-text (IMDB) Classification Dataset")
    name = "imdb"
    dataset_obj = IMDB("imdb")
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trainloader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    testloader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    valloader = None

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = 135  # We should probably stop this from being hard-coded.
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trainloader,
        valloader,
        testloader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )

def create_lra_listops_classification_dataset(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, batch_size: int = 50, seed: int = 42
):
    print("[*] Generating LRA-listops Classification Dataset")

    name = "listops"
    dir_name = "./raw_datasets/lra_release/lra_release/listops-1000"

    dataset_obj = ListOps(name, data_dir=dir_name)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = 20
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )

def create_lra_aan_classification_dataset(
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
    batch_size: int = 50,
    seed: int = 42,
):
    print("[*] Generating LRA-AAN Classification Dataset")

    name = "aan"
    dir_name = "./raw_datasets/lra_release/lra_release/tsv_data"
    kwargs = {
        "n_workers": 1,  # Multiple workers seems to break AAN.
    }

    dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = len(dataset_obj.vocab)
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )




@jax.vmap
def create_mask(x, length):
    L = x.shape[0]
    mask = (jnp.arange(L) >= length[0]) * (jnp.arange(L) < length[1])
    return mask

def prep_batch(batch, seq_len, in_dim):
    """Take a batch and convert it to a standard x/y format"""
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Unhandled data type. ")

    inputs = jnp.array(inputs.numpy()).astype(float)  # convert to jax (float32)
    targets = jnp.array(targets.numpy())  # convert to jax (int32)
    lengths = aux_data.get("lengths", None)  # get lengths from aux if it is there.

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        inputs = jnp.pad(inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,))

    # Inputs size is [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(inputs, in_dim)

    if lengths is not None:
        lengths = jnp.array(lengths)
        if len(lengths.shape) == 1:  # If lengths only give last
            lengths = jnp.stack([jnp.zeros((inputs.shape[0],)), lengths], axis=1)
        masks = create_mask(inputs, lengths)
    else:
        masks = jnp.ones((inputs.shape[0], inputs.shape[1]))

    return inputs, targets, masks


def setup_random_seeds(seed=None):
    """Initialize random seeds for reproducibility."""
    seed = seed if seed is not None else 42
    key = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return key, seed


def parse_experiment_config(args):
    """Parse and prepare experiment configuration."""
    hidden_dim = [args.hidden_dim] * args.n_layers
    latent_dim = [args.latent_dim] * args.n_layers
    args.warmup_epochs = args.warmup_frac * args.n_epochs
    
    print(f"Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")
    assert len(hidden_dim) == len(latent_dim) == args.n_layers, \
        "Hidden and latent dimensions must match the number of layers"
    
    return hidden_dim, latent_dim


def generate_experiment_id(args, hidden_dim, latent_dim, seed):
    """Generate a unique experiment identifier string."""
    # Helper function for boolean to string conversion
    bool_to_str = lambda x: 'T' if x else 'F'
    
    # Basic config strings
    hidden_dim_str = hidden_dim[0]
    latent_dim_str = latent_dim[0] if latent_dim[0] is not None else 'F'
    skip_str = f"skipL{bool_to_str(args.layer_skip)}E{bool_to_str(args.element_skip)}"
    postnorm_str = bool_to_str(args.postnorm)
    
    # Convolution config string
    if args.conv == 'dcls':
        delay_type_str = 'syn' if args.delay_type == 'synaptic' else 'ax'
        delay_ker_str = 'gaus' if args.delay_kernel == 'gaussian' else 'exp'
        hete_pos_str = f'hetP{bool_to_str(args.heterogeneous_positions)}'
        conv_str = f'DCLS{args.kernel_size}{delay_type_str}{delay_ker_str}{args.init_std}{hete_pos_str}{args.kernel_n_elems}'
    else:
        wavenet_str = 'eerf' if args.wavenet_dilation else 'lerf'
        schedule_str = args.dilation_schedule if args.dilation_schedule is not None else 'F'
        conv_str = f'conv{args.kernel_size}{wavenet_str}sch{schedule_str}'
    
    # Channel mixing config string
    if args.channel_mixing == 'glu':
        cm_str = f"cm{args.channel_mixing}{args.glu_type}"
    elif args.channel_mixing == 'mlp':
        cm_str = f"cm{args.channel_mixing}{args.cm_act}"
    else:
        cm_str = "cmF"
    
    # Heterogeneous config string
    hetero_parts = [
        f"hetW{bool_to_str(args.heterogeneous_weights)}",
        f"S{bool_to_str(args.heterogeneous_std)}",
        f"trainW{bool_to_str(args.train_weights)}",
        f"S{bool_to_str(args.train_std)}",
        f"P{bool_to_str(args.train_positions)}"
    ]
    hetero_str = "".join(hetero_parts)
    
    # Combine all parts
    id_parts = [
        f"general/{args.dataset}",
        f"H{hidden_dim_str}L{args.n_layers}B{args.batch_size}",
        f"do{args.do_rate}",
        f"lr{args.lr}wd{args.weight_decay}we{args.warmup_epochs}",
        skip_str,
        conv_str,
        f"rec{args.rec_act}",
        cm_str,
        f"HpreReg{args.reg_factor}",
        f"C{latent_dim_str}{args.comp_act}",
        f"p{postnorm_str}",
        hetero_str,
        f"s{seed}"
    ]
    
    return "_".join(id_parts)


def create_experiment_directories(base_id):
    """Create unique experiment directories, handling conflicts."""
    def get_unique_id(base_id):
        current_id = base_id
        ckpt_dir = os.path.join(os.getcwd(), f"checkpoints/{current_id}")
        
        if not os.path.exists(ckpt_dir):
            return current_id, ckpt_dir
        
        # Handle conflicts by appending numbers
        current_id += "_training_1"
        ckpt_dir = os.path.join(os.getcwd(), f"checkpoints/{current_id}")
        
        counter = 2
        while os.path.exists(ckpt_dir):
            current_id = current_id[:-1] + str(counter)
            ckpt_dir = os.path.join(os.getcwd(), f"checkpoints/{current_id}")
            counter += 1
        
        return current_id, ckpt_dir
    
    final_id, ckpt_dir = get_unique_id(base_id)
    wu_dir = os.path.join(ckpt_dir, "warmup")
    result_dir = os.path.join(os.getcwd(), f"results/{final_id}")
    plt_dir = os.path.join(os.getcwd(), f"plots/{final_id}")
    
    # Create directories
    for directory in [ckpt_dir, result_dir, plt_dir]:
        os.makedirs(directory, exist_ok=True)
        print(directory)
    
    return final_id, ckpt_dir, wu_dir, result_dir, plt_dir


def compute_class_weights(train_loader, num_classes):
    """Compute class weights based on the training dataset."""
    class_counts = np.zeros(num_classes, dtype=np.float32)
    
    for batch in train_loader:
        labels = batch[1]  # batch can be a tuple (inputs, labels) or (inputs, labels, aux_data)
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        class_counts[unique] += counts
    
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    return tuple(class_weights.tolist())

def create_lra_path32_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											 bsz: int = 50,
											 seed: int = 42):
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-Pathfinder32 Classification Dataset")
	name = 'pathfinder'
	resolution = 32
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE

def create_lra_pathx_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											bsz: int = 50,
											seed: int = 42):
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-PathX Classification Dataset")
	name = 'pathfinder'
	resolution = 128
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE
