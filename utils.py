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
from lra import IMDB

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
