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

PX = 1/plt.rcParams['figure.dpi']



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