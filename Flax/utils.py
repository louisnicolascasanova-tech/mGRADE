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

PX = 1/plt.rcParams['figure.dpi']


@jax.custom_jvp
def gr_than(x, thr):
    """ Thresholding function for spiking neurons. """
    return (x > thr).astype(jnp.float32)

@gr_than.defjvp
def gr_jvp(primals, tangents):
    """ Surrogate gradient function for thresholding. """
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = gr_than(x, thr)
    tangent_out = x_dot / (10 * jnp.absolute(x - thr) + 1)**2
    return primal_out, tangent_out

@jax.jit
def threshold(x,thr):
    return gr_than(x,thr)

@jax.jit
def scaled_shifted_sigmoid(x, scale, center):
    """
    A scaled and shifted version of the sigmoid function.
    Args:
        x: Input array.
        scale: Scaling factor for the sigmoid output.
        shift: Shift applied to the sigmoid output.
    Returns:
        Scaled and shifted sigmoid output.
    """
    s=scale
    c=center
    return jax.nn.sigmoid((x-c)*s)

@jax.jit
def scaled_shifted_tanh(x, scale, center):
    """
    A scaled and shifted version of the tanh function.
    Args:
        x: Input array.
        scale: Scaling factor for the tanh output.
        shift: Shift applied to the tanh output.
    Returns:
        Scaled and shifted tanh output.
    """
    s=scale
    c=center
    return jax.nn.tanh((x-c)*s)


def create_cifar_gs_classification_dataset(bsz=128):
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

    # Split the training dataset into train and validation sets
    train_size = int((1 - 5/6) * len(train))
    val_size = len(train) - train_size
    train, val = torch.utils.data.random_split(train, [train_size, val_size])

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

def create_cifar_classification_dataset(bsz=128):
    print("[*] Generating CIFAR-10 Classification Dataset")
    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 3
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    # Split the training dataset into train and validation sets
    train_size = int((1 - 5/6) * len(train))
    val_size = len(train) - train_size
    train, val = torch.utils.data.random_split(train, [train_size, val_size])

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


def g(x):
    return jnp.where(x > 0, x+0.5, jax.nn.sigmoid(x))

def identity_weight_init():
    def init(key, shape, dtype=jnp.float32):
        w = jnp.eye(shape[0], shape[1], dtype=dtype) * 0.9
        w = w.at[-1, :].set(1.0)
        return w
    return init

def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def bimodal_gaussian(key, size, mean1, mean2, std1, std2, weight1=0.5):
    """
    Draw samples from a bimodal Gaussian distribution.
    
    Args:
        key: JAX random key.
        size: Number of samples to draw.
        mean1: Mean of the first Gaussian.
        mean2: Mean of the second Gaussian.
        std1: Standard deviation of the first Gaussian.
        std2: Standard deviation of the second Gaussian.
        weight1: Weight of the first Gaussian (between 0 and 1).
                 The second Gaussian will have weight (1 - weight1).
    
    Returns:
        Samples from the bimodal Gaussian distribution.
    """
    # Split the random key
    key1, key2 = jax.random.split(key)
    
    # Draw samples from two Gaussian distributions
    samples1 = jax.random.normal(key1, shape=size) * std1 + mean1
    samples2 = jax.random.normal(key2, shape=size) * std2 + mean2
    
    # Randomly choose between the two distributions based on the weights
    mix_key = jax.random.uniform(key, shape=size)
    mask = mix_key < weight1  # True for samples from the first Gaussian
    
    # Combine the samples
    samples = jnp.where(mask, samples1, samples2)
    return samples

def plot_dynamics(model, params, batch_inputs, batch_labels, dataset_version='sequential',
                    id_sample=0, nb_inputs_to_plot=5, nb_components_to_plot=5, model_type='srn', variable_to_plot='h', zoom=True):
    
    model_LUT = {
        'lstm': {'h': 0, 'c': 1},
        'gru': {'h': 0, 'z': 1, 'r': 2},
        'mgu': {'h': 0, 'f': 1},
        'mingru': {'h': 0, 'z': 1, 'z_preact': 2},
        'mingru_heinsen': {'h': 0, 'z': 1, 'z_preact': 1, 'h_tilde': 2, 'h_tilde_preact': 2, 'h_preact': 3},
    }
    
    # check if params has the key 'params' or not
    if 'params' not in params:
        params = {'params': params}

    n_inputs_to_simulate = 5 if id_sample < 5 else id_sample+1

    if model_type == 'srn':
        state_hist, out_hist = model.apply(params, batch_inputs[:n_inputs_to_simulate])
        n_layers = len(state_hist)
    elif model_type in ['lstm', 'gru', 'mgu', 'mingru', 'mingru_heinsen']:
        state_hist, out_hist = model.apply(params, batch_inputs[:n_inputs_to_simulate])
        n_layers = len(state_hist)
        print(len(state_hist))
        print(len(state_hist[0]))
        print(state_hist[0][0].shape)
        if variable_to_plot == 'all':
            n_cols = len(model_LUT[model_type])
        else: 
            id_component = model_LUT[model_type][variable_to_plot]
            n_cols = 1

    t = jnp.arange(784) if dataset_version == "sequential" else jnp.arange(28)

    if dataset_version == "sequential":
        inputs_to_plot = batch_inputs[id_sample, :, 0]
        labels_input = f'id_sample={id_sample}'
    else:
        inputs_to_plot = batch_inputs[id_sample, :, 14-nb_inputs_to_plot//2:14+(nb_inputs_to_plot+1)//2]
        labels_input = [f'input_{i}' for i in range(nb_inputs_to_plot)]

    fig, axs = plt.subplots(len(state_hist)+2, n_cols, figsize=(((1200+(n_cols-1)*600)*PX, (600+200*n_layers)*PX)))

    legend_ncol = 5 if nb_components_to_plot > 5 else nb_components_to_plot
    if n_cols > 1: 
        for n in range(n_cols):
            axs[0,n].plot(t, inputs_to_plot, label=labels_input)
            axs[0,n].set_title(f'Input Sequence = {batch_labels[id_sample]}')
            axs[0,n].legend(ncol=14, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            axs[0,n].grid(True)
            for i in range(n_layers):
                variable_name = list(model_LUT[model_type].keys())[n]
                id_variable = model_LUT[model_type][variable_name]
                data_to_plot = state_hist[i][id_variable][id_sample, :, :nb_components_to_plot]
                if model_type == 'mingru_heinsen':
                    if variable_name == 'z': 
                        data_to_plot = jax.nn.sigmoid(data_to_plot)
                    elif variable_name == 'h_tilde':
                        data_to_plot = g(data_to_plot)
                axs[i+1,n].plot(t, data_to_plot, label=[f'state_{i}' for i in range(nb_components_to_plot)])
                axs[i+1,n].set_title(f'{variable_name} - Layer {i}')
                if nb_components_to_plot < 10:
                    axs[i+1,n].legend(ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0.5, -0.1))
                # axs[i+1].set_ylim(0, 1)
                # axs[i+1].set_yticks(np.arange(0, 1, 0.1))
                axs[i+1,n].grid(True)
                if zoom: 
                    if variable_to_plot in ['z', 'r', 'f']:
                        axs[i+1,n].set_ylim(0, 1)
                        axs[i+1,n].set_yticks(np.arange(0, 1, 0.1))
            axs[-1,n].plot(t, out_hist[id_sample, :, :], label=[f'out_{i}' for i in range(10)])
            axs[-1,n].set_title('Output History')
            axs[-1,n].legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            axs[-1,n].grid(True)
    else:
        axs[0].plot(t, inputs_to_plot, label=labels_input)
        axs[0].set_title(f'Input Sequence = {batch_labels[id_sample]}')
        axs[0].legend(ncol=14, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        axs[0].grid(True)
        for i in range(n_layers):
            data_to_plot = state_hist[i][id_component][id_sample, :, :nb_components_to_plot]
            if model_type == 'mingru_heinsen':
                if variable_to_plot == 'z': 
                    data_to_plot = jax.nn.sigmoid(data_to_plot)
                elif variable_to_plot == 'h_tilde':
                    data_to_plot = g(data_to_plot)

            axs[i+1].plot(t, data_to_plot, label=[f'state_{i}' for i in range(nb_components_to_plot)])
            axs[i+1].set_title(f'{variable_to_plot} - Layer {i}')
            if nb_components_to_plot < 10:
                axs[i+1].legend(ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            # axs[i+1].set_ylim(0, 1)
            # axs[i+1].set_yticks(np.arange(0, 1, 0.1))
            axs[i+1].grid(True)
            if zoom: 
                if variable_to_plot in ['z', 'r', 'f']:
                    axs[i+1].set_ylim(0, 1)
                    axs[i+1].set_yticks(np.arange(0, 1, 0.1))

        axs[-1].plot(t, out_hist[id_sample, :, :], label=[f'out_{i}' for i in range(10)])
        axs[-1].set_title('Output History')
        axs[-1].legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        axs[-1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_loss_and_acc(train_losses, val_losses, train_accuracies, val_accuracies, offset=4):
    '''
    Plot the loss and accuracy of the model during training and validation
    # ax[0] entire loss - ax[0] entire accuracy
    # ax_in0 zoomed loss - ax_in0 zoomed accuracy
    '''

    t = np.arange(len(val_losses))
    fig, ax = plt.subplots(1, 2, figsize=(1200*PX, 600*PX))

    ax[0].plot(t, train_losses, label='train')
    ax[0].plot(t, val_losses, label='val')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].set_yticks(np.arange(0, np.max(train_losses), 0.3))
    ax[0].set_xticks(np.arange(0, len(val_losses), 5))
    ax[0].grid()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Value')

    ax_in0 = inset_axes(ax[0], width="60%", height="60%", loc="upper right")
    ax_in0.plot(t[offset:], train_losses[offset:], label='train')
    ax_in0.plot(t[offset:], val_losses[offset:], label='val')
    ax_in0.legend()
    # ax_in0.set_yticks(np.arange(0, np.max(train_losses[offset:]).round(2), 0.1))
    ax_in0.set_xticks(np.arange(offset, len(val_losses), 5))
    # set ylim
    y_min = min(np.min(train_losses[offset:]), np.min(val_losses[offset:])) * 0.9
    y_max = max(np.max(train_losses[offset:]), np.max(val_losses[offset:])) * 1.1
    ax_in0.set_ylim(y_min, y_max)
    ax_in0.set_yticks(np.arange(round(y_min, 2), y_max, 0.025))
    ax_in0.grid()


    ax[1].plot(t, train_accuracies, label='train')
    ax[1].plot(t, val_accuracies, label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].set_yticks(np.arange(0, 1, 0.1))
    ax[1].set_xticks(np.arange(0, len(val_accuracies), 5))
    ax[1].grid()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')

    ax_in1 = inset_axes(ax[1], width="60%", height="50%", loc="lower right", borderpad=2)
    ax_in1.plot(t[offset:], train_accuracies[offset:], label='train')
    ax_in1.plot(t[offset:], val_accuracies[offset:], label='val')
    ax_in1.legend()
    ax_in1.set_xticks(np.arange(offset, len(val_accuracies), 5))
    # set ylim
    y_min = min(np.min(val_accuracies[offset:]), np.min(train_accuracies[offset:])) * 0.999
    y_max = max(np.max(val_accuracies[offset:]), np.max(train_accuracies[offset:])) * 1.001
    ax_in1.set_ylim(y_min, y_max)
    ax_in1.set_yticks(np.arange(round(y_min, 2), y_max, 0.0025))
    ax_in1.grid()
            
    plt.tight_layout()
    plt.show()

def plot_digits(digit, model, params, model_type, batch_inputs, batch_labels, plt_dir, 
                variable_to_plot='o', layer_to_plot=0, nb_components_to_plot=5):
    

    y_digit = batch_labels == digit
    x_to_plot = batch_inputs[y_digit]
    
    model_LUT = {
        'lstm': {'h': 0, 'c': 1},
        'gru': {'h': 0, 'z': 1, 'r': 2},
        'mgu': {'h': 0, 'f': 1},
        'mingru': {'h': 0, 'z': 1, 'z_preact': 2},
        'mingru_heinsen': {'h': 0, 'z': 1, 'z_preact': 1, 'h_tilde': 2, 'h_tilde_preact': 2},
    }
    
    # check if params has the key 'params' or not
    if 'params' not in params:
        params = {'params': params}


    state_hist, out_hist = model.apply(params, x_to_plot)

    if variable_to_plot == 'o':
        full_data_to_plot = out_hist[:9]
    else: 
        full_data_to_plot = state_hist[layer_to_plot][model_LUT[model_type][variable_to_plot]][:9, :, :nb_components_to_plot]
        if variable_to_plot == 'z': 
            full_data_to_plot = jax.nn.sigmoid(full_data_to_plot)
        elif variable_to_plot == 'h_tilde':
            full_data_to_plot = g(full_data_to_plot)

    min_data = jnp.min(full_data_to_plot)*0.95
    max_data = jnp.max(full_data_to_plot)*1.05
    print(full_data_to_plot.shape)
    legend_ncol = 5 
    fig, axs = plt.subplots(6, 3, figsize=(2400*PX, 1400*PX))
    t = jnp.arange(784)
    id_sample_x = 0
    id_sample_var = 0
    for i in range(3):
        for j in range(6):
            if id_sample_x >= len(x_to_plot) or id_sample_var >= len(x_to_plot):
                break
            if j%2 == 0:
                axs[j, i].plot(t, x_to_plot[id_sample_x])
                id_sample_x += 1
            else: 
                if variable_to_plot == 'o':
                    axs[j, i].plot(t, full_data_to_plot[id_sample_var], label=[f'out_{i}' for i in range(10)])
                else:                
                    axs[j, i].plot(t, full_data_to_plot[id_sample_var][:, :nb_components_to_plot], label=[f'{variable_to_plot}_{i}' for i in range(nb_components_to_plot)])

                axs[j, i].set_ylim(min_data, max_data)
                axs[j, i].grid(True)
                id_sample_var += 1

            
        axs[j, i].legend(ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0.5, -0.1))

        if id_sample_x >= len(x_to_plot) or id_sample_var >= len(x_to_plot):
            break
    plt.suptitle(f'Digit {digit}')
    plt.tight_layout()

    os.makedirs(plt_dir, exist_ok=True)
    if variable_to_plot == 'o':
        fig_name = f'digit_{digit}_o.png'
    else:
        fig_name = f'digit_{digit}_{variable_to_plot}_l{layer_to_plot}.png' 
    plt.savefig(os.path.join(plt_dir, fig_name))
    plt.show()


def plot_all_digits(model, params, model_type, batch_inputs, batch_labels, plt_dir,
                    variable_to_plot='o', layer_to_plot=0, nb_components_to_plot=5):
    x_to_plot = jnp.zeros_like(batch_inputs[:10])
    for d in range(10):
        y_digit = batch_labels == d
        x_digit = batch_inputs[y_digit]
        x_to_plot = x_to_plot.at[d].set(x_digit[0])

    
    model_LUT = {
        'lstm': {'h': 0, 'c': 1},
        'gru': {'h': 0, 'z': 1, 'r': 2},
        'mgu': {'h': 0, 'f': 1},
        'mingru': {'h': 0, 'z': 1, 'z_preact': 2},
        'mingru_heinsen': {'h': 0, 'z': 1, 'z_preact': 1, 'h_tilde': 2, 'h_tilde_preact': 2},
    }
    
    # check if params has the key 'params' or not
    if 'params' not in params:
        params = {'params': params}

    
    state_hist, out_hist = model.apply(params, x_to_plot)
    
    if variable_to_plot == 'o':
        full_data_to_plot = out_hist
    else:
        full_data_to_plot = state_hist[layer_to_plot][model_LUT[model_type][variable_to_plot]][:, :, :nb_components_to_plot]
        if variable_to_plot == 'z': 
            full_data_to_plot = jax.nn.sigmoid(full_data_to_plot)
        elif variable_to_plot == 'h_tilde':
            full_data_to_plot = g(full_data_to_plot)

    min_data = jnp.min(full_data_to_plot)*0.95
    max_data = jnp.max(full_data_to_plot)*1.05
    print(full_data_to_plot.shape)

    legend_ncol = 5
    fig, axs = plt.subplots(6, 4, figsize=(2400*PX, 1600*PX))
    t = jnp.arange(784)
    id_sample_x = 0
    id_sample_var = 0
    for j in range(6):
        for i in range(4):
            if id_sample_x >= len(x_to_plot) or id_sample_var >= len(x_to_plot):
                break
            if j%2 == 0:
                axs[j, i].set_title(f'Digit {id_sample_x}')
                axs[j, i].plot(t, x_to_plot[id_sample_x])
                id_sample_x += 1
            else: 
                if variable_to_plot == 'o':
                    axs[j, i].plot(t, full_data_to_plot[id_sample_var], label=[f'out_{i}' for i in range(10)])
                else:
                    axs[j, i].plot(t, full_data_to_plot[id_sample_var], label=[f'{variable_to_plot}_{i}' for i in range(nb_components_to_plot)])
                axs[j, i].set_ylim(min_data, max_data)
                axs[j, i].grid(True)
                id_sample_var += 1
        if id_sample_var >= len(x_to_plot):
            break

        
    axs[-1, 0].plot(t, full_data_to_plot[8], label=[f'out_{i}' for i in range(10)])
    axs[-1, 0].set_ylim(min_data, max_data)
    axs[-1, 0].grid(True)
    axs[-1, 1].plot(t, full_data_to_plot[9], label=[f'out_{i}' for i in range(10)])
    axs[-1, 1].set_ylim(min_data, max_data)
    axs[-1, 1].grid(True)
    
    for i in range(4):
        axs[-1, i].legend(ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    if variable_to_plot == 'o':
        fig_name = f'all_digits_o.png'
    else:
        fig_name = f'all_digits_{variable_to_plot}_l{layer_to_plot}.png'
    os.makedirs(plt_dir, exist_ok=True)
    plt.savefig(os.path.join(plt_dir, fig_name))
    plt.show()

def plot_cm(lbls, preds, plt_dir):
    # plot confusion matrix
    cm = confusion_matrix(lbls, preds)
    mask = 1 - np.eye(cm.shape[0], dtype=bool)
    cm_masked = cm * mask
    # find the 90th percentile
    percentile = np.percentile(cm_masked, 90)
    fig, axs = plt.subplots(1,2,figsize=(1000*PX, 500*PX))
    cax = axs[0].matshow(cm, cmap='Blues_r')
    cax_offdiag = axs[1].matshow(cm * mask, cmap='Blues_r')
    fig.colorbar(cax)
    fig.colorbar(cax_offdiag)
    # add the text
    for i in range(10):
        for j in range(10):
            color = 'black' if i == j else 'white'
            axs[0].text(i, j, str(cm[j, i]), va='center', ha='center', color=color)
            count = cm_masked[j, i]
            if count > percentile:
                color = 'red'
            axs[1].text(i, j, str(count), va='center', ha='center', color=color)

    # make x_tick and y_tick every digit
    for i in range(2):
        axs[i].set_xticks(np.arange(10))
        axs[i].set_yticks(np.arange(10))
        axs[i].set_xlabel('Predicted')
        axs[i].set_ylabel('True')
    plt.suptitle('Confusion Matrix')
    plt.tight_layout()
    os.makedirs(plt_dir, exist_ok=True)
    fig_name = 'confusion_matrix.png'
    plt.savefig(os.path.join(plt_dir, fig_name))

    plt.show()

def compute_classifcation_report(lbls, preds, sort=True):
    df = classification_report(lbls, preds, output_dict=True)
    df = pd.DataFrame(df).T
    # multiply by 100 to get percentage all columns except support
    df.iloc[:, :-1] *= 100
    df = df.round(2)
    df_summary = df.iloc[-2:, :-1]
    acc = df.iloc[-3, -2]
    df = df.iloc[:-3, :-1]
    df = df.sort_values(by='f1-score', ascending=False) if sort else df
    return df, df_summary, acc
