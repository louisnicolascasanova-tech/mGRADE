import jax
import jax.numpy as jnp
import torch
import torchaudio
import torchvision
from torch.utils.data import DataLoader, Dataset
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

import urllib.request
import tarfile
from pathlib import Path
import random
from typing import Tuple, Optional, List
import numpy as np

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


def write_config_yaml(config_file_path, CKPT_DIR):
    """
    Copy the original configuration YAML file to CKPT_DIR.
    """
    import shutil
    config_path = os.path.join(CKPT_DIR, 'config.yaml')
    shutil.copy2(config_file_path, config_path)
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


def prep_batch(batch, seq_len, in_dim, dtype=jnp.float32):
    """Take a batch and convert it to a standard x/y format"""
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Unhandled data type. ")

    inputs = jnp.array(inputs.numpy()).astype(dtype)  # convert to jax (float32)
    targets = jnp.array(targets.numpy()).astype(jnp.float32)  # convert to jax (int32)
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

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(dtype), lengths.astype(dtype))
    else:
        full_inputs = inputs.astype(dtype)

    return full_inputs, targets


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
    elif args.conv == 'conv':
        wavenet_str = 'eerf' if args.wavenet_dilation else 'lerf'
        schedule_str = args.dilation_schedule if args.dilation_schedule is not None else 'F'
        conv_str = f'conv{args.kernel_size}{wavenet_str}sch{schedule_str}'
    else:
        conv_str = 'F'
    
    # Channel mixing config string
    if args.channel_mixing == 'glu':
        cm_str = f"cm{args.channel_mixing}{args.glu_type}"
    elif args.channel_mixing == 'mlp':
        cm_str = f"cm{args.channel_mixing}{args.cm_act}"
    else:
        cm_str = "cmF"
    
    # Heterogeneous config string
    hetero_parts = [
        f"hetW{bool_to_str(getattr(args, 'heterogeneous_weights', False))}",
        f"S{bool_to_str(getattr(args, 'heterogeneous_weights', False))}",
        f"trainW{bool_to_str(getattr(args, 'train_weights', False))}",
        f"S{bool_to_str(getattr(args, 'train_std', False))}",
        f"P{bool_to_str(getattr(args, 'train_positions', False))}"
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


class SpeechCommandsDataset(Dataset):
    """
    Google Speech Commands Dataset for PyTorch
    
    Args:
        root (str): Root directory where dataset will be stored
        subset (str): 'training', 'validation', or 'testing'
        version (str): 'v1' or 'v2' 
        download (bool): Whether to download the dataset if not found
        transform (callable, optional): Optional transform to be applied on audio
        sample_rate (int): Target sample rate for audio
        max_length (int): Maximum length of audio in samples (pad/truncate)
    """
    
    # Dataset URLs
    URLS = {
        'v1': 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
        'v2': 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    }
    
    # Class labels for each version
    CLASSES_V1 = [
        'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow'
    ]
    
    CLASSES_V2 = CLASSES_V1 + ['backward', 'forward', 'follow', 'learn', 'visual']
    
    def __init__(self, 
                 root: str = './data',
                 subset: str = 'training',
                 version: str = 'v2',
                 download: bool = True,
                 transform: Optional[callable] = None,
                 sample_rate: int = 16000,
                 max_length: int = 16000):
        
        self.root = Path(root)
        self.subset = subset
        self.version = version.lower()
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Set class labels based on version
        self.classes = self.CLASSES_V2 if version == 'v2' else self.CLASSES_V1
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / f'speech_commands_{self.version}'
        
        if download:
            self._download()
        
        # Load file paths and labels
        self.data = self._load_data()
        
    def _download(self):
        """Download and extract the dataset"""
        if self.data_dir.exists() and len(list(self.data_dir.glob('*'))) > 0:
            print(f"Dataset already exists at {self.data_dir}")
            return
            
        url = self.URLS[self.version]
        filename = url.split('/')[-1]
        filepath = self.root / filename
        
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        
        print(f"Extracting {filepath}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(self.data_dir)
        
        # Clean up
        filepath.unlink()
        print("Download complete!")
        
    def _load_data(self) -> List[Tuple[Path, int]]:
        """Load file paths and corresponding labels"""
        data = []
        
        # Load validation and test splits if they exist
        validation_list = self.data_dir / 'validation_list.txt'
        testing_list = self.data_dir / 'testing_list.txt'
        
        validation_files = set()
        testing_files = set()
        
        if validation_list.exists():
            with open(validation_list, 'r') as f:
                validation_files = set(line.strip() for line in f)
                
        if testing_list.exists():
            with open(testing_list, 'r') as f:
                testing_files = set(line.strip() for line in f)
        
        # Collect all audio files
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for audio_file in class_dir.glob('*.wav'):
                relative_path = f"{class_name}/{audio_file.name}"
                
                # Determine subset
                if self.subset == 'validation' and relative_path in validation_files:
                    data.append((audio_file, class_idx))
                elif self.subset == 'testing' and relative_path in testing_files:
                    data.append((audio_file, class_idx))
                elif self.subset == 'training' and relative_path not in validation_files and relative_path not in testing_files:
                    data.append((audio_file, class_idx))
        
        # Add background noise as a separate class (optional)
        background_dir = self.data_dir / '_background_noise_'
        if background_dir.exists() and 'silence' not in self.classes:
            # You can add silence/noise as an additional class if needed
            pass
            
        print(f"Loaded {len(data)} samples for {self.subset} subset")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            audio (torch.Tensor): Audio waveform of shape (1, max_length)
            label (int): Class label
        """
        audio_path, label = self.data[idx]
        
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or truncate to fixed length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply transforms if any
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform.transpose(1,0), label

def create_speechcommands35_classification_dataset(
        bsz: int = 32,
        root: str = './data',
        dtype: jnp.dtype = jnp.float32,
        sample_rate: int = 16000,
        max_length: int = 16000,
        download: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """
    Create DataLoaders for training, validation, and testing sets
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    N_CLASSES, SEQ_LENGTH, IN_DIM = 35, max_length, 1
    # Common transforms (you can customize these)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )
    
    # Create datasets
    train_dataset = SpeechCommandsDataset(
        root=root, subset='training', version='v2',
        download=download, transform=None,  # Apply transforms later if needed
        sample_rate=sample_rate, max_length=max_length
    )
    
    val_dataset = SpeechCommandsDataset(
        root=root, subset='validation', version='v2',
        download=False, transform=None,
        sample_rate=sample_rate, max_length=max_length
    )
    
    test_dataset = SpeechCommandsDataset(
        root=root, subset='testing', version='v2',
        download=False, transform=None,
        sample_rate=sample_rate, max_length=max_length
    )
    
    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        labels = jnp.array(transposed_data[1])
        waveforms = jnp.array(transposed_data[0], dtype=dtype)  # Shape: (batch_size, max_length, 1)

        return waveforms, labels       

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=bsz, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=False
    )

    return train_loader, val_loader, test_loader, N_CLASSES, SEQ_LENGTH, IN_DIM


def create_uea_classification_dataset(dataset_name, bsz=128, data_dir="./data_dir", dtype=jnp.float32, seed=42):
    """
    Create PyTorch dataloaders for any UEA dataset.

    Args:
        dataset_name: Name of the UEA dataset (e.g., 'EigenWorms', 'Heartbeat', 'SCP1', 'SCP2', 'MotorImagery', 'Ethanol')
        bsz: Batch size
        data_dir: Root directory containing processed data
        dtype: Data type for JAX arrays (float16 or float32)
        seed: Random seed for reproducibility

    Returns:
        trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM
    """
    import pickle

    print(f"[*] Generating {dataset_name} Classification Dataset...")

    # Load the processed data
    dataset_path = Path(data_dir) / "processed" / "UEA" / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    with open(dataset_path / "data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(dataset_path / "labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # Convert to numpy arrays if needed
    data = np.array(data)
    labels = np.array(labels)

    # Get dataset constants
    N_SAMPLES, SEQ_LENGTH, IN_DIM = data.shape
    N_CLASSES = len(np.unique(labels))

    print(f"    Dataset size: {N_SAMPLES} samples")
    print(f"    Sequence length: {SEQ_LENGTH}")
    print(f"    Input dimension: {IN_DIM}")
    print(f"    Number of classes: {N_CLASSES}")

    # Create a simple Dataset class
    class SimpleDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Split the dataset into train, val, test (70%, 15%, 15%)
    n_train = int(N_SAMPLES * 0.7)
    n_val = int(N_SAMPLES * 0.15)

    # Shuffle indices with seed for reproducibility
    np.random.seed(seed)
    indices = np.random.permutation(N_SAMPLES)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Create datasets
    train = SimpleDataset(data[train_idx], labels[train_idx])
    val = SimpleDataset(data[val_idx], labels[val_idx])
    test = SimpleDataset(data[test_idx], labels[test_idx])

    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        labels = jnp.array(transposed_data[1])
        sequences = jnp.array(transposed_data[0], dtype=dtype)

        return sequences, labels

    # Return data loaders, with the provided batch size
    trainloader = DataLoader(
        train, batch_size=bsz, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
    )
    valloader = DataLoader(
        val, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )
    testloader = DataLoader(
        test, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM
