import jax 
import jax.numpy as jnp
from functools import partial
import optax
from tqdm import tqdm
import numpy as np
from flax.training import train_state
from time import time
import wandb
from flax.traverse_util import flatten_dict, unflatten_dict
from utils import prep_batch

@jax.jit
def update_model(state, grads, kernel_size):
    state = state.apply_gradients(grads=grads)
    def clip_negative_positions(params):
        def clip_fn(path, v):
            # path is a tuple of keys, e.g. ('Dense_0', 'DCLS', 'positions')
            if path[-1] == 'positions':
                v = jnp.where(v < 0, 0, v)
                v = jnp.where(v > kernel_size, kernel_size, v)
            return v
        flat_params = flatten_dict(params, sep='/')
        clipped_flat = {k: clip_fn(k.split('/'), v) for k, v in flat_params.items()}
        return unflatten_dict(clipped_flat, sep='/')

    state = state.replace(params=clip_negative_positions(state.params))
    return state


@partial(jax.jit, static_argnames=('model','reg_factor'))
def apply_model(state, model, x, y, reg_factor, do_key):
    """Computes gradients, loss and accuracy for a single batch."""
    # do_key = jax.random.fold_in(do_key, state.step)
    def loss_fn(params):
        net_dyn, out_hist = model.apply({'params': params}, x, rngs={'dropout': do_key})
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(y, model.out_dim)
        batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        reg = 0.0
        for layers in net_dyn:
            reg += jnp.where(jnp.abs(layers[2]) > 1, layers[2]**2, 0.0).sum() # h_tilde_preact
            # reg += jnp.where(jnp.abs(layers[1]) > 1, layers[1]**2, 0.0).sum() # z_preact
        reg += jnp.where(jnp.abs(out_hist) > 1, out_hist**2, 0.0).sum()
        loss = jnp.mean(batch_loss) #+ reg_factor * reg
        return loss, {'logits': logits, 'batch_loss': batch_loss, 'net_dyn': net_dyn}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_dict), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(aux_dict['logits'], -1) == y)
    aux_dict.pop('logits')
    return grads, loss, accuracy, aux_dict

def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn

def run_epoch(state, model_cls, train_dl, key, reg_factor, kernel_size, lim_batch=None, keys_to_track=None, inner_keys_to_track=None, lr_fn=None,
              wandb_gradients=False, in_dim=None, seq_len=None):
    """Train for a single epoch."""
    model = model_cls(training=True)
    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = {k: [] for k in keys_to_track+inner_keys_to_track}
    batch_id = 0
    break_flag = False
    key, do_key = jax.random.split(key)
    grads_previous = None
    for batch in progress_bar:
        if in_dim is None: 
            batch_x, batch_y = batch
        else:
            batch_x, batch_y, mask = prep_batch(batch, seq_len, in_dim)
        # start = time()
        grads, loss, accuracy, aux_dict = apply_model(state, model, batch_x, batch_y, reg_factor=reg_factor, do_key=do_key)
        # stop = time()
        # print("forward pass time:", stop-start)
        for k in keys_to_track:
            aux_dict_hist[k].append(locals()[k])
        for k in inner_keys_to_track:
            aux_dict_hist[k].append(aux_dict[k])

        if wandb_gradients:
            # Flatten the gradient tree into key-value pairs
            flat_grads = flatten_dict(grads, sep='/')
            for key, grad in flat_grads.items():
                if grad is not None:
                    if jnp.isnan(grad).sum() > 0:
                        print(f'NaN in gradients: {key}')
                    flat_grad = jnp.ravel(grad)
                    wandb.log({f"gradients/{key}": wandb.Histogram(flat_grad)}, step=state.step)

        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
        
        if jnp.isnan(loss).sum() > 0:
            print('NAN Loss')
            # check if ther is a NaN in the gradients
            # if grads_previous is not None:
            #     for k, v in grads_previous.items():
            #         if jnp.isnan(v).sum() > 0:
            #             print(f'NaN in gradients: {k}')
            flat_grads = flatten_dict(grads, sep='/')
            for key, grad in flat_grads.items():
                if grad is not None:
                    if jnp.isnan(grad).sum() > 0:
                        print(f'NaN in gradients: {key}')
                    # flat_grad = jnp.ravel(grad)
                    # wandb.log({f"gradients/{key}": wandb.Histogram(flat_grad)}, step=state.step)
            for l, ldyn in enumerate(aux_dict['net_dyn']):
                for i in range(3):
                    print(f"Layer {l} | {i} | {ldyn[i].shape} | {ldyn[i].min()} | {ldyn[i].max()} | {ldyn[i].mean()}")

            break_flag = True
            break
        # start = time()
        state = update_model(state, grads, kernel_size)
        grads_previous = grads
        # print(f"lr: {lr_fn(state.step)}")
        # stop = time()
        # print("update pass time:", stop-start)
        batch_id += 1

        if batch_id % 3 == 0:
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())       

        if lim_batch is not None and batch_id >= lim_batch:
            break_flag = True
            break 
        
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy, (break_flag, aux_dict_hist)

@partial(jax.jit, static_argnames=('model', 'out_dim'))
def eval_model(state, model, images, labels, out_dim):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        _, out_hist = model.apply({'params': params}, images)
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(labels, out_dim)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    loss, logits = loss_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy

def validate(state, model, testloader, seq_len, in_dim, out_dim):
    # Compute average loss & accuracy
    model = model(training=False) # needed when using dropout
    losses, accuracies = [], []
    for batch_idx, batch in enumerate(testloader):
        inputs, labels, _ = prep_batch(batch, seq_len, in_dim)
        loss, acc = eval_model(
            state, model, inputs, labels, out_dim # from S4D: , model, classification=classification
        )
        losses.append(loss)
        accuracies.append(acc)
    return np.mean(losses), np.mean(accuracies)


def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., 
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch)
    
    cosine_epochs = max(config.n_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch,
        alpha=config.alpha_cosine)
    print(config.alpha_cosine)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])
    return schedule_fn


def create_learning_rate_map(args, steps_per_epoch):
    lr_fn = create_learning_rate_fn(args, args.lr, steps_per_epoch) if args.scheduler else args.lr
    print(lr_fn)
    none_params = []
    adam_params = []
    adamw_params = []
    if args.train_std:
        adam_params.append('std')
    else: 
        none_params.append('std')
    if args.train_weights:
        adamw_params.append('weights')
    else:
        none_params.append('weights')
    if args.train_positions:
        adam_params.append('positions')
    else:
        none_params.append('positions')
    adam_params.append('bias')
    adamw_params.append('kernel')
    lr_map = {
        'none': {'keys': none_params, 'tx': optax.set_to_zero()},
        'adam': {'keys': adam_params, 'tx': optax.adam(lr_fn)},
        'adamw': {'keys': adamw_params, 'tx': optax.adamw(lr_fn, weight_decay=args.weight_decay)}
    }
    return lr_map, lr_fn


def init_model(key, model_cls, dataset_version, in_dim, seq_len, batch_size):
    
    
    init_x = jnp.ones((batch_size, seq_len, in_dim)) if dataset_version == "sequential" else jnp.ones((batch_size, jnp.sqrt(seq_len), jnp.sqrt(seq_len)))

    model = model_cls(training=True)
    key, pkey, do_key = jax.random.split(key, 3)
    params = model.init({'params': pkey, 'dropout': do_key}, init_x)['params']
    return model, params


def create_train_state(key, model_cls, lr_map, dataset_version, in_dim, seq_len, batch_size, wd=0.05):
    
    """Creates initial `TrainState`."""
    model, params = init_model(key, model_cls, dataset_version, in_dim, seq_len, batch_size)
    
    # Debugging: Print parameter structure
    print("Initialized parameter structure:", jax.tree_util.tree_map(jnp.shape, params))

    param_sizes = map_nested_fn(lambda k, param: param.size)(params)
    n_params = sum(jax.tree_util.tree_leaves(param_sizes))
    print(f"[*] Trainable Parameters: {n_params}")

    def label_fn(params):
        flat = flatten_dict(params, sep='/')
        labels = {}
        for path, _ in flat.items():
            name = path.split('/')[-1]  # last part of the path
            if name in lr_map['none']['keys']:
                labels[path] = 'none'
            elif name in lr_map['adam']['keys']:
                labels[path] = 'adam'
            else:
                labels[path] = 'adamw'
        return unflatten_dict(labels, sep='/')

    # 2. Define optimizer transformations
    tx = optax.multi_transform(
        {
            'adamw': lr_map['adamw']['tx'],  # weight decay
            'adam': lr_map['adam']['tx'],  # adam
            'none': lr_map['none']['tx'],  # frozen
        },
        param_labels=label_fn  # returns pytree of labels
    )

    key, do_key = jax.random.split(key)
        
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        # key=do_key
    ), n_params, params