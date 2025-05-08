import jax 
import jax.numpy as jnp
from functools import partial
import optax
from tqdm import tqdm
import numpy as np
from flax.training import train_state
from time import time
@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@partial(jax.jit, static_argnames=('reg_factor','model'))
def apply_model(state, model, x, y, reg_factor, do_key):
    """Computes gradients, loss and accuracy for a single batch."""
    # do_key = jax.random.fold_in(do_key, state.step)
    def loss_fn(params):
        net_dyn, out_hist = model.apply({'params': params}, x, rngs={'dropout': do_key})
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(y, 10)
        batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        reg = 0.0
        for layers in net_dyn:
            reg += jnp.where(jnp.abs(layers[2]) > 1, layers[2]**2, 0.0).sum() # h_tilde_preact
            # reg += jnp.where(jnp.abs(layers[1]) > 1, layers[1]**2, 0.0).sum() # z_preact
        reg += jnp.where(jnp.abs(out_hist) > 1, out_hist**2, 0.0).sum()
        loss = jnp.mean(batch_loss) + reg_factor * reg
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

def run_epoch(state, model_cls, train_dl, key, reg_factor, lim_batch=None, keys_to_track=None, inner_keys_to_track=None):
    """Train for a single epoch."""
    model = model_cls(training=True)
    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = {k: [] for k in keys_to_track+inner_keys_to_track}
    batch_id = 0
    break_flag = False
    key, do_key = jax.random.split(key)
    for batch_x, batch_y in progress_bar:
        # start = time()
        grads, loss, accuracy, aux_dict = apply_model(state, model, batch_x, batch_y, reg_factor=reg_factor, do_key=do_key)
        # stop = time()
        # print("forward pass time:", stop-start)
        for k in keys_to_track:
            aux_dict_hist[k].append(locals()[k])
        for k in inner_keys_to_track:
            aux_dict_hist[k].append(aux_dict[k])

        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
        
        if jnp.isnan(loss).sum() > 0:
            print('NAN Loss')
            break_flag = True
            break
        # start = time()
        state = update_model(state, grads)
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

@partial(jax.jit, static_argnames=('model'))
def eval_model(state, model, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        _, out_hist = model.apply({'params': params}, images)
        logits = out_hist.mean(axis=1)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    loss, logits = loss_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy

def validate(state, model, testloader):
    # Compute average loss & accuracy
    model = model(training=False) # needed when using dropout
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(testloader):
        loss, acc = eval_model(
            state, model, inputs, labels # from S4D: , model, classification=classification
        )
        losses.append(loss)
        accuracies.append(acc)
    return np.mean(losses), np.mean(accuracies)



def create_train_state(key, model_cls, lr, dataset_version, seq_len, batch_size):
    
    init_x = jnp.ones((batch_size, seq_len, 1)) if dataset_version == "sequential" else jnp.ones((batch_size, jnp.sqrt(seq_len), jnp.sqrt(seq_len)))

    model = model_cls(training=True)
    key, pkey, do_key = jax.random.split(key, 3)
    params = model.init({'params': pkey, 'dropout': do_key}, init_x)['params']
    
    # Debugging: Print parameter structure
    print("Initialized parameter structure:", jax.tree_util.tree_map(jnp.shape, params))

    param_sizes = map_nested_fn(
        lambda k, param: param.size
        # if lr_layer.get(k, lr) > 0.0
        # else 0
    )(params)
    n_params = sum(jax.tree_util.tree_leaves(param_sizes))
    print(f"[*] Trainable Parameters: {n_params}")

    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adamw(lr, weight_decay=1e-2),
    )
    key, do_key = jax.random.split(key)
    # class TrainState(train_state.TrainState):
    #     key: jax.Array
        
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        # key=do_key
    ), n_params, params