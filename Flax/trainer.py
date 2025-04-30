import jax
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm
from flax.training import train_state, checkpoints
import optax
import os
from utils import map_nested_fn, round_weights_to_nearest
from model import apply_model, update_model, eval_model


def run_epoch(state, train_dl, rng, args, lim_batch=None, keys_to_track=None, inner_keys_to_track=None):
    """Train for a single epoch."""

    epoch_loss = []
    epoch_accuracy = []
    progress_bar = tqdm(train_dl, desc="Training", leave=True)
    aux_dict_hist = {k: [] for k in keys_to_track+inner_keys_to_track}
    batch_id = 0
    break_flag = False
    for batch_x, batch_y in progress_bar:
        grads, loss, accuracy, aux_dict = apply_model(state, batch_x, batch_y, reg_factor=args.reg_factor)
        
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

        state = update_model(state, grads)
        
        # clipping time constants
        if args.train_gate_decay:
            lower_bound = 1.0
            upper_bound = 1.5 * args.time_decay_mean[0]
            state = state.replace(params=jax.tree_map(
                lambda p: jnp.clip(p, lower_bound, upper_bound) if 'Time_constants_gate' in p else p,
                state.params
            ))

        batch_id += 1

        if batch_id % 3 == 0:
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())       

        if lim_batch is not None and batch_id >= lim_batch:
            break_flag = True
            break 
        
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy, (break_flag, aux_dict_hist)


def validate(state, testloader):
    # Compute average loss & accuracy
    # model = model(training=False) # needed when using dropout
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(testloader):
        loss, acc = eval_model(
            state, inputs, labels # from S4D: , model, classification=classification
        )
        losses.append(loss)
        accuracies.append(acc)
    return np.mean(losses), np.mean(accuracies)



def create_train_state(key, model_cls, dataset_version, args):
    
    init_x = jnp.ones((args.batch_size, 784, 1)) if dataset_version == "sequential" else jnp.ones((args.batch_size, 28, 28))

    model = model_cls(output_size=10, args=args)
    params = model.init(key, init_x)['params']
    
    # Debugging: Print parameter structure
    print("Initialized parameter structure:", jax.tree_util.tree_map(jnp.shape, params))

    param_sizes = map_nested_fn(
        lambda k, param: param.size
        # if lr_layer.get(k, lr) > 0.0
        # else 0
    )(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-2),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def train(key, state, trainloader, val_loader, testloader, args, result_dir, cktp_dir):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # 'batch_loss' = [loss_sample1, loss_sample2, ..., loss_sampleBS]
    # 'loss' = [loss_batch1, loss_batch2, ..., loss_batchN] where N is the number of batches and loss_batchX = mean([loss_sample1, loss_sample2, ..., loss_sampleBS])
    keys_to_track = ['grads', 'loss', 'state'] #['grads', 'loss', 'state', 'batch_x', 'batch_y'] # ['batch_x', 'batch_y', 'state', 'grads']
    inner_keys_to_track = [] #['batch_loss', 'net_dyn']
    aux_dict_training = []

    best_val_acc = 0.0
    improvement = 0.01 # 1%, minimum improvement to save checkpoint

    async_manager = checkpoints.AsyncManager()

    for epoch in range(args.n_epochs):
        key, subkey = jax.random.split(key) # not used in run_epoch (TODO: remove?)
        state, train_loss, train_accuracy, (break_flag, aux_dict_epoch) = run_epoch(state, trainloader, key, args, lim_batch=None, keys_to_track=keys_to_track, inner_keys_to_track=inner_keys_to_track)
        aux_dict_training.append(aux_dict_epoch)
        
        if args.round_weights:
            state = state.replace(params=round_weights_to_nearest(state.params, step=0.5))

        if break_flag:
            break
        val_loss, val_acc  = validate(state, val_loader)
        if val_acc > best_val_acc + improvement:
            best_val_acc = val_acc
            if best_val_acc > 0.97: improvement = 0.001 # 0.1%
            checkpoints.save_checkpoint(ckpt_dir=cktp_dir, target=state, step=state.step, overwrite=True, async_manager=async_manager)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | train_acc: {train_accuracy*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%")


    test_loss, test_acc  = validate(state, testloader)
    print(f"Final Test | test_loss: {test_loss:.4f} | test_acc: {test_acc*100:.2f}%")

    final_state = state
    # Save training dynamics
    np.savez(os.path.join(result_dir, 'training_dynamics.npz'), 
            train_losses=train_losses, 
            train_accuracies=train_accuracies, 
            val_losses=val_losses, 
            val_accuracies=val_accuracies)
    np.savez(os.path.join(result_dir, 'test_loss.npz'),
             test_loss=test_loss, test_acc=test_acc)
    
    return train_losses, train_accuracies, val_losses, val_accuracies, test_loss, test_acc, aux_dict_training, final_state