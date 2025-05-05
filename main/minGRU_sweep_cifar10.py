import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]='0.95'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]='false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]='platform'

def run_parameter_sweep():
    import subprocess

    # Define the parameter ranges
    total_params = [500000,750000, 1000000, 1500000, 2000000]
    layers = [4, 5, 6, 7, 8, 9, 10]#[2, 3, 4, 5, 6, 7, 8]
    seeds = [5, 6]
    model_codes = ["alllayer_dyn", "firstlayer_dyn", "nolayer_dyn"]
    hidden_dims = [[287, 249, 222, 203, 188, 176, 166],
                   [352, 305, 273, 249, 230, 216, 203],
                   [407, 352, 315, 288, 266, 249, 235],
                    [498, 431, 386, 352, 326, 305, 288],
                    [576, 499, 446, 407, 377, 353, 332]]

    """[[25, 18, 15, 13, 12, 11, 10],
                  [38, 27, 23, 20, 18, 16, 15],
                  [46, 33, 27, 24, 21, 19, 18],
                  [67, 48, 39, 34, 30, 28, 26],
                  [96, 69, 56, 49, 44, 40, 37]]"""

    # Create combinations of parameters
    #param_combinations = itertools.product(total_params, layers, seeds, decay_values)
    for j in range(len(seeds)):
        for i in range(len(total_params)):
            for k in range(len(layers)):
                for l in range(len(model_codes)):
                    hidden_dim = hidden_dims[i][k]
                    n_layers = layers[k]
                    seed = seeds[j]
                    model_code = model_codes[l]
                    # Construct the command to run the main function with the specified parameters
                    reg_factor = 1e-8
                    command = [
                        'python', 'flax4b_minGRU_heinsen.py',
                        '--dataset', 'cifar10',
                        '--hidden_dim', str(hidden_dim),
                        '--n_layers', str(n_layers),
                        '--seed', str(seed),
                        '--reg_factor', str(reg_factor),
                        '--gpu_index', str(3),
                        '--model_code', str(model_code)
                    ]
                    # Execute the command
                    subprocess.run(command)

if __name__ == "__main__":
    run_parameter_sweep()