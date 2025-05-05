import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]='0.95'

def run_parameter_sweep():
    import subprocess

    # Define the parameter ranges
    total_params = [1700, 3500, 5000, 10000, 20000]
    layers = [2, 3, 4, 5, 6, 7, 8]
    seeds = [0, 1, 2]
    model_codes = ["alllayer_dyn", "firstlayer_dyn", "nolayer_dyn"]
    hidden_dims = [[25, 18, 15, 13, 12, 11, 10],
                  [38, 27, 23, 20, 18, 16, 15],
                  [46, 33, 27, 24, 21, 19, 18],
                  [67, 48, 39, 34, 30, 28, 26],
                  [96, 69, 56, 49, 44, 40, 37]]

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
                        '--dataset', 'mnist',
                        '--hidden_dim', str(hidden_dim),
                        '--n_layers', str(n_layers),
                        '--seed', str(seed),
                        '--reg_factor', str(reg_factor),
                        '--model_code', str(model_code)]
                    # Execute the command
                    subprocess.run(command)

if __name__ == "__main__":
    run_parameter_sweep()