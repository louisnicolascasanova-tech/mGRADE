import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]='0.95'

def run_parameter_sweep():
    import subprocess

    # Define the parameter ranges
    total_params = [10000, 20000]
    layers = [2, 3, 4, 5]
    seeds = [0, 1]
    model_codes = ["alllayer_dyn", "nolayer_dyn", "firstlayer_dyn"]#13 symbols
    steepness = [10, 20, 100, 1000, 10000, "hard"]
    hidden_dims = [[67, 48, 39, 34],
                  [96, 69, 56, 49]]

    # Create combinations of parameters
    #param_combinations = itertools.product(total_params, layers, seeds, decay_values)
    for j in range(len(seeds)):
        for i in range(len(total_params)):
            for k in range(len(layers)):
                for t in range(len(steepness)):
                    for l in range(len(model_codes)):

                        hidden_dim = hidden_dims[i][k]
                        n_layers = layers[k]
                        seed = seeds[j]

                        if model_codes[l] == 'firstlayer_dyn':
                            decay = "0"*n_layers
                            decay = "1" + decay[1:]
                        elif model_codes[l] == 'nolayer_dyn':
                            decay = "0"*n_layers
                        elif model_codes[l] == 'alllayer_dyn':
                            decay = "1"*n_layers
                        
                        model_code = model_codes[l]+f"_stpns_{steepness[t]}"
                        print("Decay:", str(decay))
                        
                        # Construct the command to run the main function with the specified parameters
                        reg_factor = 0.0
                        print("steepness", steepness[t], "model_code", model_code)
                        if steepness[t] == "hard":
                            command = [
                                'python', 'flax4b_minGRU_heinsen.py',
                                '--dataset', 'mnist',
                                '--hidden_dim', str(hidden_dim),
                                '--n_layers', str(n_layers),
                                '--seed', str(seed),
                                '--reg_factor', str(reg_factor),
                                '--model_code', str(model_code),
                                '--decay', str(decay),
                                '--hidden_activation', 'threshold']
                        else:
                            command = [
                                'python', 'flax4b_minGRU_heinsen.py',
                                '--dataset', 'mnist',
                                '--hidden_dim', str(hidden_dim),
                                '--n_layers', str(n_layers),
                                '--seed', str(seed),
                                '--reg_factor', str(reg_factor),
                                '--model_code', str(model_code),
                                '--decay', str(decay),
                                '--sigmoid_scale', str(steepness[t]),
                                '--hidden_activation', 'scaled_sigmoid']
                        
                        # Execute the command
                        subprocess.run(command)

if __name__ == "__main__":
    run_parameter_sweep()