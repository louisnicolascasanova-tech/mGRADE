# Den-MinGRU

# Backbone
...
# MinGRU
...
# DCLS
...
# Running the code
The configuration of each experiment is written in a specific `yaml` file. 
- `cifar_conv_eerf.yaml`: backbone with wavenet TCN style
- `cifar_conv_lerf.yaml`: backbone with vanilla TCN style
- `cifar_dcls.yaml`: DCLS style
- `cifar_tcn.yaml`: pure vanilla TCN (MISSING)
- `cifar_wavnenet.yaml`: pure wavenet (MISSING)

This organization allows to modify the independent paramters (depth, width, kernel size ...) directly in the `yaml`, while keeping the dependent parameters (conv_mode, conv_schedule ...) already set in the different `yaml` files.

**Running the code**: 

``` bash
python main.py --dataset <dataset_name> --gpu <gpu_id> --conv_mode <conv_mode> 
```
```python
dataset_name = 'mnist' or 'cifar'
gpu_id = 0 or 1 or 2 or 3
conv_mode = 'dcls' or 'conv_eerf' or 'conv_lerf'
```



# Monitoring internal variables 

`RNN_General_Backbone` only tracks the state of the minGRU block. 

`RNN_General_Backbone_Monitored` monitors every internal variable of the network. 
```python
monitor = {
    'encoder_out': (B, 4096, H),
    'final_output': (B, 4096, NUM_CLASSES),
    'layers': [
        {
            'cm_input': (B, 4096, H),
            'cm_output': (B, 4096, H),
            'cm_skip': (B, 4096, H),
            'conv_input': (B, 4096, H),
            'conv_output': (B, 4096, H),
            'conv_skip': (B, 4096, H),
            'final_layer_output': (B, 4096, H),
            'input': (B, 4096, H),
            'layer_skip_output': (B, 4096, H),
            'layer_skip_source': (B, 4096, H),
            'postnorm_output': (B, 4096, H),
            'rec_input': (B, 4096, H),
            'rec_ln_output': (B, 4096, H),
            'rec_output': (B, 4096, H),
            'rec_skip': (B, 4096, H),
        },
        ...
    ]

}