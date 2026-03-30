import numpy as np 

# HGRN
datasets = {
    'listops': {
        'd_in': 18,
        'n_classes': 10,
        'L': 6,
        'd': 32,
        'glu_expansion': 1
    },
    'imdb': {
        'd_in': 98,
        'n_classes': 2,
        'L': 4,
        'd': 128,
        'glu_expansion': 1
    },
    'aan': {
        'd_in': 135,
        'n_classes': 2,
        'L': 2,
        'd': 64,
        'glu_expansion': 2
    },
    'image': {
        'd_in': 256,
        'n_classes': 10,
        'L': 6,
        'd': 512,
        'glu_expansion': 1
    },
    'path': {
        'd_in': 256,
        'n_classes': 2,
        'L': 6,
        'd': 128,
        'glu_expansion': 1
    },
}

for dataset_name, cfg in datasets.items():
    print(dataset_name, type(cfg))
    encoding = cfg['d_in'] * cfg['d']
    hgrn_input = 2*cfg['d']**2
    hgrn_gate = hgrn_input
    hgrn_lambda = cfg['d']**2
    hgrn_output = hgrn_input
    hgrn_ln = cfg['d']
    glu_l1 = cfg['d']**2 * cfg['glu_expansion']
    glu_l2 = glu_l1
    glu_l3 = glu_l1
    glu_ln = cfg['d']
    if dataset_name == 'aan':
        decoding_1 = cfg['d']**2
        decoding_2 = 0.5 * cfg['d']**2
        decoding_out = 0.5 * cfg['d'] * cfg['n_classes']
        decoding = decoding_1 + decoding_2 + decoding_out
    else:
        decoding = cfg['d'] * cfg['n_classes']
    tot = encoding + (hgrn_input + hgrn_gate + hgrn_lambda + hgrn_output + hgrn_ln + glu_l1 + glu_l2 + glu_l3 + glu_ln) * cfg['L'] + decoding
    print(f"{dataset_name}: {tot} params, {tot/1e6}M params")
    print(f"  Encoding: {encoding}")
    print(f"  Decoding: {decoding}")



# dss
# listops: 
# imdb:
# aan:
# image:
# path: 600,320 + 128 + 256

print("\n\n")
print("DSS PARAMS")
# DSS
datasets = {
    'listops': {
        'd_in': 18,
        'n_classes': 10,
        'd': 128,
        'seq_params': 201984
    },
    'imdb': {
        'd_in': 98,
        'n_classes': 2,
        'd': 128,
        'seq_params': 134912
    },
    'aan': {
        'd_in': 135,
        'n_classes': 2,
        'd': 256,
        'seq_params': 600320
    },
    'image': {
        'd_in': 1,
        'n_classes': 10,
        'd': 512,
        'seq_params': 1985280
    },
    'path': {
        'd_in': 1,
        'n_classes': 2,
        'd': 256,
        'seq_params': 600320
    },
}

for dataset_name, cfg in datasets.items():
    print(dataset_name, type(cfg))
    encoding = cfg['d_in'] * cfg['d']
    if dataset_name == 'aan':
        decoding_1 = cfg['d']**2
        decoding_2 = 0.5 * cfg['d']**2
        decoding_out = 0.5 * cfg['d'] * cfg['n_classes']
        decoding = decoding_1 + decoding_2 + decoding_out
    else:
        decoding = cfg['d'] * cfg['n_classes']
    tot = encoding + cfg['seq_params'] + decoding
    print(f"{dataset_name}: {tot} params, {tot/1e6}M params")
    print(f"  Encoding: {encoding}")
    print(f"  Decoding: {decoding}")


print("\n\n")
print("S4 orig")
datasets = {
    'listops': {
        'd_in': 18,
        'n_classes': 10,
        'd': 128,
        'L': 6
    },
    'imdb': {
        'd_in': 98,
        'n_classes': 2,
        'd': 64,
        'L': 4
    },
    'aan': {
        'd_in': 135,
        'n_classes': 2,
        'd': 256,
        'L': 6
    },
    'image': {
        'd_in': 1,
        'n_classes': 10,
        'd': 512,
        'L': 6
    },
    'path': {
        'd_in': 1,
        'n_classes': 2,
        'd': 256,
        'L': 6
    },
}
N = 64
for dataset_name, cfg in datasets.items():
    print(dataset_name, type(cfg))
    encoding = cfg['d_in'] * cfg['d'] + cfg['d'] # Input projection + bias
    ssm = 4 * N * cfg['d'] # LegS, Lambda, P, B, C
    cm = cfg['d'] ** 2 + cfg['d'] # Channel mixing + biases
    norm = 2 * cfg['d'] # Layer norms (bias + scale)
    decoding = cfg['d'] * cfg['n_classes'] + cfg['n_classes'] # Output projection + bias
    tot = encoding + (ssm + cm + norm) * cfg['L'] + decoding
    print(f"{dataset_name}: {tot} params, {tot/1e6}M params")
    print(f"  Encoding: {encoding}")
    print(f"  Decoding: {decoding}")

print("\n\n")
print("S4 bidir")
datasets = {
    'listops': {
        'd_in': 18,
        'n_classes': 10,
        'd': 128,
        'L': 8
    },
    'imdb': {
        'd_in': 98,
        'n_classes': 2,
        'd': 256,
        'L': 6
    },
    'aan': {
        'd_in': 135,
        'n_classes': 2,
        'd': 256,
        'L': 6
    },
    'image': {
        'd_in': 1,
        'n_classes': 10,
        'd': 512,
        'L': 6
    },
    'path': {
        'd_in': 1,
        'n_classes': 2,
        'd': 256,
        'L': 6
    },
}
N = 64
for dataset_name, cfg in datasets.items():
    print(dataset_name, type(cfg))
    encoding = cfg['d_in'] * cfg['d'] + cfg['d'] # Input projection + bias
    ssm = 4 * N * 2 * cfg['d'] # LegS, Lambda, P, B, C
    cm = 2 * cfg['d'] ** 2 + cfg['d'] # Channel mixing + biases
    norm = 2 * cfg['d'] # Layer norms (bias + scale)
    decoding = cfg['d'] * cfg['n_classes'] + cfg['n_classes'] # Output projection + bias
    tot = encoding + (ssm + cm + norm) * cfg['L'] + decoding
    print(f"{dataset_name}: {tot} params, {tot/1e6}M params")
    print(f"  Encoding: {encoding}")
    print(f"  Decoding: {decoding}")