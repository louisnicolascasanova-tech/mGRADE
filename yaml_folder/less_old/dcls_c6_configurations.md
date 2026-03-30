# DCLS C6 Configuration Summary

This document provides a comprehensive overview of all DCLS C6 configurations across four datasets: IMDB, ListOps, Path, and PathX.

## Overview

| Dataset | Config Count | Config Range | Total Files |
|---------|--------------|--------------|-------------|
| IMDB | 15 | c6_0 → c6_14 | 15 |
| ListOps | 20 | c6_0 → c6_19 | 20 |
| Path | 9 | c6_0 → c6_8 | 9 |
| PathX | 4 | c6_1 → c6_4 | 4 |
| **Total** | **48** | - | **48** |

## Complete Configuration Table

| Dataset | Config | batch_size | lr | weight_decay | warmup_frac | n_layers | hidden_dim | kernel_size | kernel_n_elems | grad_clip_norm | mem_frac | do_rate | Run Name | HTTP Link | Comment | Val Acc Best |
|---------|--------|------------|----|--------------|-------------|----------|------------|-------------|----------------|----------------|----------|---------|----------|-----------|---------|--------------|
| **IMDB** | c6_0 | 32 | 0.004 | 0.1 | 0.5 | 6 | 32 | 256 | 8 | 1.0 | 0.5 | | | | | |
| IMDB     | c6_1  | 32 | 0.004 | 0.1 | 0.5 | 6 | 32 | 256 | 8 | - | 0.5 | | | | | |
| IMDB     | c6_2  | 32 | 0.003 | 0.1 | 0.3 | 6 | 32 | 256 | 8 | 1.0 | 0.5 | | | | | |
| IMDB     | c6_3  | 32 | 0.003 | 0.1 | 0.3 | 6 | 32 | 256 | 8 | 1.0 | 0.5 | | | | | |
| IMDB     | c6_4  | 64 | 0.003 | 0.1 | 0.3 | 6 | 32 | 256 | 8 | 5.0 | 0.9 | | | | | |
| IMDB     | c6_5  | 64 | 0.003 | 0.1 | 0.3 | 6 | 64 | 256 | 8 | 5.0 | 0.9 | | | | | |
| IMDB     | c6_6  | 32 | 0.003 | 0.1 | 0.3 | 6 | 144 | 128 | 8 | 5.0 | 0.9 | | | | | |
| IMDB     | c6_7  | 32 | 0.003 | 0.1 | 0.3 | 6 | 192 | 128 | 8 | 5.0 | 0.9 | | | | | |
| IMDB     | c6_8  | 32 | 0.004 | 0.1 | 0.1 | 6 | 128 | 128 | 16 | 5.0 | 0.9 | | | | | |
| IMDB     | c6_9  | 32 | 0.003 | 0.1 | 0.3 | 4 | 256 | 128 | 16 | 5.0 | 0.95 | | | | | |
| IMDB     | c6_10 | 32 | 0.004 | 0.1 | 0.2 | 6 | 32 | 256 | 8 | 5.0 | 0.5 | | | | | |
| IMDB     | c6_11 | 32 | 0.004 | 0.1 | 0.2 | 6 | 32 | 256 | 8 | 5.0 | 0.5 | | | | | |
| IMDB     | c6_12 | 32 | 0.004 | 0.1 | 0.1 | 6 | 32 | 256 | 8 | 5.0 | 0.5 | | | | | |
| IMDB     | c6_13 | 32 | 0.004 | 0.1 | 0.2 | 6 | 32 | 256 | 8 | 5.0 | 0.5 | | | | | |
| IMDB     | c6_14 | 32 | 0.003 | 0.1 | 0.2 | 6 | 32 | 256 | 8 | 5.0 | 0.5 | | | | | |
| **ListOps** | c6_0  | 32 | 0.003 | 0.1  | 0.3 | 6 | 32  | 256 | 8  | 1.0 | 0.5  | 0   | u79iui2j | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/u79iui2j/overview) |                               | 0.4077 |
| ListOps     | c6_1  | 32 | 0.003 | 0.02 | 0.3 | 6 | 128 | 256 | 8  | 1.0 | 0.5  | 0   | vjbxueza | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/vjbxueza/overview) | L6 - H128, wd                 | 0.5882 |
| ListOps     | c6_2  | 32 | 0.003 | 0.02 | 0.3 | 8 | 128 | 256 | 8  | 1.0 | 0.5  | 0   | 8z0gghgb | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/8z0gghgb/overview)| L8 - H128                      | 0.5670 |
| ListOps     | c6_3  | 32 | 0.003 | 0.02 | 0.3 | 8 | 32  | 256 | 8  | 1.0 | 0.5  | 0   | | | L8 - H32; no run     | |          
| ListOps     | c6_4  | 32 | 0.004 | 0.1  | 0.3 | 6 | 32  | 256 | 8  | 5.0 | 0.5  | 0   | m0lzwqj1 | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/m0lzwqj1/overview) | L6 - H32, gc, lr              | 0.3953 |
| ListOps     | c6_5  | 32 | 0.004 | 0.1  | 0.3 | 8 | 32  | 256 | 8  | 5.0 | 0.5  | 0   | uug756wi | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/uug756wi/overview) | L8 - H32                      | 0.3874 |
| ListOps     | c6_6  | 32 | 0.004 | 0.1  | 0.3 | 6 | 64  | 256 | 8  | 5.0 | 0.5  | 0   | 7exum465 | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/7exum465/overview) | L6 - H32                      | 0.3963 |
| ListOps     | c6_7  | 32 | 0.004 | 0.1  | 0.3 | 6 | 128 | 128 | 8  | 5.0 | 0.5  | 0   | a24igzyf | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/a24igzyf/overview) | L6 - H128 - 128dcls           | 0.3998 |
| ListOps     | c6_8  | 32 | 0.004 | 0.1  | 0.3 | 6 | 256 | 64  | 8  | 5.0 | 0.5  | 0   | roshuhta | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/roshuhta/overview) | L6 - H256 - 64dcls - collapsed| 0.5375 |
| ListOps     | c6_9  | 64 | 0.004 | 0.1  | 0.3 | 6 | 256 | 64  | 8  | 5.0 | 0.9  | 0   | dym1j3au | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/dym1j3au/overview) | L6 - H256 - 64dcls - bs64     | 0.6064 |
| ListOps     | c6_10 | 32 | 0.004 | 0.1  | 0.3 | 6 | 128 | 64  | 8  | 5.0 | 0.5  | 0   | vsw348uu | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/vsw348uu/overview) | L6 - H128 - 64dcls            | 0.4112 |
| ListOps     | c6_11 | 64 | 0.004 | 0.02 | 0.3 | 6 | 128 | 64  | 8  | 2.0 | 0.5  | 0   | | [link]() | L6 - H128 - 64dcls - bs64  | 0.5620|
| ListOps     | c6_12 | 96 | 0.004 | 0.02 | 0.3 | 8 | 128 | 128 | 8  | 2.0 | 0.9  | 0.1 | | [link]() | L6 - H128 - 128dcls - bs96 | 0.5478 |
| ListOps     | c6_13 | 64 | 0.004 | 0.1  | 0.2 | 8 | 128 | 64  | 8  | 5.0 | 0.9  | 0.1 | | [link]() | | |
| ListOps     | c6_14a | 96 | 0.004 | 0.02 | 0.3 | 8 | 128 | 128 | 16 | 2.0 | 0.9  | 0.1 | 77usmjnf | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/77usmjnf/overview) | c12 + dcls16 + Why Did I kill | |
| ListOps     | c6_14b | 96 | 0.004 | 0.02 | 0.3 | 8 | 128 | 128 | 16 | 2.0 | 0.9  | 0.3 | vli1t63c | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/vli1t63c/overview) | c12 + dcls16 + do0.3 + Why Did I kill | |
| ListOps | c6_15 = c6_14a | 96 | 0.004 | 0.02 | 0.3 | 8 | 128 | 128 | 16 | 2.0 | 0.9  | 0.1 | 56ld7zmz | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/56ld7zmz/overview) | Why diff results ? | |
| ListOps     | c6_16 | 64 | 0.003 | 0.1  | 0.2 | 8 | 96  | 256 | 8  | 5.0 | 0.99 | 0.3 | | [link]() | | |
| ListOps     | c6_17 | 64 | 0.01  | 0.1  | 0.2 | 8 | 96  | 256 | 8  | 5.0 | 0.99 | 0.3 | | [link]() | | |
| ListOps     | c6_18 | 64 | 0.004 | 0.02 | 0.2 | 8 | 128 | 128 | 8  | 2.0 | 0.99 | 0   | | [link]() | | |
| ListOps     | c6_19 | 32 | 0.003 | 0.02 | 0.3 | 6 | 128 | 256 | 8  | 1.0 | 0.5  | 0   | | | | |
| **Path** | c6_0 | 64 | 0.01 | 0.1 | 0.3 | 6 | 128 | 256 | 16 | 5.0 | 0.9 | | | | | |
| Path | c6_1  | 64 | 0.005 | 0.02 | 0.1 | 6 | 128 | 256 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_2  | 64 | 0.005 | 0.02 | 0.1 | 6 | 128 | 128 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_3  | 64 | 0.01 | 0.02 | 0.1 | 6 | 128 | 128 | 16 | 2.0 | 0.9 | | | | | |
| Path | c6_4  | 64 | 0.01 | 0.02 | 0.1 | 6 | 64 | 256 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_5  | 64 | 0.01 | 0.02 | 0.1 | 4 | 128 | 256 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_6  | 64 | 0.01 | 0.02 | 0.1 | 6 | 256 | 128 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_7  | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 256 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_8  | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 224 | 8 | 2.0 | 0.9 | | | | | |
| Path | c6_8  | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 224 | 8 | 2.0 | 0.9 | | | | Bias Init | |
| Path | c6_8  | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 224 | 8 | 2.0 | 0.9 | | | yg0qd4v7 | Bias Init + LR | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/yg0qd4v7/overview) |
| Path | c6_9  | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 192 | 8 | 2.0 | 0.9 | | | hvuqx7j5 | Bias Init + LR | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/hvuqx7j5/overview) |
| Path | c6_10 | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 160 | 8 | 2.0 | 0.9 | | | yg0qd4v7 | Bias Init + LR | [link](https://wandb.ai/torchet-tristan/DenGRU_general/runs/yg0qd4v7/overview) |
| Path | c6_11 | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 92  | 8 | 2.0 | 0.9 | | | | Bias Init + LR | [link]() | Failed | 
| Path | c6_12 | 64 | 0.005 | 0.02 | 0.1 | 8 | 92 | 92  | 8 | 2.0 | 0.9 | | | | L8 + Bias Init + LR | [link]() | Failed |
| Path | c6_13 | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 128  | 8 | 2.0 | 0.9 | | | |  Bias Init + LR | [link]() | Failed |
| Path | c6_14 | 64 | 0.005 | 0.02 | 0.1 | 6 | 64 | 160  | 8 | 2.0 | 0.9 | | | |  Bias Init + LR | [link]() | Failed |
| Path | c6_20 | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 160  | 8 | 1.0 | 0.9 | | | | c10 + gn1 | [link]() | no change |
| Path | c6_21 | 64 | 0.005 | 0.02 | 0.05 | 6 | 92 | 160  | 8 | 1.0 | 0.9 | | | | c10 + gn1 | [link]() | Failed |
| Path | c6_30 | 64 | 0.005 | 0.02 | 0.1 | 6 | 92 | 160  | 8 | 1.0 | 0.9 | | | | c10 + state tracking | [link]() | no change |

| **PathX** | c6_1 | 16 | 0.005 | 0.02 | 0.1 | 6 | 96 | 192 | 8 | 2.0 | 0.9 | | | | | |
| PathX | c6_2 | 8 | 0.1 | 0.02 | 0.01 | 6 | 128 | 512 | 8 | 2.0 | 0.9 | | | | | |
| PathX | c6_3 | 16 | 0.1 | 0.02 | 0.01 | 6 | 128 | 1024 | 16 | 2.0 | 0.9 | | | | | |
| PathX | c6_4 | 16 | 0.1 | 0.02 | 0.01 | 6 | 128 | 2048 | 256 | 2.0 | 0.9 | | | | | |

## DCLS-Specific Parameters (Consistent Across All Configurations)

| Parameter | Value |
|-----------|-------|
| delay_type | 'axonal' |
| delay_kernel | 'gaussian' |
| init_std | 0.7 |
| heterogeneous_weights | True |
| heterogeneous_positions | False |
| heterogeneous_std | False |
| train_weights | True |
| train_positions | True |
| train_std | False |
| n_epochs | 100 |
| scheduler | True |
| alpha_cosine | 0 |
| encoder | True |
| layer_skip | False |
| element_skip | True |
| enable_rec | True |
| rec_act | relu |
| enable_conv | True |
| conv | 'dcls' |
| enable_cm | True |
| channel_mixing | 'mlp' |
| cm_act | 'relu' |
| postnorm | True |
| do_rate | 0 |
| reg_factor | 0 |
| seed | 0 |

## Configuration Patterns by Dataset

### IMDB (15 configs)
- **Learning rates**: 0.003-0.004
- **Batch sizes**: 32-64  
- **Hidden dimensions**: 32-256
- **Kernel sizes**: 128-256
- **Memory fractions**: 0.5-0.95
- **Focus**: Systematic exploration of model capacity

### ListOps (20 configs)
- **Learning rates**: 0.003-0.01
- **Batch sizes**: 32-96 (highest variety)
- **Hidden dimensions**: 32-256
- **Kernel sizes**: 64-256
- **Memory fractions**: 0.5-0.99 (highest)
- **Focus**: Extensive hyperparameter exploration

### Path (9 configs)
- **Learning rates**: 0.005-0.01 (consistently higher)
- **Batch sizes**: 64 (consistent)
- **Hidden dimensions**: 64-256 (includes unique 92 hidden_dim)
- **Kernel sizes**: 128-256 (includes unique 224 kernel_size)
- **Kernel elements**: 8-16
- **Memory fractions**: 0.9 (consistent)
- **Focus**: Architectural exploration with varied hidden dims and kernel sizes

### PathX (4 configs)
- **Learning rates**: 0.005-0.1 (extreme values)
- **Batch sizes**: 8-16 (smallest)
- **Hidden dimensions**: 96-128
- **Kernel sizes**: 192-2048 (largest)
- **Kernel elements**: 8-256 (most variation)
- **Memory fractions**: 0.9 (consistent)
- **Focus**: Large kernel sizes for long sequence processing

## Notes

- All configurations use the same DCLS architecture with axonal delays and Gaussian kernels
- PathX configurations are optimized for extremely long sequences with very large kernel sizes
- ListOps shows the most architectural diversity with varied layer counts (6-8)
- Memory fractions generally increase with model complexity
- Gradient clipping norms vary significantly (1.0-5.0) across configurations