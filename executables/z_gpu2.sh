#!/bin/bash
# # python main.py --dataset cifar --gpu 2 --conv_mode dcls --dcls_config 6 --file_nb 4
# python main.py --resume_from checkpoints/general/cifar_H128L6B64_do0_lr0.003wd0.02we10.0_skipLFET_DCLS256axgaus0.7hetPF8_reclinear_cmmlpgelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s1_training_1 --dataset cifar --gpu 2

# python main.py --dataset cifar --gpu 2 --conv_mode dcls --dcls_config 6 --file_nb 5

# python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 2 --sim_name aan_MLPh+x_h64_8dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed0_hetP
# python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 3 --sim_name aan_MLPh+x_h64_8dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed1_hetP
# python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 5 --sim_name aan_MLPh+x_h32_8dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed0
python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 6 --sim_name aan_MLPh+x_h64_8dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed1_3L
python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 7 --sim_name aan_MLPh+x_h64_8dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed2_3L
python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 8 --sim_name aan_MLPh+x_h64_8dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed3_3L
python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 9 --sim_name aan_MLPh+x_h64_8dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed4_3L

