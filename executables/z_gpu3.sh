#!/bin/bash
# python main.py --dataset cifar --gpu 1 --conv_mode dcls --dcls_config 6 --file_nb 2
# python main.py --resume_from checkpoints/general/cifar_H128L6B64_do0_lr0.005wd0.1we10.0_skipLFET_DCLS256axgaus0.7hetPF8_reclinear_cmmlpgelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s1 --dataset cifar --gpu 1 
# python main.py --dataset cifar --gpu 1 --conv_mode dcls --dcls_config 6 --file_nb 3
# python main.py --dataset aan --gpu 2 --conv_mode dcls --file_nb 0 --sim_name aan_MLPh+x_h64_16dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed0
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 1 --sim_name aan_MLPh+x_h64_16dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed1
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 2 --sim_name aan_MLPh+x_h64_16dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed2

# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 0 --sim_name aan_MLPh+x_h64_16dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed0
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 1 --sim_name aan_MLPh+x_h64_16dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed1
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 5 --sim_name aan_MLPh+x_h32_8dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed0_hetP
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 7 --sim_name aan_MLPh+x_h64_8dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1zero_h1Zero_EWgn10_seed0
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 41 --sim_name aan_MLPh+x_h32_8dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed3_hetP
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 42 --sim_name aan_MLPh+x_h32_8dcls2wi0.25_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed4_hetP
# python main.py --dataset aan --gpu 3 --conv_mode dcls --file_nb 10 --sim_name aan_MLPh+x_h92_8dcls2wi0.05_3e-3lr0.1_wd0.1_bs32_z1UGI_h1Zero_EWgn10_seed4_2L
python main.py --resume_from /home/tristan/Den-minGRU/checkpoints/general/aan_H64L3B32_do0_lr0.003wd0.1we5.0_skipLFET_DCLS8axgaus0.7hetPF2_reclinear_cmmlpgelu_HpreReg0_CFNone_pT_hetWTSTtrainWTSFPT_s0 --gpu 3 --dataset aan
python main.py --resume_from /home/tristan/Den-minGRU/checkpoints/general/aan_H64L3B32_do0_lr0.003wd0.1we5.0_skipLFET_DCLS8axgaus0.7hetPF2_reclinear_cmmlpgelu_HpreReg0_CFNone_pT_hetWTSTtrainWTSFPT_s2 --gpu 3 --dataset aan
python main.py --resume_from /home/tristan/Den-minGRU/checkpoints/general/aan_H64L3B32_do0_lr0.003wd0.1we5.0_skipLFET_DCLS8axgaus0.7hetPF2_reclinear_cmmlpgelu_HpreReg0_CFNone_pT_hetWTSTtrainWTSFPT_s3 --gpu 3 --dataset aan


