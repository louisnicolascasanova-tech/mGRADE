#!/bin/bash
# python main.py --dataset cifar --gpu 0 --conv_mode dcls --dcls_config 6 --file_nb 0
# python main.py --resume_from checkpoints/general/cifar_H128L6B64_do0_lr0.005wd0.02we10.0_skipLFET_DCLS256axgaus0.7hetPF8_reclinear_cmmlpgelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s1 --dataset cifar --gpu 0 
# python main.py --dataset cifar --gpu 0 --conv_mode dcls --dcls_config 6 --file_nb 1
python main.py --dataset aan --gpu 0 --conv_mode dcls --dcls_config 6 --file_nb 0 --sim_name aan_MLPsigh+x_h64_64dcls8
python main.py --dataset aan --gpu 0 --conv_mode dcls --dcls_config 6 --file_nb 1 --sim_name aan_MLPh+x_h64_64dcls8
