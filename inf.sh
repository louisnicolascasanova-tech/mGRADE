#!/bin/bash
python inf.py \
    --ckpt_dir /home/tristan/Den-minGRU/checkpoints/general/listops_H128L6B32_do0_lr0.003wd0.02we30.0_skipLFET_DCLS256axgaus0.7hetPF8_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0 \
    --dataset listops \
    --gpu 0
python inf.py \
    --ckpt_dir /home/tristan/Den-minGRU/checkpoints/general/listops_H256L6B64_do0_lr0.004wd0.1we30.0_skipLFET_DCLS64axgaus0.7hetPF8_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0_training_1 \
    --dataset listops \
    --gpu 0
python inf.py \
    --ckpt_dir /home/tristan/Den-minGRU/checkpoints/general/cifar/H32L6B32_do0_lr0.004wd0.1we50.0_skipLFET_DCLS128axgaus0.7hetPT_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0_training_2 \
    --dataset cifar \
    --gpu 0
python inf.py \
    --ckpt_dir /home/tristan/Den-minGRU/checkpoints/general/path_H92L6B64_do0_lr0.005wd0.02we10.0_skipLFET_DCLS224axgaus0.7hetPF8_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0 \
    --dataset path \
    --gpu 0
python inf.py \
    --ckpt_dir /home/tristan/Den-minGRU/checkpoints/general/path_H128L6B64_do0_lr0.005wd0.02we10.0_skipLFET_DCLS256axgaus0.7hetPF8_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWTSFtrainWTSFPT_s0 \
    --dataset path \
    --gpu 0

#     --config /home/tristan/Den-minGRU/checkpoints/general/cifar/H32L6B32_do0_lr0.004wd0.1we50.0_skipLFET_DCLS32axgaus0.7hetPF4_recrelu_cmmlprelu_HpreReg0_CFNone_pT_hetWFSFtrainWFSFPT_s0_training_2/config.yaml  \
