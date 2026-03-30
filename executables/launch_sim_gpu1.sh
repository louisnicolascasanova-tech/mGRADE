#!/bin/bash
python3 main.py --dataset listops --gpu 1 --conv_mode dcls --file_nb 1 --dcls_config 6
python3 main.py --dataset listops --gpu 1 --conv_mode dcls --file_nb 2 --dcls_config 6
python3 main.py --dataset listops --gpu 1 --conv_mode dcls --file_nb 3 --dcls_config 6


# python3 main.py --dataset imdb --gpu 1 --conv_mode dcls --file_nb 10 --dcls_config 6
# python3 main.py --dataset imdb --gpu 1 --conv_mode dcls --file_nb 11 --dcls_config 6
# python3 main.py --dataset imdb --gpu 1 --conv_mode dcls --file_nb 5 --dcls_config 6 # h=64
# python3 main.py --dataset imdb --gpu 1 --conv_mode dcls --file_nb 6 --dcls_config 6 # L=4, h=32
# python3 main.py --dataset imdb --gpu 1 --conv_mode dcls --file_nb 7 --dcls_config 6 # L=4, h=64

# python3 main.py --dataset cifar --gpu 1 --conv_mode dcls --file_nb 36
# python3 main.py --dataset cifar --gpu 1 --conv_mode dcls --file_nb 39
# python3 main.py --dataset cifar --gpu 1 --conv_mode dcls --file_nb 42
# python3 main.py --dataset imdb --gpu 3 --conv_mode tcn_eerf --file_nb 0

