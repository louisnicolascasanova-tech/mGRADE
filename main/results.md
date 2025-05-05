
The following is the table of results

| Model         | n layers | n_hidden | Task     | Train Acc  | Train Loss | Val Acc | Val Loss | Epochs | time / epoch (hh:mm:ss) | Params     |
|---------------|----------|----------|----------|------------|------------|---------|----------|--------|-------------------------|------------|
| (slow) SRN    |        1 |      256 | rowMNIST |      81.18 |            |         |          |      1 |                00:02:00 ||
| SRN           |        1 |      256 | rowMNIST |      78.16 |            |         |          |      1 |                00:01:02 ||
| SRN           |        1 |      256 | sMNIST   |      12.23 |            |         |          |      1 |                00:01:16 ||
| SRN           |        3 |      256 | rowMNIST |      86.71 |            |         |          |      1 |                00:02:58 ||
| SRN           |        3 |      256 | sMNIST   |      15.24 |            |         |          |      1 |                00:03:39 ||
| SRN jit       |        1 |      256 | sMNIST   |      19.32 |     2.1902 |   23.04 |   2.0303 |      5 |                00:00:14 |     68'874 | # concat was slow, orthogonal makes it learn
| SRN jit       |        3 |      256 | sMNIST   |      31.40 |     1.9007 |   33.77 |   1.8096 |      5 |                00:00:38 |    332'042 |
| DiagReLURNN   |        1 |      256 | sMNIST   |      52.46 |     1.4105 |   45.69 |   1.6003 |      5 |                00:00:07 |      3'338 |
| DiagReLURNN   |        3 |      256 | sMNIST   |      73.05 |     0.8077 |   74.36 |   0.8166 |      5 |                00:00:29 |    135'434 | 
| LSTM          |        1 |      256 | sMNIST   |      87.39 |     0.3794 |   87.69 |   0.3802 |      5 |                00:00:19 |    267'786 |
| LSTM          |        3 |      256 | sMNIST   |      87.06 |     0.3793 |   88.95 |   0.3324 |      5 |                00:00:55 |  1'320'458 |
| flax OptiLSTM |        1 |      256 | sMNIST   |      64.06 |     1.1030 |   67.72 |   1.0441 |      5 |                00:00:22 |    266'762 |
| flax OptiLSTM |        3 |      256 | sMNIST   |      86.90 |     0.3926 |   89.11 |   0.3271 |      5 |                00:01:05 |  1'317'386 |
| GRU           |        1 |      256 | sMNIST   |      87.39 |     0.3794 |   87.69 |   0.3802 |      5 |                00:00:17 |    201'482 |
| GRU           |        1 |       16 | sMNIST   |      41.83 |     1.9047 |   44.94 |   1.7925 |     10 |                00:00:13 |      1'082 |
| GRU           |        3 |      256 | sMNIST   |      87.06 |     0.3793 |   88.95 |   0.3324 |      5 |                00:00:48 |    990'986 |
| GRU           |        3 |       16 | sMNIST   |      x.xxx |     x.xxxx |   xx.xx |   x.xxxx |      6 |                00:00:39 |      4'346 | early stopping for 10, destroyed
| MGU           |        1 |      256 | sMNIST   |      94.51 |     0.1852 |   94.91 |   0.1680 |     10 |                00:00:18 |    135'178 |
| MGU           |        3 |      256 | sMNIST   |      95.48 |     0.1465 |   95.90 |   0.1274 |      5 |                00:00:46 |    661'514 |
| MGU           |        1 |       16 | sMNIST   |      44.48 |     1.6063 |   45.83 |   1.5520 |     10 |                00:00:12 |        778 |
| MGU           |        3 |       16 | sMNIST   |      52.18 |     1.4083 |   97.18 |   1.3234 |      5 |                00:00:35 |      2'954 | 
| minGRU        |        1 |       16 | sMNIST   |      26.21 |     2.0369 |   27.08 |   2.0222 |     10 |                00:00:07 |        234 |
| minGRU        |        3 |       16 | sMNIST   |      63.48 |     1.0852 |   65.92 |   1.0180 |      5 |                00:00:xx |      x'xxx |
| minGRU        |        1 |       64 | sMNIST   |      31.78 |     1.9111 |   31.79 |   1.9051 |     10 |                00:00:07 |        906 |
| minGRU        |        3 |       64 | sMNIST   |      81.84 |     0.5569 |   81.87 |   0.5561 |      5 |                00:00:28 |     17'546 |
| minGRU        |        1 |      256 | sMNIST   |      33.75 |     1.8220 |   33.82 |   1.8185 |     10 |                00:00:08 |      3'594 |
| minGRU        |        3 |      256 | sMNIST   |      88.98 |     0.3387 |   89.54 |   0.3179 |      5 |                00:00:36 |    266'762 |
| minGRU        |        4 |       64 | sMNIST   |      92.05 |     0.2503 |   90.13 |   0.3374 |     10 |                00:00:38 |     25'866 |

Parameters complexity:
- N_h: number of hidden units
- N_o: number of output units
- N_i: number of input units
- L: number of layers
GRU:
- W^{zx}, W^{rx}, W^{hx}: N_h x N_i or N_h x N_h
- W^{zh}, W^{rh}, W^{hh}: N_h x N_h
- b^z, b^r, b^h: N_h
- W^{ho}: N_o x N_h
- b^o: N_o
- Total:
  - Single layer: 
    - W^{zx}    + W^{rx}    + W^{hx}    + W^{zh}    + W^{rh}    + W^{hh}    + W^{ho}    + b^z + b^r + b^h + b^o 
    - N_h x N_i + N_h x N_i + N_h x N_i + N_h x N_h + N_h x N_h + N_h x N_h + N_o x N_h + 3 x N_h + N_o
    - 3 x N_h x N_i + 3 x (N_h)^2  + (N_o + 3) x N_h + N_o
    - 3 x (N_h)^2 + (3 x N_i + N_o + 3) x N_h + N_o
    - Using: N_h = 256, N_i = 16, N_o = 10: 3 x (256^2) + (3 x 1 + 10 + 3) x 256 + 10 = 200'714
  - L layers:
    - N_h x (3 x N_i + (3 + 6(L - 1)) x N_h + N_o + 3 x L) + N_o
    - (3 + 6(L - 1)) x (N_h)^2 + (3 x N_i + N_o + 4 * L) x N_h + N_o
    - Using: N_h = 256, N_i = 16, N_o = 10, L = 3: (3 + 6(3 - 1)) x (256)^2 + (3 x 16 + 10 + 3 x 3) x 256 = 987'146


1d_SRN - rowMNIST
Training: 100%|██████████| 468/468 [02:02<00:00,  3.83it/s, accuracy=0.852, loss=0.567]
Epoch 0 | Loss: 0.589563250541687 | Accuracy: 0.8118823170661926
Training: 100%|██████████| 468/468 [01:59<00:00,  3.91it/s, accuracy=0.797, loss=0.542]

1e_scanSRN - rowMNIST
Training: 100%|██████████| 468/468 [01:04<00:00,  7.24it/s, accuracy=0.914, loss=0.25] 
Epoch 0 | Loss: 0.7187735438346863 | Accuracy: 0.781617283821106

1e_scanSRN - sMNIST
Training: 100%|██████████| 468/468 [01:16<00:00,  6.09it/s, accuracy=0.0781, loss=2.3] 
Epoch 0 | Loss: 2.293076276779175 | Accuracy: 0.12234575301408768

1e_scanSRN - 3 layers - rowMNIST
Training: 100%|██████████| 468/468 [02:58<00:00,  2.62it/s, accuracy=0.953, loss=0.195] 
Epoch 0 | Loss: 0.44230347871780396 | Accuracy: 0.8671040534973145

1e_scanSRN - 3 layers - sMNIST
Training: 100%|██████████| 468/468 [03:39<00:00,  2.13it/s, accuracy=0.211, loss=2.11] 
Epoch 0 | Loss: 2.2509360313415527 | Accuracy: 0.15246060490608215

1e_scanSRN with jit(apply_model())- sMNIST - 4.2x speedup
Training: 100%|██████████| 468/468 [00:18<00:00, 25.21it/s, accuracy=0.0859, loss=2.3] 
Epoch 0 | Loss: 2.2912023067474365 | Accuracy: 0.12393162399530411
Training: 100%|██████████| 468/468 [00:18<00:00, 25.85it/s, accuracy=0.18, loss=2.28]  
Epoch 1 | Loss: 2.282238483428955 | Accuracy: 0.1280215084552765

1e_scanSRN with jit(apply_model()) - 3 layers - sMNIST - 4.6x speedup
Training: 100%|██████████| 468/468 [00:49<00:00,  9.55it/s, accuracy=0.125, loss=2.29] 
Epoch 0 | Loss: 2.304612398147583 | Accuracy: 0.11965811997652054
Training: 100%|██████████| 468/468 [00:48<00:00,  9.66it/s, accuracy=0.133, loss=2.3]  
Epoch 1 | Loss: 2.2971558570861816 | Accuracy: 0.11733774095773697

======= FROM here everymodel jits apply_model() =========

DiagReLURNN - sMNIST
Training: 100%|██████████| 390/390 [00:08<00:00, 46.86it/s, accuracy=0.312, loss=1.83] 
Epoch 0 | train_loss: 2.2349 | train_acc: 23.86% | val_loss: 1.8919 | val_acc: 35.30%
Training: 100%|██████████| 390/390 [00:07<00:00, 49.75it/s, accuracy=0.383, loss=1.68]
Epoch 1 | train_loss: 1.7272 | train_acc: 40.21% | val_loss: 1.6351 | val_acc: 40.35%
Training: 100%|██████████| 390/390 [00:07<00:00, 49.88it/s, accuracy=0.453, loss=1.5] 
Epoch 2 | train_loss: 1.5606 | train_acc: 47.11% | val_loss: 1.4817 | val_acc: 49.96%

DiagReLURNN - 3 layers - sMNIST
Training: 100%|██████████| 390/390 [00:28<00:00, 13.58it/s, accuracy=0.406, loss=1.76]
Epoch 0 | train_loss: 6.8655 | train_acc: 27.04% | val_loss: 1.7030 | val_acc: 41.33%
Training: 100%|██████████| 390/390 [00:28<00:00, 13.47it/s, accuracy=0.586, loss=1.22]
Epoch 1 | train_loss: 1.5775 | train_acc: 46.19% | val_loss: 1.2984 | val_acc: 57.81%
Training: 100%|██████████| 390/390 [00:29<00:00, 13.42it/s, accuracy=0.625, loss=1.12] 
Epoch 2 | train_loss: 1.2642 | train_acc: 56.97% | val_loss: 0.9963 | val_acc: 66.90%
Training: 100%|██████████| 390/390 [00:29<00:00, 13.43it/s, accuracy=0.695, loss=0.936]
Epoch 3 | train_loss: 0.9438 | train_acc: 67.85% | val_loss: 0.7772 | val_acc: 74.95%
Training: 100%|██████████| 390/390 [00:28<00:00, 13.50it/s, accuracy=0.781, loss=0.699]
Epoch 4 | train_loss: 0.7595 | train_acc: 74.40% | val_loss: 0.7431 | val_acc: 73.69%





=====================================================================================
OLD RESULTS
=====================================================================================



1f_scanMLSRN - rowMNIST (interrupted)
Training:   6%|▌         | 26/468 [00:31<08:57,  1.22s/it, accuracy=0.492, loss=1.91]


2d_DiagReLURNN - rowMNIST
Training: 100%|██████████| 468/468 [02:32<00:00,  3.07it/s, accuracy=0.75, loss=0.706] 
Epoch 0 | Loss: 1.1938234567642212 | Accuracy: 0.6174045205116272
Training:  84%|████████▍ | 393/468 [02:09<00:24,  3.03it/s, accuracy=0.828, loss=0.49]

2f_MLDiagReLURNN - rowMNIST
Training: 100%|██████████| 468/468 [06:39<00:00,  1.17it/s, accuracy=0.938, loss=0.22] 
Epoch 0 | Loss: 0.6568479537963867 | Accuracy: 0.7925847768783569


1e_scanSRN - sMNIST - doesn't learn

1f_scanMLSRN - sMNIST - doesn't learn

2d_DiagReLURNN - sMNIST
Training: 100%|██████████| 468/468 [03:14<00:00,  2.40it/s, accuracy=0.438, loss=1.58]
Epoch 0 | Loss: 1.978589653968811 | Accuracy: 0.3096120357513428
Training: 100%|██████████| 468/468 [03:14<00:00,  2.41it/s, accuracy=0.508, loss=1.46]
Epoch 1 | Loss: 1.5807161331176758 | Accuracy: 0.4659288227558136
Training: 100%|██████████| 468/468 [03:17<00:00,  2.37it/s, accuracy=0.477, loss=1.54]
Epoch 2 | Loss: 1.4591313600540161 | Accuracy: 0.5098824501037598

2e_IRLM - sMNIST
Training: 100%|██████████| 468/468 [02:23<00:00,  3.26it/s, accuracy=0.281, loss=1.97] 
Epoch 0 | Loss: 2.28014874458313 | Accuracy: 0.15306156873703003
Training: 100%|██████████| 468/468 [02:20<00:00,  3.34it/s, accuracy=0.258, loss=1.99]
Epoch 1 | Loss: 2.032831907272339 | Accuracy: 0.23753005266189575
Training: 100%|██████████| 468/468 [02:20<00:00,  3.33it/s, accuracy=0.312, loss=1.87]
Epoch 2 | Loss: 1.8504436016082764 | Accuracy: 0.3020332455635071

2f_MLDiagReLURNN - sMNIST
Training: 100%|██████████| 468/468 [07:39<00:00,  1.02it/s, accuracy=0.523, loss=1.26]
Epoch 0 | Loss: 3.9891393184661865 | Accuracy: 0.38555020093917847
Training: 100%|██████████| 468/468 [07:48<00:00,  1.00s/it, accuracy=0.57, loss=1.15]  
Epoch 1 | Loss: 1.2117186784744263 | Accuracy: 0.5834835767745972
Training: 100%|██████████| 468/468 [07:58<00:00,  1.02s/it, accuracy=0.719, loss=0.82] 
Epoch 2 | Loss: 0.9546190500259399 | Accuracy: 0.6827924847602844
Training: 100%|██████████| 468/468 [09:32<00:00,  1.22s/it, accuracy=0.82, loss=0.637] 
Epoch 3 | Loss: 0.749786913394928 | Accuracy: 0.7476963400840759
Training: 100%|██████████| 468/468 [09:21<00:00,  1.20s/it, accuracy=0.711, loss=0.882]
Epoch 4 | Loss: 0.631913423538208 | Accuracy: 0.7879440188407898

