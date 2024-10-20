#!/bin/bash
n_gpu=1  # number of gpu to use
torchrun --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu -checkpoint "FFB6D_best.pth.tar"
