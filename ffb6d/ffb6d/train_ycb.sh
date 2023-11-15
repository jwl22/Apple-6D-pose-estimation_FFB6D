#!/bin/bash
n_gpu=1  # number of gpu to use
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu -checkpoint "FFB6D_best.pth.tar"
