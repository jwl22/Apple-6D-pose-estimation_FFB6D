#!/bin/bash

cls='ape'
# tst_mdl=train_log/linemod/checkpoints/ape/FFB6D_ape_best.pth.tar
tst_mdl=train_log/linemod/checkpoints/${cls}/FFB6D_${cls}_best.pth.tar
python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py --gpu '0' --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose # -debug

