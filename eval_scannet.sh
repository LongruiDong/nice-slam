#!/bin/bash

evalset=(
    # 0000
    0059
    0106
    0169
    0181
    0207
)

for ((i=0; i<5; i++)); do
    printf "now do scene%s \t \n" "${evalset[$i]}"
    CUDA_VISIBLE_DEVICES=2 python -W ignore run.py configs/ScanNet/scene${evalset[$i]}.yaml >log/scannet${evalset[$i]}_00.log 
    # nohup ./eval_scannet.sh >log/eval_scannet.log 2>&1 &
done

