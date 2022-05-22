#!/bin/bash

evalset=(
    office0
    office1
    # office2
    office3
    office4
)

for ((i=0; i<4; i++)); do
    printf "now do Replica-%s \t \n" "${evalset[$i]}"
    CUDA_VISIBLE_DEVICES=3 python -W ignore run.py configs/Replica/${evalset[$i]}.yaml >log/Replica-${evalset[$i]}.log 
    # nohup ./eval_replica.sh >log/eval_replica.log 2>&1 &
done