#!/bin/bash

evalset=(
    office0
    office1
    office2
    office3
    office4
    room0
    room1
    room2
)

for ((i=0; i<8; i++)); do
    printf "now make prior Replica-%s \t \n" "${evalset[$i]}"
    python -W ignore dt2depth.py configs/Replica/${evalset[$i]}gt-nodloss-dt.yaml >log/tum_Replica-${evalset[$i]}.log 
done