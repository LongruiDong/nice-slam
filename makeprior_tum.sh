#!/bin/bash

prior_root=/home/dlr/Project1/ORB_SLAM2_Enhanced/result/tum

evalset=(
    # freiburg1_desk
    freiburg2_xyz
    freiburg3_office
)

sname=(
    # fr1desk
    fr2xyz
    fr3office
)

for ((i=0; i<2; i++)); do
    printf "now make prior TUM-%s \t \n" "${evalset[$i]}"
    python -W ignore dt2depth-tum.py configs/TUM_RGBD/${evalset[$i]}-dt.yaml --orbmapdir $prior_root/${sname[$i]} > log/tumdt-${sname[$i]}.log 2>&1 &
done