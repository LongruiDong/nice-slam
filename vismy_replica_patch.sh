#!/bin/bash

# evalset=(
#     # office0
#     office1
#     office2
#     office3
#     office4
#     room0
#     room1
#     room2
# )

# for ((i=0; i<4; i++)); do
#     printf "now do Replica-%s \t \n" "${evalset[$i]}"
#     CUDA_VISIBLE_DEVICES=3 python -W ignore run.py configs/Replica/${evalset[$i]}.yaml >log/Replica-${evalset[$i]}.log 
#     # nohup ./eval_replica.sh >log/eval_replica.log 2>&1 &
# done

CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/office0gt-vismy-patchsm.yaml >log/office0gt-vismy-patchsm.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/office1gt-vismy-patchsm.yaml >log/office1gt-vismy-patchsm.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/room0gt-vismy-patchsm.yaml >log/room0gt-vismy-patchsm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/room2gt-vismy-patchsm.yaml >log/room2gt-vismy-patchsm.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -W ignore run.py configs/Replica/office2gt-visbl.yaml >log/office2gt-visbl.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/office3gt-visbl.yaml >log/office3gt-visbl.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/office4gt-visbl.yaml >log/office4gt-visbl.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -W ignore run.py configs/Replica/room0gt-visbl.yaml >log/room0gt-visbl.log 2>&1 &



# CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/room1gt-visbl.yaml >log/room1gt-visbl.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/room2gt-visbl.yaml >log/room2gt-visbl.log 2>&1 &
