#!/bin/bash

# evalset=(
#     office0
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
#     # nohup./eval_replica.sh >log/eval_replica.log 2>&1 &
# done

# CUDA_VISIBLE_DEVICES=3 nohup python -W ignore run.py configs/Replica/office0gt-rgbonly.yaml >log/office0gt-rgbonly.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python -W ignore run.py configs/Replica/office1gt-rgbonly.yaml >log/office1gt-rgbonly.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -W ignore run.py configs/Replica/office2gt-rgbonly.yaml >log/office2gt-rgbonly.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -W ignore run.py configs/Replica/office3gt-rgbonly.yaml >log/office3gt-rgbonly.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/office4gt-rgbonly.yaml >log/office4gt-rgbonly.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/room0gt-rgbonly.yaml >log/room0gt-rgbonly.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/room1gt-rgbonly.yaml >log/room1gt-rgbonly.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/room2gt-rgbonly.yaml >log/room2gt-rgbonly.log 2>&1 & 

CUDA_VISIBLE_DEVICES=3 nohup python -W ignore run.py configs/Replica/office0gt-rgbonly_imap.yaml --imap >log/office0gt-rgbonly_imap.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -W ignore run.py configs/Replica/office1gt-rgbonly_imap.yaml --imap >log/office1gt-rgbonly_imap.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore run.py configs/Replica/office2gt-rgbonly_imap.yaml --imap >log/office2gt-rgbonly_imap.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -W ignore run.py configs/Replica/office3gt-rgbonly_imap.yaml --imap >log/office3gt-rgbonly_imap.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/office4gt-rgbonly_imap.yaml --imap >log/office4gt-rgbonly_imap.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -W ignore run.py configs/Replica/room0gt-rgbonly_imap.yaml --imap >log/room0gt-rgbonly_imap.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/room1gt-rgbonly_imap.yaml --imap >log/room1gt-rgbonly_imap.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -W ignore run.py configs/Replica/room2gt-rgbonly_imap.yaml --imap >log/room2gt-rgbonly_imap.log 2>&1 &