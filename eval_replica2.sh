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


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office0-prior-lss --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-lss.yaml >log/office0gt-lss.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room2-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2gt-spcd-KL.yaml >log/room2gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room0-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0gt-spcd-KL.yaml >log/room0gt-spcd-KL.log

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixorb-room2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2orb.yaml >log/room2orbfix.log
srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixorb-room2-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2orb-spcd-KL.yaml >log/room2orbfix-spcd-KL.log