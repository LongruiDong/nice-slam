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

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office3-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office3gt-spcd-KL.yaml >log/office3gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office0-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-spcd-KL.yaml >log/office0gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office1-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office1gt-spcd-KL.yaml >log/office1gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office2-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office2gt-spcd-KL.yaml >log/office2gt-spcd-KL.log


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office4-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office4gt-spcd-KL.yaml >log/office4gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room0-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0gt-spcd-KL.yaml >log/room0gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room1-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room1gt-spcd-KL.yaml >log/room1gt-spcd-KL.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room2-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2gt-spcd-KL.yaml >log/room2gt-spcd-KL.log


srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixorb-room0 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0orb.yaml >log/room0orbfix.log
srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixorb-room0-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0orb-spcd-KL.yaml >log/room0orbfix-spcd-KL.log