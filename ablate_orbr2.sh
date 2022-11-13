#!/bin/bash


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-room2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2orb.yaml >log/room2orb.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-room2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2orb-spcd-KL.yaml >log/room2orb-prior.log

# src/tools/eval_ate.py

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-room2 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/room2orb.yaml

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-room2 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/room2orb-spcd-KL.yaml