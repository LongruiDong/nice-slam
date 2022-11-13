#!/bin/bash


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-room0 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0orb.yaml >log/room0orb.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-room0 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0orb-spcd-KL.yaml >log/room0orb-prior.log

# 对比ate

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-room0 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/room0orb.yaml

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-room0 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/room0orb-spcd-KL.yaml

# src/tools/eval_ate.py
