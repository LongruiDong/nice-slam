#!/bin/bash


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-office3 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office3orb.yaml >log/office3orb.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-office3 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office3orb-spcd-KL.yaml >log/office3orb-prior.log

# src/tools/eval_ate.py

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-office3 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office3orb.yaml

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-office3 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office3orb-spcd-KL.yaml
