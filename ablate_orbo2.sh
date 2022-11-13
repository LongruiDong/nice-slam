#!/bin/bash


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-office2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office2orb.yaml >log/office2orb.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-office2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office2orb-spcd-KL.yaml >log/office2orb-prior.log


srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-office2 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office2orb.yaml

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-office2 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office2orb-spcd-KL.yaml
