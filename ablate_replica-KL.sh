#!/bin/bash



# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=ablate-KL --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-wo-KL.yaml >log/office0gt-wo-KL.log

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=ablate-KL-rsize2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-wo-KL.yaml >log/office0gt-wo-KL-rsize2.log