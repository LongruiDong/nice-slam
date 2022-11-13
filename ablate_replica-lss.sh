#!/bin/bash



# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=ablate-lss --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-wo-lss.yaml >log/office0gt-wo-lss.log

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=ablate-lssgd-resize2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-wo-lss.yaml >log/office0gt-wo-lssgd-resize2.log

