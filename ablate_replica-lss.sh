#!/bin/bash



# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office0-prior --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-spcd-KL.yaml >log/office0gt-spcd-KL.log

