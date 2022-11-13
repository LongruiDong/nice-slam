#!/bin/bash


srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=gtbl-o0 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-bl.yaml >log/office0gt-bl.log