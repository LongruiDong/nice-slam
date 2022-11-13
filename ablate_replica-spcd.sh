#!/bin/bash

# only wo spd
# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=ablate-spd --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-wo-spcd.yaml >log/office0gt-wo-spd.log

# only with spd
srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=only-spd-wowt --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-only-spd.yaml >log/office0gt-only-spdwowt.log