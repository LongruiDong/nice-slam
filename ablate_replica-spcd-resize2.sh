#!/bin/bash

# only wo spd
srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=ablate-spd-resize2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-wo-spcd-rs2.yaml >log/office0gt-wo-spd-resize2.log

# only with spd
srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=only-spd-resize2 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-only-spd-rs2.yaml >log/office0gt-only-spd-resize2.log