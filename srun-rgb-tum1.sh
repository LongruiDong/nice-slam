#!/usr/bin/env sh


srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=tum1-rgb-bl --kill-on-bad-exit=1 python -W ignore run.py configs/TUM_RGBD/freiburg1_desk-rgbonly.yaml >log/tum1-rgbonly.log


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=tum2-rgb-bl --kill-on-bad-exit=1 python -W ignore run.py configs/TUM_RGBD/freiburg2_xyz-rgbonly.yaml >log/tum2-rgbonly.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=tum3-rgb-bl --kill-on-bad-exit=1 python -W ignore run.py configs/TUM_RGBD/freiburg3_office-rgbonly.yaml >log/tum3-rgbonly.log
