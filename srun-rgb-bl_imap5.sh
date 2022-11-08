#!/usr/bin/env sh


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office0-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-rgbonly_imap.yaml --imap >log/office0gt-rgbonly_imap.log


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office0-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0gt-rgbonly_imap.yaml --imap >log/office0gt-rgbonly_imap.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office1-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office1gt-rgbonly_imap.yaml --imap >log/office1gt-rgbonly_imap.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office2-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office2gt-rgbonly_imap.yaml --imap >log/office2gt-rgbonly_imap.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office3-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office3gt-rgbonly_imap.yaml --imap >log/office3gt-rgbonly_imap.log

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=office4-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office4gt-rgbonly_imap.yaml --imap >log/office4gt-rgbonly_imap.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room0-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room0gt-rgbonly_imap.yaml --imap >log/room0gt-rgbonly_imap.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room1-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room1gt-rgbonly_imap.yaml --imap >log/room1gt-rgbonly_imap.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=room2-rgb-bl-imap --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/room2gt-rgbonly_imap.yaml --imap >log/room2gt-rgbonly_imap.log
