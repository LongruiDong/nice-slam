#!/bin/bash


srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=nplorb-office0 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0orb.yaml >log/office0orb-npl.log

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=nplorb-office1 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office1orb.yaml >log/office1orb-npl.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=priorb-office1 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office1orb-spcd-KL.yaml >log/office1orb-prior.log


# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=orb-office1 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office1orb.yaml



# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixpriorb-office0 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office0orb-spcd-KL.yaml >log/office0orbfix-prior.log

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixpriorb-office1 --kill-on-bad-exit=1 python -W ignore run.py configs/Replica/office1orb-spcd-KL.yaml >log/office1orbfix-prior.log


srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=nplorb-office0 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office0orb.yaml

srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=nplorb-office1 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office1orb.yaml

# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixpriorb-office0 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office0orb-spcd-KL.yaml
# srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=fixpriorb-office1 --kill-on-bad-exit=1 python -W ignore src/tools/eval_ate.py configs/Replica/office1orb-spcd-KL.yaml
