#!/bin/bash


CUDA_VISIBLE_DEVICES=0 nohup python -W ignore run.py configs/TUM_RGBD/freiburg1_deskgt-my.yaml > log/fr1deskgt-my.log 2>&1 &

