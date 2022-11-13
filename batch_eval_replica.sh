#!/bin/bash
# 对replica数据 3d mesh 质量的评估
# 设置gpu
# CUDA_VISIBLE_DEVICES=1

gt_dir=Datasets/Replica/cull_replica_mesh

out_dir=$1 # /Replica

suffix=$2

evalset=(
    # office0
    office1
    office2
    office3
    office4
    room0
    room1
    room2
)

for ((i=0; i<7; i++)); do
    printf "now do Replica-%s \t \n" "${evalset[$i]}"
    GT_MESH=${gt_dir}/${evalset[$i]}.ply
    OUTPUT_FOLDER=${out_dir}/${evalset[$i]}${suffix}
    python -W ignore src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
    #  >log/Replica-${evalset[$i]}.log 
    # nohup./eval_replica.sh >log/eval_replica.log 2>&1 &
done
