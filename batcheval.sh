#!/bin/bash
CUDA_VISIBLE_DEVICES=2

GT_MESH=Datasets/Replica/cull_replica_mesh/office0.ply


# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-imap
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-pix10240
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-pix20480
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-pix5k--f1k5-it2448
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d


# OUTPUT_FOLDER=output-reg/Replica/office0gt-nodloss
# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-woNsurf
# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-woNsurf-Ni32
# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-woNsurf-Ni32-ws10-p2k # output/Replica/office0gt-nodloss-woNsurf-Ni32-ws41 # output/Replica/office0gt-nodloss-woNsurf-Ni32-ws10-p2k-f1k5-it1k # output/Replica/office0gt-nodloss-woNsurf-Ni32-p1k-f4k-it4k

# OUTPUT_FOLDER=/home/dlr/Project/nice_pl/output/Replica/office0gt-nodloss-80kf-glr-d120k-minlr-guidesample-woffs-exp6
# # output/Replica/office0gt-nodloss1-Ni32-ef25-it1k-resize2 # output/Replica/office0gt-nodloss1-Ni32-ef25-resize2
# # output/Replica/office0gt-nodloss1-Ns32-ef25-guidesamp-resize2-woffs
# # output/Replica/office0gt-nodloss1-Ns32-ef25-it1k-guidesamp-resize2-woffs
# # output/Replica/office0gt-nodloss1-Ns32-ef25-it1k-guidesamp-resize2 # output/Replica/office0gt-nodloss1-Ns32-ef25-guidesamp-resize2
# # /home/dlr/Project1/nice_pl/output/Replica/office0gt-nodloss-woffs-80kf-glr-d120k-minlr-guidesample
# # output/Replica/office0gt-nodloss1-Ni32-ef25 # output/Replica/office0gt-nodloss1-woNsurf-Ni32-f4k-it4k # office0gt-nodloss1-woNsurf-Ni32-ws40 # office0gt-nodloss1-woNsurf-Ni32-ws10-p2k-it1k #office0gt-nodloss1-woNsurf-Ni32 # office0gt-nodloss1-woNsurf-Ni32-p1k-it1k # office0gt-nodloss-woNsurf-Ni32-p1k-it1k # output/Replica/office0gt-nodloss-woNsurf-Ni16-p1k-f4k-it4k # /home/dlr/Project1/nice-slam/output/Replica/office0gt-nodloss-woNsurf-Ni32-p1k-f4k-it4k-mysamp
# # python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/01950_mesh.ply --gt_mesh $GT_MESH -2d -3d
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=/home/dlr/Project/nice_pl/output/Replica/office0gt-nodloss-80kf-glr-d120k-minlr-guidesample-woffs-exp7
# # /home/dlr/Project1/nice_pl/output/Replica/office0gt-nodloss-80kf-glr-d120k-minlr-guidesample
# # output/Replica/office0gt-nodloss1-Ni32-ef25-it1k-resize2-woffs # output/Replica/office0gt-nodloss1-Ni32-ef25-resize2-woffs
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=/home/dlr/Project/nice_pl/output/Replica/office0gt-nodloss-80kf-glr-d120k-minlr-guidesample-woffs-exp8
# # /home/dlr/Project1/nice_pl/output/Replica/office0gt-nodloss-noshuffle-80kf-glr-d120k-minlr-guidesample-woffs
# # /home/dlr/Project1/nice_pl/output/Replica/office0gt-nodloss-woffs-80kf-glr-d120k-minlr
# # # output/Replica/office0gt-nodloss1-Ns32-ef25-guidesamp # /home/dlr/Project2/nice-slam/output/Replica/office0gt-ndloss1-woNsurf-Ni32-p1k-f4k-it4k-mysamp
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica_prior/office0gt-nogtd-resize2-debug-KLloss
# # output/Replica_prior/office0gt-nogtd-resize2-debug-kfglobal #output/Replica_prior/office0gt-nogtd-resize2-debug
# # output/Replica_prior/office0gt-nodloss1-resize2-onlyorbkf
# # /home/dlr/Project2/nice-slam/output/Replica/office0gt-nodloss1-Ni32-ef25-resize2
# # output/Replica_prior/office0gt-nodloss1-resize2-woguide # output/Replica_prior/office0gt-nodloss1-resize2-ertune-it1k
# # output/Replica_prior/office0gt-nodloss1-resize2
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica_prior/office0gt-nogtd-debug-resize2-kfglobal-less-regs
# # output/Replica_prior/office0gt-nogtd-debug-kfglobal-less-map5hz
# # output/Replica_prior/office0gt-nogtd-debug-kfglobal-less-colormask
# # output/Replica_prior/office0gt-nogtd-resize2-debug-kfglobal-less
# # output/Replica_prior/office0gt-nogtd-resize2-orbpose-kfglobal
# # output/Replica_prior/office0gt-nogtd-resize2-orbpose
# # output/Replica_prior/office0gt-nodloss1-resize2-it1k
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica_prior/office0gt-nodloss1-resize2-woguide-onlyorbkf
# # output/Replica_prior/office0gt-nodloss1-resize2-woguide-it1k
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica_prior/office0gt-sp-5hz-KLoss-stagecolor
# # output/Replica_prior/office0gt-sp-resize2-5hz-KLossfilter-stagecolor
# # output/Replica_prior/office0gt-sp-resize2-5hz-KLoss-stagecolor
# # output/Replica_prior/office0gt-sp-resize2-5hz-KLoss
# # output/Replica_prior/office0gt-klfactor
# # output/Replica_prior/office0gt-kl
# # output/Replica/office0gt-nodloss-pix20480
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica_prior/office0gt-spkl-resize2-5hz-cdguide-lss
# # output/Replica_prior/office0gt-sp-resize2-5hz-cdguide
# # output/Replica_prior/office0gt-sp-resize2-5hz
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica_prior/office0gt-sp-resize2-5hz-cdguide-lss
# # output/Replica/office0gt-nodloss-pix4096-f1500-it1k
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

OUTPUT_FOLDER=output/Replica_prior/office0gt-resize2-5hz-lss
# output/Replica/office0gt-nodloss-pix1000-f4k-it4k
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=/home/dlr/Project/NeuralRecon-W/results/replica/{epoch:d}_epoch=0-step=29999
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/extracted_mesh_res_256_radius_1.0_colored-clip.ply --gt_mesh $GT_MESH -3d -2d

# OUTPUT_FOLDER=/home/dlr/Project/NeuralRecon-W/results/replica/train-office0_scale2_80kf-20220926_163219_epoch=10-step=21812
# /home/dlr/Project/NeuralRecon-W/results/replica/train-office0noshuffle_scale2-20220911_103840_epoch=0-step=29999
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/extracted_mesh_res_256_radius_1.0_colored-clip.ply --gt_mesh $GT_MESH -3d -2d

# OUTPUT_FOLDER=/home/dlr/Project/NeuralRecon-W/results/replica/train-office0_scale2_80kf-20220927_225134_epoch=9-step=19829
# /home/dlr/Project/NeuralRecon-W/results/replica/train-office0noshuffle_scale2-20220911_103840_epoch=5-step=228259
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/extracted_mesh_res_256_radius_1.0_colored-clip.ply --gt_mesh $GT_MESH -3d -2d

# OUTPUT_FOLDER=/home/dlr/Project/nerf_pl # nerfw_office0-noshuffle_64.ply nerfw_office0-noshuffle-ep0_64.ply
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/nerfw_office0-noshuffle-ep0_64.ply --gt_mesh $GT_MESH -3d -2d

# # assign any output_folder and gt mesh you like, here is just an example
# OUTPUT_FOLDER=output/Replica/room2
# GT_MESH=cull_replica_mesh/room2.ply
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/room2.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame

# OUTPUT_FOLDER=output/Replica/office0
# GT_MESH=cull_replica_mesh/office0.ply
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office0.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame
