#!/bin/bash
CUDA_VISIBLE_DEVICES=1

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
# OUTPUT_FOLDER=/home/dlr/Project1/nice-slam/output/Replica/office0gt-nodloss-woNsurf-Ni32-p1k-f4k-it4k-mysamp
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/01950_mesh.ply --gt_mesh $GT_MESH -2d -3d


# OUTPUT_FOLDER=output_bl/Replica/office0gtp-rgb-resize2
# # output_bl/Replica/office0gtp-rgb-map25hz-resize2
# # output_bl/Replica/office0gtp-rgb-map25hz
# # output_bl/Replica/office0gtp-rgb
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

OUTPUT_FOLDER=output_imap_bl/Replica/office0gtp-rgb
# output/Replica/office0gt-nodloss-pix5000
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix10240
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix20480
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix4096-f1500-it15
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix4096-f1500-it1k
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix1000-f4k-it4k
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=/home/dlr/Project/NeuralRecon-W/results/replica/{epoch:d}_epoch=5-step=248651
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/extracted_mesh_res_1024_radius_1.0_colored-clip.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=/home/dlr/Project/NeuralRecon-W/results/replica/train-replicaoffice0_scale1-20220827_125619_epoch=11-step=476629
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/extracted_mesh_res_256_radius_1.0_colored-clip.ply --gt_mesh $GT_MESH -3d -2d

# OUTPUT_FOLDER=/home/dlr/Project/NeuralRecon-W/results/replica/train-replicaoffice0_scale1-20220827_125619_epoch=19-step=803933
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
