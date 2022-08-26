#!/bin/bash
CUDA_VISIBLE_DEVICES=0

GT_MESH=Datasets/Replica/cull_replica_mesh/office0.ply


# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-imap
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-pix10240
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-pix20480
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

OUTPUT_FOLDER=output_imap/Replica/office0gt-nodloss-pix5k--f1k5-it2448
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d


# OUTPUT_FOLDER=output/Replica/office0gt-nodepthloss
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix5000
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix10240
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix20480
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix4096-f1500-it15
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

# OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix4096-f1500-it1k
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d

OUTPUT_FOLDER=output/Replica/office0gt-nodloss-pix1000-f4k-it4k
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d




# # assign any output_folder and gt mesh you like, here is just an example
# OUTPUT_FOLDER=output/Replica/room2
# GT_MESH=cull_replica_mesh/room2.ply
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/room2.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame

# OUTPUT_FOLDER=output/Replica/office0
# GT_MESH=cull_replica_mesh/office0.ply
# python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office0.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame
