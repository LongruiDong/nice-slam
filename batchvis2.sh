
CUDA_VISIBLE_DEVICES=2

# python src/tools/eval_ate.py configs/ScanNet/scene0000.yaml
# python visualizer.py configs/ScanNet/scene0000.yaml --output output/scannet/scans/scene0000_00 --save_rendering --vis_input_frame


# python src/tools/eval_ate.py configs/ScanNet/scene0059.yaml
# python visualizer.py configs/ScanNet/scene0059.yaml --output output/scannet/scans/scene0059_00 --save_rendering --vis_input_frame

# python src/tools/eval_ate.py configs/ScanNet/scene0106.yaml
# python visualizer.py configs/ScanNet/scene0106.yaml --output output/scannet/scans/scene0106_00 --save_rendering --vis_input_frame

# python src/tools/eval_ate.py configs/ScanNet/scene0169.yaml
# python visualizer.py configs/ScanNet/scene0169.yaml --output output/scannet/scans/scene0169_00 --save_rendering --vis_input_frame

# python src/tools/eval_ate.py configs/ScanNet/scene0181.yaml
# python visualizer.py configs/ScanNet/scene0181.yaml --output output/scannet/scans/scene0181_00 --save_rendering --vis_input_frame

# python src/tools/eval_ate.py configs/ScanNet/scene0207.yaml
# python visualizer.py configs/ScanNet/scene0207.yaml --output output/scannet/scans/scene0207_00 --save_rendering --vis_input_frame



# assign any output_folder and gt mesh you like, here is just an example
OUTPUT_FOLDER=output/Replica/room2
GT_MESH=Datasets/Replica/cull_replica_mesh/room2.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/room2.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame

OUTPUT_FOLDER=output/Replica/office0
GT_MESH=Datasets/Replica/cull_replica_mesh/office0.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office0.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame


OUTPUT_FOLDER=output/Replica/office1
GT_MESH=Datasets/Replica/cull_replica_mesh/office1.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office1.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame

OUTPUT_FOLDER=output/Replica/office2
GT_MESH=Datasets/Replica/cull_replica_mesh/office2.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office2.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame

OUTPUT_FOLDER=output/Replica/office3
GT_MESH=Datasets/Replica/cull_replica_mesh/office3.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office3.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame

OUTPUT_FOLDER=output/Replica/office4
GT_MESH=Datasets/Replica/cull_replica_mesh/office4.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply --gt_mesh $GT_MESH -2d -3d
# python visualizer.py configs/Replica/office4.yaml --output $OUTPUT_FOLDER --save_rendering --vis_input_frame