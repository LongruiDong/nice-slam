
CUDA_VISIBLE_DEVICES=0

python src/tools/eval_ate.py configs/TUM_RGBD/freiburg1_desk.yaml
python visualizer.py configs/TUM_RGBD/freiburg1_desk.yaml --output output/TUM_RGBD/rgbd_dataset_freiburg1_desk --save_rendering --vis_input_frame


python src/tools/eval_ate.py configs/TUM_RGBD/freiburg2_xyz.yaml
python visualizer.py configs/TUM_RGBD/freiburg2_xyz.yaml --output output/TUM_RGBD/rgbd_dataset_freiburg2_xyz --save_rendering --vis_input_frame


# python src/tools/eval_ate.py configs/TUM_RGBD/freiburg3_office.yaml
# python visualizer.py configs/TUM_RGBD/freiburg3_office.yaml --output output/TUM_RGBD/rgbd_dataset_freiburg1_desk --save_rendering --vis_input_frame