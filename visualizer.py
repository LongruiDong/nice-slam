import argparse
import os
import time
# -*- coding:utf-8 -*-
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # 画直方图
from src import config
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset
batch= 2 #直方图间隔
# bands = int((10000-0)/batch)+1
# bins = np.arange(0,10000+1,batch)
# x_ticks = bins - batch/2
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output
    if args.vis_input_frame:
        frame_reader = get_dataset(cfg, args, scale, device='cpu')
        frame_loader = DataLoader(
            frame_reader, batch_size=1, shuffle=False, num_workers=4)
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
            #拿出kf信息
            # selected_keyframes=ckpt['selected_keyframes']
    estimate_c2w_list[:, :3, 3] /= scale
    gt_c2w_list[:, :3, 3] /= scale
    estimate_c2w_list = estimate_c2w_list.cpu().numpy()
    gt_c2w_list = gt_c2w_list.cpu().numpy()
    # select_kfidx = selected_keyframes.keys()
    # print('selsected_kfid: \n',select_kfidx)

    frontend = SLAMFrontend(output, init_pose=estimate_c2w_list[0], cam_scale=0.3,
                            save_rendering=args.save_rendering, near=0,
                            estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list).start()
    os.makedirs(f'{output}/inputvis', exist_ok=True)
    # os.makedirs(f'{output}/depthhist', exist_ok=True)
    for i in tqdm(range(0, N+1)):
        # show every second frame for speed up 2 1 5
        if args.vis_input_frame and i % 2 == 0:
            idx, gt_color, gt_depth, gt_c2w = frame_reader[i]
            # if idx == 76:
            #     print('here')
            depth_np = gt_depth.numpy()
            # # 对于tartanair 会有深度值很大的异常值 但已经clip 0--10000
            # depth_np = np.clip(depth_np, 0 , 600)
            # depthvec = depth_np.reshape(-1,)
            # bands = int((depthvec.max()-0)/batch)+1
            # bins = np.arange(0,depthvec.max()+1,batch)
            # x_ticks = bins - batch/2
            # if (i==76 or i==82 or i==83 or
            # # (i>=96 and i<=214) or
            # (i>=237 and i<=240) or
            # (i>=254 and i<=265)):
            #     plt.hist(depthvec,bins,rwidth=0.5)
            #     plt.xticks(x_ticks)
            #     plt.xlim(0,depthvec.max()+1)
            #     plt.title(f'{i:05d}_depthhist')
            #     plt.savefig(f'{output}/depthhist/{i:05d}_hist.jpg')
            #     plt.show()
            #     plt.pause(1)
            #     plt.cla()
            # depth_np = depth_np.astype(np.float)*100
            # depth_np = np.clip(depth_np, 0, 65535)
            # depth_np = depth_np.astype(np.uint16)
            gt_alldepthv = np.unique(depth_np)
            gt_alldepthv1 = gt_alldepthv[np.argsort(-gt_alldepthv)] #降序排列
            secondmax_depth = gt_alldepthv1[1] # 次大值 保证不是sky 无限远
            color_np = (gt_color.numpy()*255).astype(np.uint8)
            depth_np = depth_np/secondmax_depth*255 #np.max(depth_np)
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            # 转为3通道
            depth_np = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2RGB)
            # depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            color_np = np.clip(color_np, 0, 255)
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            tH = H//2
            tW = W//2 #为了保证ffmpeg拿到的是大小能被2整除
            whole = cv2.resize(whole, (tW+tW%2, tH+tH%2)) #//4
            cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            # 保存图像来可视化
            cv2.imwrite(f'{output}/inputvis/{i:05d}_rgbd.jpg',whole[:, :, ::-1])
            cv2.waitKey(1)
        time.sleep(0.03)
        meshfile = f'{output}/mesh/{i:05d}_mesh.ply'
        if os.path.isfile(meshfile):
            frontend.update_mesh(meshfile)
        frontend.update_pose(1, estimate_c2w_list[i], gt=False)
        if not args.no_gt_traj:
            frontend.update_pose(1, gt_c2w_list[i], gt=True)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames) 10 2
        if i % 10 == 0:
            frontend.update_cam_trajectory(i, gt=False)
            if not args.no_gt_traj:
                frontend.update_cam_trajectory(i, gt=True)

    if args.save_rendering:
        time.sleep(1)
        os.system( # 30 10 4
            f"/usr/bin/ffmpeg -f image2 -r 10 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
    if args.vis_input_frame:
        time.sleep(1)
        # os.system( # 30 10
        #     f"/usr/bin/ffmpeg -f image2 -r 10 -pattern_type glob -i '{output}/inputvis/*.jpg' -y {output}/inrgbd.mp4")
