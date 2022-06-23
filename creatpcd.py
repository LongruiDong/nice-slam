# -*- coding:utf-8 -*-
"""_summary_
按照zihan的建议
为了确定mapping的bound参数
使用opend3d 把rgbd通过内外参转为world frame下的点云
"""
import os
import argparse
import random
import copy
import numpy as np
import torch
import open3d as o3d
from matplotlib import pyplot as plt
import cv2

from src import config
# from src.NICE_SLAM import NICE_SLAM
from src.utils.datasets import get_dataset
from src.utils.datasets import bilinear_interp

def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    frame_reader = get_dataset(cfg, args, 1.0)
    n_img = len(frame_reader)
    pcd_combined = o3d.geometry.PointCloud() # 用于储存多帧合并后的点云
    count = 0
    for idx, gt_color, gt_depth, gt_c2w in frame_reader:
        # if idx != 708: #debug 只看1帧 > 10 break
        #     if idx > 708:
        #         break
        #     else:
        #         continue
        # if idx > 1000:
        #     break
        
        if idx % 15 != 0 : #debug 间隔 10 15 30 60 100 
            continue
        count = count+1
        print('read frame {}'.format(idx))
        
        rgbfile = frame_reader.color_paths[idx]
        depthfile = frame_reader.depth_paths[idx]
        color_data = cv2.imread(rgbfile) # uint8 
        # color_raw = o3d.io.read_image(rgbfile) # 1242 375 3
        savescale = None
        if '.png' in depthfile:
            depth_data = cv2.imread(depthfile, cv2.IMREAD_UNCHANGED) # replica uint16
            # depth_raw = o3d.io.read_image(depthfile)
            # depth_data = np.clip(depth_data, 0, 65535) # vkitti2 原始数据uint16 全是整数 因为是cm单位 可以
            
            depth_trunc = 25.0 #655 200 100 50 25
            skydepth = depth_trunc # 65535
            # # for replica:
            # depth_trunc = 20.0 # 
            # skydepth = depth_trunc
        elif '.npy' in depthfile: #对于tartanqir来说 gt depth 保存真实数值 m 0~ 10000（infinite）
            depth_data = np.load(depthfile) #float32
            # 实际上给的深度是有大于 10000 即使是office 所以先clip吧
            depth_data = np.clip(depth_data, 0, 10000)
            skydepth = 10000
            depth_trunc = 20.0 # 20 25 30
            savescale = 100.
            # # 对于office0 76 82 83 上的3个外点做插值处理
            # if idx == 76 or idx == 82 or idx == 83:
            #     qx, qy = np.argwhere(depth_data>30)[0]
            #     print('bf interp, max: {}'.format(np.max(depth_data)))
            #     print('depth[{}, {}] = {}'.format(qx, qy, depth_data[qx, qy]))
            #     # 在 [qx, qy]处进行双线性插值
            #     depth_data = bilinear_interp(depth_data, qx, qy)
            #     print('aft interp, max: {}'.format(np.max(depth_data)))
            
        
        depth_data = depth_data.astype(np.float32) / frame_reader.png_depth_scale #真实单位
        # cv2.imwrite('tmpraw.png', (depth_data*frame_reader.png_depth_scale).astype(np.uint16)) #没有处理sky的深度    
        # skydepth = skydepth / frame_reader.png_depth_scale #单位 m
        depth_data1 = copy.deepcopy(depth_data)
        # 对于深度中 sky的处理  depth_residual[gt_depth_np == 0.0] = 0.0
        # 对于office0 depth限制在 45mi之内就行
        depth_data1[depth_data > depth_trunc] = 0.0 #就0 即 不会被转为点云 float32 skydepth
        # 保存为tmp.png 为了uint16不损失小数 
        if savescale is None: # 非tartanair
            savescale = frame_reader.png_depth_scale # 1. # 1 100
        depth_data2 = depth_data1*savescale
        cv2.imwrite('tmp.png', depth_data2.astype(np.uint16)) # np.float32 .astype(np.uint16)保存未16位 无符号png 会截断 这里已经是m了 就出现深度 阶段  不应该
        max_depth = np.max(depth_data)
        max_depth1 = np.max(depth_data1)
        # fig1, axs1 = plt.subplots(1, 2)
        # fig2, axs2 = plt.subplots(1, 2)
        # axs2[0].imshow(depth_data, cmap="plasma", vmin=0, vmax=max_depth1) #
        # axs2[0].set_title('raw depth')
        # axs2[0].set_xticks([])
        # axs2[0].set_yticks([])
        # axs2[1].imshow(depth_data1, cmap="plasma", vmin=0, vmax=max_depth1) #
        # axs2[1].set_title('sky=0 depth')
        # axs2[1].set_xticks([])
        # axs2[1].set_yticks([])
        # fig2.tight_layout()
        
        # 在用o3d读取 这就是已经 clip 并转换真实单位的深度
        depth_raw = o3d.io.read_image("tmp.png") 
        H, W = np.asarray(depth_raw).shape
        color_data = cv2.resize(color_data, (W, H))
        # 保存
        cv2.imwrite('tmp-color.jpg', color_data)
        color_raw = o3d.io.read_image('tmp-color.jpg')
        # color_raw = cv2.resize(color_raw, (W, H))
        gtpose = frame_reader.poses[idx].cpu().numpy() #Twc nerf frame SE3
        # http://www.open3d.org/docs/release/python_api/open3d.geometry.RGBDImage.html#open3d.geometry.RGBDImage.create_from_color_and_depth
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=savescale, depth_trunc=skydepth, convert_rgb_to_intensity=False)
        # 缩放深度1000.0 1.0 截断3.0 skydepth 到0 rgb 2 intensity True
        if idx == 0:
            print(rgbd_image) 
        
        # axs1[0].imshow(rgbd_image.color) #
        # axs1[0].set_title('rgbd_image.color')
        # axs1[0].set_xticks([])
        # axs1[0].set_yticks([])
        # axs1[1].imshow(rgbd_image.depth, cmap="plasma")#,
        #                         #  vmin=0, vmax=secondmax_depth) #max_depth
        # axs1[1].set_title('rgbd_image.depth')
        # axs1[1].set_xticks([])
        # axs1[1].set_yticks([])
        # fig1.tight_layout()
        # plt.show()
        
        # plt.subplot(1, 2, 1)
        # plt.title('in grayscale image')
        # plt.imshow(rgbd_image.color)
        # plt.subplot(1, 2, 2)
        # plt.title('in depth image')
        # plt.imshow(rgbd_image.depth)
        
        
        
        inter = o3d.camera.PinholeCameraIntrinsic()
        inter.set_intrinsics(frame_reader.W, frame_reader.H, frame_reader.fx, frame_reader.fy, frame_reader.cx, frame_reader.cy)

        
        gt_c2w = gt_c2w.cpu().numpy()
        gt_w2c = np.linalg.inv(gt_c2w)
        invgtpose = np.linalg.inv(gtpose)
        if (True in np.isinf(gt_c2w)) or (True in np.isnan(gt_c2w)): #scannet里 会有-inf
            print('c2w nan or inf at ', idx)
            continue
        # 上面nerf 坐标系 转会来
        gt_c2w[:3, 1] *= -1
        gt_c2w[:3, 2] *= -1
        pcd_idx = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, inter,
        extrinsic=np.linalg.inv(gt_c2w)) #测试这里就用外参 哦好像是得用 w2c!!
        pcd_idx_w = pcd_idx# .transform(gt_c2w)
        
        # # 最粗暴的合并点云
        # for pt_id in range(len(pcd_idx_w)):
        #     pcd_combined += pcd_idx_w[pt_id] 
        pcd_combined += pcd_idx_w
        
        # if count >= 2:
        #     break
    
    # Flip it(to nerf style), otherwise the pointcloud will be upside down  
    # pcd_combined.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    #获取真实bound
    # 计算轴对齐边界框
    abox = pcd_combined.get_axis_aligned_bounding_box()
    print('axis_aligned_bounding_box: \n', abox)
    # aabb.color = (1, 0, 0)
    pcdarray = np.asarray(pcd_combined.points)
    print('pcd shape: \n', pcdarray.shape) #(n,3)
    # 输出 xyz 的区域
    print('x y z min:\n', pcdarray.min(axis=0))
    print('x y z max:\n', pcdarray.max(axis=0))
    #保存最终点云
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.15) # .voxel_down_sample(voxel_size=0.7) 0.4 0.15 0.2 0.01 0.02
    # o3d.io.write_point_cloud(os.path.join(frame_reader.input_folder,"truncd-gtrawextrsub.ply"), pcd_combined_down)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]) #显示坐标系 1.0 20.0
    o3d.visualization.draw_geometries([pcd_combined_down, mesh_frame])
    
        

if __name__ == '__main__':
    main()
