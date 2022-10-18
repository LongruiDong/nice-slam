"""
利用~/Project1/ORB_SLAM2  拿出来的处理过的 inlierpcd.ply
再这里进行三角剖分 并投影到图像 注意是 所有点 和 有效的三角形
并估计有效区域的depth 保存并可视化
"""

import argparse, os, copy
import random
# -*- coding:utf-8 -*-
import numpy as np
import torch

import matplotlib.pyplot as plt
import open3d as o3d
from mathutils import Matrix
import mathutils

from src import config
from src.utils.datasets import get_dataset

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def TQtoSE3(inputs):
    """
    x y z i j k w (tum)转为 SE3 matrix 
    注意 tum四元数 要先转为 w i j k
    """
    t, quad1 = inputs[:3], inputs[3:]
    quad = [quad1[3], quad1[0], quad1[1], quad1[2]]
    R = mathutils.Quaternion(quad).to_matrix()
    SE3 = np.eye(4)
    SE3[0:3, 0:3] = np.array(R)
    SE3[0:3,3] = t
    SE3 = Matrix(SE3)
    return SE3

def update_cam(cfg):
    """
    Update the camera intrinsics according to pre-processing config, 
    such as resize or edge crop.
    """
    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'] #相机参数
    # resize the input images to crop_size (variable name used in lietorch)
    if 'crop_size' in cfg['cam']:
        crop_size = cfg['cam']['crop_size']
        sx = crop_size[1] / W
        sy = crop_size[0] / H
        fx = sx*fx
        fy = sy*fy
        cx = sx*cx
        cy = sy*cy
        W = crop_size[1]
        H = crop_size[0]

    # croping will change H, W, cx, cy, so need to change here
    if cfg['cam']['crop_edge'] > 0:
        H -= cfg['cam']['crop_edge']*2
        W -= cfg['cam']['crop_edge']*2
        cx -= cfg['cam']['crop_edge']
        cy -= cfg['cam']['crop_edge']
    
    K = np.eye(3)
    K[0] = fx # fx # 按照比例 得到新的内参
    K[1, 1] = fy # fy
    K[0, 2] = cx # cx
    K[1, 2] = cy # cy
    
    return copy.deepcopy(K), H, W

def main(cfg, args, orbpcdfile = "/home/dlr/Project1/ORB_SLAM2/inlierpcd.ply", # align_inlierpcd inlierpcd.ply office0_orb_mappts.txt
         kf_orb_file = "/home/dlr/Project1/ORB_SLAM2/KeyFrameTrajectory.txt",  # alignKeyFrameTrajectory.txt KeyFrameTrajectory
         align_kf_orb_file = "/home/dlr/Project1/ORB_SLAM2/alignKeyFrameTrajectory.txt",
         alignorbpcdfile = "/home/dlr/Project1/ORB_SLAM2/align_inlierpcd.ply"):
    
    
    if '.ply' in orbpcdfile:
        alignorbpcdfile = "/home/dlr/Project1/ORB_SLAM2/align_inlierpcd.ply"
        inlier_cloud = o3d.io.read_point_cloud(orbpcdfile)
        print('load ply from {}'.format(orbpcdfile))
        print('readed: ', inlier_cloud)
    elif '.txt' in orbpcdfile:
        alignorbpcdfile = "/home/dlr/Project1/ORB_SLAM2/office0_orbalign_mappts.txt"
        print('load raw point cloud txt from {}'.format(orbpcdfile))
        pcdarr = np.loadtxt(orbpcdfile)
        npt = pcdarr.shape[0]
        print('pts size: ', pcdarr.shape)
        inlier_cloud = o3d.geometry.PointCloud()
        inlier_cloud.points = o3d.utility.Vector3dVector(pcdarr[:, :3])

    
    # 对该点云 再次三角化
    dec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(inlier_cloud, 0.5) # 此算法不需要normal 0.03 0.06 0.08 0.16 0.18
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]) #显示坐标系 1.0 20.0
    
    vis_lst = [inlier_cloud, dec_mesh, mesh_frame]
    o3d.visualization.draw_geometries(vis_lst)
    
    # 三角剖分内的元素
    triangles = np.asarray(dec_mesh.triangles) # 每个三角形 顶点组成 id 对应下面 的坐标值 (#,3) 0 6216
    vertices =np.asarray(dec_mesh.vertices) # (#=6217, 3)
    # vertices = np.asarray(inlier_cloud.points) # 现就看所有的点
    n_vert = vertices.shape[0]
    print('total vert to project: ', n_vert)
    verts_h = np.concatenate([vertices, np.ones((n_vert, 1))], -1) # (M,4) 齐次表示
        
    
    # 投影的话还是要有每帧位姿 再跑前面的orbslam 插值 不准 就先只用kf的吧
    kf_orb_pose = np.loadtxt(kf_orb_file)
    nkf = kf_orb_pose.shape[0]
    print('load orb-mono pose(tum): {} \n size: {}, {}'.format(kf_orb_file, kf_orb_pose.shape[0], kf_orb_pose.shape[1]))
    dic_est = dict([(int(float(format(kf_orb_pose[i, 0], '.1f'))*10), kf_orb_pose[i, 1:]) for i in range(nkf)]) # key 就是 frame id
    
    scale = cfg['scale']
    # 逐帧看 image
    frame_reader = get_dataset(cfg, args, scale)
    frame_loader = DataLoader(
            frame_reader, batch_size=1, shuffle=False, num_workers=1)
    
    K, H, W = update_cam(cfg) # 得到内参矩阵
    
    idx_map = [] # 记录每张图上 有效triangulate 以及 所有verts 和之前原3d 数据的idx 的映射关系 list 里是多个字典 每个字典是个List 里面两个字典
    # 分别表示 有效triangulate 和 verts
    
    for idx, gt_color, _, _ in frame_loader:
        idx = idx.item()
        if (not idx in dic_est.keys()):
            continue # 非kf 暂时不投影
        if idx > 0:
            break
        print('process frame {}'.format(idx))
        map_i = {'tri':{}, 'vet':{}}
        # 3d triangle 投影到2d当前视角后 的数组
        tri2d = [] # 当前图像里 
        vert2d = []
        idx_vt3d = {} # 用字典记录已经涉及的 点的 原(3d index:2d index) 
        count = 0 # 累计当前view下已有的2d vert 
        
        gt_color_np = gt_color.cpu().numpy()[0] # (1, h,w,3)
        # 拿出orb 的 pose
        orb_tq = dic_est[idx]
        orb_c2w = np.array(TQtoSE3(orb_tq))
        orb_w2c = np.linalg.inv(orb_c2w)
        # 把 3d 点 投影到当前坐标系 并拿出当前图上的点 
        xyz_cam_i = (verts_h @ orb_w2c.T)[:, :3] # xyz in the ith cam coordinate 批量转换 (M,4)
        # xyz_cam_i_in = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam ,this id 这里的序号就不是之前的了
        flag_i = xyz_cam_i[:, 2]>0 # 表示 所有定点中 投在当前view下的 bool array (6217,)
        
        
        
        # 再计算图像上像素坐标
        z_cam_i = xyz_cam_i[:,2].reshape(-1,1) # (6217,) -> (6217, 1)
        z3_cam_i = np.hstack((z_cam_i, z_cam_i, z_cam_i))
        xyz_cam_i_h = np.divide(xyz_cam_i, z3_cam_i) # (x/z, y/z, 1) # (n,3)
        # 相机投影
        uv_i = (xyz_cam_i_h @ K.T).astype(np.int32)[:, :2] # 这才是像素坐标 (n,2)
        # uv_i = xyz_cam_i_h[:, :2] # 像素坐标 艹艹艹艹 弱智Bug! 
        # 可视化
        fig, axs = plt.subplots()
        fig.tight_layout()
        axs.imshow(gt_color_np, cmap="plasma")
        axs.set_title('Input RGB projected triangles')
        axs.set_xticks([])
        axs.set_yticks([])
        
        # 把3d点的投影画出来
        # axs.scatter(uv_i[:, 0], uv_i[:, 1], c='red', s=2)
        for k in range(uv_i.shape[0]):
            pix = uv_i[k]
            if not flag_i[k]:
                continue
            if pix[0] >= 0 and pix[0] < W and pix[1] >= 0 and pix[1] < H:
                axs.scatter(pix[0], pix[1], c='red', s=2)
        # 根据 保存的三角形 用edge 连起来 
        edge_x = []
        edge_y = []
        for t in range(triangles.shape[0]):
            vts = triangles[t] # 3个 顶点 的 index
            bin0 = flag_i[vts[0]]
            bin1 = flag_i[vts[1]]
            bin2 = flag_i[vts[2]]
            vt_0 = uv_i[vts[0]] # (2)
            vt_1 = uv_i[vts[1]]
            vt_2 = uv_i[vts[2]]
            bat0 = vt_0[0] >= 0 and vt_0[0] < W and vt_0[1] >= 0 and vt_0[1] < H
            bat1 = vt_1[0] >= 0 and vt_1[0] < W and vt_1[1] >= 0 and vt_1[1] < H
            bat2 = vt_2[0] >= 0 and vt_2[0] < W and vt_2[1] >= 0 and vt_2[1] < H
            if (bin0 and bin1 and bin2 and bat0 and bat1 and bat2): # 该三角的 3个顶点都在该图像内
                # 拿出这3个顶点的 像素坐标
                edge_x += [[vt_0[0], vt_1[0]], [vt_1[0], vt_2[0]], [vt_0[0], vt_2[0]]]
                edge_y += [[vt_0[1], vt_1[1]], [vt_1[1], vt_2[1]], [vt_0[1], vt_2[1]]]
                my_tri = Polygon([(vt_0[0],vt_0[1]), (vt_1[0],vt_1[1]), (vt_2[0],vt_2[1])])
                a = axs.add_patch( my_tri )
                a.set_fill(False)
                # axs.scatter(vt_0[0], vt_0[1], c='red', s=2)
                # axs.scatter(vt_1[0], vt_1[1], c='red', s=2)
                # axs.scatter(vt_2[0], vt_2[1], c='red', s=2)
                
    #             # 保存2d 到对应
    #             if (not vts[0] in idx_vt3d.keys()):
    #                 idx_vt3d[vts[0]] = count
    #                 vert2d += [vt_0]
    #                 map_i['vet'][count] = vts[0]
    #                 count += 1
    #             if (not vts[1] in idx_vt3d.keys()):
    #                 idx_vt3d[vts[1]] = count
    #                 map_i['vet'][count] = vts[1]
    #                 vert2d += [vt_1]
    #                 count += 1 
    #             if (not vts[2] in idx_vt3d.keys()):
    #                 idx_vt3d[vts[2]] = count
    #                 map_i['vet'][count] = vts[2]
    #                 vert2d += [vt_2]
    #                 count += 1
                
    #             # 保存三角形组成
    #             tri2d += [np.array([idx_vt3d[vts[0]], idx_vt3d[vts[1]], idx_vt3d[vts[2]]])]
        
    #     # axs.plot(edge_x, edge_y) # 画所有的边  
        
        plt.savefig(os.path.join('triangulation', '{:04d}big.png'.format(idx)))
    #     # show 
        plt.show()
    #     plt.cla()
    
    #     # 估计粗糙的深度图 to do
        
        
        
        




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Arguments for running the triangulation projection.'
        )
    parser.add_argument('--orbpcdfile', type=str, help='Path to filtered ply file', default="/home/dlr/Project1/ORB_SLAM2/office0_orb_mappts.txt") # /home/dlr/Project1/ORB_SLAM2/office0_orb_mappts.txt inlierpcd.ply
    parser.add_argument('--predpose', type=str, help='orb 估计的位姿路径',default="/home/dlr/Project1/ORB_SLAM2/KeyFrameTrajectory.txt")
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    parser.set_defaults(nice=True)
    args = parser.parse_args()
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    # 读取参数
    orbpcdfile = args.orbpcdfile
    predpose = args.predpose
    
    main(cfg, args, orbpcdfile=orbpcdfile, kf_orb_file=predpose)