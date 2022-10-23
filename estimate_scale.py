"""
利用~/Project1/ORB_SLAM2  拿出来的所有需要的数据
/home/dlr/Project1/ORB_SLAM2_Enhanced/result:
####.txt (所有kf 但以frameid为名) (首行是 tum 格式 Tcw (8,)) 其余 kptid u v 3dpid(-1表示没有对应) x y z 
mappts.txt 3dpid x y z [obsv_frameid kptid ....(repeat) ]
KeyFrameTrajectory.txt tum 格式 Twc kf位姿
office0_orb_mappts.txt 仅是所有3d点坐标 (验证过各文件 一致 除了pose还没验证)
计算每帧深度是在dt2depth.py里面, 这里只是为了计算 输入的点云 对应的kf pose 和gt 之间的尺度变换
尺度问题 放在nice-slam 里的scale 参数得了，那这里就需要用位姿估计尺度了

subdiv https://zhuanlan.zhihu.com/p/340510482
"""

import argparse, os, copy
import random, math
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

import cv2
from numba import jit


fps = 10 # 生成伪时间 但要根据设置的帧率

def align(model,data,calc_scale=True):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn) 应该得是gt吧 这是为了 最后评估时的尺度是在gt上 否则结果数字失真
    data -- second trajectory (3xn) est
    
    Output:
    rot -- rotation matrix (3x3)  此SE3变换应该是把 model 变 为 data
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    s -- model 相对于 data 的尺度, 即 data*s --> model
    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    if calc_scale:
        rotmodel = rot*model_zerocentered
        dots = 0.0
        norms = 0.0
        for column in range(data_zerocentered.shape[1]):
            dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
            normi = np.linalg.norm(model_zerocentered[:,column])
            norms += normi*normi
        # s = float(dots/norms)  
        s = float(norms/dots)
    else:
        s = 1.0  

    # trans = data.mean(1) - s*rot * model.mean(1)
    # model_aligned = s*rot * model + trans
    # alignment_error = model_aligned - data

    # scale the est to the gt, otherwise the ATE could be very small if the est scale is small
    trans = s*data.mean(1) - rot * model.mean(1)
    model_aligned = rot * model + trans
    data_alingned = s * data
    alignment_error = model_aligned - data_alingned
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans,trans_error, s

def SE3toQT(RT):
    """
    Convert transformation matrix to quaternion and translation. (tum 格式)
    x y z i j k w
    """
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quadw = rot.to_quaternion() # w i j k https://docs.blender.org/api/current/mathutils.html#mathutils.Quaternion
    # quad = quadw[1, 2, 3, 0] # i j k w
    quad = [quadw[1], quadw[2], quadw[3], quadw[0]]
    tq = np.concatenate([T, quad], 0) # x y z i j k w
    return tq

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

def load_traj(gtpath, save=None, firstI = False):
    """
    firstI: True 表示 要把首帧归一化， 默认false
    """
    with open(gtpath, "r") as f:
        lines = f.readlines()
    n_img = len(lines)
    gtpose = []
    c2ws = []
    inv_pose = None
    for i in range(n_img):
        timestamp = float(i * 1.0/fps) # 根据帧率设置伪时间
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        #测试首帧归一化
        if inv_pose is None: # 首帧位姿归I 但是好像只有tum做了归一化处理 why
            inv_pose = np.linalg.inv(c2w) #T0w
            c2w = np.eye(4)
        else:
            c2w = inv_pose@c2w # T0w Twi = T0i
        # 转为tum
        tq = SE3toQT(c2w)
        ttq = np.concatenate([np.array([timestamp]), tq], 0) #  带上时间戳
        gtpose += [ttq]
        c2ws += [c2w]
    
    gtposes = np.stack(gtpose, 0) # (N, 8)
    c2ws = np.stack(c2ws, 0)
    
    if save is not None:
        np.savetxt(save, gtposes, fmt="%.1f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")
        if firstI:
            print('first c2w is I, save tum-format gt: {}'.format(save))
        else:
            print('save tum-format gt: {}'.format(save))
        
        
    return gtposes, c2ws

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches    


def display_inlier_outlier(cloud, ind, vis=True):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # 统计内外点数目
    nin = np.asarray(inlier_cloud.points).shape[0]
    nout = np.asarray(outlier_cloud.points).shape[0]
    print('inlier: {}, outlier: {}'.format(nin, nout))
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    if vis:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
                                        #   ,
                                        #   zoom=0.3412,
                                        #   front=[0.4257, -0.2125, -0.8795],
                                        #   lookat=[2.6172, 2.0475, 1.532],
                                        #   up=[-0.0694, -0.9768, 0.2024])
    return inlier_cloud


@jit(nopython=True) # (nopython=True) # (nopython=True)
def locate_triangle(qt, trangleList):
    """自己写一个粗暴遍历 当前所有三角形 判断当前点的位置
       有多种情况 [0,]表示就在内部
       -1---不在任何一个三角内部
       -2---就是某个顶点
       -3---就在某条edge上 # 这个情况比较棘手
    Args:
        pt (np.array): 查询的像素坐标 (u, v)
        trangleList (np.array): cv2.subdiv.getTriangleList().astype(np.int32) 返回的结果 (n, 6)

    Returns:
        int: pt 所在的triangle trangleList里的 索引 ; 若不在任意三角形，就返回 -1
    https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    """
    
    tri_id = [-1]
    
    n_tri = trangleList.shape[0]
    px, py = qt[0], qt[1]
    
    for i in range(n_tri):
        pt1 = trangleList[i, 0:2] # (t[0],t[1]) # (u,v)
        pt2 = trangleList[i, 2:4] # (t[2],t[3])
        pt3 = trangleList[i, 4:6] # (t[4],t[5])
        # # 这里不能用相等
        # if qt==pt1:
        #     return [-2, 0]
        # if qt==pt2:
        #     return [-2, 1]
        # if qt==pt3:
        #     return [-2, 2]
        # if (qt==pt1 or qt==pt2 or qt==pt3):
        #     return -2  # 表示就是某个顶点
        
        # # 3条边
        # edg12 = pt1-pt2
        # edg13 = pt1-pt3
        # edg23 = pt2-pt3
        # ref https://blog.csdn.net/dageda1991/article/details/77875637
        p0x, p0y = pt1[0], pt1[1]
        p1x, p1y = pt2[0], pt2[1]
        p2x, p2y = pt3[0], pt3[1]
        Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y) # 会出现0
        if Area == 0.:
            print('debug')
        u = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
        v = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)
        
        # 根据u v 的值 分情况
        if (u>0 and v>0 and u+v<1):
            # 在此三角形内部
            return [i]
        # elif (u<0 or v<0):
        #     # 不在此三角内 跳过
        #     continue
        elif (u==0 and v==0):
            # 顶点A
            return [-2, i]
        elif (u+v==1):
            if u==1:
                # 顶点B
                return [-2, i]
            elif v==1:
                # 顶点C
                return [-2, i]
            
        elif (u>0 and v>0 and u+v==1):
            # 在BC上
            return [-3, 1,2]
        elif (u+v<1 and u>=0 and v==0):
            # 在AB边
            return [-3, 0,1]
        elif (u+v<1 and u==0 and v>=0):
            # 在AC边上
            return [-3, 0,2]
        else:
            # 在当前三角外部 跳过
            continue 
    
    return tri_id


def RayCastTriPlane(ray_origin, ray_dir, vert, plane_norm):
    """计算ray_origin为起点, ray_dir(单位向量)为方向的射线 到 三角平面(三角一顶点, 法向) 的交点
    返回该点3d坐标 注意都是世界系下
    ref: https://blog.csdn.net/qq_41524721/article/details/103490144
    
    Args:
        ray_origin (np.array): (3,)
        ray_dir (np.array): (3,)
        vert (np.array): (3,)
        plane_norm (np.array): (3,)
    
    return intrersected 3d point (3,)
    """
    
    test_para = np.dot(ray_dir, plane_norm)
    fenzi = np.dot((vert-ray_origin), plane_norm)
    if np.isclose(test_para, 0):
        if np.isclose(fenzi, 0): 
            #  ray_origin 就在此平面 所以交点就是它
            return -2
        else:
            # ray 于平面平行 无交点
            return -1 
    else:
        # 正常情况
        t = fenzi / test_para 
        intersec = ray_origin + ray_dir * t # (3,)
        if t<0: # debug 应该都是大于0吧 果然 t 负值 对应于下面深度负值
            # print('[RayCastTriPlane] t<0: {}'.format(t))
            return -3 # 此情况也是舍弃的
        
        return intersec # .reshape(3, -1)    

def main(cfg, args, orbmapdir="/home/dlr/Project1/ORB_SLAM2_Enhanced/result"):
    
    mapptsfile = os.path.join(orbmapdir, 'mappts.txt')
    kftrajfile = os.path.join(orbmapdir, 'KeyFrameTrajectory.txt')
    gttrajfile = os.path.join(cfg['data']['input_folder'], 'traj.txt')
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyzs[:, :3]) # 相机位置的点云
    # pcd.normals = o3d.utility.Vector3dVector(xyzs[:, :3] / np.linalg.norm(xyzs[:, :3], axis=-1, keepdims=True)) # 每个位置点到原点的距离  每个位置单位向量        
    # # 对点云滤波 open3d的函数
    # # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
    # # 原点云 边界
    # print('raw pcd box: \n', pcd.get_axis_aligned_bounding_box())
    # print("Statistical oulier removal")
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5) # 20,30 2
    # inlier_cloud = display_inlier_outlier(pcd, ind, vis=False) # , vis=True False
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    # size=1.0, origin=[0, 0, 0]) #显示坐标系 1.0 20.0
    # vis_lst = [inlier_cloud, mesh_frame]
    # # o3d.visualization.draw_geometries(vis_lst)
    # # inlier 的边界：
    # print('inlier pcd box: \n', inlier_cloud.get_axis_aligned_bounding_box())

    estpose = np.loadtxt(kftrajfile) # 本身是kf pose
    print('load pred pose from {}'.format(kftrajfile))
    gtpose, _ = load_traj(gttrajfile, save=os.path.join(cfg['data']['input_folder'], 'tum_gt.txt'), firstI=False)
    print('load gt traj from {}'.format(gttrajfile))
    # 转变为 字典 key 为 时间戳
    n_gt = gtpose.shape[0]
    n_est = estpose.shape[0]
    print('gt pose: ', gtpose.shape)
    print('est pose: ', estpose.shape)
    dic_gt = dict([(gtpose[i, 0], gtpose[i, 1:]) for i in range(n_gt)])
    dic_est = dict([(float(format(estpose[i, 0], '.1f')), estpose[i, 1:]) for i in range(n_est)])
    matches = associate(dic_gt, dic_est)
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?")
    first_xyz = np.matrix(
        [[float(value) for value in dic_gt[a][0:3]] for a, b in matches]).transpose()
    second_xyz = np.matrix([[float(value) for value in dic_est[b][0:3]] for a, b in matches]).transpose()
    
    # 对齐 得到尺度变换
    rot, trans, trans_error, s = align(first_xyz, second_xyz) # RT把前者 变为后者, s 把后者变前者
    if True:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))

        print("absolute_translational_error.rmse %f m" % np.sqrt(
            np.dot(trans_error, trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m" %
              np.mean(trans_error))
        print("absolute_translational_error.median %f m" %
              np.median(trans_error))
        print("absolute_translational_error.std %f m" % np.std(trans_error))
        print("absolute_translational_error.min %f m" % np.min(trans_error))
        print("absolute_translational_error.max %f m" % np.max(trans_error))
    scale = float(s) # float(1./s)
    print('est map should x {}'.format(scale))
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Arguments for running the triangulation projection.'
        )
    parser.add_argument('--orbmapdir', type=str, help='dir Path to saved map from orb', default="/home/dlr/Project1/ORB_SLAM2_Enhanced/result")
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
    orbmapdir = args.orbmapdir
    # # 自己制作一个profile工具，并且传入要分析的代码
    # profile = lp.LineProfiler(main)
    # # 起始分析
    # profile.enable()
    main(cfg, args, orbmapdir=orbmapdir) # 调用函数，这里还是正常传入参数的哦
    # # 停止分析，这里就相当于只分析get_url_txt这个函数
    # profile.disable()
    # # 打印结果
    # profile.print_stats()
    