"""
利用~/Project1/ORB_SLAM2  拿出来的所有需要的数据
/home/dlr/Project1/ORB_SLAM2_Enhanced/result:
####.txt (所有kf 但以frameid为名) (首行是 tum 格式 Tcw (8,)) 其余 kptid u v 3dpid(-1表示没有对应) x y z 
mappts.txt 3dpid x y z [obsv_frameid kptid ....(repeat) ]
KeyFrameTrajectory.txt tum 格式 Twc kf位姿
office0_orb_mappts.txt 仅是所有3d点坐标 (验证过各文件 一致 除了pose还没验证)
再这里进行三角剖分 并投影到图像 注意是 所有点 和 有效的三角形
并估计有效区域的depth 保存并可视化

尺度问题 放在nice-slam 里的scale 参数得了，那这里就需要用位姿估计尺度了
也用来保存某几帧的结果，放到pipeline里面了
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
# 用来分析时间
import line_profiler as lp

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

#Check if a point is insied a rectangle
def rect_contains(rect,point):
    if point[0] <rect[0]:
        return False
    elif point[1]<rect[1]:
        return  False
    elif point[0]>rect[2]:
        return False
    elif point[1] >rect[3]:
        return False
    return True

# Draw a point
def draw_point(img,p,color):
    cv2.circle(img,p,5,color) # 2 5

#Draw delaunay triangles
def draw_delaunay(img,trangleList,delaunay_color):
    # trangleList = np.around(subdiv.getTriangleList()).astype(np.int32) # .astype(np.int32) # (8630,6 ?) 应该四舍五入而不是 Int
    size = img.shape
    r = (0,0,size[1],size[0])
    for t in  trangleList:
        pt1 = (t[0],t[1]) # (u,v)
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        if (rect_contains(r,pt1) and rect_contains(r,pt2) and rect_contains(r,pt3)):
            cv2.line(img,pt1,pt2,delaunay_color,1)
            cv2.line(img,pt2,pt3,delaunay_color,1)
            cv2.line(img,pt3,pt1,delaunay_color,1)

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

# @jit # (nopython=True) # (nopython=True) 
def filterkpt(kfarr, inlier_list): # 加速下  改用数组 运算 改进
    data = kfarr # copy.deepcopy(kfarr)
    # for k in range(kfarr.shape[0]):
    #     mpid = data[k, 3]
    #     if mpid >= 0: # 作为第二次过滤 ! 0也是
    #         if not(mpid in inlier_list): # pcd filter 认为是 离群点 也认为不做三角剖分
    #             data[k, 3] = -1
    inlier = np.array(inlier_list)
    data[~np.isin(data[:,3], inlier),3] = -1
    return data

def com_norm(arr):
    return math.sqrt(arr[0]*arr[0] + arr[1]*arr[1])

@jit # (nopython=True) # (nopython=True) 
def findloop(arr, data, qt): # 加速下
    tsmax = 100000.
    locidx = -1
    for i in range(arr.shape[0]):
        k = arr[i]
        flag = data[k, 3]
        if flag < 0: # 无效点
            continue
        kpt = data[k, 1:3]
        subarr = kpt - qt
        norm = math.sqrt(subarr[0]*subarr[0] + subarr[1]*subarr[1])
        if norm < tsmax:
            locidx = k
            tsmax = norm
    return locidx # 会有多个？


def getkptidx(qt, kfarr): # 加速下
    '''
    在kfarray kptid u v 3dpid(-1表示没有对应) x y z
    中 找到 qt 像素点的位置
    '''
    data = kfarr # [:, 1:3] #np.around(kfarr[:, 1:3]).astype(np.int32)
    # tsmax = 100000.
    locidx = -1
    # for k in range(kfarr.shape[0]):
    #     flag = data[k, 3]
    #     if flag < 0: # 无效点
    #         continue
    #     kpt = data[k, 1:3]
    #     # if (int(round(kpt[0])) == qt[0]) and (int(round(kpt[1])) == qt[1]) : # 是此点
    #     #     print("debug")
    #     subarr = kpt - qt
    #     if math.sqrt(subarr[0]*subarr[0] + subarr[1]*subarr[1]) < tsmax :
    #         # locidx += [k]
    #         locidx = k
    #         tsmax = math.sqrt(subarr[0]*subarr[0] + subarr[1]*subarr[1])
    #         # return locidx
    # 改为 np.where
    arr1 = np.where(np.round(data[:, 1])==qt[0])
    if type(arr1) == tuple:
        arr1 = arr1[0]
    
    if arr1.shape[0] == 0 : # 连四舍五入后 都没有接近的 认为没有 不应该发生的
        return locidx
    # locidx = findloop(arr1, data, qt)
    sub = data[arr1, 1:3] - qt # (x,2)
    # 还要避免 flag==-1的
    if sub[data[arr1, 3]<0].shape[0] > 0:
        sub[data[arr1, 3]<0] += 1000000.
    norms = np.linalg.norm(sub, axis=1) # (x,)
    # 拿出上面数组 最小值索引
    min_norm_idx = np.argmin(norms)
    locidx = arr1[min_norm_idx]
    # for i in range(arr1.shape[0]):
    #     k = arr1[i]
    #     flag = data[k, 3]
    #     if flag < 0: # 无效点
    #         continue
    #     kpt = data[k, 1:3]
    #     subarr = kpt - qt
    #     norm = math.sqrt(subarr[0]*subarr[0] + subarr[1]*subarr[1])
    #     if norm < tsmax:
    #         locidx = k
    #         tsmax = norm
    return locidx # 会有多个？

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
    
    # 载入 mapptsfile 以字典方式存吧  每行长度不一
    dic_mappts = {}
    with open(mapptsfile, 'r') as f:
        lines = f.readlines()
    n_pts = len(lines) # 总3d点数
    xyzs = []
    for i in range(n_pts):
        line = lines[i]
        raw = line.split(' ')
        if len(raw) <= 4:
            print('[ERROR] this 3d pt has no 2d obs!')
            assert False
        num_obs = int((len(raw)-4) / 2)
        assert (len(raw)-4) % 2 == 0
        dic_mappts[int(raw[0])] = []
        xyz = np.array(raw[1:4], dtype=float) # (3)
        dic_mappts[int(raw[0])].append(xyz)
        xyz.reshape(1, -1)
        xyzs += [xyz]
        if len(raw) > 4:
            obsarr = np.array(list(map(int, raw[4:]))).reshape(num_obs, 2) # (n,2)
            dic_mappts[int(raw[0])].append(obsarr)
    # (mapptidx:[xyz, obsarr])
    xyzs = np.stack(xyzs, 0) # (n,3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs[:, :3]) # 相机位置的点云
    pcd.normals = o3d.utility.Vector3dVector(xyzs[:, :3] / np.linalg.norm(xyzs[:, :3], axis=-1, keepdims=True)) # 每个位置点到原点的距离  每个位置单位向量        
    # 对点云滤波 open3d的函数
    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
    # 原点云 边界
    print('raw pcd box: \n', pcd.get_axis_aligned_bounding_box())
    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5) # 20,30 2
    inlier_cloud = display_inlier_outlier(pcd, ind, vis=False) # , vis=True False
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]) #显示坐标系 1.0 20.0
    vis_lst = [inlier_cloud, mesh_frame]
    # o3d.visualization.draw_geometries(vis_lst)
    # inlier 的边界：
    print('inlier pcd box: \n', inlier_cloud.get_axis_aligned_bounding_box())
    # return -1
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
    

    # 投影的话还是要有每帧位姿 再跑前面的orbslam 插值 不准 就先只用kf的吧
    kf_orb_pose = copy.deepcopy(estpose)
    print('load orb-mono pose(tum): {} \n size: {}, {}'.format(kftrajfile, kf_orb_pose.shape[0], kf_orb_pose.shape[1]))
    dic_est = dict([(int(float(format(kf_orb_pose[i, 0], '.1f'))*10), kf_orb_pose[i, 1:]) for i in range(n_est)]) # key 就是 frame id
    
    # 逐帧看 image
    
    frame_reader = get_dataset(cfg, args, 1)
    n_img = frame_reader.__len__()
    K, H, W = update_cam(cfg) # 得到内参矩阵
    invK = np.linalg.inv(K)
    DEPTHSCALE = cfg['cam']['png_depth_scale']
    idx_map = [] # 记录每张图上 有效triangulate 以及 所有verts 和之前原3d 数据的idx 的映射关系 list 里是多个字典 每个字典是个List 里面两个字典
    # 分别表示 有效triangulate 和 verts
    
    # for idx, gt_color, _, _ in frame_loader:
    for idx in range (n_img):
        # idx = idx.item()
        if (not idx in dic_est.keys()):
            continue # 非kf 暂时不投影
        if idx != 813: # 813 > 0
            continue # break
        # if idx == 0: # 813 > 0
        #     continue # break
        if idx > 813: # 813 > 0 45
            break # break
        rgbpath = frame_reader.color_paths[idx]
        color_data = cv2.imread(rgbpath) # h,w,3 unit8
        print('process frame {}'.format(rgbpath))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB) 
        #Rectangle to be used with Subdiv2D
        size = color_data.shape # h, w, 3
        rect = (0,0,size[1],size[0])
        # 读入 ####.txt
        kffile = os.path.join(orbmapdir, "%04d.txt" % idx)
        # kptid u v 3dpid(-1表示没有对应) x y z
        kfarr = np.loadtxt(kffile, skiprows=1) # (N,7) , fmt='%d %d %d %d %.6f %.6f %.6f'
        # 转为字典 no
        kpts = kfarr[:, 1:3] # .astype(np.int32) # (n,2)
        kfarr1 = filterkpt(kfarr, ind) # 3d 点云上的过滤1次 看作再一次过滤 kfarr1 只是 mptid 那栏 falg 变了 看过到后面那帧时  是会变的
        # continue
        kpts_wdepth = kpts[kfarr1[:, 3]>0] # 那些有深度的点才用 
        # kfarr = copy.deepcopy(kfarr1) # debug
        # 还要就取出首行 Tcw
        tum_i = np.loadtxt(kffile, max_rows=1) # , fmt='%.1f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'
        assert int(tum_i[0]*10) == idx
        kf_tq = tum_i[1:]
        kf_w2c = np.array(TQtoSE3(kf_tq))
        kf_c2w = np.linalg.inv(kf_w2c)
        R_c2w = kf_c2w[0:3, 0:3]
        # 对 kpts 做2d上的三角剖分
        #Create an instance of Subdiv2d
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(np.around(kpts_wdepth))

        #Draw points
        for p in np.around(kpts_wdepth).astype(np.int32): #.astype(np.int32):
            draw_point(color_data,p,(255,0,0))
        win_delaunary = "%04d-delaunay triangulation" % idx
        #Show results
        # cv2.imshow(win_delaunary,color_data)
        # cv2.waitKey(0) # 注意 光标在窗口上时 按空格键 才能正常退出
        # save img dt kpt(只画 kpt)
        outvisfile = os.path.join('tridebug1', "kpt%04d.jpg" % idx)
        cv2.imwrite(outvisfile, cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB))
        
        # 估计粗糙的深度图 to do
        est_depth = np.zeros((size[0], size[1])) # 初始深度全0
        # 由于还没搞懂 subdiv locate 的含义 这里就简单粗暴的遍历来Locate 在哪个三角形吧
        trangleList = np.around(subdiv.getTriangleList()).astype(np.int32) # .astype(np.int32) # (8630,6 ?)
        # #Draw delaunary triangles 先用白色画全部的
        # draw_delaunay(color_data,trangleList,(255,255,255))
        tri_flag = np.ones(trangleList.shape[0]) # 记录 哪些三角面是有效的
        # 遍历每个像素
        curr_pcd = []
        for u in range(size[1]):
            for v in range(size[0]):
                # if (u != 538 or v != 87) and (u != 68 or v != 222):
                #     continue
                # if (u != 551 or v != 85): # kfarr中有两个 接近
                #     continue
                # if (u != 385 or v != 204): # 没找到
                #     continue

                qt = [u, v]
                ret = locate_triangle(np.array(qt), trangleList) # 对于在顶点上的判断有错误
                if ret[0] < 0 and ret[0] != -2: # ret[0] < 0 and 
                    continue
                
                if ret[0] == -2:
                    kpidxq = getkptidx(qt, kfarr1) # 造成下面gug是由于 这里返回-1
                    if kpidxq < 0:
                        print('ERROR. frame {}, qt: {}, {}'.format(idx, u, v))
                        assert False
                    qmapt = kfarr1[kpidxq, 4:7]
                    curr_pcd += [qmapt]
                    if kfarr1[kpidxq, 3] < 0:
                        print('ERROR') # 按道理前面三角形顶点已经过滤过了 为何这里还有
                        assert False
                else:
                    tri_valid = trangleList[ret[0]] # 当前像素在三级哦行内 所在三角形  (6) 各顶点坐标
                    pt1 = tri_valid[0:2] # (t[0],t[1]) # (u,v)
                    pt2 = tri_valid[2:4] 
                    pt3 = tri_valid[4:6]
                    # pts = np.stack([pt1, pt2, pt3], 0) # (3,2)
                    # 先找这顶点像素 在 原 kfarr 中的下标 np.around(kfarr[:, 1:3]).astype(np.int32)
                    kpidx1 = getkptidx(pt1, kfarr1)
                    if kpidx1 < 0:
                        print('ERROR. frame {}, pt1: {}, {}'.format(idx, pt1[0], pt1[1]))
                        assert False
                    kpidx2 = getkptidx(pt2, kfarr1) # 0000.jpg [551,85] 两个
                    if kpidx2 < 0:
                        print('ERROR. frame {}, pt2: {}, {}'.format(idx, pt2[0], pt2[1]))
                        assert False
                    kpidx3 = getkptidx(pt3, kfarr1)
                    if kpidx3 < 0:
                        print('ERROR. frame {}, pt3: {}, {}'.format(idx, pt3[0], pt3[1]))
                        assert False
                    if kfarr1[kpidx1, 3] == -1 or kfarr1[kpidx2, 3] == -1 or kfarr1[kpidx3, 3] == -1:
                        print('error')
                        assert False
                    # if not(kfarr1[kpidx1, 3] in ind) or not(kfarr1[kpidx2, 3] in ind) or not(kfarr1[kpidx3, 3] in ind): # pcd filter 认为是 离群点
                    #     print('error') # 上面这个判断 太耗时！ 关掉！
                    #     assert False
                    mapt1 = kfarr1[kpidx1, 4:7] # (3,)  为啥这里找到的点 还是 没有3d对应的？
                    mapt2 = kfarr1[kpidx2, 4:7]
                    mapt3 = kfarr1[kpidx3, 4:7]
                    # 计算 改平面法线 以及顶点0, 当前像素 的ray 的方向, ray 起点（当前pose的twc）
                    ray_origin = kf_c2w[:3, 3] # (3,)
                    qt += [1]
                    qt = np.array(qt) # (3,)
                    ray_dir = np.matmul(invK, qt) # 注意这里是 cam 坐标系 得转换到世界系
                    ray_dir_w = np.matmul(R_c2w, ray_dir)
                    ray_dirwunit = ray_dir_w / np.linalg.norm(ray_dir_w) # / np.linalg.norm(ray_dir_w) 测试不是单位方向
                    plane_norm = np.cross(mapt1-mapt2, mapt1-mapt3) # 在这里出现nan了 是因为此三角形 2d上很近 面积很小 3d 上有两个点很接近
                    # debug = (np.linalg.norm(plane_norm)==np.linalg.norm(plane_norm)) # (True in np.isnan(plane_norm))
                    
                    plane_norm = plane_norm / np.linalg.norm(plane_norm)
                    qmapt = RayCastTriPlane(ray_origin, ray_dirwunit, mapt1, plane_norm) # (3,)
                    if type(qmapt) == int:
                        if qmapt == -3:
                            if ( idx <= 45 ): 
                                print('t<0, skip')
                            continue # 此交点无效
                        if qmapt==-2 or qmapt==-1:
                            print('RayCastTriPlane ERROR')
                            assert False
                if (True in np.isnan(qmapt)):
                    # 设置此三角面无效
                    tri_flag[ret[0]] = -1
                    # print('plane has nan, skip this, 3mapt: \n', mapt1, mapt2, mapt3)
                    continue
                # 将上面求得 估计的 交点3d点(世界系) 投到相机上 就得到深度
                qmapt_h = np.concatenate([qmapt, np.array([1])], 0) # (4,)
                qmapt_c = np.matmul(kf_w2c, qmapt_h)[:3] # (3,)
                dd = qmapt_c[2]
                if dd < 0 or dd > 1.5: # 10 9 2.5(因为这本来就是尺度变小近3倍)
                    if ( True ): # idx <= 45
                        print('query is kpt: {}, invalid depth: {}'.format( (ret[0] == -2), dd))
                    continue
                if np.isclose(dd, 0):
                    print('query is kpt: {}, coarse depth close 0: {}'.format( (ret[0] == -2), dd))
                    break
                # 验证该点 是否投到当前像素点
                qmapt_c_img = np.around(np.matmul(K, (qmapt_c/dd))) # [0:2] 会出现全 nan
                contain_nan = (True in np.isnan(qmapt_c_img))
                if contain_nan:
                    print('proj has nan')
                debug = abs(qmapt_c_img[0]-u) <= 3. and abs(qmapt_c_img[1]-v) <= 3.
                if not debug and (ret[0] >= 0 ):
                    print('query: ', qt)
                    print('depth proj: ', qmapt_c_img)
                
                # assert debug
                # assert ( abs(qmapt_c_img[0]-u) <= 1. and abs(qmapt_c_img[1]-v) <= 1.)
                # 对深度赋值  值会有问题
                est_depth[v, u] = qmapt_c[2]
        
        curr_pcd = np.stack(curr_pcd, 0) # (n,3)
        # 转为点云pcd保存
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(curr_pcd[:, :3]) # 相机位置的点云
        pcd.normals = o3d.utility.Vector3dVector(curr_pcd[:, :3] / np.linalg.norm(curr_pcd[:, :3], axis=-1, keepdims=True)) # 每个位置点到原点的距离  每个位置单位向量
        o3d.io.write_point_cloud(os.path.join('tridebug1', "pts%04d.ply" % idx), pcd)
        #Draw delaunary triangles 再用绿色 画出保留下来的
        draw_delaunay(color_data,trangleList[tri_flag>0],(0,255,0))
        # draw_delaunay(color_data,trangleList[tri_flag<0],(255,255,255)) # 再画出白色的边 表示 被舍弃的
        # 对当前估计的深度图 可视化 可视化有点问题
        fig, axs = plt.subplots(1, 1)
        fig.tight_layout()
        max_depth = np.max(est_depth)
        axs.imshow(est_depth, cmap="plasma",
                   vmin=0, vmax=2.5)
        # axs.set_title('coarse depth')
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(os.path.join('tridebug1', "pltcd%04d.png" % idx), dpi = 200)
        plt.show()
        # est_depth_vis = est_depth/np.max(est_depth)*255
        # est_depth_vis = np.clip(est_depth_vis, 0, 255).astype(np.uint8)
        # est_depth_vis = cv2.applyColorMap(est_depth_vis, cv2.COLORMAP_JET)
        # est_depth_vis = cv2.cvtColor(est_depth.astype(np.float32), cv2.COLOR_GRAY2RGB)
        # 在深度图上画 关键点位置
        # # Draw points
        # for p in np.around(kpts_wdepth).astype(np.int32): #.astype(np.int32):
        #     draw_point(est_depth_vis,p,(0,0,255))
        # est_depth_vis = cv2.cvtColor(est_depth_vis, cv2.COLOR_GRAY2RGB)
        # whole = np.concatenate([color_data, est_depth_vis], axis=1)
        # tH = H #//2
        # tW = W #//2 #为了保证ffmpeg拿到的是大小能被2整除
        # whole  = cv2.resize(whole, (tW+tW%2, tH+tH%2))
        # cv2.imshow(f'dt and coarse depth', whole[:, :, ::-1])
        # cv2.waitKey(0)
        # 保存为 uint 格式 和原gt depth一样格式吧 6553.5
        est_depth = est_depth * DEPTHSCALE
        # 对于实际深度> 65535 的就按最大值 就会成为白色
        depthclip = np.clip(est_depth, 0 , 65535) # 10000
        # cv2.imshow(f'keypoint depth vis', depthclip)
        # cv2.waitKey(0)
        savepath = os.path.join('tridebug1', "cd%04d.png" % idx) # triangulation/coarsedepth
        # cv2.imwrite(savepath, depthclip.astype(np.uint16))
        outvisfile = os.path.join('tridebug1', "dt%04d.jpg" % idx)
        cv2.imwrite(outvisfile, cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB))
        print('save coasrse depth: {}'.format(savepath))



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
    