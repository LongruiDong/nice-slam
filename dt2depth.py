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

subdiv https://zhuanlan.zhihu.com/p/340510482
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

import cv2

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
    cv2.circle(img,p,2,color)

#Draw delaunay triangles
def draw_delaunay(img,subdiv,delaunay_color):
    trangleList = subdiv.getTriangleList().astype(np.int32) # (8630,6 ?)
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


def main(cfg, args, orbmapdir="/home/dlr/Project1/ORB_SLAM2_Enhanced/result"):
    
    mapptsfile = os.path.join(orbmapdir, 'mappts.txt')
    kftrajfile = os.path.join(orbmapdir, 'KeyFrameTrajectory.txt')
    gttrajfile = os.path.join(cfg['data']['input_folder'], 'traj.txt')
    
    # 载入 mapptsfile 以字典方式存吧  每行长度不一
    dic_mappts = {}
    with open(mapptsfile, 'r') as f:
        lines = f.readlines()
    n_pts = len(lines) # 总3d点数
    
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
        if len(raw) > 4:
            obsarr = np.array(list(map(int, raw[4:]))).reshape(num_obs, 2) # (n,2)
            dic_mappts[int(raw[0])].append(obsarr)
    # (mapptidx:[xyz, obsarr])
            
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
    
    
    # 投影的话还是要有每帧位姿 再跑前面的orbslam 插值 不准 就先只用kf的吧
    kf_orb_pose = copy.deepcopy(estpose)
    print('load orb-mono pose(tum): {} \n size: {}, {}'.format(kftrajfile, kf_orb_pose.shape[0], kf_orb_pose.shape[1]))
    dic_est = dict([(int(float(format(kf_orb_pose[i, 0], '.1f'))*10), kf_orb_pose[i, 1:]) for i in range(n_est)]) # key 就是 frame id
    
    # 逐帧看 image
    frame_reader = get_dataset(cfg, args, scale)
    # frame_loader = DataLoader(
    #         frame_reader, batch_size=1, shuffle=False, num_workers=1)
    n_img = frame_reader.__len__()
    K, H, W = update_cam(cfg) # 得到内参矩阵
    
    idx_map = [] # 记录每张图上 有效triangulate 以及 所有verts 和之前原3d 数据的idx 的映射关系 list 里是多个字典 每个字典是个List 里面两个字典
    # 分别表示 有效triangulate 和 verts
    
    # for idx, gt_color, _, _ in frame_loader:
    for idx in range (n_img):
        # idx = idx.item()
        if (not idx in dic_est.keys()):
            continue # 非kf 暂时不投影
        # if idx > 0:
        #     break
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
        # 还要就取出首行 Tcw
        tum_i = np.loadtxt(kffile, max_rows=1) # , fmt='%.1f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'
        assert int(tum_i[0]*10) == idx
        kf_tq = tum_i[1:]
        kf_w2c = np.array(TQtoSE3(kf_tq))
        # 拿出orb 的 pose
        orb_tq = dic_est[idx]
        orb_c2w = np.array(TQtoSE3(orb_tq))
        orb_w2c = np.linalg.inv(orb_c2w)
        debug = np.matmul(kf_w2c, orb_c2w)
        print('test kf w2c: \n', debug)
        # 对 kpts 做2d上的三角剖分
        #Create an instance of Subdiv2d
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(kpts)
        #Draw delaunary triangles
        draw_delaunay(color_data,subdiv,(255,255,255))

        #Draw points
        for p in kpts.astype(np.int32):
            draw_point(color_data,p,(0,0,255))
        win_delaunary = "%04d-delaunay triangulation" % idx
        #Show results
        # cv2.imshow(win_delaunary,color_data)
        # cv2.waitKey(0)
        # save img
        outvisfile = "dt%04d.jpg" % idx
        cv2.imwrite(outvisfile, color_data)
        
        # 估计粗糙的深度图 to do
        
        
        
        




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
    
    main(cfg, args, orbmapdir=orbmapdir)