# -*- coding:utf-8 -*-
"""_summary_
按照zihan的建议
可视化 gt mesh 和我自己得到的点云！
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


def main():
    # truncd-gtrawextr.ply
    # Datasets/Replica/office0_mesh.ply  output/Replica/office0/mesh/final_mesh_eval_rec.ply
    gtmeshf = '/home/dlr/Project/nice-slam/output/tartanair/office/P001-gt/mesh/final_mesh_eval_rec.ply' # office0 room0 Datasets/Replica/cull_replica_mesh/office0.ply
    print('load replica gt mesh:\n {}'.format(gtmeshf))
    gtmesh = o3d.io.read_triangle_mesh(gtmeshf)
    
    #得到我之前creatpcd的点云
    mypcd = 'Datasets/Replica/office0/truncd-gtrawextr.ply'
    print('load my ply:\n {}'.format(mypcd))
    myply = o3d.io.read_point_cloud(mypcd)
    #获取真实bound
    bbox = gtmesh.get_axis_aligned_bounding_box()
    print('gtmesh axis_aligned_bounding_box: \n', bbox)
    # 计算轴对齐边界框
    abox = myply.get_axis_aligned_bounding_box()
    print('myply axis_aligned_bounding_box: \n', abox)
    # aabb.color = (1, 0, 0)
    pcdarray = np.asarray(myply.points)
    print('myply shape: \n', pcdarray.shape) #(n,3)
    # 输出 xyz 的区域
    print('x y z min:\n', pcdarray.min(axis=0))
    print('x y z max:\n', pcdarray.max(axis=0))
    gtmpcd = gtmesh.sample_points_uniformly(number_of_points=pcdarray.shape[0])
    cbox = gtmpcd.get_axis_aligned_bounding_box()
    print('gtmpcd axis_aligned_bounding_box: \n', cbox)
    #保存最终点云
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.15) # .voxel_down_sample(voxel_size=0.7) 0.4 0.15 0.2 0.01 0.02
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]) #显示坐标系 1.0 20.0
    gtmesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([mesh_frame, gtmesh]) #m gtmesh mypcd, mesh_frame gtmpcd
    
        

if __name__ == '__main__':
    main()
