# -*- coding:utf-8 -*-
"""_summary_
从图像文件夹得到视频
"""
import os
import argparse
import copy
import numpy as np
import cv2
import glob


def image_to_video(filedir, outfile, H=480, W=640, fps=1):
    # file = 'de/test4/1/'  # 图片目录
    output = outfile+'.mp4'  # 生成视频路径
    num = os.listdir(filedir)  # 生成图片目录下以图片名字为内容的列表
    height = H
    weight = W
    
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv2.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    left_paths = sorted(glob.glob(os.path.join( # Datasets/TartanAir/office/Easy/P000/ image_left
            filedir, '*.jpg')), key=lambda x: int(os.path.basename(x)[0:9])) #整个timestamp
    for i in range(len(left_paths)):
        rawpath = left_paths[i]
        rawidx = int(rawpath[-13:-9])
        # if i<48:
        #     continue
        # print()
        if int(rawpath[-8:-4]) < 25 and (rawidx>0): #只看最后次迭代 40 30 25
            continue
        elif (rawidx==0) and int(rawpath[-8:-4]) < 1470:
            continue
        else:
            pass
        path = rawpath #filedir + str(i) + '.jpg'
        print(path)
        frame = cv2.imread(path)
        videowriter.write(frame)

    videowriter.release()


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    # parser.add_argument('--output', type=str,
    #                     help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()
    
    outname = os.path.join(args.input_folder,'track_vis') # track_vis  map_vis
    indir = os.path.join(args.input_folder,'tracking_vis') #mapping_vis tracking_vis
    image_to_video(indir, outname, H=419, W=612, fps=2)
    
        

if __name__ == '__main__':
    main()
