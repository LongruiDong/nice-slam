import glob
from logging import raiseExceptions
import os
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import copy
# 是否用双目生成新的depth 对于双目数据
makedepth = False # False True
usegtdepth = True # 对于合成数据vktti tartanair 是否使用原始完美深度
add_depthnoise = False #对真实depth是否添加噪声

#在给定2d array上某个索引位置进行双线性插值
def bilinear_interp(dataary, qx, qy):
    dst = copy.deepcopy(dataary)
    h ,w = dataary.shape
    # 计算在源图上 4 个近邻点的位置
    # i,j h w
    i = int(np.floor(qx))
    j = int(np.floor(qy))
    # u = qx-i
    # v = qy-j
    if j == w-1:
        j = w-2
    if i == h-1:
        i = h-2
    dst[qx, qy] = (1/3)*(#dataary[i, j] + \
                dataary[i+1, j] + dataary[i, j+1]+ dataary[i+1, j+1])
    return dst

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.npy' in depth_path: #对于tartanqir来说 gt depth 保存真实数值 m 0~ 10000（infinite）
            depth_data = np.load(depth_path)
            # 实际上给的深度是有大于 10000 即使是office 所以先clip吧
            depth_data = np.clip(depth_data, 0, 10000)
            # 对于office0 76 82 83 上的3个外点做插值处理
            if index == 76 or index == 82 or index == 83:
                qx, qy = np.argwhere(depth_data>30)[0]
                print('bf interp, max: {}'.format(np.max(depth_data)))
                print('depth[{}, {}] = {}'.format(qx, qy, depth_data[qx, qy]))
                # 在 [qx, qy]处进行双线性插值
                depth_data = bilinear_interp(depth_data, qx, qy)
                print('aft interp, max: {}'.format(np.max(depth_data)))
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data)*self.scale
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)


class Replica(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Azure(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Azure, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(
            self.input_folder, 'scene', 'trajectory.log'))
        self.plot_traj()

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
    
    # 也画出轨迹 想看看bound to do
    def plot_traj(self):
        self.trajzx = []
        for i in range(len(self.poses)):
            c2w = self.poses[i].numpy()
            self.trajzx.append([c2w[0,3], c2w[1,3], c2w[2,3]]) # (x,y,z)
        traj = np.array(self.trajzx) # n,3
        # 输出 xyz 的区域
        print('x y z min:\n', traj.min(axis=0))
        print('x y z max:\n', traj.max(axis=0))
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(traj[:,0],traj[:,2], linestyle='dashed',c='k') # x,z
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.legend(['Ground Truth'])
        plt.axis('equal')
        savefigname = os.path.join(self.input_folder, 'gtwnerf.pdf')    
        plt.savefig(savefigname)
        plt.close(fig)


class ScanNet(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(CoFusion, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'colour', '*.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth_noise', '*.exr')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'trajectories'))

    def load_poses(self, path):
        # We tried, but cannot align the coordinate frame of cofusion to ours.
        # So here we provide identity matrix as proxy. ?? 相机没动？
        # But it will not affect the calculation of ATE since camera trajectories can be aligned.
        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None: # 首帧位姿归I 但是好像只有tum做了归一化处理 why
                inv_pose = np.linalg.inv(c2w) #T0w
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w # T0w Twi = T0i
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

# 增加数据api for vkitti2
class vKITTI2(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(vKITTI2, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(glob.glob(os.path.join( # Datasets/vkitti2/Scene01/clone/frames/depth/Camera_0/depth_00000.png
            self.input_folder, 'frames', 'rgb', 'Camera_0', '*.jpg')), key=lambda x: int(os.path.basename(x)[-9:-4])) #00000
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'frames', 'depth', 'Camera_0', '*.png')), key=lambda x: int(os.path.basename(x)[-9:-4]))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/extrinsic.txt') # Datasets/vkitti2/Scene01/clone/extrinsic.txt
        self.plot_traj() # 保存 世界系下 (已转为算法需要的nerf 坐标系)下的轨迹 来估计Bound
        self.input_folder = os.path.join(self.input_folder, 'frames') # Datasets/vkitti2/Scene01/clone/frames
    
    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        inv_pose = None
        for i in range(self.n_img): # 文件中 左右目都在
            j = 2*i+1 # 首行跳过 表示使用 Camera_0 Camera_1 : 2*(i+1)
            line = lines[j] # 18个值 只要后16
            w2c = np.array(list(map(float, line.split())))[2:].reshape(4, 4) #Tiw
            c2w = np.linalg.inv(w2c) #Twi
            if inv_pose is None: # 首帧位姿归I
                inv_pose = w2c  # np.linalg.inv(c2w) #T0w
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w # T0w Twi = T0i
            c2w[:3, 1] *= -1 # nerf-pytorch y和z取反 此操作（首帧对齐后）和完整的坐标系变换结果是等价 
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
    # 也画出轨迹 想看看bound to do
    def plot_traj(self):
        self.trajzx = []
        for i in range(len(self.poses)):
            c2w = self.poses[i].numpy()
            self.trajzx.append([c2w[0,3], c2w[1,3], c2w[2,3]]) # (x,y,z)
        traj = np.array(self.trajzx) # n,3
        # 输出 xyz 的区域
        print('x y z min:\n', traj.min(axis=0))
        print('x y z max:\n', traj.max(axis=0))
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(traj[:,0],traj[:,2], linestyle='dashed',c='k') # x,z
        # plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.legend(['Ground Truth'])
        # plt.title(title)

        plt.axis('equal')
        savefigname = os.path.join(self.input_folder, 'gtwnerf.pdf')    
        plt.savefig(savefigname)
        
        # if vis:
        #     plt.show()

        plt.close(fig)


# 增加数据api for tcsvt
class tcsvt(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(tcsvt, self).__init__(cfg, args, scale, device)
        self.left_paths = sorted(glob.glob(os.path.join( # Datasets/tcsvt/whole-floor/W2/glass/cam0/data/00000072406369452870.png
            self.input_folder, 'cam0', 'data', '*.png')), key=lambda x: int(os.path.basename(x)[:-4])) #整个timestamp
        self.right_paths = sorted(glob.glob(os.path.join( # Datasets/tcsvt/whole-floor/W2/glass/cam0/data/00000072406369452870.png
            self.input_folder, 'cam1', 'data', '*.png')), key=lambda x: int(os.path.basename(x)[:-4])) #整个timestamp
        self.colordir = os.path.join(self.input_folder, 'color0') #双目去畸变并rectify后的左目图路径
        self.colordir1 = os.path.join(self.input_folder, 'color1')
        self.depthdir = os.path.join(self.input_folder, 'depth') #通过双目得到的左目深度图路径
        self.dispdir = os.path.join(self.input_folder, 'disp') #通过双目得到的左目深度图路径
        if not os.path.exists(self.colordir):
            os.makedirs(self.colordir)

        if not os.path.exists(self.colordir1):
            os.makedirs(self.colordir1)

        if not os.path.exists(self.depthdir):
            os.makedirs(self.depthdir)
        
        if not os.path.exists(self.dispdir):
            os.makedirs(self.dispdir)
            
        # 对双目图像处理 并保存rgb-d
        self.rectifystereo()
        
        self.color_paths, self.depth_paths, self.poses = self.loadtcsvt(
            self.input_folder, frame_rate=32) # 实际30Hz
        
        self.n_img = len(self.color_paths)
        print('seq lenth: ', self.n_img)
        self.plot_traj() # 保存 世界系下 (已转为算法需要的 矫正相机nerf 坐标系)下的轨迹 来估计Bound
    
    
    def undistort_fisheye(self, img_path,K,D,DIM,scale=1,imshow=False):
        img = cv2.imread(img_path)
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if dim1[0]!=DIM[0]:
            img = cv2.resize(img,DIM,interpolation=cv2.INTER_LINEAR)
        Knew = K.copy()
        # if scale:#change fov  这里pengzhen说可以s>1 之后尝试
        #     Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_32FC1) #CV_32FC1 CV_16SC2
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) #BORDER_CONSTANT BORDER_TRANSPARENT  INTER_LINEAR INTER_CUBIC
        # if imshow:
        #     cv2.imshow("undistorted/", undistorted_img)
        return undistorted_img
    
    # 参照之前写过的处理代码
    def rectifystereo(self):
        # 读取该数据下sensor.yaml
        skip_lines = 2
        leftymp = os.path.join(self.input_folder, 'cam0', 'sensor.yaml')
        rightymp = os.path.join(self.input_folder, 'cam1', 'sensor.yaml')
        with open(leftymp) as f:
            for i in range(skip_lines):
                f.readline()
            yl = yaml.safe_load(f)
            raw_size = yl['resolution'] # [W, H] 640,400
            intrinsics_vec = yl["intrinsics"] # fx fy cx cy
            distortion_vec = yl["distortion_coefficients"]
            T_Bl = np.array(yl['T_BS']['data']).reshape(4,4)
            K_l = np.array([intrinsics_vec[0], 0.0, intrinsics_vec[2], 0.0, intrinsics_vec[1], intrinsics_vec[3], 0.0, 0.0, 1.0]).reshape(3,3)
            d_l = np.array(distortion_vec).reshape(8)
        
        with open(rightymp) as f:
            for i in range(skip_lines):
                f.readline()
            yr = yaml.safe_load(f)
            intrinsics_vec = yr["intrinsics"]
            distortion_vec = yr["distortion_coefficients"]
            T_Br = np.array(yr['T_BS']['data']).reshape(4,4)
            K_r = np.array([intrinsics_vec[0], 0.0, intrinsics_vec[2], 0.0, intrinsics_vec[1], intrinsics_vec[3], 0.0, 0.0, 1.0]).reshape(3,3)
            d_r = np.array(distortion_vec).reshape(8)
        print('\n')
        print('K_l: \n',K_l)
        print('K_r: \n',K_r)
        print('d_l: \n', d_l)
        print('d_r: \n', d_r)
        print('T_Bl: \n', T_Bl)
        print('T_Br: \n', T_Br)
        
        T_rl = np.linalg.inv(T_Br).dot(T_Bl)
        R_rl = T_rl[0:3,0:3]
        t_rl = T_rl[0:3,3]
        
        R_l, R_r, P_l, P_r, _,_,_= cv2.stereoRectify(K_l, np.zeros(4), K_r, np.zeros(4), (raw_size[0], raw_size[1]), R_rl, t_rl,
                                                 flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(raw_size[0], raw_size[1])) #为了不改变fov 就设为1  0就是去掉所有黑边
        intrinsics_vec = [P_l[0,0], P_l[1,1], P_l[0,2], P_l[1,2]] #？
        map_l = cv2.initUndistortRectifyMap(K_l, np.zeros(5), R_l, P_l[:3,:3], (raw_size[0], raw_size[1]), cv2.CV_32FC1) #需要W,H [:3,:3]
        map_r = cv2.initUndistortRectifyMap(K_r, np.zeros(5), R_r, P_r[:3,:3], (raw_size[0], raw_size[1]), cv2.CV_32FC1)
        T_l43 = np.vstack((R_l, np.array([0,0,0])))
        T_l = np.hstack((T_l43, np.array([[0],[0],[0],[1]]))) #Trect_l
        T_l_inv = np.linalg.inv(T_l) #Tl_rect
        # global T_Brect
        T_Brect = T_Bl.dot(T_l_inv) # 矫正相机系 到 imu(gt)
        T_rectB = np.linalg.inv(T_Brect) # 原gt 到 矫正相机系
        self.T_rectB = T_rectB
        if True:
            print("T_rl: \n",T_rl)
            print("R_l: \n",R_l)
            print("R_r: \n",R_r)
            print("P_l: \n",P_l)
            print("P_r: \n",P_r)
            print("intrinsics vec: \n", intrinsics_vec) #要写入到 nice-slam 对应数据yaml
            print("T_Brect: \n", T_Brect)
            print('(H,W): ({},{})'.format(raw_size[1],raw_size[0]))
        ht0, wd0 = [raw_size[1],raw_size[0]]
        self.intrinsics_vec = intrinsics_vec
        self.rbaseline = - P_r[0,3] / intrinsics_vec[0] # --fb/f
        
        return
        # read all png images in folder 以左目为准
        images_left = sorted(glob.glob(os.path.join(self.input_folder, 'cam0/data/*.png')))
        images_right = [x.replace('cam0', 'cam1') for x in images_left]
        rgblist = os.path.join(self.input_folder, 'rgb.txt')
        depthlist = os.path.join(self.input_folder, 'depth.txt')
        # fr = open(rgblist, "w")
        # fd = open(depthlist,"w")
        for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
            if True and not os.path.isfile(imgR):
                continue
            tstamp = int(float(imgL.split('/')[-1][:-4]) )
            #测试 先去畸变 得到正常双目 再正常双目校正
            imgl_un = self.undistort_fisheye(imgL,K_l,d_l[:4],raw_size,scale=1,imshow=False) 
            imgr_un = self.undistort_fisheye(imgR,K_r,d_r[:4],raw_size,scale=1,imshow=False) 
                
            imgl_rect = cv2.remap(imgl_un, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR) #(400,640,3)
            imgr_rect = cv2.remap(imgr_un, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)
            testfile = str(tstamp)+'.png'
            # print("an img size : ", images[0].shape)

            # 把处理后的stereo保存
            stereo0path = os.path.join(self.colordir,testfile)
            stereo1path = stereo0path.replace('color0', 'color1')
            cv2.imwrite(stereo0path, imgl_rect)
            cv2.imwrite(stereo1path, imgr_rect)
            
            #得到深度图
            #再次读取 注意读为8位单通道
            imgl_rect0 = cv2.imread(stereo0path,0) # 400 640
            imgr_rect0 = cv2.imread(stereo1path,0)
            disp, depth = self.makergbd(imgl_rect0, imgr_rect0, testfile)
            #视差图normalize下
            dispnorm= cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(os.path.join(self.dispdir,testfile), dispnorm)
            if t < 2: #查看中间去畸变 和 校正对比
                distortst = np.hstack((imgl_un, imgr_un))
                testimgs = np.hstack((imgl_rect, imgr_rect))
                cv2.imwrite(os.path.join('output','rect_'+testfile), testimgs)
                cv2.imwrite(os.path.join('output','und_'+testfile), distortst)
                # cv2.imwrite(os.path.join('output','disp_'+testfile), dispnorm)
                cv2.imwrite(os.path.join('output','depth_'+testfile), depth)
                np.savetxt(os.path.join('output','depth_'+str(tstamp)+'.txt'), depth)
                np.savetxt(os.path.join('output','disp_'+str(tstamp)+'.txt'), disp)
                print('savetxt')
            # #写入文件名文件
            # fr.write('{} {}\n'.format(str(tstamp), testfile))
            # fd.write('{} {}\n'.format(str(tstamp), testfile))
        
        # fr.close()
        # fd.close()
        
    # 得到左目深度图 并保存
    def makergbd(self, imgl, imgr, filename):
        min_disp = 0 #216
        num_disp = 32 - min_disp
        blocksize = 11
        # window_size = 5
        #StereoBM_create() StereoSGBM_create minDisparity=min_disp, 
        stereo = cv2.StereoBM_create(numDisparities=num_disp , blockSize=blocksize
                                    #    P1 = 8*1*blocksize**2,
                                    #     P2 = 32*1*blocksize**2,
                                    #    speckleWindowSize = 100
                                    #    speckleRange = 1
                                       )
        disparity = stereo.compute(imgl, imgr) # 400 640 int16 CV_16S
        disptmp = disparity.astype(np.float32)/16
        # disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # disparity = disparity.astype(np.uint8) # uint8 负值截断位0
        fx = self.intrinsics_vec[0]
        baseline = self.rbaseline
        # 转换
        height = disparity.shape[0]
        width = disparity.shape[1]
        depth = np.zeros((height, width), dtype=np.float) # uint16 np.float
        for i in range(height):
            for j in range(width):
                if disptmp[i,j] > 0: #视差为负值怎么处理？
                    depth[i,j] = fx* baseline / float(disptmp[i,j])
        
        # depth = depth.astype(np.ushort)
        # 归一化
        
        # depth = fx* baseline / disparity.astype(np.float32) # 这里的单位是？ 会有- inf
        DEPTHSCALE = 800. #注意深度尺度1000
        depth = depth * DEPTHSCALE
        # 对于实际深度> 65535 的就按最大值 就会成为白色
        depthclip = np.clip(depth, 0 , 65535)
        #保存深度图
        cv2.imwrite(os.path.join(self.depthdir,filename), depthclip.astype(np.uint16)) # 保存未16位 无符号png
        # indepth = cv2.imread(os.path.join(self.depthdir,filename),cv2.IMREAD_UNCHANGED) #验证恢复
        return disptmp, depth/DEPTHSCALE
        # pass
    
    def parse_list(self, filepath, skiprows=0, delimiter = ' '):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=delimiter,
                          dtype=np.unicode_, skiprows=skiprows)
        return data
    
    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtcsvt(self, datapath, frame_rate=-1):
        """ read video data in tcsvt-rgbd format """
        pathseg = datapath.split('/') #Datasets/tcsvt/whole-floor/W2/glass
        if os.path.isfile(os.path.join('Datasets/tcsvt/gt_result2022', pathseg[2], pathseg[3], 'glass/gba_pose.csv')):
            pose_list = os.path.join('Datasets/tcsvt/gt_result2022', pathseg[2], pathseg[3], 'glass/gba_pose.csv')
        else:
            raiseExceptions(True)
            
        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, delimiter=',')
        pose_vecs = pose_data[:, 1:].astype(np.float64) # tx ty tz w x y z

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose, max_dt=20000000.)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate: # 30Hz 0.03 
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        T_rectb = self.T_rectB
        T_brect = np.linalg.inv(T_rectb)
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, 'color0', image_data[i, 1])]
            depths += [os.path.join(datapath, 'depth', depth_data[j, 1])]
            b2w = self.pose_matrix_from_quaternion(pose_vecs[k]) #gt 是imu系 Twb
            # 转为校正相机系 Trectb Twb Tbrect
            c2w = (T_rectb.dot(b2w)).dot(T_brect)
            if inv_pose is None: # 首帧位姿归I 但是好像只有tum做了归一化处理 why
                inv_pose = np.linalg.inv(c2w) #T0w
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w # T0w Twi = T0i
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4) #对于tcsvt的接口
        qxyzw = [pvec[4],pvec[5],pvec[6],pvec[3]] # pvec: tx ty tz w x y z -> x y z w
        pose[:3, :3] = Rotation.from_quat(qxyzw).as_matrix() # xyz w
        pose[:3, 3] = pvec[:3]
        return pose

    # 也画出轨迹 想看看bound to do
    def plot_traj(self):
        self.trajzx = []
        for i in range(len(self.poses)):
            c2w = self.poses[i].numpy()
            self.trajzx.append([c2w[0,3], c2w[1,3], c2w[2,3]]) # (x,y,z)
        traj = np.array(self.trajzx) # n,3
        # 输出 xyz 的区域
        print('x y z min:\n', traj.min(axis=0))
        print('x y z max:\n', traj.max(axis=0))
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(traj[:,0],traj[:,2], linestyle='dashed',c='k') # x,z
        # plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.legend(['Ground Truth'])
        # plt.title(title)

        plt.axis('equal')
        savefigname = os.path.join(self.input_folder, 'gtwnerf.pdf')    
        plt.savefig(savefigname)
        
        # if vis:
        #     plt.show()

        plt.close(fig)

# 添加数据api for TartanAir
class TartanAir(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TartanAir, self).__init__(cfg, args, scale, device)
        self.left_paths = sorted(glob.glob(os.path.join( # Datasets/TartanAir/office/Easy/P000/ image_left
            self.input_folder, 'image_left', '*.png')), key=lambda x: int(os.path.basename(x)[0:6])) #整个timestamp
        self.right_paths = sorted(glob.glob(os.path.join( # Datasets/TartanAir/office/Easy/P000/ image_right
            self.input_folder, 'image_right', '*.png')), key=lambda x: int(os.path.basename(x)[0:6])) #整个timestamp
        # self.colordir = os.path.join(self.input_folder, 'color0') #双目去畸变并rectify后的左目图路径
        # self.colordir1 = os.path.join(self.input_folder, 'color1')
        self.gtdepth_paths = sorted(glob.glob(os.path.join( # Datasets/TartanAir/office/Easy/P000/ depth_left
            self.input_folder, 'depth_left', '*.npy')), key=lambda x: int(os.path.basename(x)[0:6]))
        # self.depthdir = os.path.join(self.input_folder, 'depth') #通过双目得到的左目深度图路径
        # self.dispdir = os.path.join(self.input_folder, 'disp') #通过双目得到的左目视差图路径
        # self.noisedepthdir = os.path.join(self.input_folder, 'noisedepth') #增加噪声后的深度
        # if not os.path.exists(self.noisedepthdir):
        #     os.makedirs(self.noisedepthdir)
        # if not os.path.exists(self.depthdir):
        #     os.makedirs(self.depthdir)
        
        # if not os.path.exists(self.dispdir):
        #     os.makedirs(self.dispdir)
            
        # # 对双目图像处理 并保存rgb-d
        # self.rectifystereo()
        if makedepth:
            self.makergbd()
        if add_depthnoise:
            print('use gtdepth add noise!')
            # self.noisedepth() #已经生成过了
        self.color_paths, self.depth_paths, self.poses = self.loadtcsvt(
            self.input_folder) # 实际
        
        self.n_img = len(self.color_paths)
        print('seq lenth: ', self.n_img)
        self.plot_traj() # 保存 世界系下 (已转为算法需要的 矫正相机nerf 坐标系)下的轨迹 来估计Bound
        
    # 得到左目深度图 并保存
    def makergbd(self):
        for t in range(len(self.left_paths)):
            imgl = cv2.imread(self.left_paths[t], 0) # 480 640 3->480 640 读为灰度图
            imgr = cv2.imread(self.right_paths[t], 0)
            frameid = '%06d' % t
            filename = frameid + '.png'
            min_disp = 7 #216
            num_disp = 71 - min_disp
            blocksize = 11
            # window_size = 5
            #StereoBM_create() StereoSGBM_create minDisparity=min_disp, 
            # stereo = cv2.StereoBM_create(numDisparities=num_disp , blockSize=blocksize)
            stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, 
                                         blockSize=blocksize,
                                         P1 = 8*1*blocksize**2,
                                        #  P2 = 32*1*blocksize**2,
                                         speckleWindowSize = 100,
                                         speckleRange = 1
                                        )
            disparity = stereo.compute(imgl, imgr) # 400 640 int16 CV_16S
            disptmp = disparity.astype(np.float32)/16
            # disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # disparity = disparity.astype(np.uint8) # uint8 负值截断位0
            fx = self.fx
            baseline = 0.25 #m
            # 转换
            height = disparity.shape[0]
            width = disparity.shape[1]
            depth = np.zeros((height, width), dtype=np.float) # uint16 np.float
            for i in range(height):
                for j in range(width):
                    if disptmp[i,j] > 0: #视差为负值怎么处理？
                        depth[i,j] = fx* baseline / float(disptmp[i,j])
            
            # depth = depth.astype(np.ushort)
            # 归一化
            
            # depth = fx* baseline / disparity.astype(np.float32) # 这里的单位是？ 会有- inf
            DEPTHSCALE = 1. #注意深度尺度1000
            depth = depth * DEPTHSCALE
            # 对于实际深度> 65535 的就按最大值 就会成为白色
            depthclip = np.clip(depth, 0 , 10000)
            #保存深度图
            cv2.imwrite(os.path.join(self.depthdir,filename), depthclip.astype(np.uint16)) # 保存未16位 无符号png
            # indepth = cv2.imread(os.path.join(self.depthdir,filename),cv2.IMREAD_UNCHANGED) #验证恢复

            #视差图normalize下
            dispnorm= cv2.normalize(disptmp, disptmp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(os.path.join(self.dispdir,filename), dispnorm)
            if t < 1: #查看中间结果
                # distortst = np.hstack((imgl_un, imgr_un))
                testimgs = np.hstack((imgl, imgr))
                cv2.imwrite(os.path.join('output','stereo_'+filename), testimgs)
                # cv2.imwrite(os.path.join('output','und_'+testfile), distortst)
                # cv2.imwrite(os.path.join('output','disp_'+testfile), dispnorm)
                cv2.imwrite(os.path.join('output','depth_'+filename), depth/DEPTHSCALE)
                # np.savetxt(os.path.join('output','depth_'+str(tstamp)+'.txt'), depth)
                # np.savetxt(os.path.join('output','disp_'+str(tstamp)+'.txt'), disptmp)
                #读取gt depth 存为视差并保存可视化
                gtdepth = np.load(os.path.join(self.input_folder, 'depth_left', frameid+'_left_depth.npy'))
                gtdisp = 80.0 / gtdepth # 7.8 55.4
                gtdispnorm= cv2.normalize(gtdisp, gtdisp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(os.path.join('output','gtdisp_'+filename), gtdispnorm)
                print('savetxt')
            else: 
                break
                
        # return disptmp, depth/DEPTHSCALE
        # pass
    
    def noisedepth(self, percent=0.2, timebase=40, aa=0.75):
        """对每个gt depth得到加上高斯噪声的深度 并保存

        Args:
            percent (float, optional): 对深度降序排列 最大的分位数部分施加噪声. Defaults to 0.2.
            timebase (int, optional): 施加噪声的界限min= timebase*baseline. Defaults to 40.
            aa (float, optional): 标准差是关于深度二次函数 这是最外的系数 *z^2/(b*f). Defaults to 0.75.
        """
        min_dth = timebase * 0.25 #10
        ptidx = int(self.W * self.H * percent) # 307200 * 0.2
        for t in range(len(self.gtdepth_paths)):
            frameid = '%06d' % t
            gtdepth = np.load(os.path.join(self.input_folder, 'depth_left', frameid+'_left_depth.npy'))
            gtdepth1 = gtdepth.reshape(-1)
            gtdepth2 = gtdepth1[np.argsort(-gtdepth1)] #降序排列
            percent_depth = gtdepth2[ptidx] #分为树上的深度阈值
            newdepth = copy.deepcopy(gtdepth)
            # noise = np.random.normal(0., , gtdepth.shape)
            for i in range(gtdepth.shape[0]):
                for j in range(gtdepth.shape[1]):
                    gtz = gtdepth[i,j]
                    if (gtz>=percent_depth and gtz >= min_dth):
                        stdij = aa * gtz * gtz /(0.25*self.fx)
                        gaussdelta = np.random.normal(0, stdij, (1))
                        newdepth[i,j] = newdepth[i,j] + gaussdelta
            
            # 保存
            newdepth = np.clip(newdepth, 0 , 10000)
            np.save(os.path.join(self.noisedepthdir,frameid+'_noise_depth.npy'),newdepth.astype(np.uint16))
            if t<2:
                gtdisp = 80.0 / newdepth # 7.8 55.4
                gtdispnorm= cv2.normalize(gtdisp, gtdisp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(os.path.join('output','noisedisp_'+frameid+'.png'), gtdispnorm)
                    

            
    
    def parse_list(self, filepath, skiprows=0, delimiter = ' '):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=delimiter,
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def loadtcsvt(self, datapath, frame_rate=-1):
        """ read video data in tcsvt-rgbd format """
        pathseg = datapath.split('/') #Datasets/TartanAir/office/Easy/P000/ pose_left.txt
        if os.path.isfile(os.path.join(datapath, 'pose_left.txt')):
            pose_list = os.path.join(datapath, 'pose_left.txt')
            pose_listr = os.path.join(datapath, 'pose_right.txt')
        else:
            raiseExceptions(True)
            
        pose_data = self.parse_list(pose_list)
        pose_vecs = pose_data[:, 0:].astype(np.float64) # tx ty tz  x y z qw
        pose_datar = self.parse_list(pose_listr)
        pose_vecsr = pose_datar[:, 0:].astype(np.float64) # tx ty tz  x y z qw

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        T_rectb = np.array([ [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1] 
        ]) #从tartanair坐标系 到标准相机系
        print('T_rectned:\n',T_rectb)
        T_brect = np.linalg.inv(T_rectb)
        for i in range(pose_vecs.shape[0]):
            frameid = '%06d' % i
            images += [os.path.join(datapath, 'image_left', frameid+'_left.png')]
            if usegtdepth and not add_depthnoise:
                depths += [os.path.join(datapath, 'depth_left', frameid+'_left_depth.npy')]
            elif add_depthnoise: # 增加噪声的实验
                depths += [os.path.join(self.noisedepthdir, frameid+'_noise_depth.npy')]
            else: #不适用gtdepth 使用opencv的depth 注意是png
                depths += [os.path.join(datapath, 'depth', frameid+'.png')]
            b2w = self.pose_matrix_from_quaternion(pose_vecs[i]) #gt 是imu系 Twbl
            r2w = self.pose_matrix_from_quaternion(pose_vecsr[i]) # Twbr
            w2l = np.linalg.inv(b2w) #Tlw
            r2l = w2l.dot(r2w) # Tlr
            # print('Tlr: \n',r2l) #验证gt pose是否stereo 是rectify 应该只有y上的平移 通过验证
            # 转为校正相机系 Trectb Twb Tbrect
            c2w = (T_rectb.dot(b2w)).dot(T_brect)
            if inv_pose is None: # 首帧位姿归I 但是好像只有tum做了归一化处理 why
                inv_pose = np.linalg.inv(c2w) #T0w
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w # T0w Twi = T0i
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4) #对于tcsvt的接口
        qxyzw = pvec[3:] # pvec: tx ty tz  x y z w
        pose[:3, :3] = Rotation.from_quat(qxyzw).as_matrix() # xyz w
        pose[:3, 3] = pvec[:3]
        return pose

    # 也画出轨迹 想看看bound to do
    def plot_traj(self):
        self.trajzx = []
        for i in range(len(self.poses)):
            c2w = self.poses[i].numpy()
            self.trajzx.append([c2w[0,3], c2w[1,3], c2w[2,3]]) # (x,y,z)
        traj = np.array(self.trajzx) # n,3
        # 输出 xyz 的区域
        print('x y z min:\n', traj.min(axis=0))
        print('x y z max:\n', traj.max(axis=0))
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(traj[:,0],traj[:,2], linestyle='dashed',c='k') # x,z
        # plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.legend(['Ground Truth (nerf coord)'])
        # plt.title(title)

        plt.axis('equal')
        savefigname = os.path.join(self.input_folder, 'gtwnerf.pdf')    
        plt.savefig(savefigname)
        
        # if vis:
        #     plt.show()

        plt.close(fig)




dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "cofusion": CoFusion,
    "azure": Azure,
    "tumrgbd": TUM_RGBD,
    "tcsvt": tcsvt,
    "vkitti2": vKITTI2,
    "tartanair": TartanAir
}
