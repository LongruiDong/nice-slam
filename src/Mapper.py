import os
import time
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples, get_rays_by_weight,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False
                 ):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper #coarse level的优化
        self.rgbonly = slam.rgbonly
        self.guide_sample = slam.guide_sample
        self.use_prior = slam.use_prior
        self.idx = slam.idx
        self.nice = slam.nice
        self.c = slam.shared_c
        if self.use_prior:
            self.prior_xyzs_dict = slam.prior_xyzs_dict
            self.prior_3dobs_dict = slam.prior_3dobs_dict # 还有对应关系
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame
        # 增加记录每次迭代时在当前帧选择的点的坐标
        self.slecti = torch.zeros(0,)
        self.slectj = torch.zeros(0,)
        self.gmstep = int(0) # 记录总的优化迭代次数 记录用来 decay 退火ray上bound from regnerf

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy'] # nice-slam 才为true
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']
        self.fix_fine = cfg['mapping']['fix_fine']
        self.eval_rec = cfg['meshing']['eval_rec'] # 默认只有replica时是true
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.fix_color = cfg['mapping']['fix_color'] # 固定颜色？
        self.mapping_pixels = cfg['mapping']['pixels'] # lba所用的像素数
        self.num_joint_iters = cfg['mapping']['iters']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame'] #输入参数每隔x帧进行mapping
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.w_depth_loss = cfg['mapping']['w_depth_loss'] # 给depth loss 加权
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level'] #默认false
        self.mapping_window_size = cfg['mapping']['mapping_window_size'] # 论文里的K 参与lba的kf总数
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        self.prior_camera = cfg['tracking']['prior_camera'] # 控制当前触发mapping的帧初值是否用prior kf的位姿
        self.less_sample_space = slam.less_sample_space # 是否在 surface 附近区间多重采样,对应不同的render_batch
        self.use_KL_loss = slam.use_KL_loss # 是否使用ds-nerf的sigma loss
        self.sigma_lambda = float(cfg['rendering']['sigma_lambda']) # KL loss 的权重参数
        self.color_loss_mask = cfg['mapping']['color_loss_mask'] # color loss 是否用depth>0 mask
        self.use_regulation = cfg['mapping']['use_regulation'] # 是否增加对于前景的nice sigma 的正则化
        self.use_sparse_loss = cfg['mapping']['use_sparse_loss'] # 是否增加使用稀疏点 depth color loss
        self.use_sparse_color = cfg['mapping']['use_sparse_color'] # 上面就特指sparse depth 这里kpt上的 color 就在吧
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'

        self.keyframe_dict = [] # 还是个列表 里面每个元素是dict
        self.keyframe_list = []
        # self.frame_reader = get_dataset(
        #     cfg, args, self.scale, device=self.device)
        self.frame_reader = slam.frame_reader # 何必再初始化一次呢？
        self.n_img = len(self.frame_reader)
        if 'Demo' not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         verbose=self.verbose, depth_trunc=cfg['cam']['depth_trunc'], device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color, _, _ = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def keyframe_selection_prior(self, keypt, keyframe_dict, k):
        """
        当没有gt depth 后， 可以使用来自orb的 2d-3d对应关系 给定某帧kpt数据, 来给当前kf选择给定数量的历史kf
        注意 返回的仍是 kfid 列表 而不是 其普通帧id 这才和后面代码对应
        Args:
            keypt (tensor): (n, 4) 包含了kpt的信息 以及其对应的3d点: kptid u v 3dpid(-1表示没有对应)
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of keyframes to select. 因为能看到局部3d点的历史帧可能有很多 从中挑k个出来
        """
        
        # 先从kpt中拿出所有3d点 id 作为局部地图点
        curr_mappt_idx = torch.unique(keypt[:, 3]) # 咋还在cpu上 float64
        curr_mappt_idx = curr_mappt_idx[curr_mappt_idx >= 0] # 去掉 -1 表示关键点无对应 地图点
        # 从 prior_3dobs_dict 拿出上面地图点 的所有观测
        n_curr_mappt = curr_mappt_idx.shape[0]
        curr_obs = []
        for i in range(n_curr_mappt):
            key = int(curr_mappt_idx[i])
            obsarr = self.prior_3dobs_dict[key] # (m,2)
            curr_obs += [obsarr]
        curr_obs = torch.cat(curr_obs, 0) # (#allobs from local mapts, 2)
        curr_obs_frame = curr_obs[:, 0].int() # (#allobs from local mapts) int64
        # 遍历历史kf 统计它出现在curr_obs_frame的次数 表示某kf 和当前kf的共视 点数量
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            kf_fid = int(keyframe['idx']) # kf 对应的frame id
            # 统计kf_fid出现在curr_obs_frame的次数
            cov_num = (curr_obs_frame==kf_fid).sum()
            if cov_num>0:
                list_keyframe.append( # 这里的id是关键帧id！
                    {'id': keyframeid, 'cov_num': cov_num, 'frame_id': kf_fid})
        # 对含有共视的kf排序  这里的kf都是有kpt的帧(来自orb)
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['cov_num'], reverse=True) #降序排列 根据共视点数量
        list_keyframe_id = [dic['id'] for dic in list_keyframe]
        selected_keyframe_list = list_keyframe_id[:k] # 拿出前k帧 的关键帧id !
        
        return selected_keyframe_list   
    
    def get_last_priorkf(self, keyframe_dict, keyframe_list):
        """
        从历史Kf中找到最近的一个含有kpt的kf; 用于当mapping到非kpt kf时  如何选取含有共视的Kf
        返回 id 以及对应的 keypt 
        Args:
            keyframe_dict (list): a list containing info for each keyframe.
            keyframe_list (list): list of keyframe index.
        """
        # 先判断上一个kf是否是kpt kf
        last_kf_dic = keyframe_dict[-1]
        last_kf_kpt = last_kf_dic['keypt']
        if not torch.isnan(last_kf_kpt).all().item():
            # 若上一帧就是 就返回它 以及kpt
            return -1, last_kf_kpt
        # 不是 那就逆向遍历 先对kf id 降序排列 且排除上一个kf
        inv_keyframe_list = sorted(keyframe_list[:-1], reverse=True)
        # for keyframeid, _ in enumerate(inv_keyframe_list):
        for keyframeid in reversed(range(len(keyframe_list)-1)):
            keyframe_dic =  keyframe_dict[keyframeid]
            kf_kpt = keyframe_dic['keypt']
            if not torch.isnan(kf_kpt).all().item():
                # 若此帧就是 就返回它 以及kpt
                return keyframeid, kf_kpt
        
        
    
    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth,
                     gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w, cur_keypt, cur_keypt_wt):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera. / coarse prior depth map (will be updated)
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 当使用prior pose来track 这里可能为nan 
            cur_keypt (tensor): prior key point & 2d-3d correspondences (data api), kptid u v 3dpid(-1表示没有对应)
            cur_keypt_wt (tensor): 和图像大小相同 只有在有3d对应的位置上有权重>0 其余为 -0 用来拿出关键点像素 从而能得到ray

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c = self.c
        if self.use_prior:
            prior_xyzs_dict = self.prior_xyzs_dict # 取出当前最新的稀疏点
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global': # imap 使用5个kf
                num = self.mapping_window_size-2
                if torch.isnan(cur_c2w).all().item():
                    num = self.mapping_window_size-1
                optimize_frame = random_select(len(self.keyframe_dict)-1, num) # 按照它论文 这里不应该采用一定的选择策略？
            elif self.keyframe_selection_method == 'overlap': #按共视区域 注意这里也涉及深度  后续可以用2d-3d对应来找overlap的kf
                num = self.mapping_window_size-2 # K-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)
            elif self.keyframe_selection_method == 'prior':
                num = self.mapping_window_size-2 # K-2
                if not torch.isnan(cur_keypt).all().item(): # 若使用先验 且 当前帧关键点非空
                    optimize_frame = self.keyframe_selection_prior(cur_keypt, keyframe_dict[:-1], num)
                else: # 若当前帧不含kpt 其他策略 此时既没有coarse depth 也没有track pose初值 只能都以上个有效kf来做 且当前帧不加入窗口
                    # optimize_frame = random_select(len(self.keyframe_dict)-1, num) # 就退化为简单随机选择
                    # 拿出上一个含有kpt的kf 以他的共视拿来用
                    last_priorkf_id, last_keypt = self.get_last_priorkf(keyframe_dict, keyframe_list)
                    if True: # 之后就为true self.prior_camera
                        num = self.mapping_window_size-1 # K-1 此时由于触发帧不加入窗口，为保证窗口内共K，就这里从历史多拿一帧！
                    optimize_frame = self.keyframe_selection_prior(last_keypt, keyframe_dict[:-1], num)
                    
                    

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] #recent kf
            oldest_frame = min(optimize_frame)
        if not torch.isnan(cur_c2w).all().item(): # 否则 不含此帧进入窗口！
            optimize_frame += [-1] # add curr frame 只要当前帧track的位姿非全nan 表示gt pose or 使用prior时 本帧就是orbkf 那就包含此帧

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else: # 当使用prior pose 时 窗口内不在包含当前帧 下面的就不使用了
                    frame_idx = idx # curr frame 的frame id 
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            # 直接打印得了
            print('current frame {:d}, frame in opt window: '.format(idx))
            for il, kdic in enumerate(keyframes_info):
                fidx = kdic['idx']
                if il>0:
                    print(', ',end="")
                print('{:d}'.format(fidx),end="")
            print(' ')
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame) # 每个kf的像素数
        # print(f'Fid {idx:d} , pixs_per_image: {pixs_per_image:d}')
        decoders_para_list = []
        coarse_grid_para = []
        middle_grid_para = []
        fine_grid_para = []
        color_grid_para = []
        if not torch.isnan(cur_gt_depth).all().item(): # cur_gt_depth is not None:
            gt_depth_np = cur_gt_depth.cpu().numpy()
        else:
            gt_depth_np = None # 只是用与 FFS
        if self.nice:
            if self.frustum_feature_selection:
                masked_c_grad = {}
                mask_c2w = cur_c2w
            for key, val in c.items():
                if not self.frustum_feature_selection or (gt_depth_np is None):
                    val = Variable(val.to(device), requires_grad=True)
                    c[key] = val
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val)
                    elif key == 'grid_color':
                        color_grid_para.append(val)

                elif gt_depth_np is not None:
                    mask = self.get_mask_from_c2w(
                        mask_c2w, key, val.shape[2:], gt_depth_np)
                    mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(
                        0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    val_grad = val[mask].clone()
                    val_grad = Variable(val_grad.to(
                        device), requires_grad=True)
                    masked_c_grad[key] = val_grad
                    masked_c_grad[key+'mask'] = mask
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val_grad)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val_grad)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val_grad)
                    elif key == 'grid_color':
                        color_grid_para.append(val_grad)

        if self.nice:
            if not self.fix_fine:
                decoders_para_list += list(
                    self.decoders.fine_decoder.parameters())
            if not self.fix_color:
                decoders_para_list += list(
                    self.decoders.color_decoder.parameters())
        else:
            # imap*, single MLP
            decoders_para_list += list(self.decoders.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else: # 当使用prior pose 时 窗口内不在包含当前帧 下面的就不使用了
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)
                    if frame == -1: #  idx -1 就是指当前帧
                        initial_loss_camera_tensor = torch.abs(
                            gt_camera_tensor.to(device)-camera_tensor).mean().item()

        if self.nice:
            if self.BA:
                # The corresponding lr will be set according to which stage the optimization is in
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else: # 若关闭self.BA 就不优化相机位姿
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0}])
        else:
            # imap*, single MLP
            if self.BA:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam(
                    [{'params': decoders_para_list, 'lr': 0}])
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)
        kl_loss = None
        regulation_loss = None # 为了观察大小
        for joint_iter in range(num_joint_iters):
            if self.nice:
                if self.frustum_feature_selection:
                    for key, val in c.items():
                        if (self.coarse_mapper and 'coarse' in key) or \
                                ((not self.coarse_mapper) and ('coarse' not in key)):
                            val_grad = masked_c_grad[key]
                            mask = masked_c_grad[key+'mask']
                            val = val.to(device)
                            val[mask] = val_grad
                            c[key] = val

                if self.coarse_mapper:
                    self.stage = 'coarse' #这是零一个线程在做
                elif joint_iter <= int(num_joint_iters*self.middle_iter_ratio): #default 0.4
                    self.stage = 'middle'
                elif joint_iter <= int(num_joint_iters*self.fine_iter_ratio): # 0.4~0.6
                    self.stage = 'fine'
                else: #0.6~1
                    self.stage = 'color'
                # 学习率设置
                optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor #color 的decoder
                optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['coarse_lr']*lr_factor
                optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['middle_lr']*lr_factor
                optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr']*lr_factor
                optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr']*lr_factor
                if self.BA:
                    if self.stage == 'color':
                        optimizer.param_groups[5]['lr'] = self.BA_cam_lr
            else:
                self.stage = 'color'
                optimizer.param_groups[0]['lr'] = cfg['mapping']['imap_decoders_lr']
                if self.BA:
                    optimizer.param_groups[1]['lr'] = self.BA_cam_lr

            if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output):
                if torch.isnan(cur_gt_depth).all().item() and (not torch.isnan(cur_c2w).all().item()): # depth全nan 且位姿非nan 就画全0
                    # cur_gt_depth_0 = torch.zeros_like(cur_gt_color).to(device)
                    self.visualizer.vis( #每一次优化前去可视化rendered的结果
                    idx, joint_iter, torch.full(size=cur_gt_color.shape[:2], fill_value=float(0)).to(device), cur_gt_color, cur_c2w, self.c, self.decoders,
                    selecti=self.slecti, selectj=self.slectj)
                elif (not torch.isnan(cur_gt_depth).all().item()): # 本帧depth非nan 意味着pose也有 正常画 而且多重采样
                    # cur_gt_depth_0 = cur_gt_depth
                    self.visualizer.vis( #每一次优化前去可视化rendered的结果
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.c, self.decoders,
                    selecti=self.slecti, selectj=self.slectj, confidence=cur_keypt_wt)
            # 也打印一下 各个 kl loss    
            if (joint_iter % self.visualizer.inside_freq == 0) and (joint_iter>0): # (idx % self.visualizer.freq == 0) and 
                if self.use_KL_loss and self.use_regulation:
                    print(f'[{self.stage}] Fid {idx:d} -- {joint_iter:d}, Re-rendering loss: {initial_loss:.2f}->{loss:.2f}. KL_loss: {initial_kl_loss:.2f}->{kl_loss:.2f}. Reg_loss: {initial_regulation_loss:.2f}->{regulation_loss:.2f}')
                elif self.use_KL_loss and not self.use_regulation:
                    print(f'[{self.stage}] Fid {idx:d} -- {joint_iter:d}, Re-rendering loss: {initial_loss:.2f}->{loss:.2f}. KL_loss: {initial_kl_loss:.2f}->{kl_loss:.2f}. sp_factor: {sp_factor:.2f} ')
                elif not self.use_KL_loss and self.use_regulation:
                    print(f'[{self.stage}] Fid {idx:d} -- {joint_iter:d}, Re-rendering loss: {initial_loss:.2f}->{loss:.2f}. Reg_loss: {initial_regulation_loss:.2f}->{regulation_loss:.2f}')
                else:
                    print(f'[{self.stage}] Fid {idx:d} -- {joint_iter:d}, Re-rendering loss: {initial_loss:.2f}->{loss:.2f}')
            optimizer.zero_grad() #梯度归零
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []
            
            if self.use_sparse_loss or self.use_sparse_color: # 两者分别对应 kpt depth 和 color
                # 对于有效稀疏关键点 并行地进行提取射线和渲染
                prior_rays_d_list = []
                prior_rays_o_list = []
                prior_depth_list = []
                prior_color_list = []
                prior_kpt_wt_list = []
            camera_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1: # 最下面目前 还可能有 没有 prior depth 的kf构成
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    kpt_weight = keyframe_dict[frame]['keypt_wt'].to(device)
                    if self.BA and frame != oldest_frame: # 最老帧位姿固定
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']
                # 当使用prior pose 时 窗口内不在包含当前帧 下面的就不使用了
                else: # cur_gt_depth 可能是nan tensor！为了之后batch_depth 那就在这里把gt_depth设为全0
                    if not torch.isnan(cur_gt_depth).all().item():
                        gt_depth = cur_gt_depth.to(device)
                    else: # 当前帧是非orb kf # 得到的gt_depth 可能是 nan-->改为全0表示
                        gt_depth = torch.full(size=cur_gt_color.shape[:2], fill_value=float(0)).to(device)
                    gt_color = cur_gt_color.to(device)
                    kpt_weight = cur_keypt_wt.to(device) # 可能为 -0
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                        loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                        if (joint_iter % self.visualizer.inside_freq == 0) and (joint_iter>0): # (idx % self.visualizer.freq == 0) and
                            print(f'Fid {idx:d} -- {joint_iter:d}, camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    else:
                        c2w = cur_c2w

                if self.use_sparse_loss or self.use_sparse_color:
                    # 取出当前帧现在位姿后 用单独函数来提取出 对应的射线
                    prior_rays_o, prior_rays_d, prior_depth, prior_color, prior_weight, kpt_i, kpt_j = get_rays_by_weight(
                        H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, kpt_weight, device
                    )
                    
                # 每次batch iter 就要随机采样一次 batch_gt_depth 里面可能有nan
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, sti, stj = get_samples(
                    0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                if frame == -1: #记录当前帧 本次循环的采样点 #  idx -1 就是指当前帧
                    self.slecti = sti
                    self.slectj = stj
                    if (self.use_sparse_loss or self.use_sparse_color) and (not torch.isnan(cur_gt_depth).all().item()):
                        self.selcti = kpt_i
                        self.selectj = kpt_j
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())
                if self.use_sparse_loss or self.use_sparse_color:
                    prior_rays_o_list.append(prior_rays_o.float())
                    prior_rays_d_list.append(prior_rays_d.float())
                    prior_depth_list.append(prior_depth.float())
                    prior_color_list.append(prior_color.float())
                    prior_kpt_wt_list.append(prior_weight.float())
                

            batch_rays_d = torch.cat(batch_rays_d_list) # 不是归一化的！
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list) # 6000 类似于batchsize batch_gt_depth 里面可能有nan
            batch_gt_color = torch.cat(batch_gt_color_list) # 6000,3
            
            if self.use_sparse_loss or self.use_sparse_color:
                prior_rays_o = torch.cat(prior_rays_o_list) # (m)
                prior_rays_d = torch.cat(prior_rays_d_list)
                prior_depth = torch.cat(prior_depth_list)
                prior_color = torch.cat(prior_color_list)
                prior_weight = torch.cat(prior_kpt_wt_list)

            if self.nice: # and (not torch.isnan(batch_gt_depth).all().item())
                # should pre-filter those out of bounding box depth value 还不涉及 ray上积分的bound
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to( # (3,2) (1,3,2) N 3 2
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1) # ->(N,3)-> N 改视线 最早和box相交时 d的系数（深度？ 但这里）
                    inside_mask = t >= batch_gt_depth # 该系数 当景物在bound内时  才考虑使用！
                    intmask = inside_mask.long()
                    samp = intmask.sum().cpu().numpy()
                    x = int(2/samp)
                    # print('2/samp: \t', x)
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
                
                if self.use_sparse_loss or self.use_sparse_color:
                    with torch.no_grad():
                        det_rays_o = prior_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                        det_rays_d = prior_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                        t = (self.bound.unsqueeze(0).to( # (3,2) (1,3,2) N 3 2
                            device)-det_rays_o)/det_rays_d
                        t, _ = torch.min(torch.max(t, dim=2)[0], dim=1) # ->(N,3)-> N 改视线 最早和box相交时 d的系数（深度？ 但这里）
                        inside_mask = t >= prior_depth # 该系数 当景物在bound内时  才考虑使用！
                        # intmask = inside_mask.long()
                    prior_rays_d = prior_rays_d[inside_mask]
                    prior_rays_o = prior_rays_o[inside_mask]
                    prior_depth = prior_depth[inside_mask]
                    prior_color = prior_color[inside_mask]
                    prior_weight = prior_weight[inside_mask]
                
                # 对于prior 的也捋一遍
            # batch_gt_depth 要么 全nan 要么正常 不是的 当前帧可能无深度 但前面已经转为0了 所以就是0和非0混合
            if torch.isnan(batch_gt_depth).all().item() or batch_gt_depth is None:
                print('[debug] batch_gt_depth should not here!')
                batch_gt_depth = None # 为了下面render 直接 去除表面引导的方便
            tmp = torch.zeros(batch_gt_depth.shape, device=self.device)
            if torch.equal(tmp, batch_gt_depth): # 若当前帧是非orb kf 就不用它来render 否则全黑
                validflag = False
            else:
                validflag = True
            sigma_loss = None
            if self.less_sample_space: # 在会更小范围内 对surface多重采样 这里 lss 应该和kl loss 独立才行 方便配置控制
                ret = self.renderer.render_batch_ray_tri(c, self.decoders, batch_rays_d,
                                                    batch_rays_o, device, self.stage,
                                                    gt_depth=None if ( self.coarse_mapper or (not self.guide_sample) or (not validflag) ) else batch_gt_depth,
                                                    confidence=None) # 突然意识到之前这里 还是会根据gt depth 采样ray上表面附近的
                
            else: # 原始情况
                ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                    batch_rays_o, device, self.stage, #)
                                                    gt_depth=None if ( self.coarse_mapper or (not self.guide_sample) or (not validflag) ) else batch_gt_depth,
                                                    confidence=None) # 这里是因为都是 coarse prior 暂时用的估计的 权重做方差
            depth, uncertainty, color, weights, sigma_loss, _ = ret # 这里的输出不同 但sigma_loss可能仍是None
            if self.use_sparse_loss or self.use_sparse_color:
                # 对于稀疏点的 数据 都用多重采样 也要用到weight 也改为选择两种方式  这样 稀疏点 kl loss 采样策略都独立了！
                if self.less_sample_space:
                    prior_ret = self.renderer.render_batch_ray_tri(c, self.decoders, prior_rays_d,
                                                            prior_rays_o, device, self.stage,
                                                            gt_depth=prior_depth, # 对于有效稀疏点 coarse深度和置信度都有
                                                            confidence=prior_weight) #
                else:
                    prior_ret = self.renderer.render_batch_ray(c, self.decoders, prior_rays_d,
                                                        prior_rays_o, device, self.stage,
                                                        gt_depth=prior_depth, # 对于有效稀疏点 coarse深度和置信度都有
                                                        confidence=prior_weight) #
                depth_sp, uncertainty_sp, color_sp, weights_sp, sigma_loss_sp, _ = prior_ret
            # 和 tracker中的改动一致 现在已经不会是None
            if batch_gt_depth is not None:
                if validflag: # 非全0
                    depth_mask = (batch_gt_depth > 0)# & (batch_gt_depth < 600) #只考虑mask内的像素参与误差 # 对于outdoor 加上 不属于无穷远 vkitti 655.35
                else: # 全0 就都用 其实目前rgb only 时 depth_mask没用上
                    depth_mask = torch.full(size=batch_gt_depth.shape, fill_value=True)
            # 这里测试 loss 改为 color only loss
            if self.rgbonly:
                loss = torch.tensor(0.0, requires_grad=True).to(device)  # 其他阶段都没color 下面还要加
                # self.w_color_loss = 1.
                # loss = torch.abs(
                #     batch_gt_color[depth_mask]-color[depth_mask]).sum() # batch_gt_depth[depth_mask]-depth[depth_mask] batch_gt_color[depth_mask]-color[depth_mask]
            else: # 否则还是原来 rgb-d
                depth_loss = torch.abs(
                    batch_gt_depth[depth_mask]-depth[depth_mask]).sum()
                loss = self.w_depth_loss * depth_loss
            if self.use_sparse_loss or self.use_sparse_color: # 对稀疏点loss 加权 使用归一化后的置信度 这里只看一个flag
                prior_mask = (prior_depth>0)
                prior_weight = (prior_weight / prior_weight.max() + 1e-10) # (0, 1)
                sp_factor = float(depth.shape[0]) / depth_sp.shape[0] # 数量上平衡 一般点 和 系数深度点的数量上的比值
                if self.use_sparse_loss:
                    sp_depth_loss = (torch.abs( #
                        prior_depth-depth_sp) * prior_weight)[prior_mask].sum()
                    loss += sp_factor* self.w_depth_loss * sp_depth_loss
            if ((not self.nice) or (self.stage == 'color')): # 才发现 其他stage color是0 所以之前 rgb only 时应该 都改为color stage,反正其他stage color loss是错的
                if self.color_loss_mask: # 是否使用depth 的mask
                    color_loss = torch.abs(batch_gt_color[depth_mask] - color[depth_mask]).sum()
                else:
                    color_loss = torch.abs(batch_gt_color - color).sum()
                weighted_color_loss = self.w_color_loss*color_loss
                # if self.rgbonly:
                #     self.w_color_loss = 1.
                #     weighted_color_loss = color_loss
                loss += weighted_color_loss
                if self.use_sparse_color: # self.use_sparse_loss 稀疏点 颜色始终有
                    sp_color_loss = torch.abs(
                        prior_color[prior_mask] - color_sp[prior_mask]).sum()
                    loss += sp_factor* self.w_color_loss * sp_color_loss
            
            # 当使用KL loss 时 继续添加
            kl_loss = 0
            if self.use_KL_loss and (sigma_loss is not None or sigma_loss_sp is not None):
                # kl_loss = 0
                if sigma_loss is not None:
                    sigma_sum = torch.sum(sigma_loss, dim=1) # 避免计入那些 较大的值一般是 深度+ 标准差接近边界
                    kl_loss += sigma_sum[sigma_sum<500.].mean() # (B,) -> 平均到一个值了
                    # loss += self.sigma_lambda*kl_loss # 权重求和
                # if self.use_sparse_loss or self.use_sparse_color: # 这些位置已经用稀疏点depth直接监督过了
                #     sigma_sum_sp= torch.sum(sigma_loss_sp, dim=1)
                #     kl_loss_sp = sigma_sum_sp[sigma_sum_sp<500.].mean()
                #     kl_loss += kl_loss_sp
                loss += self.sigma_lambda*kl_loss

            if self.use_regulation and validflag: # 对于 nice 当先验深度非全0时 也像下面那样同样正则化 前景
                point_sigma = self.renderer.regulation_occ(
                    c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                regulation_loss = torch.abs(point_sigma).sum()
                loss += 0.1*regulation_loss
            
            # for imap*, it uses volume density
            regulation = (not self.occupancy)
            if regulation and batch_gt_depth is not None: # 增加条件 但其实 不会none了 总是0与非0
                point_sigma = self.renderer.regulation(
                    c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                regulation_loss = torch.abs(point_sigma).sum()
                loss += 0.0005*regulation_loss
            
            loss.backward(retain_graph=False) #反向传播计算梯度
            optimizer.step() #梯度下降1次进行参数更新
            if not self.nice:
                # for imap*
                scheduler.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            if self.nice and self.frustum_feature_selection:
                for key, val in c.items():
                    if (self.coarse_mapper and 'coarse' in key) or \
                            ((not self.coarse_mapper) and ('coarse' not in key)):
                        val_grad = masked_c_grad[key]
                        mask = masked_c_grad[key+'mask']
                        val = val.detach()
                        val[mask] = val_grad.clone().detach()
                        c[key] = val
            if joint_iter==0:
                initial_loss = loss
                initial_kl_loss = kl_loss
                initial_regulation_loss = regulation_loss
            
            # 更新 全局 mapping step regnerf 但其实 coarse mapping 没用到weight decay
            self.gmstep += 1

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else: # 当使用prior pose 时 窗口内不在包含当前帧 下面的就不使用了 
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w # 当使用prior pose 时 窗口内不在包含当前帧 那么这里就还是nan tensor
        else:
            return None

    def run(self):
        cfg = self.cfg # gt_depth 其实可以是 粗糙的初始深度 
        idx, gt_color, gt_depth, gt_c2w, _, _, _ = self.frame_reader[0]

        self.estimate_c2w_list[0] = gt_c2w.cpu()
        init = True # 初始 需要做初始化
        prev_idx = -1
        while (1):
            while True:
                idx = self.idx[0].clone()
                if type(idx) == torch.Tensor:
                    idx = idx.item()
                if idx == self.n_img-1: #最后一帧也会优化的
                    break
                
                if self.sync_method == 'strict': #输入参数每隔x帧进行mapping 增加 若是orb kf 也开启  之后可以再是 只有在kf map  idx % self.every_frame == 0 or 试验后还是得加上更好一些
                    if self.use_prior:
                        if ( idx % self.every_frame == 0 or (idx in self.frame_reader.prior_poses.keys()) ) and idx != prev_idx:
                            break #否则就一直在里面循环
                    else:
                        if idx % self.every_frame == 0 and idx != prev_idx:
                            break #否则就一直在里面循环

                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                print(prefix+"Mapping Frame ", idx)
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w, _, keypt, prior_weight = self.frame_reader[idx] # prior_c2w

            if not init: #初始化已结束
                lr_factor = cfg['mapping']['lr_factor'] #非初始化 
                num_joint_iters = cfg['mapping']['iters']

                # here provides a color refinement postprocess 在这里额。。 
                if idx == self.n_img-1 and self.color_refine and not self.coarse_mapper:# 非coarse线程
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2 # 关键帧窗口*2
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True # color refine时 要固定住decoder 只优化grid
                    self.frustum_feature_selection = False # 不在只限制视锥内的grid feature
                    if self.verbose:
                        print(Fore.GREEN)
                        print("color refinement postprocess... ")
                        print(Style.RESET_ALL)
                else:
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3 # 对于imap* 外层循环3次 为何 难道是想和nice-slam 有3个优化阶段对等？

            else: # 初始化
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor'] #初始化时学习率times
                num_joint_iters = cfg['mapping']['iters_first'] #初始的总迭代次数很大
            # 当使用prior pose track时 这里可能nan tensor 但还是要开启mapping!
            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            num_joint_iters = num_joint_iters//outer_joint_iters # 注意这里除法 300/3=100 imap* 这个策略很莫名其妙。。
            for outer_joint_iter in range(outer_joint_iters):

                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and (
                    not self.coarse_mapper)

                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w, 
                                      cur_keypt=keypt, cur_keypt_wt=prior_weight) # 增加了keypt的输入 有效 keypt的weight
                if self.BA:
                    cur_c2w = _ # 用返回值更新pose 当使用prior pose 时 窗口内不在包含当前帧 那么这里就还是nan tensor!
                    self.estimate_c2w_list[idx] = cur_c2w

                # add new frame to keyframe set 这里也不是靠information gain来生成kf啊
                # keyfrrame dict 要增加保存当前最新的coarse depth, 还有不变的keypt
                if outer_joint_iter == outer_joint_iters-1: # idx % self.keyframe_every == 0 or 删去此条件 为了kf基本等同orb 的kf
                    if ((idx == self.n_img-2) or (not torch.isnan(gt_depth).all().item()) ) \
                            and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx) #关键帧列表 以 frame_id组成
                        self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                        ), 'keypt': keypt.cpu(), # 前面这些都是定值，后面的是需要优化后不断更新
                        'keypt_wt': prior_weight.cpu(), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False
            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1

            if not self.coarse_mapper:
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                    selected_keyframes=self.selected_keyframes
                                    if False else None) # self.save_selected_keyframes_info

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                if idx == self.n_img-1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(
                        f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True) #只在评估重建质量时
                    break

            if idx == self.n_img-1:
                break
