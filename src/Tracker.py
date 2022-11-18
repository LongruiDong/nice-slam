import copy
import os
import time
# -*- coding:utf-8 -*-
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        self.rgbonly = slam.rgbonly
        self.usepriormap = slam.usepriormap
        self.onlyvis = slam.onlyvis
        self.guide_sample = slam.guide_sample
        self.use_prior = slam.use_prior
        self.less_sample_space = slam.less_sample_space # 是否在 surface 附近区间多重采样,对应不同的render_batch
        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        # 增加记录每次迭代时在当前帧选择的点的坐标
        self.slecti = torch.zeros(0,)
        self.slectj = torch.zeros(0,)
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        if self.usepriormap and self.onlyvis:
            # 别和mapper的搞混
            self.keyframe_dict = [] # 还是个列表 里面每个元素是dict 注意这只是tracker的
            self.keyframe_list = []

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.prior_camera = cfg['tracking']['prior_camera'] # 控制初值是否用prior kf的位姿
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        # self.frame_reader = slam.frame_reader # 何必再初始化一次呢？ 因为是不同线程 还是得重开
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, depth_trunc=cfg['cam']['depth_trunc'], device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H #frame 25这里 get_samples全是空的返回
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, selecti, selectj = get_samples( #(n,3) (n,3) (n) (n,3)
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        self.slecti = selecti #记录
        self.slectj = selectj
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d # (1,3,2)-(n,3,1) =(n,3,2)   (n,3,2)
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1) #(n,3) -> (n)
                inside_mask = t >= batch_gt_depth #问题只可能在这里 没有满足的样本 [n]
                intmask = inside_mask.long()
                samp = intmask.sum().cpu().numpy()
                x = int(2/samp)
            batch_rays_d = batch_rays_d[inside_mask] #(m,3)
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]
        
        tmp = torch.zeros(batch_gt_depth.shape, device=self.device)
        if torch.equal(tmp, batch_gt_depth): # 若当前帧是非orb kf 就不用它来render 否则全黑
            validflag = False
        else:
            validflag = True
        if self.less_sample_space:
            ret = self.renderer.render_batch_ray_tri( # 因为只是优化pose 这里不需要sigma loss 输出
                self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=None if ( self.coarse_mapper or (not self.guide_sample) or (not validflag) ) else batch_gt_depth) # 突然意识到之前这里 还是会根据gt depth 采样ray上表面附近的
        else:
            ret = self.renderer.render_batch_ray(
                self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=None if ( self.coarse_mapper or (not self.guide_sample) or (not validflag) ) else batch_gt_depth) # 突然意识到之前这里 还是会根据gt depth 采样ray上表面附近的
        depth, uncertainty, color, weights, sigma_loss, _ = ret # 这里的输出不同 但sigma_loss可能仍是None

        uncertainty = uncertainty.detach()
        if self.handle_dynamic: # 处理运动物体
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else: #只考虑mask内的像素参与误差 # 对于outdoor 加上 不属于无穷远 vkitti 655.35
            mask = (batch_gt_depth > 0) # & (batch_gt_depth < 600)
        # 这里测试 loss 改为 color only loss
        if self.rgbonly:
            loss = torch.tensor(0.0, requires_grad=True).to(device) #下面要加
            # loss = ( torch.mean(torch.abs(batch_gt_color - color), dim=1) / # torch.abs(batch_gt_depth-depth) torch.mean(torch.abs(batch_gt_color - color), dim=1)
            #         torch.sqrt(uncertainty+1e-10))[mask].sum()
        else:   # 否则还是原来 rgb-d
            loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            if self.rgbonly:
                loss += color_loss
            else:
                loss += self.w_color_loss*color_loss


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        self.c = {}
        # 对于只是用载入的地图来评估的话 一次性的吧 c 和 decoders 赋值好 然后不再需要update_para_from_mapping()
        if self.usepriormap and self.onlyvis:
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)
        # 记录所有帧的 depth_l1 和 psnr
        depth_error = []
        img_psnr = []
        # gt_depth 这里可能是prior coarse depth, prior_c2w(后续可以以此代替gt位姿), keypt, prior_weight
        for idx, gt_color, gt_depth, gt_c2w, prior_c2w, _, _ in pbar: 
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0] #
            if len(prior_c2w.shape) >2:
                prior_c2w = prior_c2w[0]
            if type(idx) == torch.Tensor:
               idx = idx.item() 
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            if self.sync_method == 'strict':
                # strictly mapping and then tracking idx % self.every_frame == 1 or self.every_frame == 1 or 去掉这些就是 只管orb kf
                # initiate mapping every self.every_frame frames 这里增加 对于kf都tracking 否则 mapping那边更是receive不到
                if self.use_prior:
                    if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1 or ((idx-1) in self.frame_reader.prior_poses.keys())):
                        if (not self.usepriormap) and (not self.onlyvis): # 此时就没有mapping线程
                            while self.mapping_idx[0] != idx-1:
                                time.sleep(0.1)
                        pre_c2w = self.estimate_c2w_list[idx-1].to(device)
                else:
                    if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                        if (not self.usepriormap) and (not self.onlyvis):
                            while self.mapping_idx[0] != idx-1:
                                time.sleep(0.1)
                        pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass
            
            if (not self.usepriormap) and (not self.onlyvis):
                self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx)
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera or self.prior_camera: # 选择prior kf pose
                if idx == 0:
                    c2w = gt_c2w
                elif self.gt_camera and (not self.prior_camera):
                    c2w = gt_c2w
                elif (not self.gt_camera) and self.prior_camera:
                    c2w = prior_c2w # 可能为nan tensor ?
                elif self.gt_camera and self.prior_camera:
                    assert False, f"prior pose not along with gt pose !"
                    
                # if not self.no_vis_on_first_frame and (not torch.isnan(gt_depth).all().item()):
                #     self.visualizer.vis(
                #         idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders,
                #         selecti=self.slecti, selectj=self.slectj)
                
                # 这里进行 vis 的代码
                
                if self.usepriormap and self.onlyvis:
                    if idx % self.every_frame == 0 or idx==self.n_img-1:
                        retvis = self.renderer.evavis_img(self.c, self.decoders, idx, c2w, self.device, 
                                                stage='color', 
                                                gt_depth=gt_depth,
                                                gt_color=gt_color)
                        depth_l1, psnr = retvis
                        depth_error += [depth_l1]
                        img_psnr += [psnr] 

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0: #匀速运动得到track当前帧的初始位姿
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else: # 就拿上一帧的位姿  所以输入还是要有时序才行
                    estimated_new_cam_c2w = pre_c2w

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR: #? 分开优化旋转 平移 好像 
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)
                    if not torch.isnan(gt_depth).all().item():
                        self.visualizer.vis(
                            idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders,
                            selecti=self.slecti, selectj=self.slectj)

                    loss = self.optimize_cam_in_batch( # frame 25(375,1242)
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        # if cam_iter == self.num_cam_iters-1:
                        if (cam_iter % self.visualizer.inside_freq == 0) and (cam_iter>0): # (idx % self.visualizer.freq == 0) and 
                            print(
                                f'T_iter {cam_iter:d}, Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            self.idx[0] = idx 
            if self.usepriormap and self.onlyvis:
                if (idx == self.n_img-2) or (idx % self.keyframe_every == 0) \
                    and (idx not in self.keyframe_list):
                    self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                                ), 'depth': gt_depth.cpu(), 'est_c2w': c2w.clone()})
                    self.keyframe_list.append(idx) #关键帧列表 以 frame_id组成
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
            
            # 若是最后一帧 这里进行mesh的提取吧
            if self.usepriormap and self.onlyvis:
                if idx==self.n_img-1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                    
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True) #只在评估重建质量时
                    # pass
        
        # 计算所有 涉及到的帧 误差平均
        depth_errors = np.array(depth_error)
        img_psnrs = np.array(img_psnr)
        
        # from m to cm
        print('Tracked Depth L1 (cm): ', depth_errors.mean()*100)
        print('Tracked PSNR (dB): ', img_psnrs.mean())
        
        if self.low_gpu_mem:
            torch.cuda.empty_cache()
        
        
        
        
