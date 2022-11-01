from audioop import reverse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
# -*- coding:utf-8 -*-

class Visualizer(object):
    """
    Visualize intermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, depth_trunc=-1,device='cuda:0'):
        self.freq = freq #输入参数
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq #输入参数
        os.makedirs(f'{vis_dir}', exist_ok=True)
        self.depth_trunc = depth_trunc #用于区分可视化

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,
            decoders, selecti = torch.zeros(0,), selectj = torch.zeros(0,)):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in  算法中的pose是 以世界系为中心
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                            torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor
                
                tmp = torch.zeros(gt_depth.shape, device=self.device)
                if torch.equal(tmp, gt_depth): # 若当前帧是非orb kf 就不用它来render
                    vis_depth = None
                else:
                    vis_depth = gt_depth
                depth, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    self.device,
                    stage='color', #这个渲染来可视化的stage是color！ 但其实也会得到fine下的occupancy
                    gt_depth=vis_depth) # 可能为none
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                # gt_depth_np = gt_depth_np/max_depth*255
                # gt_depth_np = np.clip(gt_depth_np, 0, 255).astype(np.uint8)
                # gt_depth_np = gt_depth_np.astype(np.float)*100
                # gt_depth_np = np.clip(gt_depth_np, 0, 65535)
                # gt_depth_np = gt_depth_np.astype(np.uint16)
                # 对于outdoor 要排除掉无穷远
                gt_alldepthv = np.unique(gt_depth_np)
                gt_alldepthv1 = gt_alldepthv[np.argsort(-gt_alldepthv)] #降序排列
                # secondmax_depth = gt_alldepthv1[1] # 次大值 保证不是sky 无限远
                # depth_np = depth_np/max_depth*255 #这种方式本质上和源代码等效 近处的深度仍看不清楚
                # depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
                # depth_np = depth_np.astype(np.float)*100
                # depth_np = np.clip(depth_np, 0, 65535)
                # depth_np = depth_np.astype(np.uint16)
                # depth_residual = depth_residual/max_depth*255
                # depth_residual = np.clip(depth_residual, 0, 255).astype(np.uint8)
                # depth_residual = depth_residual.astype(np.float)*100
                # depth_residual = np.clip(depth_residual, 0, 65535)
                # depth_residual = depth_residual.astype(np.uint16)
                if self.depth_trunc>0: # 如果depth截断过 就用最大值 ！
                    secondmax_depth = max_depth
                axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth) #max_depth
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma",
                                 vmin=0) #max_depth , vmax=max_depth
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma",
                                 vmin=0, vmax=max_depth) #max_depth
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                selecti = selecti.cpu().numpy()
                selectj = selectj.cpu().numpy()
                if selecti.shape[0] > 0 : #当有采样点时 在gtcolor画散点
                    # print('sample pixels: \t',selecti.shape[0])
                    if iter==0: #指在iter0可视化
                        axs[1,0].scatter(selecti, selectj, c='red', s=2)
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(
                    f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                plt.clf()

                # if self.verbose:
                #     print(
                #         f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')
