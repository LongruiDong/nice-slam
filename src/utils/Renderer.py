from copy import copy
import torch, os
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf, get_rays_by_weight
# from src.utils.Visualizer import Visualizer.
# -*- coding:utf-8 -*-

class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance'] #用处是啥
        self.emperical_range = float(cfg['rendering']['emperical_range'])
        self.less_sample_space = slam.less_sample_space # 是否在 surface 附近区间多重采样,对应不同的render_batch
        self.use_KL_loss = slam.use_KL_loss # 是否使用ds-nerf的sigma loss, 对应于调用不同的底层raw2outputs 函数
        self.up_sample_steps = cfg['rendering']['up_sample_steps'] # 在表面附件多重采样的次数 每次 N_surface//up_sample_steps
        # Ref: https://github.com/dunbar12138/DSNeRF/blob/main/loss.py KL 散度loss sigma_loss 的参数
        self.raw_noise_std = cfg['rendering']['raw_noise_std'] # 默认1.0 其实对于occpancy用不上
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        self.nice = slam.nice
        self.bound = slam.bound
        self.guide_sample = slam.guide_sample

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            if self.nice: #注意这里stage
                ret = decoders(pi, c_grid=c, stage=stage)
            else: # imap* 
                ret = decoders(pi, c_grid=None)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def finer_sample(self, c, decoders, z_vals, rays_d, rays_o, N_importance, device, stage):
        """
        因为目前涉及到多次coarse to fine 采样 所以搞一个专门的函数来做
        在给定先前采样的下标, 再此render的结果上再进行按分布进一步采样
        注意这里的输入 是否有深度 要对应上！
        Args:
            c (_type_): _description_
            decoders (_type_): _description_
            z_vals (_type_): _description_
            rays_d (_type_): _description_
            rays_o (_type_): _description_
            N_importance (_type_): 本次细化需要增加的采样数
            device (_type_): _description_
        """
        N_rays = rays_o.shape[0]
        N_samples = z_vals.shape[1]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3] z_val 才是实际的t
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples, -1)
        # 只是为了拿到更细采样点 并不需要得到KL loss
        _, _, _, weights, _ = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
        
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
        z_samples = z_samples.detach()
        # z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 返回的是增加的样本位置
        return z_samples
        
        
    def render_batch_ray_tri(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None,
                             confidence=None):
        """
        此函数是在使用深度prior的版本
        Render color, depth and uncertainty of a batch of rays. imap 和 nice-slam公共的代码 在query mlp 时会区分开
        将原batch 分为 由深度 和无深度两类 采用不同的采样策略,还是基础的coarse全都做 这样方便拼接
        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.
            confidence (tensor, optional): 可信度 权重 对于稀疏点深度的权值 越大越可靠

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = self.N_samples
        N_surface = self.N_surface # 不同于 original nerf
        N_importance = self.N_importance

        N_rays = rays_o.shape[0]

        if stage == 'coarse' or (not self.guide_sample):
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01 # 没有深度时 这表示 几乎从rays_O开始吧
        else: # 还是会用深度 得到 near 相对有个先验 吧
            gt_depth = gt_depth.reshape(-1, 1) # (N,1)
            gt_none_zero_mask = gt_depth > 0 # (N,1)
            # 分为两部分 暂时还是需要coarse level 大range
            near_base = torch.full(size=gt_depth.shape, fill_value=0.01, device=device) # (N,1) 0.1*torch.min(gt_depth)
            near_base[gt_none_zero_mask] = 0.01 * gt_depth[gt_none_zero_mask] # 0.1 0.8
            near = near_base.repeat(1, N_samples) # (N,sam#)

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2) 这段代码在mapping判断scaene是否在bound内 出现过
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1) # 是根据设置的cube bound 得到t- far
            far_bb = far_bb.unsqueeze(-1) # (N, 1)
            far_bb += 0.01

        if gt_depth is not None: #gt_depth (4089,1) why 4089
            # in case the bound is too large 每个ray的上界不至于太离谱的大 缩小sample空间
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2)) # (N, 1)
        else:
            far = far_bb
        
        t_vals = torch.linspace(0., 1., steps=N_samples, device=device) # [0,1] 等距采样N_samp个点

        # 有了初步的near far 后 可以选择是否退火 to do
        if not self.lindisp: #默认false
            z_vals = near * (1.-t_vals) + far * (t_vals) # (N,N_sam)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand
        
        if N_surface > 0:
            # gt_none_zero_mask = gt_depth > 0
            # 看了其他的代码 还都是没有那样分开 分成有无深度的区域来进行
            gt_none_zero = gt_depth[gt_none_zero_mask] # (N) 对于render img是 某一个batch可能全是0
            gt_none_zero = gt_none_zero.unsqueeze(-1) # (N_,1)
            gt_none_zero_mask = gt_none_zero_mask.squeeze(-1) # ->(N)要不要放这里？
            z_vals_surface = torch.zeros(
                gt_depth.shape[0], N_surface).to(device).double()
            if z_vals[~gt_none_zero_mask].shape[0]>0:
                # 对于depth zero的部分 就在初次采样基础上 直接再来一次
                z_samples_zero = self.finer_sample(c, decoders, z_vals[~gt_none_zero_mask],
                                                rays_d[~gt_none_zero_mask], rays_o[~gt_none_zero_mask],
                                                N_surface, device, stage)
                z_vals_surface[~gt_none_zero_mask,
                                :] = z_samples_zero # to debug shape
            
            if gt_none_zero.shape[0]>0: # 当遇到此情况 不再进行下面的小区域内coarse2fine
                # 开始循环 >=2多次来采样 这里只是对于 none_zero的部分
                N_surface_i = N_surface // self.up_sample_steps # 每次循环时需要的采样次数
                gt_depth_surface = gt_none_zero.repeat(1, N_surface_i)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface_i).double().to(device)
                # emperical range 0.05*depth # 这个超参数本来就是经验性的 改为 0.15 0.2 因为要在这里 多重采样
                z_vals_surface_depth_none_zero = (1-self.emperical_range)*gt_depth_surface * \
                    (1.-t_vals_surface) + (1+self.emperical_range) * \
                    gt_depth_surface * (t_vals_surface) # (N, Surf) 每行都是给定的区间 在 当前depth
                z_surface_last = z_vals_surface_depth_none_zero.detach() # 这是先得到表面附近区间 初始样本位置
                
                if self.up_sample_steps>1:    
                    for i in range(1, self.up_sample_steps):
                        if self.perturb > 0.:
                            # get intervals between samples
                            mids = .5 * (z_surface_last[..., 1:] + z_surface_last[..., :-1])
                            upper = torch.cat([mids, z_surface_last[..., -1:]], -1)
                            lower = torch.cat([z_surface_last[..., :1], mids], -1)
                            # stratified samples in those intervals
                            t_rand = torch.rand(z_surface_last.shape).to(device)
                            z_surface_last = lower + (upper - lower) * t_rand
                        z_surface_i = self.finer_sample(c, decoders, z_surface_last,
                                                rays_d[gt_none_zero_mask], rays_o[gt_none_zero_mask],
                                                N_surface_i, device, stage)
                        z_surface_last, _ = torch.sort(torch.cat([z_surface_last, z_surface_i], -1), -1)
                        
                # 经过上面多次采样, 得到非0部分的样本        
                z_vals_surface[gt_none_zero_mask,
                                :] = z_surface_last

        if N_surface > 0:
            z_vals, _ = torch.sort( # 每一行 把 各自采样拼起来 concat 并每行排序  [N_rays, N_samples+N_surface]
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3] z_val 才是实际的t
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)
        sigma_loss = None # 只有在最后 要输出最后结果时 才设置是否使用KL loss
        depth, uncertainty, color, weights, sigma_loss = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device,
            coarse_depths=gt_depth, confidence=confidence) # 内部会判读是否none
        
        # # 根据regnerf 由weights 进一步得到 rendering['acc'] 来用到 depth patch loss
        # regacc = weights.sum(axis=-1) # (batchsize)
        if N_importance > 0: # 在整个所有前面基础上, 再重复采样 
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf( # 多层采样
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

            depth, uncertainty, color, weights, sigma_loss = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device,
                coarse_depths=gt_depth, confidence=confidence) # if self.use_KL_loss else None
            return depth, uncertainty, color, weights, sigma_loss, z_vals
        # 增加输出采样的深度 为了画图
        return depth, uncertainty, color, weights, sigma_loss, z_vals

    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays. imap 和 nice-slam公共的代码 在query mlp 时会区分开

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        N_samples = self.N_samples
        N_surface = self.N_surface # 不同于 original nerf
        N_importance = self.N_importance

        N_rays = rays_o.shape[0]

        if stage == 'coarse' or (not self.guide_sample):
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01 # 没有深度时 这表示 几乎从rays_O开始吧
        else: # 还是会用深度 得到 near 相对有个先验 吧
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples*0.01 # (N,sam#)

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2) 这段代码在mapping判断scaene是否在bound内 出现过
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1) # 是根据设置的cube bound 得到t- far
            far_bb = far_bb.unsqueeze(-1) # (N, 1)
            far_bb += 0.01

        if gt_depth is not None: #gt_depth (4089,1) why 4089
            # in case the bound is too large 每个ray的上界不至于太离谱的大 缩小sample空间
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface) # H*W, N_surf
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface) # H*W, N_surf 各列就是深度附近的0.05D区间采样后的深度
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth # 这个超参数本来就是经验性的 当prior depth 可否放松一些
                z_vals_surface_depth_none_zero = (1-self.emperical_range)*gt_depth_surface * \
                    (1.-t_vals_surface) + (1+self.emperical_range) * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(
                    gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,
                               :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(
                    0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,
                               :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=device) # [0,1] 等距采样N_samp个点

        # 有了初步的near far 后 可以选择是否退火 to do
        if not self.lindisp: #默认false
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3] z_val 才是实际的t
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)
        # 此函数里 就不使用 sigma_loss
        depth, uncertainty, color, weights, sigma_loss = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device, coarse_depths=gt_depth)
        # # 根据regnerf 由weights 进一步得到 rendering['acc'] 来用到 depth patch loss
        # regacc = weights.sum(axis=-1) # (batchsize)
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

            depth, uncertainty, color, weights, sigma_loss = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device, coarse_depths=gt_depth)
            return depth, uncertainty, color, weights, sigma_loss, z_vals
        # 增加输出采样的深度 为了画图
        return depth, uncertainty, color, weights, sigma_loss, z_vals

    def render_img(self, c, decoders, c2w, device, stage, gt_depth=None, weight_vis_dir=None,
                   prefix=None, confidence=None, gt_color=None):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.
            confidence (tensor, optional): 该帧的关键点的置信度图

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []
            weights_list = []
            sigma_loss_list = []
            z_vals_list = []

            ray_batch_size = self.ray_batch_size
            if gt_depth is not None:
                gt_depth = gt_depth.reshape(-1)
                confidence = confidence.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    if self.less_sample_space:
                        ret = self.render_batch_ray_tri(
                            c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                    else:
                        ret = self.render_batch_ray(
                            c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    # confidence_batch = confidence[i:i+ray_batch_size]
                    # kpt_mask = confidence_batch > -0
                    # # 强行令 非关键点的深度为0 只是为了可视化
                    # gt_depth_batch[~kpt_mask] = 0.
                    # tmp = torch.zeros_like(gt_depth_batch)
                    # if torch.equal(tmp, gt_depth_batch): # 若当前帧是非orb kf 就不用它来render
                    #     vis_depth = None
                    #     vis_con = None
                    # else:
                    #     vis_depth = gt_depth_batch
                    #     vis_con = confidence_batch
                    if self.less_sample_space: #  or (vis_depth is not None)
                        ret = self.render_batch_ray_tri(
                            c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch, confidence=None) # 只需要前3个输出 不要loss
                    else:
                        ret = self.render_batch_ray(
                            c, decoders, rays_d_batch, rays_o_batch, device, stage, gt_depth=gt_depth_batch)

                depth, uncertainty, color, weights, sigma_loss, z_vals = ret # 这里的输出不同 但sigma_loss可能仍是None
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)
                weights_list.append(weights)
                sigma_loss_list.append(sigma_loss)
                z_vals_list.append(z_vals)
                    

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)
            weights = torch.cat(weights_list, dim=0)
            
            z_vals = torch.cat(z_vals_list, dim=0)
            # img 内的 weights 切片可视化检验 debug
            if gt_depth is not None: # 即当前img是orb kf情况下 gt_depth is not None
                sigma_loss = torch.cat(sigma_loss_list, dim=0)
                if sigma_loss.shape[0] > 0:
                    
                    gt_none_zero_mask = confidence > 0 # 只管 kpt 的位置
                    gt_depth_surface = gt_depth[gt_none_zero_mask]
                    weights_valid = weights[gt_none_zero_mask]
                    z_vals_valid = z_vals[gt_none_zero_mask]
                    # sigma_loss_valid = sigma_loss[gt_none_zero_mask]
                    confidence_valid = confidence[gt_none_zero_mask]
                    
                    n_valid = gt_depth_surface.shape[0]
                    
                    interval = 100 # 500 100
                    m_plot0 = n_valid // interval # 每间隔100个像素 取出 画图
                    m_plot = min(m_plot0, 10)
                    for k in range(m_plot):
                        self.visualize_weights(gt_depth_surface[k*interval].cpu().numpy(),
                                               weights_valid[k*interval, :].cpu().numpy(),
                                               z_vals_valid[k*interval, :].cpu().numpy(),
                                               sigma_loss[k*interval, :].cpu().numpy(),
                                               os.path.join(weight_vis_dir, prefix+f'_ray_%d.png'%k),
                                               confidence=confidence_valid)
                
                # 由于前面 是分batch的 不能统一只关注kpt 这里单独渲染kpt 的ray
                
                # 取出当前帧现在位姿后 用单独函数来提取出 稀疏点 对应的射线
                prior_rays_o, prior_rays_d, prior_depth, prior_color, prior_weight, kpt_i, kpt_j = get_rays_by_weight(
                    H, W, self.fx, self.fy, self.cx, self.cy, c2w, gt_depth.reshape(H,W), gt_color, confidence, device
                )
                if prior_rays_o.shape[0] > 0 :
                    sp_ret = self.render_batch_ray_tri(c, decoders, prior_rays_d,
                                                            prior_rays_o, device, stage,
                                                            gt_depth=prior_depth, # 对于有效稀疏点 coarse深度和置信度都有
                                                            confidence=prior_weight) #
                    
                    depth_sp, uncertainty_sp, color_sp, weights_sp, sigma_loss_sp, z_vals_sp = sp_ret
                    gt_none_zero_mask = prior_depth > 0 # 只管 kpt 的位置
                    prior_depth = prior_depth[gt_none_zero_mask]
                    z_vals_sp = z_vals_sp[gt_none_zero_mask]
                    weights_sp = weights_sp[gt_none_zero_mask]
                    prior_weight = prior_weight[gt_none_zero_mask]
                    n_valid = prior_depth.shape[0]
                    
                    interval = 100 # 500 100
                    m_plot0 = n_valid // interval # 每间隔100个像素 取出 画图
                    m_plot = min(m_plot0, 10)
                    for k in range(m_plot):
                        self.visualize_weights(prior_depth[k*interval].cpu().numpy(),
                                            weights_sp[k*interval, :].cpu().numpy(),
                                            z_vals_sp[k*interval, :].cpu().numpy(),
                                            sigma_loss_sp[k*interval, :].cpu().numpy(),
                                            os.path.join(weight_vis_dir, prefix+f'_keyray_%d.png'%k), # 单独命名
                                            confidence=prior_weight[k*interval].cpu().numpy())
                

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    def visualize_weights(self, prior_depth, weights, z_vals, sigma_loss,
                          plotpath,
                          confidence=None):
        """
        可视化某个ray上 个采样点深度 、render 的weight、 prior_depth 的位置之间的关系 

        Args:
            prior_depth (_type_): _description_
            weights (_type_): _description_
            z_vals (_type_): _description_
            sigma_loss: 
            plotpath (_type_): _description_
        """
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        axs[0].plot(z_vals, weights)
        axs[0].scatter(z_vals, weights, c='green', s=2)
        axs[0].axvline(prior_depth, linestyle='--', color='red') # 竖线表示先验深度的位置
        # 再画个 其代表的高斯分布吧
        # err = 0.02 # 暂时以此为方差吧 之后是以实际方差来代替
        err = (1. / confidence) ** 2
        prior_y = 1./(np.sqrt(2*np.pi*err)) * np.exp(- (z_vals - prior_depth) ** 2 / (2*err))
        axs[0].plot(z_vals, prior_y, c='tab:orange')
        # axs[0].vlines(prior_depth, linestyle='dashed', color='red')
        axs[1].plot(z_vals, sigma_loss) # 画loss
        axs[1].scatter(z_vals, sigma_loss, c='tab:orange', s=2) # 画loss
        axs[0].set_title('ray weight. prior depth std: {}'.format(np.sqrt(err)))
        axs[1].set_title('KL loss: {}'.format(np.sum(sigma_loss)))
        # axs.set_xticks([])
        # axs.set_yticks([])
        plt.savefig(plotpath, dpi=200)
        
    
    # this is only for imap*
    def regulation(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.  0.85这个数字！       
        For imap, the geometry will not be as good if this loss is not added.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): sensor depth image
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, self.N_samples)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        near = 0.0
        far = gt_depth*0.85
        z_vals = near * (1.-t_vals) + far * (t_vals)
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pointsf = pts.reshape(-1, 3)
        raw = self.eval_points(pointsf, decoders, c, stage, device)
        sigma = raw[:, -1] # imap 这里的值域[0,1]
        return sigma
    
    # 对于 nice-slam 的 occupancy
    def regulation_occ(self, c, decoders, rays_d, rays_o, gt_depth, device, stage='color'):
        """
        Regulation that discourage any geometry from the camera center to 0.85*depth.  0.85这个数字！       
        For imap, the geometry will not be as good if this loss is not added.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            gt_depth (tensor): 应该是非全0的 coarse depth prior
            device (str): device name to compute on.
            stage (str, optional):  query stage. Defaults to 'color'.

        Returns:
            sigma (tensor, N): volume density of sampled points.
        """
        gt_depth = gt_depth.reshape(-1, 1)
        gt_none_zero_mask = gt_depth > 0
        gt_none_zero = gt_depth[gt_none_zero_mask]
        gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
        gt_none_zero = gt_none_zero.unsqueeze(-1)
        gt_depth_surface = gt_none_zero.repeat(1, self.N_samples)
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        near = 0.0
        far = gt_depth_surface*0.85
        z_vals = near * (1.-t_vals) + far * (t_vals)
        perturb = 1.0
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[gt_none_zero_mask, None, :] + rays_d[gt_none_zero_mask, None, :] * \
            z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pointsf = pts.reshape(-1, 3)
        raw = self.eval_points(pointsf, decoders, c, stage, device) # (N_rays*N_samples, 4)
        # sigma = raw[:, -1] # (N_rays*N_samples)
        # 或者返回 占据概率的值  两者的值最后abs sum后不同吧
        sigma = torch.sigmoid(10*raw[:, -1])
        return sigma
