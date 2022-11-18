import numpy as np
import torch
import torch.nn.functional as F
# -*- coding:utf-8 -*-
import traceback
from mathutils import Matrix
import mathutils

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

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.
    使用内参得到射线的起点 和 方向 在世界系下的  (n,3)
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack( # K^{-1} [u, v, 1] 得到从光心发出的射线 这里是转到 nerf-pytorch坐标系  因为下面 c2w 是用的此坐标系 
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device) #(1000,3)
    dirs = dirs.reshape(-1, 1, 3) #(n,1,3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1) #(n,3) 世界系下射线
    rays_o = c2w[:3, -1].expand(rays_d.shape) #(n,3) 射线的起点 在世界系的坐标 也就是位姿的平移部分
    return rays_o, rays_d


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n uv from dense uv. 他这里没按I-map那样采样啊。。 均匀分布来采样
    depth 可能是 nan tensor 而且shape还是[4,4]
    i和j其实就是 图像裁剪后区域每像素的 下标 (u,v)
    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device) # 值域[0，总像素数) 的size为(n)的向量
    indices = indices.clamp(0, i.shape[0]) # 确保值域范围 （多此一举？ 上句已经保证了）
    i = i[indices]  # (n) 按随机的索引的索引 拿出 选择的像素的下标
    j = j[indices]  # (n)
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth[indices]  # (n)  当depth是nan tensor shape 不对时，在这里 out of bounds
    color = color[indices]  # (n,3) 选中的像素上的深度 和 颜色
    
    return i, j, depth, color
    
    # depth = depth.reshape(-1)
    # color = color.reshape(-1, 3) # sky==0 为了限制深度有效区域拿出来 for outdoor
    # finind = (depth>0).nonzero().reshape(-1) # 有限深度的总索引depth<600 (n) 655.35
    # indices0 = torch.randint(finind.shape[0], (n,), device=device)
    # indices0 = indices0.clamp(0, finind.shape[0])
    # indices1 = finind[indices0] # 对应的原始index
    # indices = indices1
    # i = i.reshape(-1) # 335*1202
    # j = j.reshape(-1)
    # i = i[indices]  # (n) 按随机的索引的索引 拿出 选择的像素的下标
    # j = j[indices]  # (n)
    
    # depth = depth[indices]  # (n)
    # color = color[indices]  # (n,3) 选中的像素上的深度 和 颜色
    # # dnn = depth.cpu().numpy().shape[0]
    # # cnn = color.cpu().numpy().shape
    # # print('[select_uv] color size: \t', cnn)
    # # try:
    # #     # print('[select_uv] 2/depth.shape[0]: \t', 2/dnn)
    # #     x = 2/dnn
    # # except Exception as e:
    # #     traceback.print_exc()
    # return i, j, depth, color


def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1
    H0= edge_ H1=H-edge_ W0= edge_ W1=W-edge_ n 需要多少像素点参与优化 来自设置文件pixels:
    depth 可能是 nan tensor
    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1] #裁剪后的区域 各减 2*edge (335, 1202,3)
    # dnn = depth.cpu().numpy().shape
    # dnnsum = dnn[0] + dnn[1]
    # try:
    #     # print('[get_sample_uv2] 2/depth.shapesum: \t', 2/dnnsum)
    #     x = 2/dnnsum
    # except Exception as e:
    #     traceback.print_exc()
    i, j = torch.meshgrid(torch.linspace( #目标图像区域 网格 list[20, 1221] list[20, 354] 
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose (1202,335)-> (h,w)
    j = j.t()  # i和j其实就是 图像区域每像素的 下标 (u,v) 矩阵
    i, j, depth, color = select_uv(i, j, n, depth, color, device=device) #拿出随机采样的像素 的 u v d c 向量们
    return i, j, depth, color


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1. 图像裁掉了边缘一些像素
    c2w is its camera pose and depth/color is the corresponding image tensor.
    depth 可能是nan tensor 而且shape 还是[4,4] 反正nan了
    (n,3) (n,3)  (n) (n,3)
    """
    i, j, sample_depth, sample_color = get_sample_uv( # 先采样n个像素 对应的下标 深度 颜色 这块不会有我呢提 和 Bound无关
        H0, H1, W0, W1, n, depth, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth, sample_color, i, j #增加返回采样点坐标 for debug


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion() # w x y z https://docs.blender.org/api/current/mathutils.html#mathutils.Quaternion
    if Tquad:
        tensor = np.concatenate([T, quad], 0) # tx y z qw qx qy qz
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, occupancy=False,
                           coarse_depths=None,
                        #    valid_depth_mask=None,
                           confidence=None, # 1 0.0025 0.01 (0.02) | 0.05 |0.1
                           raw_noise_std=1e0,
                           device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays,N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.
        coarse_depths: 当前batch的粗糙深度 可能含有0 用来构建 sigma_loss. 默认关闭
        confidence: to 计算 构建sigma_loss (KL 散度) 时需要的方差tensor 本来应该是各深度kpt reprojection err 算的,暂时设为1

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays,3): estimated RGB color of a ray.
        weights (tensor, N_rays,N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std # 变换后的 标准正态 噪声
        noise = noise.to(device)
    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1] #注意颜色是其中的一部分
    if occupancy: # 这个alpha 都是占据概率 原本sigma 对于nice是不在[0,1] 0,100
        raw[..., 3] = torch.sigmoid(10*(raw[..., -1] + noise))
        alpha = raw[..., -1] # 才是0，1
    else:
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1] + noise, dists)  # (N_rays, N_samples)
    # ray termination probability = alpha(occupancy) * accumulated transmittance
    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1] # # (N_rays, N_samples)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays) # 方差
    
    sigma_loss = None # 返回值可能为空
    if coarse_depths is not None: # 非空 意味着启用 kl loss
        if len(coarse_depths.shape)==1:
            coarse_depths = coarse_depths.reshape(-1, 1)
        gt_none_zero_mask = (coarse_depths>0).squeeze(-1) # (N,1) 只对 那些有粗糙深度的区域使用此Loss [:,None]
        # gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
        # 由置信度 计算 方差  两者是反相关
        if confidence is not None:
            err = (0.1 / confidence) ** 2
        else:
            err = torch.ones_like(depth_var) * 0.0225 # 0.04 0.0225
        err = err.reshape(-1, 1)
        # # 对远方差进行数值调整 最大值不超 0.0225
        # fact = 0.0225/err.max()
        # err *= fact
        err = err.repeat(1, z_vals.shape[1]) # (b, N_samples) 才能参与到下面运算
        # weights 里会有0 导致下面计算失败nan https://github.com/dunbar12138/DSNeRF/issues/69 加上很小的数
        prior_gauss = torch.exp(-(z_vals[gt_none_zero_mask] - coarse_depths[gt_none_zero_mask]) ** 2 / (2 * err[gt_none_zero_mask]))
        gaus_factor = 1. / ((2*torch.pi*err[gt_none_zero_mask])**(0.5)) # (n_valid)
        if occupancy: # 当是完整kl_loss 1e-5 换成 1e-0 - prior_gauss
            sigma_loss = -torch.log(weights[gt_none_zero_mask] + 1e-5) * \
                gaus_factor * \
                torch.exp(-(z_vals[gt_none_zero_mask] - coarse_depths[gt_none_zero_mask]) ** 2 / (2 * err[gt_none_zero_mask])) * \
                    dists[gt_none_zero_mask] # 应为 (N_rays[valid depth], N_samples)
        else:
            sigma_loss = -torch.log(weights[gt_none_zero_mask] + 1e-5) * \
                gaus_factor * \
                torch.exp(-(z_vals[gt_none_zero_mask] - coarse_depths[gt_none_zero_mask]) ** 2 / (2 * err[gt_none_zero_mask])) * \
                    dists[gt_none_zero_mask] # 应为 (N_rays[valid depth], N_samples)
        assert torch.isnan(sigma_loss).any() == False # 会出现nan？
        assert torch.isinf(sigma_loss).any() == False # weight不加扰动就会
        # sigma_loss = torch.sum(sigma_loss, dim=1) # 应为 (N_rays[valid depth])
        
    return depth_map, depth_var, rgb_map, weights, sigma_loss


def dataclass_map(fn, x):
  """Behaves like jax.tree_map but doesn't recurse on fields of a dataclass."""
  return x.__class__(**{k: fn(v) for k, v in vars(x).items()})

def subsample_patches(images, patch_size, batch_size, batching='all_images'): # 需要移植吧
    """Subsamples patches.
        images 这里可能是 rays()
    """
    n_patches = batch_size // (patch_size ** 2)

    scale = np.random.randint(0, len(images)) # 就1个 0
    images = images[scale]

    if isinstance(images, np.ndarray):
        shape = images.shape
    else: # 是ray 类
        shape = images.origins.shape #(100,378,504,3)

    # Sample images
    if batching == 'all_images': # 对于ray train 是这里
        idx_img = np.random.randint(0, shape[0], size=(n_patches, 1)) # 每个patch选择的 view(pose) 每个patch属于不同虚拟image
    elif batching == 'single_image':
        idx_img = np.random.randint(0, shape[0])
        idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
    else:
        raise ValueError('Not supported batching type!')

    # Sample start locations 每个patch在各自图片的左上位置
    x0 = np.random.randint(0, shape[2] - patch_size + 1, size=(n_patches, 1, 1))
    y0 = np.random.randint(0, shape[1] - patch_size + 1, size=(n_patches, 1, 1))
    xy0 = np.concatenate([x0, y0], axis=-1) # (n_patches, 1, 2)
    patch_idx = xy0 + np.stack(
        np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
        axis=-1).reshape(1, -1, 2) # 位置 (n_p, 64,2)

    # Subsample images
    if isinstance(images, np.ndarray):
        out = images[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
    else:
        out = dataclass_map(
            lambda x: x[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(  # pylint: disable=g-long-lambda
                -1, x.shape[-1]), images) # (128=n_p*p_s,3) rays() 选出的patch的ray
        # 改为直接得到 最后显式的数据  
    return out, np.ones((n_patches, 1), dtype=np.float32) * scale # 若只有原scle 这里就是 0向量

def compute_tv_norm(values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
    """Returns TV norm for input values. geometry reg 的 loss 需要移植！ 直接copy注释
    values n_p, ps, ps, 1
    weighting n_p, ps-1, ps-1, 1
    Note: The weighting / masking term was necessary to avoid degenerate 但默认值
    solutions on GPU; only observed on individual DTU scenes.
    """
    v00 = values[:, :-1, :-1] # (n_p,ps-1,ps-1,1)
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = np.abs(v00 - v01) + np.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')

    if weighting is not None:
        loss = loss * weighting
    return loss # (n_p,ps-1,ps-1,1)

def compute_tvnorm_weight(step, max_step, weight_start=0.0, weight_end=0.0):
    """Computes loss weight for tv norm.
        这里的step 是 mapping 全局的计数器 包含init 
    """ # 从regnerf移植过来
    w = np.clip(step * 1.0 / (1 if (max_step < 1) else max_step), 0, 1)
    return weight_start * (1 - w) + w * weight_end


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.
    c2w 应该是 4,4
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound): # 这和NDC一样吗？ 感觉不是吧
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p


def get_rays_by_weight(H, W, fx, fy, cx, cy, c2w, depth, color,
                       weight, device):
    """
    由给出的 weight map 中 >0 的位置 得到射线 用来监督 稀疏关键点的位置
    和 对应的 coarse depth weght color u, v
    Args:
        H (_type_): _description_
        W (_type_): _description_
        fx (_type_): _description_
        cx (_type_): _description_
        cy (_type_): _description_
        c2w (_type_): _description_
        weight (_type_): _description_
        device (_type_): _description_
    """
    # kpt_tup = torch.where(weight > -0) # tuple 每个是 [2] (v,u)
    # sample_depth = depth[kpt_tup]
    # sample_color = depth[kpt_tup]
    
    # coords = torch.stack(kpt_tup, 0) # (n,2)
    
    # kpt_u = coords[:, 1]
    # kpt_v = coords[:, 0]

    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth).to(device)
        color = torch.from_numpy(color).to(device)
        weight = torch.from_numpy(weight).to(device)
        

    depth = depth.reshape(-1)
    weight = weight.reshape(-1) # (h*w)
    color = color.reshape(-1, 3) # (h*w,3)
    
    indices = (weight>-0).nonzero().squeeze(-1) # (m) 这里已经是去掉了本身由于 置信度太低 而没有保留下来的关键点
    i, j = torch.meshgrid(torch.linspace( #目标图像区域 网格 list[20, 1221] list[20, 354] 
        0, W-1, W).to(device), torch.linspace(0, H-1, H).to(device))
    i = i.t()  # transpose (w,h)-> (h,w)
    j = j.t()  # i和j其实就是 图像区域每像素的 下标 (u,v) 矩阵
    i = i.reshape(-1)
    j = j.reshape(-1)
    kpt_u = i[indices]  # (n) 按随机的索引的索引 拿出 选择的像素的下标
    kpt_v = j[indices]  # (n)
    sample_depth = depth[indices]  # (n)  当depth是nan tensor shape 不对时，在这里 out of bounds
    sample_color = color[indices]  # (n,3) 选中的像素上的深度 和 颜色
    sample_weight = weight[indices]
    rays_o, rays_d = get_rays_from_uv(kpt_u, kpt_v, c2w, H, W, fx, fy, cx, cy, device)
    
    return rays_o, rays_d, sample_depth, sample_color, sample_weight, kpt_u, kpt_v


def calc_depth_l1(est_depth, gt_depth):
    """
    计算给定图 的 depth L1 error
    copy from src/tools/eval_recon.py calc_2d_metric.py
    Args:
        est_depth (np.array): 估计的深度图
        dt_depth (np.array): 真实的深度图
    """
    
    assert est_depth.shape == gt_depth.shape, 'est_depth shape != gt_depth shape'
    # 对于tum 还会有 sensor depth 残缺的情况
    valid_mask = (gt_depth > 0.)
    error = np.abs(gt_depth[valid_mask] - est_depth[valid_mask]).mean()
    
    return error

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return value.mean()
    return value

def calc_psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    """
    PSNR, borrow from nerf-pl: metrics.py

    Args:
        image_pred (np.array): _description_
        image_gt (np.array): _description_
        valid_mask (np.array, optional): _description_. Defaults to None.
        reduction (str, optional): _description_. Defaults to 'mean'.

    Returns:
        float: _description_
    """
    assert image_pred.shape == image_gt.shape, 'image_pred shape != image_gt shape'
    
    return -10*np.log10(mse(image_pred, image_gt, valid_mask, reduction))
    