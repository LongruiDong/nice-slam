import os
import time
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args
        self.nice = args.nice
        # 这些变量来自与yaml哪里
        # 和 prior coarse geometry 有关的设置参数
        self.usepriormap = cfg['usepriormap'] # 使用之前建好的grid以及color decoder
        self.onlyvis = cfg['onlyvis'] # 是否只进行可视化
        self.use_prior = cfg['use_prior'] # 是否载入 est depth 作为 gt deptj
        self.guide_sample = cfg['guidesample']
        self.less_sample_space = cfg['rendering']['less_sample_space'] # 是否在 surface 附近区间多重采样,对应不同的render_batch
        self.use_KL_loss = cfg['rendering']['use_KL_loss'] # 是否使用ds-nerf的sigma loss, 对应于调用不同的raw2output 函数
        self.rgbonly = cfg['rgbonly']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge'] #含义 coarselevel 是其他层级bound coarse_bound_enlarge 倍
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'] #相机参数
        self.update_cam()

        model = config.get_model(cfg,  nice=self.nice)
        self.shared_decoders = model #which scale?

        self.scale = cfg['scale'] #?

        self.load_bound(cfg) 
        if self.nice:
            if self.usepriormap:
                self.load_priormap(cfg)
            else:
                self.load_pretrain(cfg)
                self.grid_init(cfg)
        else:
            if self.usepriormap:
                pass
            self.shared_c = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        # print('use noise depth {}'.format(self.frame_reader.wnoise))
        print('use depth guide sample? : {}'.format(self.guide_sample))
        self.n_img = len(self.frame_reader)
        
        if self.use_prior: # 若是先验模式 就增加slam的成员变量保存 3d点 都已经是tensor
            # 对3d点的处理都是仿照 shared_c
            # self.prior_xyzs = self.frame_reader.prior_xyzs
            self.prior_xyzs_dict = self.frame_reader.prior_xyzs_dict
            self.prior_3dobs_dict = self.frame_reader.prior_3dobs_dict

        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()
        # 关于位姿
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['mapping']['device']) #在grid 很大时 会出现out of memory
            val.share_memory_()
            self.shared_c[key] = val
        if self.use_prior:
            for key, val in self.prior_xyzs_dict.items(): # 这样总的3d点是否又会占很多显存？
                val = val.to(self.cfg['mapping']['device'])
                val.share_memory_()
                self.prior_xyzs_dict[key] = val
            for key, val in self.prior_3dobs_dict.items(): # 现有的3d到2d的对应关系
                val = val.to(self.cfg['mapping']['device'])
                val.share_memory_()
                self.prior_3dobs_dict[key] = val
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)
        if self.coarse and (not self.rgbonly): # (not self.rgbonly) False
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # print('raw config bound: \n',np.array(cfg['mapping']['bound']))
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound'])*self.scale) #bound也会根据scle设置调整
        bound_divisible = cfg['grid_len']['bound_divisible']
        # enlarge the bound a bit to allow it divisible by bound_divisible 为啥要除以bound_divisible
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]
        if self.nice: #2 level 和颜色 共用一个bound
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse: #coarse 都是单独
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'],
                              map_location=cfg['mapping']['device'])
            coarse_dict = {}
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids. 这个看看到底是怎么弄
        grid_len 不受scale的参数影响
        Args:
            cfg (dict): parsed config dict.
        """
        if self.coarse: #先设置每层级 网格实际scale大小
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}
        c_dim = cfg['model']['c_dim'] # map 每个voxel 32维
        xyz_len = self.bound[:, 1]-self.bound[:, 0] #bound 各维度scale

        # If you have questions regarding the swap of axis 0 and 2,
        # please refer to https://github.com/cvg/nice-slam/issues/24
        
        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_val_shape = list(
                map(int, (xyz_len*self.coarse_bound_enlarge/coarse_grid_len).tolist())) #由于coarse_bound_enlarge=coarse_grid_len 这里出来和xyzlen值类似 [17,8,13]
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0] #x 和 z互换 why ?
            self.coarse_val_shape = coarse_val_shape
            val_shape = [1, c_dim, *coarse_val_shape] #[1,32,13,8,7]
            coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01) #地图 各voxel 0高斯初始化
            c[coarse_key] = coarse_val
            print('grid_coarse: \n', val_shape)

        middle_key = 'grid_middle'
        middle_val_shape = list(map(int, (xyz_len/middle_grid_len).tolist())) # [53,26,40]
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape] #[1,32,40,26,53]
        middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[middle_key] = middle_val
        # print('grid_middle: \n', val_shape)

        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len/fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)
        c[fine_key] = fine_val
        print('grid_fine: \n', val_shape)

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[color_key] = color_val
        # print('grid_color: \n', val_shape)

        self.shared_c = c
    
    def load_priormap(self, cfg):
        """
        从之前已经建立好的map文件 ckpt 载入需要的 grid_feature 和 所有decoder 特别是color decoder
        反正另外3个decodera是pretrained
        Args:
            cfg (dict): parsed config dict
        """
        ckpt_path = cfg['tracking']['priormap'] # yaml设置好的ckpt/中最后一个文件
        print('Get ckpt :', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=cfg['mapping']['device'])
        print('Load prior map ckpt done !')
        # 直接拿出 grid tmp_c
        self.shared_c = ckpt['c']
        # 需要更改 c里 各level requires_grad 属性为valse
        for key, val in self.shared_c.items():
            val.requires_grad = False
            self.shared_c[key] = val.cpu()
        print('[Debug] grid_color: \n',self.shared_c['grid_color'].shape)
        # # 再读入 几何decoder
        # self.load_pretrain(cfg)
        # 一次性读入去全部 
        self.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])
        print('Load prior decoder all state_dict done !')
    
    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """
        if (not self.usepriormap) and (not self.onlyvis): # 若只是为了可视化就不等了
            # should wait until the mapping of first frame is finished 初始化
            while (1):
                if self.mapping_first_frame[0] == 1:
                    break
                time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(3): #共3个线程 
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                if self.usepriormap and self.onlyvis:
                    continue
                else: # 当使用已有地图 且只只进行可视化时 就不开启mapping线程了！
                    p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 2:
                if self.coarse and (not self.rgbonly): # (not self.rgbonly) False
                    p = mp.Process(target=self.coarse_mapping, args=(rank, )) # coarse mapping独立出来线程
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
