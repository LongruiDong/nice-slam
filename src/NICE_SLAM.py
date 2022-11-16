import os
import time
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

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
        self.rgbonly = cfg['rgbonly']
        self.usepriormap = cfg['usepriormap'] # 使用之前建好的grid以及color decoder
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

        model = config.get_model(cfg,  nice=self.nice) # 初始化 包含 coarse(if true) middile fine  color 4个 decoder mlp
        self.shared_decoders = model # 还不带weight?

        self.scale = cfg['scale'] #?

        self.load_bound(cfg) 
        if self.nice: # nice  需要训好的 decoder
            if self.usepriormap:
                # to do color权重 的载入
                # to do 整个shared_c的载入
                # pass
                # self.grid_init(cfg) # debug 就是因为我目前 对 grid的载入问题
                self.load_priormap(cfg)
            else:
                self.load_pretrain(cfg) # 没有color的权重
                self.grid_init(cfg)
        else:
            if self.usepriormap: # 就MLP 所以shared_c还空
                # to do 单个的MLP权值的载入
                pass
            self.shared_c = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        print('use noise depth {}'.format(self.frame_reader.wnoise))
        self.n_img = len(self.frame_reader)
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
        self.shared_decoders = self.shared_decoders.to(
            self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        self.renderer = Renderer(cfg, args, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(cfg, args, self) # 存储参数
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)
        if self.coarse and False: # (not self.rgbonly)
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
        # enlarge the bound a bit to allow it divisible by bound_divisible 为啥要除以bound_divisable
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
        注意只是表示几何的那3个decoder
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
        Initialize the hierarchical feature grids. 这个看看到底是怎么弄 都是随机初始化
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
                p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 2:
                if self.coarse and False: # (not self.rgbonly)
                    p = mp.Process(target=self.coarse_mapping, args=(rank, )) # coarse mapping独立出来线程
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # def justreconmesh(self, outdir):
    #     """
    #     载入grid后 只是得到对应的mesh 用来直接 eval
    #     """
    #     frame_loader = DataLoader(
    #         self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
    #     pbar = frame_loader
    #     for idx, gt_color, gt_depth, gt_c2w in pbar:
    #         gt_c2w = gt_c2w[0]
        
    #     self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
        
        

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
