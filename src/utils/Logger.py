import os
# -*- coding:utf-8 -*-
import torch


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, slam
                 ):
        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.shared_c = slam.shared_c
        self.gt_c2w_list = slam.gt_c2w_list
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx)) # 保存Map参数的
        torch.save({
            'c': self.shared_c,
            'decoder_state_dict': self.shared_decoders.state_dict(), # state_dict是什么
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            # 'keyframe_dict': keyframe_dict, # to save keyframe_dict into ckpt, uncomment this line
            'selected_keyframes': selected_keyframes, #包含有关键帧id等信息
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
