# Copyright (c) OpenMMLab. All rights reserved.
import random

from mmcv.runner import get_dist_info

from mmdet.models.builder import DETECTORS
from .yolox_kd import YOLOX_KD
from ..algorithm.searchbase import SearchBase


@DETECTORS.register_module()
class SearchableYOLOX_KD(SearchBase, YOLOX_KD):
    def __init__(self,
                 *args,
                 search_space=None,
                 bn_training_mode=True,
                 num_sample_training=4,
                 divisor=4,
                 retraining=False,
                 **kwargs
                 ):
        YOLOX_KD.__init__(self, *args, **kwargs)
        SearchBase.__init__(self, bn_training_mode=bn_training_mode, num_sample_training=num_sample_training, divisor=divisor, retraining=retraining)
        self._random_size_interval = self._random_size_interval*self.num_sample_training

        self.search_space = search_space
        if self.search_space:
            self.backbone_widen_factor_range = search_space['backbone_widen_factor_range']
            self.backbone_deepen_factor_range = search_space['backbone_deepen_factor_range']
            self.neck_widen_factor_range = search_space['neck_widen_factor_range']
            self.head_widen_factor_range = search_space['head_widen_factor_range']
        
    def sample_arch(self, mode='random'):
        assert mode in ('max', 'min', 'random')
        arch = {}
        if mode in ('max', 'min'):
            fn = eval(mode)
            arch['widen_factor_backbone'] = tuple([fn(self.backbone_widen_factor_range)]*5)
            arch['deepen_factor_backbone'] = tuple([fn(self.backbone_deepen_factor_range)]*4)
            arch['widen_factor_neck'] = tuple([fn(self.neck_widen_factor_range)]*8)
            arch['widen_factor_neck_out'] = max(self.head_widen_factor_range)
        elif mode == 'random':
            arch['widen_factor_backbone'] = tuple(random.choices(self.backbone_widen_factor_range, k=5))
            arch['deepen_factor_backbone'] = tuple(random.choices(self.backbone_deepen_factor_range, k=4))
            arch['widen_factor_neck'] = tuple(random.choices(self.neck_widen_factor_range, k=8))
            arch['widen_factor_neck_out'] = random.choice(self.head_widen_factor_range)
        else:
            raise NotImplementedError
        return arch

    def set_arch(self, arch_dict):
        self.backbone.set_arch(arch_dict, divisor=self.divisor)
        self.neck.set_arch(arch_dict, divisor=self.divisor)
        self.bbox_head.set_arch(arch_dict, divisor=self.divisor)