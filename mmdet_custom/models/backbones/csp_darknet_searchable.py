# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm, BatchNorm2d

from mmdet.models.builder import BACKBONES
from ..utils.usconv import set_channel_ratio, set_channels
from mmdet.models.backbones.csp_darknet import Focus, SPPBottleneck, CSPDarknet
from mmdet.models.utils import CSPLayer

@BACKBONES.register_module()
class SearchableCSPDarknet(CSPDarknet):
    arch_settings = {
        'P5': [[64, 128, 9, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 9, False, True]],
    }
    def __init__(self,
                 arch='P5',
                 deepen_factor=[1.0]*4,
                 widen_factor=[1.0]*5,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        BaseModule.__init__(self, init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor # todo

        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor[0]),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor[i])
            out_channels = int(out_channels * widen_factor[i + 1])
            num_blocks = max(round(num_blocks * deepen_factor[i]), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def set_arch(self, arch, divisor=8):
        # widen_factor = [c/d for c,d in zip(arch['widen_factor_backbone'], self.widen_factor)]
        # deepen_factor = [c/d for c,d in zip(arch['deepen_factor_backbone'], self.deepen_factor)]
        widen_factor = [c for c in arch['widen_factor_backbone']]
        deepen_factor = [c for c in arch['deepen_factor_backbone']]
        
        last_out = getattr(self, 'stem').conv.conv.in_channels
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            set_channel_ratio(layer, widen_factor[i], divisor=divisor)
            if layer_name == "stem":
                layer.conv.conv.in_channels = last_out
                last_out = layer.conv.conv.out_channels
            else:
                layer[0].conv.in_channels = last_out
                last_out = layer[-1].final_conv.bn.num_features
                layer[-1].blocks.num_layers = round(len(layer[-1].blocks) * deepen_factor[i - 1]) # deepfactor