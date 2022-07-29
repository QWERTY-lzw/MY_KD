import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS

from ..quantizers import LSQQuantizerV1


@CONV_LAYERS.register_module('QConv2d')
class QConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        w_bit=8,
        channel_wise=True
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.w_bit = w_bit
        self.channel_wise = channel_wise
        scale_num = out_channels if channel_wise else 1
        self.w_quantizer = LSQQuantizerV1(self.w_bit, scale_num, True)

    def forward(self, x):
        w = self.w_quantizer(self.weight)
        out = F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)
        return out

    @classmethod
    def build_from_original(cls, m_fp, w_bit, channel_wise):
        m = cls(m_fp.in_channels, m_fp.out_channels, m_fp.kernel_size,
                m_fp.stride, m_fp.padding, m_fp.dilation, m_fp.groups,
                m_fp.bias is not None, m_fp.padding_mode,
                w_bit, channel_wise)
        return m

# class QLinear(nn.Linear):
#     def __init__(
#         self,
#         w_bit,
#         a_bit,
#         in_features,
#         out_features,
#         bias=True,
#         channel_wise=False
#     ):
#         super(QLinear, self).__init__(in_features, out_features, bias)
#         self.w_bit = w_bit
#         self.a_bit = a_bit
#         self.channel_wise = channel_wise
#         scale_num = out_features if channel_wise else 1
#         self.w_quantizer = LSQQuantizerV1(self.w_bit, scale_num, True)
#         self.a_quantizer = LSQQuantizerV1(self.a_bit, 1, False)


#     def forward(self, x):
#         x = self.a_quantizer(x)
#         w = self.w_quantizer(self.weight)
#         out = F.linear(x, w, self.bias)
#         return out
