import numpy as np
import torch
import torch.nn as nn
from mmdet_custom.models.utils import QReLU


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    if hasattr(conv, 'w_quantizer'):
        conv.w_quantizer.scale.data *= factor
    else:
        conv.weight = nn.Parameter(conv_w *
            factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv

def _fuse_conv_relu(conv, relu):
    if hasattr(relu, 'a_quantizer'):
        conv.oup_scale = relu.a_quantizer.scale
        conv.out_min.fill_(relu.a_quantizer.min_q)
        conv.out_max.fill_(relu.a_quantizer.max_q)
    return conv

def fuse_conv_bn_relu(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
        elif isinstance(child, QReLU):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_relu(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn_relu(child)
    return module

def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=False):
    shift_bits = np.floor(np.log2((2 ** mult_bits - 1) / fp32_scale))
    if limit:
        shift_bits[shift_bits > mult_bits] = mult_bits
    return shift_bits.astype('int32')

def get_scale_approximation_mult(fp32_scale, shift_bits):
    return np.floor(fp32_scale * (2 ** shift_bits)).astype('int32')

def get_scale_approximation_params(fp32_scale, mult_bits, limit=False):
    shift_bits = get_scale_approximation_shift_bits(
        fp32_scale, mult_bits, limit=limit)
    multiplier = get_scale_approximation_mult(fp32_scale, shift_bits)
    return multiplier, shift_bits

def approx_scale_as_mult_and_shift(fp32_scale, mult_bits, limit=False):
    """
    approx scale
    """
    multiplier, shift_bits = get_scale_approximation_params(
        fp32_scale, mult_bits, limit=limit)
    if not isinstance(shift_bits, torch.Tensor):
        shift_bits = torch.tensor(shift_bits)
    return multiplier, shift_bits


def lp_loss(pred, tgt, p=2.0):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).mean()


