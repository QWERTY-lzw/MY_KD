import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper
from onnx import numpy_helper


def build_torch_layers(layers):
    torch_layers = []
    for info in layers:
        if info['type'] == 'Conv':
            torch_layer = QConv2d
        elif info['type'] == 'Add':
            torch_layer = QAdd
        elif info['type'] == 'Concat':
            torch_layer = QConcat
        elif info['type'] == 'Focus':
            torch_layer = Focus
        elif info['type'] == 'SPP':
            torch_layer = SPP
        else:
            raise NotImplementedError
        torch_layers.append(torch_layer(info))
    return torch_layers


class QConv2d(nn.Module):
    def __init__(self, info):
        super(QConv2d, self).__init__()
        self.info = info
        extra_params = info['extra_params']
        self.weight_int = torch.as_tensor(extra_params['weight_int']).float()
        self.bias_int = torch.as_tensor(extra_params['bias_int']).float()
        self.A = torch.as_tensor(extra_params['A'])
        self.N = torch.as_tensor(extra_params['N'])
        self.out_min = torch.as_tensor(extra_params['min_yq'])
        self.out_max = torch.as_tensor(extra_params['max_yq'])

        for attr in self.info['attribute']:
            if attr.name == 'pads':
                self.padding = tuple([i for i in attr.ints[2:]])
            elif attr.name == 'strides':
                self.stride = tuple([i for i in attr.ints])
            elif attr.name == 'dilations':
                self.dilation = tuple([i for i in attr.ints])
            elif attr.name == 'group':
                self.groups = attr.i

    def forward(self, x):
        out = F.conv2d(x.int().float(), self.weight_int.int().float(),
                        self.bias_int.int().float(), self.stride, self.padding,
                        self.dilation, self.groups).round()
        out *= self.A.int().float().view(1, -1, 1, 1)
        out >>= self.N.int().float().view(1, -1, 1, 1)
        if self.out_min >= 0:
            out = out.round().clamp(self.out_min, self.out_max)
        return out


class QAdd(nn.Module):
    def __init__(self, info):
        super(QAdd, self).__init__()
        self.info = info
        extra_params = info['extra_params']
        self.out_min = self.out_max = None
        if 'min_yq' in extra_params:
            self.out_min = torch.as_tensor(extra_params['min_yq'])
        if 'max_yq' in extra_params:
            self.out_max = torch.as_tensor(extra_params['max_yq'])

    def forward(self, xs):
        out = 0
        for i in xs:
            out += i.int().float()
        if self.out_min is not None:
            out = out.clamp(min=self.out_min)
        if self.out_max is not None:
            out = out.clamp(max=self.out_max)
        return out


class QConcat(nn.Module):
    def __init__(self, info):
        super(QConcat, self).__init__()
        self.info = info
        extra_params = info['extra_params']
        self.out_min = self.out_max = None
        if 'min_yq' in extra_params:
            self.out_min = torch.as_tensor(extra_params['min_yq'])
        if 'max_yq' in extra_params:
            self.out_max = torch.as_tensor(extra_params['max_yq'])

    def forward(self, xs):
        xs = [i.int().float() for i in xs]
        out = torch.cat(xs, 1)
        if self.out_min is not None:
            out = out.clamp(min=self.out_min)
        if self.out_max is not None:
            out = out.clamp(max=self.out_max)
        return out


class Focus(nn.Module):
    def __init__(self, info):
        super(Focus, self).__init__()
        self.info = info
        extra_params = info['extra_params']
        self.out_min = self.out_max = None
        if 'min_yq' in extra_params:
            self.out_min = torch.as_tensor(extra_params['min_yq'])
        if 'max_yq' in extra_params:
            self.out_max = torch.as_tensor(extra_params['max_yq'])

    def forward(self, x):
        x = x.int().float()
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        if self.out_min is not None:
            out = out.clamp(min=self.out_min)
        if self.out_max is not None:
            out = out.clamp(max=self.out_max)
        return x


class Upsample(nn.Module):
    def __init__(self, info):
        super(Upsample, self).__init__()
        params = info['params']
        self.scale_factor = numpy_helper.to_array(params[0])[2:].astype('int8').tolist()

    def forward(self, x):
        x = x.int().float()
        out = F.upsample_nearest(x, scale_factor=self.scale_factor)
        return out

class SPP(nn.Module):
    def __init__(self, info):
        super(SPP, self).__init__()
        self.info = info
        kernel_sizes = (5, 9, 13)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        extra_params = info['extra_params']
        self.out_min = self.out_max = None
        if 'min_yq' in extra_params:
            self.out_min = torch.as_tensor(extra_params['min_yq'])
        if 'max_yq' in extra_params:
            self.out_max = torch.as_tensor(extra_params['max_yq'])

    def forward(self, x):
        x = x.int().float()
        out = [x] + [pool(x).int().float() for pool in self.poolings]
        out = torch.cat(out, 1)
        if self.out_min is not None:
            out = out.clamp(min=self.out_min)
        if self.out_max is not None:
            out = out.clamp(max=self.out_max)
        return out
