import warnings
import torch.nn as nn

from mmdet.models.builder import DETECTORS


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    model = DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

    if hasattr(cfg, 'quant_cfg') and cfg.quant_cfg['quant']:
        from .utils import extract_names, setattr_dot, getattr_dot, QConv2d, QReLU
        conv2d_names = []
        relu_name = []
        extract_names(model, nn.Conv2d, conv2d_names)
        extract_names(model, nn.ReLU, relu_name)
        for name in conv2d_names:
            old_m = getattr_dot(model, name)
            new_m = QConv2d.build_from_original(
                old_m, cfg.quant_cfg['w_bit'], cfg.quant_cfg['channel_wise'])
            setattr_dot(model, name, new_m)
        for name in relu_name:
            old_m = getattr_dot(model, name)
            new_m = QReLU.build_from_original(
                old_m, cfg.quant_cfg['a_bit'])
            setattr_dot(model, name, new_m)
    return model
