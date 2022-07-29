_base_ = './retinanet_r18_fpn_1x_voc.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet101')),
    neck=dict(in_channels=[256, 512, 1024, 2048]))