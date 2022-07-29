_base_ = 'yolox_tiny_300e_ship8.py'
custom_imports=dict(imports='mmdet_custom.datasets', allow_failed_imports=False) 

widen_scale=1/3
# model settings
model = dict(
    backbone=dict(widen_factor=0.375*widen_scale, deepen_factor=0.1),
    neck=dict(
        in_channels=[round(96*widen_scale), round(192*widen_scale), round(384*widen_scale)],
        out_channels=round(96*widen_scale),
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=8, in_channels=round(96*widen_scale), feat_channels=round(96*widen_scale)),
)