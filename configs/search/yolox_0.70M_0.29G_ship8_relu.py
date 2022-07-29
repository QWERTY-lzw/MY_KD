_base_ = './yolox_0.70M_0.29G_ship8.py'

act_type = 'ReLU'

# model settings
model = dict(
    backbone=dict(act_cfg=dict(type=act_type)),
    neck=dict(act_cfg=dict(type=act_type)),
    bbox_head=dict(act_cfg=dict(type=act_type)))
