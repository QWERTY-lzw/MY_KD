_base_ = 'yolox_tiny_300e_ship8.py'

# model settings
model = dict(
    type='Point_YOLOX',
    bbox_head=dict(
        num_classes=5,
    ),
    point_head=dict(
        type='PointHead',
        in_channel=96,
        feat_channel=96,
        num_classes=3,
        test_cfg=dict(topk=100, local_maximum_kernel=3),
    ),
)

evaluation = dict(metric=['mAP', 'pts_mAP'])
custom_hooks=[]