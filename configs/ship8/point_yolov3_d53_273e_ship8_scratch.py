_base_ = 'yolov3_d53_273e_ship8_scratch.py'

# model settings
model = dict(
    type='Point_YOLOV3',
    bbox_head=dict(
        num_classes=5,
    ),
    point_head=dict(
        type='PointHead',
        in_channel=128,
        feat_channel=128,
        num_classes=3,
        test_cfg=dict(topk=100, local_maximum_kernel=3),
    ),
)

custom_hooks=[]
evaluation = dict(metric=['mAP', 'pts_mAP'])