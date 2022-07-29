_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/mmdet_runtime.py'
]


from mmcv import ConfigDict
img_scale = (640, 640)
model = dict(
    type='mmdet.YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=80, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

algorithm_cfg = ConfigDict(
    type='AutoSlim',
    architecture=dict(type='MMDetArchitecture', model=model),
    pruner=dict(
        type='RatioPruner',
        ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                10 / 12, 11 / 12, 1.0)),
    retraining=False,
    bn_training_mode=True,
    input_shape=None)

from mmrazor.models.builder import build_algorithm
algorithm = build_algorithm(algorithm_cfg)