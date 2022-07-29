_base_ = ['../../_base_/default_runtime.py', '../../_base_/datasets/ship8.py']

custom_imports=dict(imports=['mmdet_custom.datasets', 'mmrazor_custom.models.pruners'], allow_failed_imports=False) 

img_scale = (256, 320)
# model settings
model = dict(
    type='mmdet.YOLOX',
    input_size=img_scale,
    random_size_range=(8, 8), 
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.375*2),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96*2, 192*2, 384*2],
        out_channels=96*2,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=8, in_channels=96*2, feat_channels=96*2),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# optimizer
# default 4 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)
optimizer_config = None # for pruning algorithm https://github.com/open-mmlab/mmrazor/issues/117

max_epochs = 300
num_last_epochs = 5
resume_from = None
interval = 50

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = []
checkpoint_config = dict(interval=interval)
evaluation = dict(metric='mAP', interval=25)
log_config = dict(interval=50)

algorithm = dict(
    type='AutoSlim',
    architecture=dict(type='MMDetArchitecture', model=model),
    pruner=dict(
        # type='RatioPruner',
        type='DetRatioPruner',
        # except_start_keys=[
        #     'neck.',
        #     'bbox_head.',
        # ],
        ratios=[1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0]),
    retraining=False,
    bn_training_mode=True,
    input_shape=None)

# runner = dict(type='EpochBasedRunner', max_epochs=50)

use_ddp_wrapper = True
