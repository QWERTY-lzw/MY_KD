_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/ship8.py']

custom_imports=dict(imports=['mmdet_custom.datasets', 'mmdet_custom.models', 'mmcv_custom.runner'], allow_failed_imports=False) 

img_scale = (256, 320)

# checkpoint_config = dict(type='CheckpointHook_nolog', interval=1)
checkpoint_config = dict(interval=50)

# based on yolox-tiny
default_widen_factor = 0.375
default_deepen_factor = 0.33

# [0.125, 0.25, 0.375, 0.5]
widen_factor_range = [1/6, 2/6, 3/6, 4/6, 5/6, 1]
# [0.11, 0.22, 0.33]
deepen_factor_range = [0, 1/3, 2/3, 1]
search_space = dict(
    backbone_widen_factor_range = widen_factor_range,
    backbone_deepen_factor_range = deepen_factor_range,
    neck_widen_factor_range = widen_factor_range,
    head_widen_factor_range = widen_factor_range
)

widen_factor_backbone = [1]*5
deepen_factor_backbone = [1]*4
widen_factor_neck = [1]*8
widen_factor_head = [1]*1

widen_factor_backbone = [default_widen_factor*alpha for alpha in widen_factor_backbone]
deepen_factor_backbone = [default_deepen_factor*alpha for alpha in deepen_factor_backbone]
in_channels = [int(c*alpha) for c,alpha in zip([256, 512, 1024], widen_factor_backbone[-3:])]
head_channels = int(256*default_widen_factor*widen_factor_head[0])

find_unused_parameters=True

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

optimizer_config = dict(grad_clip=None)

# model settings
model = dict(
    type='SearchableYOLOX',
    bn_training_mode=True,
    retraining=False,
    search_space=search_space,
    divisor=4,
    input_size=img_scale,
    random_size_range=(8, 8), 
    random_size_interval=10,
    backbone=dict(
        type='SearchableCSPDarknet',
        deepen_factor=deepen_factor_backbone,
        widen_factor=widen_factor_backbone),
    neck=dict(
        type='SearchableYOLOXPAFPNv2', # new using v1, that hidden sizes of neck only determined by both `widen_factor_neck`
        in_channels=in_channels,
        out_channels=head_channels,
        widen_factor=widen_factor_neck, 
        num_csp_blocks=1),
    bbox_head=dict(
        type='SearchableYOLOXHead', 
        num_classes=8,
        in_channels=head_channels,
        feat_channels=head_channels,
        ),
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
optimizer_config = dict(grad_clip=None)

max_epochs = 300
num_last_epochs = 5
resume_from = None
interval = 50

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)


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

custom_hooks = [
    # dict(
    #     type='YOLOXModeSwitchHook',
    #     num_last_epochs=num_last_epochs,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP')
log_config = dict(interval=10)




