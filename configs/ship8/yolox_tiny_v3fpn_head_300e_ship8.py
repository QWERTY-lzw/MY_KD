_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/ship8.py']

# model settings
model = dict(
    type='YOLOV3',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.375),
    # neck=dict(
    #     type='YOLOXPAFPN',
    #     in_channels=[96, 192, 384],
    #     out_channels=96,
    #     num_csp_blocks=1),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[384, 192, 96],
        out_channels=[384, 192, 96]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=8,
        in_channels=[384, 192, 96],
        out_channels=[384, 192, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))


# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])

# runtime settings
max_epochs = 273
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(interval=50, metric=['bbox'])
num_last_epochs = 5
interval = 50
checkpoint_config = dict(interval=interval)
evaluation = dict(
    metric='mAP', 
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    save_best='mAP')


custom_hooks = [
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
]